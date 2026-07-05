"""CF fitting, validation, and body-map readout for one subject / hemisphere.

Validation follows the three-stage plan from the Methods section:

  (I)   Split-half noise ceiling -- odd vs even TypeD betas, Pearson r per
        vertex.  Both halves used the full GLMsingle pipeline (GLMdenoise +
        ridge), giving the highest-quality reliability estimate available
        without held-out data.

  (II)  Leave-one-run-out CV -- CF model fit on the average of 3 TypeB runs,
        tested on the single held-out TypeB run.  Repeated for all 4 runs and
        averaged.  CV-R² is then expressed relative to the noise ceiling.

  (III) Null-model comparison -- same LOSO procedure with the mean-S1 null
        (Haak et al., 2013) as the prediction model.  A vertex is considered
        to show topographically specific connectivity only if the CF model's
        CV-R² exceeds the null's.

TypeB (per-run) betas are used for the LOSO because single runs cannot reach
the full TypeD pipeline -- GLMdenoise and ridge regression both require >=2
runs for their internal leave-one-run-out cross-validation step.  TypeD
(odd/even and condavg) betas are used for the noise ceiling and the final
body-map readout because they are the highest-quality estimates available.
"""
import _compat  # noqa: F401  (numpy alias shim; must precede prfpy import)

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm

from prfpy.stimulus import CFStimulus
from prfpy.model import CFGaussianModel
from prfpy.fit import CFFitter

import config
import data_io


_RUNS = [f"run-{r}" for r in range(1, config.N_RUNS + 1)]


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Internal helpers
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

def _null_r2(target_data, design_matrix):
    """Signed R² of predicting each target from the mean S1 profile (the null)."""
    null_pred = np.nanmean(design_matrix, axis=0)      # (n_conditions,)
    zt        = zscore(target_data, axis=1)
    zp        = zscore(null_pred)
    r         = (zt @ zp) / target_data.shape[1]
    return np.sign(r) * r ** 2


def _split_half_reliability(subject, hemi):
    """Pearson r between odd- and even-run condition profiles per vertex.

    Both halves use the full TypeD pipeline (GLMdenoise + ridge), making
    this the highest-quality reliability estimate available without held-
    out data.  Returns (N_VERTS_PER_HEMI,), values in [-1, 1].
    """
    odd  = np.nan_to_num(data_io.load_betas(subject, hemi, split="odd"))
    even = np.nan_to_num(data_io.load_betas(subject, hemi, split="even"))
    zo, ze = zscore(odd, axis=1), zscore(even, axis=1)
    return (zo * ze).sum(axis=1) / odd.shape[1]


def _batched_quick_grid_fit(target_data, stim, model):
    """quick_grid_fit over target_data in batches (memory safety).

    An unbatched fit allocates an intermediate (n_candidates x n_targets)
    matrix that can exceed available RAM for large source regions (empirically
    ~16 GB for all-four-S1-subfields at full hemisphere scale).  Batching
    limits that matrix to config.BATCH_SIZE rows at a time.

    Returns (v0, sigma, grid_r2), each shape (n_targets,).
    v0 values are whole-brain vertex indices into s1_inds -- NOT positions
    within s1_inds.  Use np.searchsorted(s1_inds, v0) to convert.
    """
    n                              = target_data.shape[0]
    v0_parts, sigma_parts, r2_parts = [], [], []
    for start in range(0, n, config.BATCH_SIZE):
        end    = min(start + config.BATCH_SIZE, n)
        fitter = CFFitter(data=target_data[start:end], model=model,
                          n_jobs=config.N_JOBS)
        fitter.quick_grid_fit(config.SIGMA_GRID)
        p = fitter.quick_gridsearch_params
        v0_parts.append(p[:, 0].astype(int))
        sigma_parts.append(p[:, 1])
        r2_parts.append(p[:, 2])
    return (np.concatenate(v0_parts),
            np.concatenate(sigma_parts),
            np.concatenate(r2_parts))


def _gaussian_cv_prediction(v0, sigma, s1_inds, dist_mat, test_source):
    """CF-predicted profile for each target, using the FITTED sigma (not just v0).

    FIX (see project notes): a previous version of this function used
    test_source[pos, :] directly -- a raw single-vertex lookup, discarding the
    fitted sigma entirely. That is mathematically the sigma -> 0 special case
    of proper CF pooling (confirmed numerically: as sigma -> 0 this formula
    reduces exactly to the old single-vertex copy). For any vertex whose
    fitted sigma was not tiny, the old code compared the target's held-out
    data against a single noisy neighbouring vertex's held-out data, instead
    of the sigma-weighted pool of several source vertices that
    CFGaussianModel.return_prediction actually computes. That single-vertex
    comparison is far noisier, which systematically deflates xval_r2 and can
    manufacture spurious sparsity in the retained-vertex map independent of
    whether real topographic connectivity exists.

    This function reproduces prfpy's CFGaussianModel.return_prediction exactly
    (verified numerically to ~1e-6 against the real prfpy implementation),
    vectorised and batched across all targets at once for speed -- calling
    return_prediction in a per-vertex Python loop would additionally be slow,
    since it re-scans subsurface_verts with np.where on every call.

    Parameters
    ----------
    v0, sigma   : (n_targets,) fitted parameters from the TRAINING fit.
                  v0 are whole-brain vertex indices into s1_inds.
    s1_inds     : (n_source,) whole-brain indices of the source region.
    dist_mat    : (n_source, n_source) geodesic distances among source verts.
    test_source : (n_source, n_conditions) HELD-OUT run's source-region betas.

    Returns
    -------
    (n_targets, n_conditions) predicted condition profile per target,
    pooling the held-out source signal with a Gaussian kernel of the
    target's own fitted sigma, centred at its own fitted v0.
    """
    pos      = np.searchsorted(s1_inds, v0)                  # (n_targets,)
    n        = len(pos)
    preds    = np.empty((n, test_source.shape[1]), dtype=np.float64)

    for start in range(0, n, config.BATCH_SIZE):
        end = min(start + config.BATCH_SIZE, n)
        # rf: (batch, n_source) Gaussian weights, unnormalised (peak=1 at d=0),
        # matching prfpy's gauss1D_cart exactly. Per-row scale differences from
        # not normalising to sum-to-1 are removed by the zscore step in
        # _xval_r2_for_fold, so this matches prfpy's own output exactly.
        rf = np.exp(-(dist_mat[pos[start:end], :] ** 2)
                    / (2 * sigma[start:end, None] ** 2))
        preds[start:end] = rf @ test_source
    return preds


def _xval_r2_for_fold(v0, sigma, s1_inds, dist_mat, test_betas):
    """Evaluate sigma-weighted CF predictions against the held-out test betas.

    The prediction for each target vertex is the Gaussian-pooled condition
    profile (per its own fitted v0 AND sigma) computed from the TEST run's
    own S1 betas -- i.e. the learned topographic map is applied to the
    held-out source signal, not the training signal, ensuring the evaluation
    is genuinely out-of-sample. See _gaussian_cv_prediction for why sigma
    must be used here, not just v0.
    """
    test_source = test_betas[s1_inds, :]           # (n_source, n_cond)
    predicted   = _gaussian_cv_prediction(v0, sigma, s1_inds, dist_mat, test_source)
    zt          = zscore(test_betas, axis=1)
    zp          = zscore(predicted, axis=1)
    r           = np.nansum(zt * zp, axis=1) / test_betas.shape[1]
    return np.sign(r) * r ** 2


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Main pipeline
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

def fit_hemisphere(subject, hemi):
    """Fit + validate the CF model for one subject / hemisphere.

    Returns a DataFrame with one row per target vertex:
        vertex         -- 0-indexed whole-brain vertex number
        v0             -- best source centre (from TypeD condavg fit)
        sigma          -- CF size in mm (from TypeD condavg fit)
        grid_r2        -- in-sample R² (from TypeD condavg fit)
        xval_r2        -- mean CV-R² across 4 LOSO folds (TypeB betas)
        null_r2        -- mean null CV-R² across 4 LOSO folds (TypeB betas)
        noise_ceiling  -- split-half Pearson r (TypeD odd / even betas)
        bodypart       -- preferred condition index at v0 (0..N_CONDITIONS-1)
    """
    # --- shared geometry (exact geodesic; cached after first run) ---
    s1_inds  = data_io.get_s1_source_indices(hemi)
    dist_mat = data_io.geodesic_distance_matrix(hemi, s1_inds)

    # === Stage I: split-half noise ceiling (TypeD odd / even) =================
    print(f"  [{hemi}] Stage I: split-half reliability (TypeD odd / even)...")
    noise_ceil = _split_half_reliability(subject, hemi)

    # === Stages II + III: leave-one-run-out CV (TypeB per-run) ================
    # Detect which runs actually exist for THIS subject -- not every subject
    # has config.N_RUNS runs (e.g. sub-05 has only 3), same issue and same
    # fix pattern as derive_splits.py's _detect_n_runs.
    available_runs = [r for r in _RUNS
                      if data_io._betas_path(subject, hemi, r).exists()]
    if len(available_runs) < config.N_RUNS:
        print(f"  [{hemi}] WARNING: {len(available_runs)}/{config.N_RUNS} runs "
              f"available for {subject} -- LOSO uses only {available_runs}")
    if len(available_runs) < 2:
        raise ValueError(
            f"{subject} {hemi}: only {len(available_runs)} run(s) available -- "
            "leave-one-run-out CV needs at least 2.")

    print(f"  [{hemi}] Loading per-run betas ({len(available_runs)} runs available)...")
    run_betas = [
        np.nan_to_num(data_io.load_betas(subject, hemi, split=r))
        for r in available_runs
    ]

    cv_r2_folds, null_r2_folds = [], []
    for held_out_idx, held_out_run in enumerate(available_runs):
        print(f"  [{hemi}] Stages II+III: LOSO fold {held_out_idx+1}/{len(available_runs)} "
              f"(held out: {held_out_run})...")
        train_betas = np.mean(
            np.stack([b for i, b in enumerate(run_betas) if i != held_out_idx]),
            axis=0)                                # (N_VERTS, 22) -- average of other runs
        test_betas  = run_betas[held_out_idx]

        stim  = CFStimulus(data=train_betas, vertinds=s1_inds, distances=dist_mat)
        model = CFGaussianModel(stim)
        v0_fold, sigma_fold, _ = _batched_quick_grid_fit(train_betas, stim, model)

        cv_r2_folds.append(
            _xval_r2_for_fold(v0_fold, sigma_fold, s1_inds, dist_mat, test_betas))
        null_r2_folds.append(
            _null_r2(test_betas, stim.design_matrix))

    xval_r2 = np.mean(np.stack(cv_r2_folds),   axis=0)
    null_r2  = np.mean(np.stack(null_r2_folds), axis=0)

    # === Final body-map readout: TypeD condavg (most reliable estimate) ========
    print(f"  [{hemi}] Final fit on TypeD condavg for body-map readout...")
    betas_condavg = np.nan_to_num(data_io.load_betas(subject, hemi))
    stim_final    = CFStimulus(data=betas_condavg, vertinds=s1_inds, distances=dist_mat)
    model_final   = CFGaussianModel(stim_final)
    v0, sigma, grid_r2 = _batched_quick_grid_fit(betas_condavg, stim_final, model_final)

    source_pref   = np.argmax(stim_final.design_matrix, axis=1)
    pos_in_source = np.searchsorted(s1_inds, v0)
    bodypart      = source_pref[pos_in_source]

    return pd.DataFrame({
        "vertex":        np.arange(config.N_VERTS_PER_HEMI),
        "v0":            v0,
        "sigma":         sigma,
        "grid_r2":       grid_r2,
        "xval_r2":       xval_r2,
        "null_r2":       null_r2,
        "noise_ceiling": noise_ceil,
        "bodypart":      bodypart,
    })


def bodymap_from_params(params):
    """Thresholded body-part map (N_VERTS_PER_HEMI,), NaN where unreliable.

    A vertex is retained only if BOTH conditions hold:
      (a) split-half reliability  >= config.SPLIT_HALF_THRESHOLD
      (b) xval_r2 - null_r2       >  config.RELIABILITY_THRESHOLD
    Condition (a) masks out vertices where no model could be expected to fit
    (non-interpretable, not model failures; Schoppe et al., 2016).
    Condition (b) further restricts to topographically specific connectivity.
    """
    noise_ok = params["noise_ceiling"].values >= config.SPLIT_HALF_THRESHOLD
    cv_ok    = ((params["xval_r2"].values - params["null_r2"].values)
                > config.RELIABILITY_THRESHOLD)
    bm       = params["bodypart"].values.astype(float).copy()
    bm[~(noise_ok & cv_ok)] = np.nan
    return bm