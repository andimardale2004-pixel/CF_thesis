"""Group-level results analysis: body map, summary statistics, cross-subject reproducibility"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as _pearsonr  # kept for potential future use

import _compat   # noqa: F401 -- must precede prfpy import
import cortex
import config
import data_io
import cf_pipeline


UNIQUE_EFFECTORS = [
    "leg",
    "hand_fing5", "hand_fing4", "hand_fing3", "hand_fing2", "hand_fing1",
    "eyebrows", "eyes", "mouth", "tongue",
]
N_EFFECTORS = len(UNIQUE_EFFECTORS)   # 10

# Map each of the 22 raw condition indices → its unique effector index (0-9)
COND_TO_EFFECTOR = np.full(config.N_CONDITIONS, -1, dtype=int)
for eff_idx, eff in enumerate(UNIQUE_EFFECTORS):
    for cond_idx, cond in enumerate(config.CONDITION_LABELS):
        if eff in cond:
            COND_TO_EFFECTOR[cond_idx] = eff_idx
assert (COND_TO_EFFECTOR >= 0).all(), "every condition must map to an effector"


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Helpers
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def _cache_path(subject, hemi):
    return config.OUTPUT_DIR / f"{subject}_{hemi}_params.parquet"


def load_or_compute(subject, hemi):
    """Load cached params parquet, or run fit_hemisphere and cache it."""
    p = _cache_path(subject, hemi)
    if p.exists():
        return pd.read_parquet(p)
    print(f"  [{subject} {hemi}] no cached parquet -- running fit_hemisphere...")
    params = cf_pipeline.fit_hemisphere(subject, hemi)
    params.to_parquet(p, index=False)
    return params


def retained_mask(params):
    """Boolean array: True where both validation gates pass."""
    noise_ok = params["noise_ceiling"].values >= config.SPLIT_HALF_THRESHOLD
    cv_ok    = ((params["xval_r2"].values - params["null_r2"].values)
                > config.RELIABILITY_THRESHOLD)
    return noise_ok & cv_ok


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 1. Load all params
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
print("Loading / computing params for all subjects...")
all_params = {}   # (subject, hemi) -> DataFrame
for sub in config.SUBJECTS:
    for hemi in config.HEMIS:
        print(f"  {sub} {hemi}...", end=" ", flush=True)
        all_params[(sub, hemi)] = load_or_compute(sub, hemi)
        print("OK")
print(f"Loaded {len(all_params)} subject/hemisphere pairs.\n")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 2. Summary statistics table
#
# frac_retained (whole hemisphere) is diluted by denominator: most of a
# hemisphere isn't S1, so a low whole-hemisphere fraction can still mean
# most of S1 itself is retained. frac_retained_S1 disambiguates this --
# report BOTH, since they answer different questions (S1 pipeline validity
# vs. how much of the whole hemisphere shows topographic connectivity).
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
print("=== Summary statistics ===")
rows = []
for sub in config.SUBJECTS:
    for hemi in config.HEMIS:
        p         = all_params[(sub, hemi)]
        mask      = retained_mask(p)
        adv       = p["xval_r2"].values - p["null_r2"].values

        s1_inds   = data_io.get_s1_source_indices(hemi)
        s1_mask   = np.zeros(len(p), dtype=bool)
        s1_mask[s1_inds] = True

        rows.append({
            "subject":              sub,
            "hemi":                 hemi,
            "n_vertices":           len(p),
            "n_s1_vertices":        int(s1_mask.sum()),
            "noise_ceil_median":    float(np.nanmedian(p["noise_ceiling"])),
            "noise_ceil_p90":       float(np.nanpercentile(p["noise_ceiling"], 90)),
            "frac_noise_ok":        float((p["noise_ceiling"] >= config.SPLIT_HALF_THRESHOLD).mean()),
            "frac_cv_ok":           float((adv > config.RELIABILITY_THRESHOLD).mean()),
            "frac_retained":        float(mask.mean()),
            "n_retained":           int(mask.sum()),
            # S1-restricted: same gates, but only counting S1 source vertices.
            # Distinguishes "sparse everywhere" from "dense in S1, sparse outside".
            "frac_retained_S1":     float(mask[s1_mask].mean()) if s1_mask.sum() > 0 else np.nan,
            "n_retained_S1":        int(mask[s1_mask].sum()),
            "mean_advantage":       float(np.nanmean(adv[mask])) if mask.sum() > 0 else np.nan,
        })

summary = pd.DataFrame(rows)
summary_path = config.OUTPUT_DIR / "results_summary.csv"
summary.to_csv(summary_path, index=False)
print(summary.groupby("hemi")[
    ["noise_ceil_median", "noise_ceil_p90",
     "frac_retained", "n_retained",
     "frac_retained_S1", "n_retained_S1",
     "mean_advantage"]
].mean().round(3).to_string())
print("\n(frac_retained_S1 restricts the same two gates to S1 source vertices")
print(" only -- compare against frac_retained to see how much of the low")
print(" whole-hemisphere fraction is just denominator dilution.)")


print("\n=== Threshold sweep (whole hemisphere, both hemispheres pooled) ===")
nc_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
cv_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
print(f"{'noise_ceil>=':>14} {'cv_adv>':>10} {'n_retained':>12} {'frac':>8}")
all_nc_vals  = np.concatenate([all_params[(s, h)]["noise_ceiling"].values
                               for s in config.SUBJECTS for h in config.HEMIS])
all_adv_vals = np.concatenate([
    (all_params[(s, h)]["xval_r2"].values - all_params[(s, h)]["null_r2"].values)
    for s in config.SUBJECTS for h in config.HEMIS])
n_total = len(all_nc_vals)
for nc_t, cv_t in zip(nc_grid, cv_grid):
    keep = (all_nc_vals >= nc_t) & (all_adv_vals > cv_t)
    print(f"{nc_t:>14.2f} {cv_t:>10.2f} {int(keep.sum()):>12,} {keep.mean():>8.3f}")
print("(current config thresholds: "
     f"noise_ceil>={config.SPLIT_HALF_THRESHOLD}, cv_adv>{config.RELIABILITY_THRESHOLD})\n")
print(f"\nPer-subject table saved to: {summary_path}\n")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 3. Cross-subject reproducibility (leave-one-out CV-advantage correlation)
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
print("=== Cross-subject reproducibility (LOO Pearson r on CV-advantage) ===")
loo_rows = []
for hemi in config.HEMIS:
    # Stack advantage maps: (n_subjects, n_verts)
    adv_stack = np.stack([
        (all_params[(sub, hemi)]["xval_r2"].values -
         all_params[(sub, hemi)]["null_r2"].values)
        for sub in config.SUBJECTS
    ])

    loo_r = []
    for i, sub in enumerate(config.SUBJECTS):
        group_loo = np.delete(adv_stack, i, axis=0).mean(axis=0)
        # Correlate over valid (finite) vertices
        valid = np.isfinite(adv_stack[i]) & np.isfinite(group_loo)
        if valid.sum() < 10:
            loo_r.append(np.nan)
            continue
        r = np.corrcoef(adv_stack[i, valid], group_loo[valid])[0, 1]
        loo_r.append(r)
        loo_rows.append({"subject": sub, "hemi": hemi, "loo_r": r})

    mean_r = np.nanmean(loo_r)
    sd_r   = np.nanstd(loo_r)
    print(f"  [{hemi}]  mean LOO r = {mean_r:.3f}  SD = {sd_r:.3f}  "
          f"(range {np.nanmin(loo_r):.3f} – {np.nanmax(loo_r):.3f})")

loo_df = pd.DataFrame(loo_rows)
loo_path = config.OUTPUT_DIR / "results_loo_reproducibility.csv"
loo_df.to_csv(loo_path, index=False)
print(f"LOO table saved to: {loo_path}\n")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 4. Individual CF body maps -- one layer per subject in the viewer
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
print("=== Building individual CF body maps for all subjects ===")

# Order best → worst by LOO r so the best subject appears first in the dropdown
mean_loo  = loo_df.groupby("subject")["loo_r"].mean()
sub_order = mean_loo.sort_values(ascending=False).index.tolist()
print(f"Subject order (best LOO r first): "
      f"{[f'{s}: {mean_loo[s]:.3f}' for s in sub_order]}\n")

def subject_cf_bodymap(subject):
    """Individual CF body map for one subject, both hemispheres concatenated.

    Returns (bodymap_whole, adv_whole):
        bodymap_whole -- (2*N_VERTS_PER_HEMI,) effector index 0-9, NaN outside
                         retained vertices
        adv_whole     -- (2*N_VERTS_PER_HEMI,) CV advantage, NaN where not finite
    """
    parts = {}
    for hemi in config.HEMIS:
        p   = all_params[(subject, hemi)]
        mask = retained_mask(p)
        bp   = p["bodypart"].values.astype(float)
        adv  = p["xval_r2"].values - p["null_r2"].values

        bodymap          = np.full(config.N_VERTS_PER_HEMI, np.nan)
        valid            = mask & np.isfinite(bp) & (bp >= 0)
        bodymap[valid]   = COND_TO_EFFECTOR[bp[valid].astype(int)].astype(float)

        adv_map          = np.full(config.N_VERTS_PER_HEMI, np.nan)
        adv_map[np.isfinite(adv)] = adv[np.isfinite(adv)]

        parts[hemi]      = (bodymap, adv_map)

    bm_whole  = np.concatenate([parts["left"][0],  parts["right"][0]])
    adv_whole = np.concatenate([parts["left"][1],  parts["right"][1]])
    n_ret     = int(np.isfinite(bm_whole).sum())
    r_str     = f"{mean_loo[subject]:.3f}"
    print(f"  {subject} (LOO r={r_str}): {n_ret} vertices retained")
    return bm_whole, adv_whole


# %% Build all layers up front (fast -- just array operations on cached params)
all_adv_finite = np.concatenate([
    (all_params[(s, h)]["xval_r2"].values - all_params[(s, h)]["null_r2"].values)
    for s in config.SUBJECTS for h in config.HEMIS
])
adv_vmax = float(np.nanpercentile(all_adv_finite[np.isfinite(all_adv_finite)], 95))
adv_vmin = float(np.nanpercentile(all_adv_finite[np.isfinite(all_adv_finite)], 5))

data_layers = {}
for sub in sub_order:
    bm_whole, adv_whole = subject_cf_bodymap(sub)
    r_label = f"{mean_loo[sub]:.3f}"
    data_layers[f"{sub}  (LOO r={r_label})  body map"] = cortex.Vertex(
        bm_whole, config.PYCORTEX_SUBJECT,
        vmin=0, vmax=N_EFFECTORS - 1, cmap="rainbow")
    data_layers[f"{sub}  (LOO r={r_label})  CV advantage"] = cortex.Vertex(
        adv_whole, config.PYCORTEX_SUBJECT,
        vmin=0, vmax=0.5, cmap="coolwarm")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 5. Save a quick matplotlib summary figure
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel A: noise ceiling distribution (pooled)
all_nc = np.concatenate([
    all_params[(sub, hemi)]["noise_ceiling"].values
    for sub in config.SUBJECTS for hemi in config.HEMIS
])
axes[0].hist(all_nc[np.isfinite(all_nc)], bins=50, color="steelblue")
axes[0].axvline(config.SPLIT_HALF_THRESHOLD, color="red", lw=1.5,
                label=f"threshold = {config.SPLIT_HALF_THRESHOLD}")
axes[0].set_xlabel("Split-half reliability (r)")
axes[0].set_ylabel("# vertices")
axes[0].set_title("Noise ceiling (all subjects)")
axes[0].legend(fontsize=8)

# Panel B: CV-advantage distribution (retained vertices only)
all_adv = np.concatenate([
    (all_params[(sub, hemi)]["xval_r2"].values -
     all_params[(sub, hemi)]["null_r2"].values)[retained_mask(all_params[(sub, hemi)])]
    for sub in config.SUBJECTS for hemi in config.HEMIS
])
axes[1].hist(all_adv[np.isfinite(all_adv)], bins=50, color="teal")
axes[1].axvline(0, color="k", lw=1)
axes[1].set_xlabel("CV-R² − null-R² (advantage)")
axes[1].set_title("CV advantage (retained vertices)")

# Panel C: LOO reproducibility per subject
for hemi, col in zip(config.HEMIS, ["steelblue", "coral"]):
    sub_r = loo_df[loo_df["hemi"] == hemi]["loo_r"].values
    axes[2].bar(
        [f"{s[-2:]}\n{hemi[:1]}" for s in loo_df[loo_df["hemi"] == hemi]["subject"]],
        sub_r, color=col, alpha=0.7, label=hemi)
axes[2].axhline(0, color="k", lw=0.8)
axes[2].set_ylabel("LOO Pearson r")
axes[2].set_title("Cross-subject reproducibility (LOO)")
axes[2].legend(fontsize=8)
axes[2].tick_params(axis="x", labelsize=7)

fig.tight_layout()
fig_path = config.OUTPUT_DIR / "results_summary_figure.png"
fig.savefig(fig_path, dpi=200)
plt.close(fig)
print(f"Summary figure saved to: {fig_path}\n")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 6. Render: S1-blanked group advantage map + per-subject layers
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
print(f"=== Building S1-blanked group advantage map ===")

# S1 masks for both hemispheres
s1_masks = {}
for hemi in config.HEMIS:
    s1_inds    = data_io.get_s1_source_indices(hemi)
    s1_mask    = np.zeros(config.N_VERTS_PER_HEMI, dtype=bool)
    s1_mask[s1_inds] = True
    s1_masks[hemi]   = s1_mask

# Group-average advantage with S1 blanked, both hemispheres
adv_blanked = {}
for hemi in config.HEMIS:
    stack = []
    for sub in config.SUBJECTS:
        p   = all_params[(sub, hemi)]
        adv = (p["xval_r2"].values - p["null_r2"].values).astype(float)
        adv[s1_masks[hemi]] = np.nan     # blank source region
        stack.append(adv)
    adv_blanked[hemi] = np.nanmean(np.stack(stack), axis=0)

group_adv_blanked_whole = np.concatenate(
    [adv_blanked["left"], adv_blanked["right"]])

# Scale to the non-S1 distribution so weak outside-S1 structure is visible
finite_outside = group_adv_blanked_whole[np.isfinite(group_adv_blanked_whole)]
grp_vmax = float(np.nanpercentile(finite_outside, 95))
grp_vmin = 0.0   # only show positive advantage (CF beats null)
print(f"Group advantage (S1 blanked): vmin={grp_vmin:.3f}  vmax={grp_vmax:.3f}")

# Also build per-subject S1-blanked layers using the same scale
for sub in sub_order:
    adv_sub_parts = []
    for hemi in config.HEMIS:
        p   = all_params[(sub, hemi)]
        adv = (p["xval_r2"].values - p["null_r2"].values).astype(float)
        adv[s1_masks[hemi]] = np.nan
        adv_sub_parts.append(adv)
    adv_sub_whole = np.concatenate(adv_sub_parts)
    r_label = f"{mean_loo[sub]:.3f}"
    data_layers[f"{sub}  (LOO r={r_label})  CV advantage (S1 blanked)"] = cortex.Vertex(
        adv_sub_whole, config.PYCORTEX_SUBJECT,
        vmin=grp_vmin, vmax=grp_vmax, cmap="magma")

# Prepend the group layer so it appears first in the dropdown
group_layer = {
    "GROUP  -- mean CV advantage (S1 blanked)": cortex.Vertex(
        group_adv_blanked_whole, config.PYCORTEX_SUBJECT,
        vmin=grp_vmin, vmax=grp_vmax, cmap="magma"),
}
data_layers = {**group_layer, **data_layers}

print(f"=== Rendering {len(data_layers)} layers in one viewer ===")
out_path = config.OUTPUT_DIR / "cf_results_all_subjects_static"
cortex.webgl.make_static(outpath=str(out_path), data=data_layers)
print(f"Static viewer saved to: {out_path}")
print("Dropdown order: group map first, then per-subject body maps and")
print("S1-blanked advantage maps, ordered best → worst LOO r.")

viewer = cortex.webshow(data_layers)
print("\nWebGL viewer launched.")
input("Press Enter here to shut down the viewer and exit...")