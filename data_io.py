"""Data loading: betas, S1 source vertices, geodesic distance matrices."""
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import config

_HEMI = {"left": "L", "right": "R"}
_MODEL_TYPE_D = "TYPED_FITHRF_GLMDENOISE_RR"    # TypeD = full pipeline (GLMdenoise + ridge);
_MODEL_TYPE_B = "TYPEB_FITHRF"                  # TypeB = HRF fitting only; Produces single runs; 
_MODEL = _MODEL_TYPE_D                          # Backward compatible with other scripts 
_ODD_EVEN    = {"odd": "oddruns", "even": "evenruns"}
_RUN_SPLITS  = {f"run-{r}" for r in range(1, config.N_RUNS + 1)}


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# GIFTI helpers
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def _gifti_to_array(path):
    """Load a .func.gii / .shape.gii -> (n_vertices, n_maps) float32 array."""
    g = nib.load(str(path))
    return np.column_stack([d.data for d in g.darrays]).astype(np.float32)


def _betas_path(subject, hemi, split=None):
    H    = _HEMI[hemi]
    # condavg lives in BETA_DIR; everything split-specific lives in JOINT_BETA_DIR
    root = config.BETA_DIR if split is None else config.JOINT_BETA_DIR
    base = root / subject / "hcp_59k"

    if split is None:
        fname = (f"{subject}_{_MODEL_TYPE_D}_betasmd_condavg"
                 f"_space-fsLR_den-59k_hemi-{H}.func.gii")
    elif split in _ODD_EVEN:
        tag   = _ODD_EVEN[split]
        fname = (f"{subject}_{_MODEL_TYPE_D}_betasmd_{tag}"
                 f"_space-fsLR_den-59k_hemi-{H}.func.gii")
    elif split in _RUN_SPLITS:
        # Also TypeD quality -- single-trial betas from the joint TypeD run,
        # averaged per run.  _MODEL_TYPEB alias kept in constants but not used here.
        fname = (f"{subject}_{_MODEL_TYPE_D}_betasmd_{split}"
                 f"_space-fsLR_den-59k_hemi-{H}.func.gii")
    else:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Use None, 'odd', 'even', or one of {sorted(_RUN_SPLITS)}.")
    return base / fname

def load_atlasroi(hemi):
    """Medial-wall mask for one hemisphere: True = cortical vertex to keep.

    Uses the {L,R}.atlasroi.59k_fs_LR.shape.gii files copied by vol2fslr59k.sh
    (identical across subjects, so the first match is fine).
    """
    H = _HEMI[hemi]
    roi = next(Path(config.BETA_DIR).rglob(f"{H}.atlasroi.59k_fs_LR.shape.gii"))
    return _gifti_to_array(roi).ravel() > 0.5


def load_betas(subject, hemi, split=None):
    """Return (N_VERTS_PER_HEMI, N_CONDITIONS) condition betas for one hemisphere.
 
    Medial-wall vertices (atlasroi == 0) are set to NaN. The files already encode
    the medial wall as NaN too, so this mainly guards against partial coverage.
    """
    betas = _gifti_to_array(_betas_path(subject, hemi, split))
    assert betas.shape == (config.N_VERTS_PER_HEMI, config.N_CONDITIONS), (
        f"{subject} {hemi} split={split}: got {betas.shape}, expected "
        f"({config.N_VERTS_PER_HEMI}, {config.N_CONDITIONS})")
    try:
        keep         = load_atlasroi(hemi)
        betas[~keep, :] = np.nan
    except StopIteration:
        pass  # no atlasroi file found; rely on the file's own NaNs
    return betas


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# S1 source region
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def _parc_path(hemi):
    """Per-hemisphere parcellation path. config.PARCELLATION may contain '{H}'."""
    H = _HEMI[hemi]
    p = str(config.PARCELLATION)
    return Path(p.format(H=H) if "{H}" in p else p)

S1_AREA_KEYS = {"3a": 53, "3b": 9, "1": 51, "2": 52}  
    # somatosensory:
    #'CS1_4': 8, 'CS2_3a': 53, 'CS3_3b': 9, 'CS4_1': 51, 'CS5_2': 52,
    # auditory:
    #'A1': 24, 'PBelt': 124, 'MBelt': 173, 'LBelt': 174, '52': 103, 'RI': 104,
    # low-level visual:
    #'V1': 1, 'V2': 4, 'V3': 5,
    # mid-level and high-level visual:
    #'V3A': 13, 'V3B': 19, 'IPS1': 17, 'LIPv': 48, 'LIPd': 95,
    #'VIP': 49, 'FEF': 10, 'MST': 2, 'MT': 23, 'LO1': 20, 'LO2': 21, 'LO3': 159
_HEMI_OFFSET = {"left": 180, "right": 0}

def get_s1_source_indices(hemi):
    """Integer vertex indices of the S1 source region for one hemisphere.
 
    Reads a per-hemisphere HCP-MMP label file (.label.gii) on the fsLR 59k mesh
    and selects the areas named in config.S1_AREA_NAMES (e.g. '3b', '1').
    """
    g      = nib.load(str(_parc_path(hemi)))
    raw    = np.asarray(g.darrays[0].data).astype(int)
    offset = _HEMI_OFFSET[hemi]
 
    wanted = [S1_AREA_KEYS[area] + offset for area in config.S1_AREA_NAMES]
    src    = np.where(np.isin(raw, wanted))[0]
    if src.size == 0:
        raise ValueError(
            f"No S1 vertices found for {hemi} / {config.S1_AREA_NAMES} "
            f"(looked for keys {wanted}). Check config.PARCELLATION path.")
    return src.astype(int)


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.
# Geodesic distance matrix (cached)
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.
def geodesic_distance_matrix(hemi, source_inds, recompute=False):
    """NxN geodesic distances (mm) among source vertices, cached to disk.
    Using pygeodesic (exact Mitchell-Mount-Papadimitriou algorithm), NOT
    pycortex's built-in geodesic_distance -- because that method uses an 
    APPROXIMATE heat-diffusion algorithm which produced geodesic
    distances shorter than the straight-line Euclidean distance for the same
    vertex pairs -- mathematically impossible result for a true geodesic,
    confirmed at ~99.7% of S1 vertices during QC.
    """
    import cortex
    import pygeodesic.geodesic as pgeo
 
    area_tag = "-".join(config.S1_AREA_NAMES)
    cache    = (config.DIST_CACHE /
                f"{config.PYCORTEX_SUBJECT}_{hemi}_s1_{area_tag}_exact_dist.npy")
    if cache.exists() and not recompute:
        return np.load(cache)
 
    pts, polys = cortex.db.get_surf(config.PYCORTEX_SUBJECT, "fiducial", hemi)
    geoalg     = pgeo.PyGeodesicAlgorithmExact(pts, polys)
 
    dm = np.vstack([
        geoalg.geodesicDistances(np.array([int(v)]), source_inds)[0]
        for v in tqdm(source_inds, desc=f"geodesic (exact) {hemi}")
    ])
    np.save(cache, dm)             
    return dm                      
