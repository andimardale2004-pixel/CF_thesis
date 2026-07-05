# %% 1. Imports + effector definitions
import numpy as np
import pandas as pd
import cortex

import config
import data_io

UNIQUE_EFFECTORS = [
    "leg",
    "hand_fing5", "hand_fing4", "hand_fing3", "hand_fing2", "hand_fing1",
    "eyebrows", "eyes", "mouth", "tongue",
]
N_EFFECTORS = len(UNIQUE_EFFECTORS)   # 10

EFFECTOR_GROUPS = {
    eff: [i for i, c in enumerate(config.CONDITION_LABELS) if eff in c]
    for eff in UNIQUE_EFFECTORS
}
assert sorted(sum(EFFECTOR_GROUPS.values(), [])) == list(range(config.N_CONDITIONS)), \
    "EFFECTOR_GROUPS must partition all 22 conditions exactly once"


# %% 2. Load the LOO results so we can label each participant's layer
#       with their reproducibility score (best first in the dropdown).
import pandas as pd
loo_path = config.OUTPUT_DIR / "results_loo_reproducibility.csv"
if loo_path.exists():
    loo_df   = pd.read_csv(loo_path)
    mean_loo = loo_df.groupby("subject")["loo_r"].mean()
    sub_order = mean_loo.sort_values(ascending=False).index.tolist()
    print("Subject order (best LOO r first):",
          {s: f"{mean_loo[s]:.3f}" for s in sub_order})
else:
    sub_order = config.SUBJECTS
    mean_loo  = {s: float("nan") for s in config.SUBJECTS}
    print("No LOO CSV found -- using default subject order.")


# %% 3. Helper: load one subject's S1 betas (condavg TypeD)
def s1_subject_betas(subject, hemi):
    s1_inds = data_io.get_s1_source_indices(hemi)
    betas   = np.nan_to_num(data_io.load_betas(subject, hemi))[s1_inds, :]
    return s1_inds, betas


# %% 4. Collapse laterality → winner-take-all with margin threshold
MARGIN_THRESHOLD_PERCENTILE = 0   # 0 = keep all vertices; raise to clean noise

def winner_take_all_s1(avg_s1_betas, s1_inds, hemi):
    """Collapse laterality → 10 effectors, argmax with margin threshold."""
    collapsed = np.stack([
        avg_s1_betas[:, EFFECTOR_GROUPS[eff]].mean(axis=1)
        for eff in UNIQUE_EFFECTORS
    ], axis=1)

    top2      = np.partition(collapsed, -2, axis=1)[:, -2:]
    margin    = top2[:, 1] - top2[:, 0]
    threshold = float(np.percentile(margin, MARGIN_THRESHOLD_PERCENTILE))
    confident = margin > threshold

    winner           = np.full(len(s1_inds), np.nan)
    winner[confident] = np.argmax(collapsed[confident], axis=1).astype(float)

    bodymap          = np.full(config.N_VERTS_PER_HEMI, np.nan)
    bodymap[s1_inds] = winner
    blank            = np.full(config.N_VERTS_PER_HEMI, np.nan)
    return (np.concatenate([bodymap, blank]) if hemi == "left"
            else np.concatenate([blank, bodymap]))


# %% 5. Build one layer per subject (best LOO r first in the dropdown)
print("Building body map layers for all subjects...")
data_layers = {}
for sub in sub_order:
    s1_inds_L, avg_L = s1_subject_betas(sub, "left")
    s1_inds_R, avg_R = s1_subject_betas(sub, "right")
    bm_L = winner_take_all_s1(avg_L, s1_inds_L, "left")
    bm_R = winner_take_all_s1(avg_R, s1_inds_R, "right")
    bm   = np.where(np.isfinite(bm_L), bm_L, bm_R)

    r_str = f"LOO r={mean_loo[sub]:.3f}" if not np.isnan(mean_loo[sub]) else ""
    label = f"{sub}  ({r_str})  S1 body map" if r_str else f"{sub}  S1 body map"
    data_layers[label] = cortex.Vertex(
        bm, config.PYCORTEX_SUBJECT,
        vmin=0, vmax=N_EFFECTORS - 1, cmap="rainbow")
    print(f"  {sub}: done")


# %% 6. Render
out_path = config.OUTPUT_DIR / "panel_a_body_map_static"
cortex.webgl.make_static(outpath=str(out_path), data=data_layers)
print(f"\nStatic viewer saved to: {out_path}")
print("Subjects are ordered best → worst LOO reproducibility.")


# %% 7. Live interactive viewer
viewer = cortex.webshow(data_layers)
print("\nWebGL viewer launched in browser.")
input("Viewer running -- press Enter here (in the terminal) when done "
     "exploring, to shut it down and exit...")