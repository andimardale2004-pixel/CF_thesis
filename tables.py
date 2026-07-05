import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import config

# APA-style rcParams
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.serif":           ["Arial"],
    "font.size":            9,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "axes.labelpad":        4,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "legend.frameon":       False,
    "legend.fontsize":      8,
    "figure.dpi":           300,
})

COL_HIST   = "#57b4e6"   
COL_ADV    = "#26e09f"   
COL_LEFT   = "#1333d1"   
COL_RIGHT  = "#ba0f15"   
COL_THRESH = "#d13238"   

print("Loading parquet files...")
nc_all, adv_retained = [], []
for sub in config.SUBJECTS:
    for hemi in config.HEMIS:
        p = config.OUTPUT_DIR / f"{sub}_{hemi}_params.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"No parquet for {sub} {hemi}. Run run.py first.")
        df = pd.read_parquet(p)
        nc_all.append(df["noise_ceiling"].values)
        adv = df["xval_r2"].values - df["null_r2"].values
        mask = ((df["noise_ceiling"].values >= config.SPLIT_HALF_THRESHOLD) &
                (adv > config.RELIABILITY_THRESHOLD))
        adv_retained.append(adv[mask])

nc_all      = np.concatenate(nc_all)
adv_retained = np.concatenate(adv_retained)
print(f"  noise ceiling pooled: {len(nc_all):,} vertices")
print(f"  CV advantage (retained): {len(adv_retained):,} vertices")

loo_df   = pd.read_csv(config.OUTPUT_DIR / "results_loo_reproducibility.csv")
loo_left  = (loo_df[loo_df["hemi"] == "left"]
             .set_index("subject").reindex(config.SUBJECTS)["loo_r"].values)
loo_right = (loo_df[loo_df["hemi"] == "right"]
             .set_index("subject").reindex(config.SUBJECTS)["loo_r"].values)
sub_labels = [s.replace("sub-", "") for s in config.SUBJECTS]


def panel_label(ax, letter):
    ax.text(-0.18, 1.10, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

def make_panel_A(ax):
    ax.hist(nc_all[np.isfinite(nc_all)], bins=60,
            color=COL_HIST, edgecolor="none", linewidth=0)
    ax.axvline(config.SPLIT_HALF_THRESHOLD, color=COL_THRESH,
               lw=1.2, label=f"{config.SPLIT_HALF_THRESHOLD}")
    ax.set_xlabel("Split-half reliability (r)")
    ax.set_ylabel("Vertex count")
    ax.set_title("Noise ceiling", pad=6)
    ax.legend()
    panel_label(ax, "A")

def make_panel_B(ax):
    ax.hist(adv_retained[np.isfinite(adv_retained)], bins=60,
            color=COL_ADV, edgecolor="none", linewidth=0)
    ax.set_xlabel(r"CV-$R^2$ $-$ null-$R^2$")
    ax.set_ylabel("Vertex count")
    ax.set_title("CV advantage (retained vertices)", pad=6)
    panel_label(ax, "B")

def make_panel_C(ax):
    x  = np.arange(len(config.SUBJECTS))
    bw = 0.38
    ax.bar(x - bw/2, loo_left,  bw, color=COL_LEFT,  label="Left",  alpha=0.9)
    ax.bar(x + bw/2, loo_right, bw, color=COL_RIGHT, label="Right", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, fontsize=7.5)
    ax.set_xlabel("Participant")
    ax.set_ylabel("LOO Pearson r")
    ax.set_title("Cross-subject reproducibility", pad=6)
    ax.set_ylim(0, 1)
    ax.legend()
    # Mean line for each hemisphere
    ax.axhline(np.nanmean(loo_left),  color=COL_LEFT,  lw=0.8,
               ls="--", alpha=0.6)
    ax.axhline(np.nanmean(loo_right), color=COL_RIGHT, lw=0.8,
               ls="--", alpha=0.6)
    panel_label(ax, "C")

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
fig.subplots_adjust(wspace=0.45, left=0.09, right=0.97,
                    bottom=0.22, top=0.88)

make_panel_A(axes[0])
make_panel_B(axes[1])
make_panel_C(axes[2])

combined_path = config.OUTPUT_DIR / "results_summary_APA.png"
fig.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nCombined figure saved: {combined_path}")

panel_specs = [
    ("A", make_panel_A, (2.8, 2.8)),
    ("B", make_panel_B, (2.8, 2.8)),
    ("C", make_panel_C, (3.2, 2.8)),
]
for letter, fn, figsize in panel_specs:
    fig_p, ax_p = plt.subplots(figsize=figsize)
    fig_p.subplots_adjust(left=0.18, right=0.95, bottom=0.22, top=0.88)
    fn(ax_p)
    out = config.OUTPUT_DIR / f"results_summary_APA_panel{letter}.png"
    fig_p.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_p)
    print(f"Panel {letter} saved:  {out}")
