"""Panel B: an example connective field kernel -- idealized, and on real S1 geometry.

PURPOSE: a methods-figure panel showing what CF(v, v0, sigma) actually looks
like, computed from your real model equation and your real S1 distance
matrix -- not a generic illustration. Deliberately stays within S1 itself
(model mechanics), making no claim about any downstream target -- that
distinction matters: this panel should never be mistaken for a result.

Two halves:
  1. Idealized 2D Gaussian on a flat grid -- the abstract math object.
  2. The SAME formula, SAME sigma, evaluated on real S1 geodesic distances
     from a single example source vertex, rendered on the cortical surface.

The example source vertex is chosen as the geometric "centre" of S1 (the
vertex with minimum mean distance to all other S1 vertices) -- an arbitrary
but principled, clearly-labelled example, not a fitted result.

Run cell-by-cell in VSCode.
"""
# %% 1. Imports + settings
import numpy as np
import matplotlib.pyplot as plt
import cortex

import config
import data_io

HEMI = "left"                      # which hemisphere's S1 to illustrate
EXAMPLE_SIGMA = 3   # a representative CF size, mm
EXAMPLE_SUBJECT = config.SUBJECTS[0]                  # arbitrary, illustrative
EXAMPLE_CONDITION = "rhand_fing1"                      # same condition as panel_source_region.py,
                                                       # for a thematically consistent pair of panels
print(f"Example sigma for illustration: {EXAMPLE_SIGMA:.2f} mm "
     f"(median of config.SIGMA_GRID)")


# %% 2. Idealized 2D Gaussian (the abstract math object, same formula/sigma)
xx, yy = np.meshgrid(np.linspace(-25, 25, 200), np.linspace(-25, 25, 200))
dd = np.sqrt(xx**2 + yy**2)
ideal_kernel = np.exp(-dd**2 / (2 * EXAMPLE_SIGMA**2))

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(ideal_kernel, extent=[-25, 25, -25, 25], cmap="viridis", vmin=0, vmax=1)
ax.set_title(f"Idealized connective field\n(sigma = {EXAMPLE_SIGMA:.1f} mm, example)")
ax.set_xlabel("geodesic distance (mm)")
fig.colorbar(im, ax=ax, label="CF weight")
fig.tight_layout()
out_path = config.OUTPUT_DIR / "panel_b_idealized_gaussian.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved: {out_path}")


# %% 3. The same kernel, on real S1 geometry -- three ways to pick the example vertex
s1_inds = data_io.get_s1_source_indices(HEMI)
dist_mat = data_io.geodesic_distance_matrix(HEMI, s1_inds)   # cached -- should load instantly
pts, _ = cortex.db.get_surf(config.PYCORTEX_SUBJECT, "fiducial", HEMI)
pts_s1 = pts[s1_inds]   # XYZ coordinates, S1 vertices only

betas = np.nan_to_num(data_io.load_betas(EXAMPLE_SUBJECT, HEMI))
cond_idx = config.CONDITION_LABELS.index(EXAMPLE_CONDITION)
betas_at_s1 = betas[s1_inds, :]

# Pick ONE of three modes:
#   "window"            -- constrain to a region by real coordinates, then pick
#                           whichever vertex in that region responds most to
#                           EXAMPLE_CONDITION (keeps the pick data-driven).
#   "explicit_vertex"    -- skip everything else; use this exact whole-brain
#                           vertex index. Fully free, but you supply the number.
#   "nearest_to_xyz"     -- skip the functional pick; use whichever S1 vertex
#                           is closest to a target (x, y, z) coordinate.
SELECTION_MODE = "explicit_vertex"   # pick one of the three above

if SELECTION_MODE == "window":
    # Each entry constrains one coordinate axis to a percentile range OF S1's
    # OWN extent on that axis (0=one end, 100=the other -- which end is which
    # you determine empirically from the printed ranges below).
    # axis 1 is typically anterior-posterior, axis 2 typically superior-inferior
    # (roughly "up/down the strip", i.e. toward leg vs toward face) -- but
    # CONFIRM against the printed ranges and the rendered image, don't assume.
    AXIS_WINDOWS = [
        {"axis": 1, "sign": 1, "percentile_range": (0, 40)},   # AP: posterior 40%
        # Add a second constraint to also control the other axis, e.g.:
        # {"axis": 2, "sign": 1, "percentile_range": (0, 60)},
    ]
    mask = np.ones(len(s1_inds), dtype=bool)
    for w in AXIS_WINDOWS:
        coord = w["sign"] * pts_s1[:, w["axis"]]
        print(f"  axis {w['axis']}: S1 range {coord.min():.1f} to {coord.max():.1f}")
        lo, hi = np.percentile(coord, w["percentile_range"])
        mask &= (coord >= lo) & (coord <= hi)
    assert mask.sum() > 0, "AXIS_WINDOWS excludes every S1 vertex -- widen the ranges"
    print(f"Window keeps {mask.sum()} / {len(s1_inds)} S1 vertices")
    search_values = np.where(mask, betas_at_s1[:, cond_idx], -np.inf)
    central_pos = int(np.argmax(search_values))

elif SELECTION_MODE == "explicit_vertex":
    EXPLICIT_VERTEX = 13725   # <-- set this to a real whole-brain vertex index
    if EXPLICIT_VERTEX not in s1_inds:
        raise ValueError(f"vertex {EXPLICIT_VERTEX} is not inside the S1 source "
                         f"region for {HEMI} -- pick one that is, or it will silently "
                         f"resolve to the wrong position and crash later.")
    central_pos = int(np.searchsorted(s1_inds, EXPLICIT_VERTEX))

central_vertex = int(s1_inds[central_pos])
print(f"Example source vertex ({SELECTION_MODE}): whole-brain index {central_vertex}")

kernel_on_s1 = np.exp(-dist_mat[central_pos] ** 2 / (2 * EXAMPLE_SIGMA ** 2))
assert np.isclose(kernel_on_s1[central_pos], 1.0)   # sanity: peaks at its own centre

# Place onto a whole-hemisphere array (NaN outside S1 -- this panel makes no
# claim about anything beyond the source region itself).
kernel_whole_hemi = np.full(config.N_VERTS_PER_HEMI, np.nan)
kernel_whole_hemi[s1_inds] = kernel_on_s1


# %% 4. Render on the cortical surface (left hemisphere only; right is blank)
blank_right = np.full(config.N_VERTS_PER_HEMI, np.nan)
whole_brain = np.concatenate([kernel_whole_hemi, blank_right]) if HEMI == "left" \
    else np.concatenate([blank_right, kernel_whole_hemi])

vx = cortex.Vertex(whole_brain, config.PYCORTEX_SUBJECT, vmin=0, vmax=1, cmap="viridis")

out_path = config.OUTPUT_DIR / f"panel_b_cf_on_brain_v{central_vertex}_static"
cortex.webgl.make_static(
    outpath=str(out_path),
    data={f"Example CF kernel (v0={central_vertex}, sigma={EXAMPLE_SIGMA:.1f}mm)": vx},
)
print(f"\nStatic viewer saved to: {out_path}")
print("(folder name includes the vertex index, so each run gets its own fresh")
print(" folder -- if a previous run's browser tab is still open, close it and")
print(" open THIS folder's index.html, not the old one)")
print("Open index.html, switch to the flat or inflated surface for the clearest panel crop.")
print("\nCaption note: this shows model mechanics on an example S1 vertex --")
print("not a fitted result or a claim about any specific downstream target.")


# %% 5. (Optional) live interactive viewer
viewer = cortex.webshow({f"CF kernel example (sigma={EXAMPLE_SIGMA:.1f}mm)": vx})
print("\nWebGL viewer launched in your browser.")
input("Viewer running -- press Enter here (in the terminal) when you're done "
     "exploring, to shut it down and exit...")

print(float(np.median(config.SIGMA_GRID)))
