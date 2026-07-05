"""One-subject, one-hemisphere dry run. Use this to time the pipeline and catch
shape / format problems BEFORE launching the full 11-subject batch.

    python one_subject_test.py                  # sub-01, left hemisphere
    python one_subject_test.py sub-03 right     # third subject, right hemisphere
"""
import sys
import time
import numpy as np

import _compat   # noqa: F401  -- must be first, before any prfpy import
import cf_pipeline
import config


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else config.SUBJECTS[0]
    hemi    = sys.argv[2] if len(sys.argv) > 2 else "left"
    print(f"\nSmoke test: {subject} / {hemi}")
    print(f"  S1 areas: {config.S1_AREA_NAMES}")
    print(f"  LOSO folds: {config.N_RUNS}  |  BATCH_SIZE: {config.BATCH_SIZE}")

    t0     = time.time()
    params = cf_pipeline.fit_hemisphere(subject, hemi)
    dt     = time.time() - t0
    print(f"\nfit_hemisphere: {dt:.1f} s  ({dt/60:.1f} min)")

    print("\n--- per-column summary (all target vertices) ---")
    print(params[["sigma", "grid_r2", "xval_r2", "null_r2",
                  "noise_ceiling", "bodypart"]].describe().round(4))

    # How many vertices pass each gate individually, then the joint mask
    noise_ok = (params["noise_ceiling"] >= config.SPLIT_HALF_THRESHOLD).sum()
    cv_ok    = ((params["xval_r2"] - params["null_r2"])
                > config.RELIABILITY_THRESHOLD).sum()
    bm       = cf_pipeline.bodymap_from_params(params)
    joint_ok = int(np.isfinite(bm).sum())
    n        = len(params)
    print(f"\n--- vertex retention ---")
    print(f"  noise_ceiling >= {config.SPLIT_HALF_THRESHOLD}:          "
          f"{noise_ok:>5} / {n}  ({noise_ok/n*100:.1f}%)")
    print(f"  xval_r2 - null_r2 > {config.RELIABILITY_THRESHOLD}:   "
          f"{cv_ok:>5} / {n}  ({cv_ok/n*100:.1f}%)")
    print(f"  joint mask (body-map vertices):  {joint_ok:>5} / {n}  "
          f"({joint_ok/n*100:.1f}%)")

    print("\nSmoke test complete.")
    print("If joint_ok is very low, inspect the noise_ceiling distribution")
    print("and consider adjusting config.SPLIT_HALF_THRESHOLD before the")
    print("full batch run (the r > .? placeholder in the Methods text).")

    import cortex

    # Blank half for the hemisphere not being shown
    blank = np.full(config.N_VERTS_PER_HEMI, np.nan)

    def _whole(arr):
        """Put a per-hemisphere array into whole-brain vertex order."""
        v = np.asarray(arr, dtype=float)
        return np.concatenate([v, blank]) if hemi == "left" \
            else np.concatenate([blank, v])

    advantage   = (params["xval_r2"] - params["null_r2"]).values
    sigma_vals  = np.where(params["sigma"].values > 0,
                           params["sigma"].values, np.nan)

    adv_finite  = advantage[np.isfinite(advantage)]
    sig_finite  = sigma_vals[np.isfinite(sigma_vals)]

    data_layers = {
        "1. Noise ceiling (split-half r)": cortex.Vertex(
            _whole(params["noise_ceiling"].values),
            config.PYCORTEX_SUBJECT,
            vmin=-1, vmax=1, cmap="magma"),

        "2. CF size sigma (mm)": cortex.Vertex(
            _whole(sigma_vals),
            config.PYCORTEX_SUBJECT,
            vmin=0,
            vmax=float(np.nanpercentile(sig_finite, 95)),
            cmap="PuBu"),

        "3. CV-R2 minus null-R2 (advantage)": cortex.Vertex(
            _whole(advantage),
            config.PYCORTEX_SUBJECT,
            vmin=float(np.nanpercentile(adv_finite, 5)),
            vmax=float(np.nanpercentile(adv_finite, 95)),
            cmap="coolwarm"),

        "4. Joint mask (retained vertices)": cortex.Vertex(
            _whole(np.where(np.isfinite(bm), 1.0, np.nan)),
            config.PYCORTEX_SUBJECT,
            vmin=0, vmax=1, cmap="hot"),
    }

    out_path = config.OUTPUT_DIR / f"smoke_vis_{subject}_{hemi}_static"
    cortex.webgl.make_static(outpath=str(out_path), data=data_layers)
    print(f"\nStatic viewer saved to: {out_path}")
    print("Open index.html in a browser.  Use the dropdown (top-left) to")
    print("switch between the four maps.")
    viewer = cortex.webshow(data_layers)
    input("\nViewer running -- press Enter here (in the terminal) to shut it down...")

if __name__ == "__main__":
    main()
