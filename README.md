# Somatotopic CF Modelling

Fits `prfpy` connective field models to per-condition (22-effector) beta-weights
from the SOMA motor-mimicry task, using S1 as the source region, and produces a
whole-brain somatotopic body map rendered with pycortex.
This code was developed before access to prior GLMsingle analyses ran by a trained master's student (Beta Weights)

## Project layout
|     File    |  What it does |
|-<>-<>-<>-<>-|--<>-<>-<>-<>--|
| `config.py` | all paths, subjects, condition labels, model settings |
| `_compat.py` | NumPy-2.0 alias shim, fixes an outdated co-dependency error for prfpy. Only use if running NumPy version > 2. 0. Initially made for google colab work-around |
| `data_io.py` | load betas (CIFTI), select S1 source verts, geodesic distance matrix |
| `cf_pipeline.py` | fit + cross-validate the CF model, null model, body-part readout |
| `visualize.py` | visualisations (body map + CF size) |
| `run.py` | resumable batch over all subjects x hemispheres |
| `one_subject_test.py` | one-subject dry run for timing + debugging |

## Setup (on macOS)
1. Install **VSCode** and **Miniforge**.
2. Create the project + environment:
   ```bash
   mkdir ~/(directoryname) && cd ~/(directoryname)    # then put these files here <('.'<)
   conda create -n cf python=3.11 -y
   conda activate cf
   pip install -r requirements.txt
   ```
3. In VSCode: **Python: Select Interpreter** -> the `cf` env.
4. Configure pycortex once: set its filestore to where your surfaces live and
   confirm `python -c "import cortex; print(cortex.db.subjects)"` lists your subject.

## Run
Can either bash, but I reccommend running them one by one for debugging reasons.
```bash                         
python one_subject_test.py      # ALWAYS do this first: times one subject, checks shapes
python run.py                   # full batch; safe to re-run after a crash (it skips finished work)
python run.py --subjects sub-01 # just one subject
python run.py --force           # refit everything
```
Outputs (one parquet per subject/hemisphere, cached distance matrices, webGL viewers)
land in `outputs/`.

## Before the first real run, fill in (in `config.py` / `data_io.py`)
- `BETA_DIR` + the filename pattern in `load_betas` (and how odd/even splits are named)
- `PARCELLATION` path + S1 label names in `get_s1_source_indices`
- `PYCORTEX_SUBJECT` matching the fsLR den-170k (59,292 verts/hemi) mesh

## Notes
- Validation (`quick_xval`) needs two independent beta estimates (e.g. odd/even runs).
- The geodesic distance matrix is the slow step; it is cached to `outputs/distance_matrices`
  after the first run per subject/hemisphere.
- 22 conditions, indices 0..21. Hand/leg effectors appear in left/both/right variants,
  so collapse or restrict laterality in the source readout if you want a cleaner axis
  (see `SOMATO_AXIS_ORDER` in `config.py`).

|     .-.
|    /   \         .-.
|   /     \       /   \       .-.     .-.     _   _
+--/-------\-----/-----\-----/---\---/---\---/-\-/-\/\/---
| /         \   /       \   /     '-'     '-'
|/           '-'         '-'
