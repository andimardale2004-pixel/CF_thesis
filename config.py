"""Central configuration: paths, subjects, model settings, condition labels.
Everything in the project reads from here, so this is the only file with machine-specific paths.
"""
from pathlib import Path
import numpy as np
import os
import cortex

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Paths : file locations 
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = Path("~/Documents/uni/3rd year/THESIS/cf_thesis").expanduser()    
BETA_DIR     = DATA_DIR / "betas/glmsingle_surface"     
SPLIT_BETA_DIR = DATA_DIR / "betas/glmsingle_joint_shift2TR_surface"        
JOINT_BETA_DIR = DATA_DIR / "betas/glmsingle_joint_shift2TR_surface"             
OUTPUT_DIR   = PROJECT_ROOT / "outputs"                    # results land here
DIST_CACHE   = OUTPUT_DIR / "distance_matrices"            # cached geodesic matrices
PARCELLATION = DATA_DIR / "atlas" / "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.59k_fs_LR.dlabel.{H}.gii"

for _d in (OUTPUT_DIR, DIST_CACHE):
    _d.mkdir(parents=True, exist_ok=True)


# .----------.
# | Subjects |
# '----------'
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
            "sub-07", "sub-08", "sub-09", "sub-11", "sub-12", "sub-13"]
HEMIS    = ["left", "right"]


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Surface / data geometry  (fMRIPrep fsLR den-59k = 59,292 cortical verts/hemi)
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
N_VERTS_PER_HEMI = 59292
PYCORTEX_SUBJECT = "hcp_999999"  


# ¸,ø¤º°`°º¤ø¤º°`°º¤ø,¸¸,ø¤º°`°º¤ø¤º°`°º¤ø,¸¸,ø¤º°
# 22 conditions, ordered as in SOMA design-matrix 
# ¸,ø¤º°`°º¤ø¤º°`°º¤ø,¸¸,ø¤º°`°º¤ø¤º°`°º¤ø,¸¸,ø¤º°
CONDITION_LABELS = [
    "eyebrows", "eyes", "mouth", "tongue",
    "lhand_fing1", "lhand_fing2", "lhand_fing3", "lhand_fing4", "lhand_fing5", "lleg",
    "bhand_fing1", "bhand_fing2", "bhand_fing3", "bhand_fing4", "bhand_fing5", "bleg",
    "rhand_fing1", "rhand_fing2", "rhand_fing3", "rhand_fing4", "rhand_fing5", "rleg",
]
N_CONDITIONS = len(CONDITION_LABELS)      # == 22

# Optional: a 1-D somatotopic ordering of the conditions for a smoother colour axis.
# Leave as None to colour targets by raw condition index (0..21). To order roughly
# foot -> hand -> face, supply a permutation of CONDITION_LABELS here.
SOMATO_AXIS_ORDER = [
    "lleg",  "bleg",  "rleg",
    "lhand_fing5", "bhand_fing5", "rhand_fing5",
    "lhand_fing4", "bhand_fing4", "rhand_fing4",
    "lhand_fing3", "bhand_fing3", "rhand_fing3",
    "lhand_fing2", "bhand_fing2", "rhand_fing2",
    "lhand_fing1", "bhand_fing1", "rhand_fing1",
    "eyebrows", "eyes", "mouth", "tongue",
]
# S1 source region: HCP-MMP areas. Selected in data_io.get_s1_source_indices.
S1_AREA_NAMES = ["3a", "3b", "1"]    # TODO: confirm which subfields you want as source


# -=-=-=-=-=-=-=-=-=-=-=-=-
# Model / fitting settings
# -=-=-=-=-=-=-=-=-=-=-=-=-
SIGMA_GRID =  np.array([0.2, 0.3, 0.5, 1, 3, 5, 7, 10, 15, 20, 30])
RELIABILITY_THRESHOLD = 0.3                # keep targets where (xval_R2 - null_R2) > this
SPLIT_HALF_THRESHOLD = 0.2              # minimum split-half r to keep a vertex as interpretable
N_RUNS = 4
BATCH_SIZE = 5000                          # vertex batch size for parallel fitting (joblib)
N_JOBS = -2                                # joblib cores for CFFitter (-2 = all but one)



# ¸,ø¤º°`°º¤ø¤º°`°º¤ø,¸
# Pycortex Filestore 
# ¸,ø¤º°`°º¤ø¤º°`°º¤ø,¸
FILESTORE = DATA_DIR / "pycortex_store"  

def register_pycortex_filestore():
    os.makedirs(FILESTORE, exist_ok=True)
    cortex.options.config.set('basic', 'filestore', str(FILESTORE))   # <- str() here
    with open(cortex.options.usercfg, 'w') as f:
        cortex.options.config.write(f)
    print("filestore set to:", FILESTORE)

if __name__ == "__main__":
    register_pycortex_filestore()

#_|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_   
# |       |       |       |       |       |       |       |       |    
#    _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_
#     |       |       |       |       |       |       |       |       |
#_|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_   
# |       |       |       |       |       |       |       |       |    
#    _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_
#     |       |       |       |       |       |       |       |       |
#_|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_   
# |       |       |       |       |       |       |       |       |    
#    _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_
#     |       |       |       |       |       |       |       |       |
#_|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_   
# |       |       |       |       |       |       |       |       |    
#    _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_
#     |       |       |       |       |       |       |       |       |
#_|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_   
# |       |       |       |       |       |       |       |       |    
#    _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_
#     |       |       |       |       |       |       |       |       |
#_|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_     _|_   
# |       |       |       |       |       |       |       |       |