"""
Prepare the training labels:
****************************
- For annotated reference samples (~1.3%, only 58), use the provided labels as-is, with increased confidence.
- For the remaining studies, use LLM-generated pseudo-labels.
- Skip samples with ambiguous laterality (~0.9%) during training.

NOTE: This script assumes the LLM output contains no missing values (validated upstream).

"""

from pathlib import Path
import numpy as np, pandas as pd


INPUT = Path("/kaggle/input/")
WORKING = Path("/kaggle/working/")


TARGETS = {
    "ACL":'''ACL''', "MCL":'''MCL''',
    "Medial Meniscus":'''Medial_Meniscus''', "Lateral Meniscus":'''Lateral_Meniscus''',
    "Medial OA":'''Medial_OA''', "Lateral OA":'''Lateral_OA''', "PF OA":'''PF_OA''',
    "Effusion":'''Effusion''', "Synovitis":'''Synovitis''',
    "Baker's":'''Bakers''',
    "Contusion":'''Contusion''', "Fracture":'''Fracture''',
}

# ==================================================
# Read all UIDs, with or without reference labels
# ====================================
y = pd.read_csv(
    INPUT / "competitions/rsna-knee-abnormality-detection/train.csv",
    index_col="StudyInstanceUID", usecols= ["StudyInstanceUID"] + list(TARGETS),
    dtype={"StudyInstanceUID":'string'}|{col:np.float32 for col in TARGETS},
    # we will use soft labels instead of hard 0-1 labeling
).rename(columns=TARGETS, index=str.strip)

ref_flag = y.notna().all(axis=1).rename("ref_flag") # annotated reference studies


LLM_out = pd.read_csv(
    INPUT/"datasets/ramitnandi/rsna-knee-labels-gpt-luna/report_labels.csv",
    index_col="StudyInstanceUID", usecols=["StudyInstanceUID","CLASS","SCORE","FINAL_LABEL","weight","ambiguous_laterality"], 
    dtype={
        "StudyInstanceUID":'string', 
        "CLASS":'category', "SCORE":np.float32, "FINAL_LABEL":bool, 
        "weight":np.float32, "ambiguous_laterality":bool
    },
).rename(index=str.strip)

LLM_out = LLM_out.pivot(
    columns="CLASS",
    values=["SCORE", "FINAL_LABEL", "weight", "ambiguous_laterality"],
)

# -------------------------

# ==================================================
# Assign sample Weights
# ====================================
w = (
    LLM_out["weight"].reindex_like(y)
    .where(y.isna(), 2.5)
    .astype(np.float32)
    .mask(LLM_out["ambiguous_laterality"].astype(bool), 0)
    .add_suffix("__w")
)

# -------------------------

# ==================================================
# Hard labels for Evaluation 
# ====================================
label = (
    y.astype("boolean")
    .fillna(LLM_out["FINAL_LABEL"].astype("boolean"))
    .add_suffix("__label")
)

# -------------------------

# ==================================================
# Soft Scoring
# ====================================
y = y.fillna(LLM_out["SCORE"].astype(np.float32))

# -------------------------


pd.concat(
    [y, ref_flag, w, label],
    axis=1,
).to_parquet(
    WORKING / "train_y.parquet", index=True, 
    compression='zstd',
)


