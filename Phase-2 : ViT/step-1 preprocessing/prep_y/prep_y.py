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
    index_col="StudyInstanceUID", usecols=["StudyInstanceUID","CLASS","SCORE","FINAL_LABEL","classification","hedged","ambiguous_laterality"], 
    dtype={
        "StudyInstanceUID":'string', 
        "CLASS":'category', "SCORE":np.float32, "FINAL_LABEL":bool, "classification":'category', 
        "hedged":bool, "ambiguous_laterality":bool
    },
).rename(index=str.strip)

# -------------------------

# ==================================================
# Assign sample Weights
# ====================================
w = y.where(y.isna(), 2.5) 
LLM_out["W"] = (
    np.where(
        LLM_out["classification"].notna(),
        1 - 0.2*LLM_out["hedged"],
        0.5
    ) 
)
w_pred = (
    LLM_out
    .pivot(
        columns="CLASS",
        values=["W", "ambiguous_laterality"],
    )
)
w.update(w_pred["W"].astype(np.float32), overwrite=False)
w.mask(
    w_pred["ambiguous_laterality"].astype(bool),
    0,
    inplace=True,
)
w.rename(columns=lambda x: str(x)+'__w', inplace=True)

# -------------------------

# ==================================================
# Hard labels for Evaluation 
# ====================================
label = y.astype("boolean")
label_pred = (
    LLM_out
    .pivot(
        columns="CLASS",
        values="FINAL_LABEL",
    )
)
label.update(label_pred, overwrite=False)
label.rename(columns=lambda x: str(x)+'__label', inplace=True)
 
# -------------------------

# ==================================================
# Soft Scoring
# ====================================
y_pred = (
    LLM_out
    .pivot(
        columns="CLASS",
        values="SCORE",
    )
)
y.update(y_pred, overwrite=False)

# -------------------------


pd.concat(
    [y, ref_flag, w, label],
    axis=1,
).to_parquet(
    WORKING / "train_y.parquet", index=True, 
    compression='zstd',
)


