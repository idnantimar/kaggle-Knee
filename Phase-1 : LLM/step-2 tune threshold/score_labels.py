"""
Design: @idnantimar
-------
Instead of asking an LLM to tag True/False for arbitrary medical abnormalities, we have decoupled the task into two subproblems:

[1.] An LLM will read the reports and answer two simple questions:
    - `classification`: does the report talk about a predefined medical abnormality (POSITIVE/NEGATIVE/NA)?
    - `severity`: if POSITIVE, what is the severity (HIGH/MEDIUM/LOW)?

[2.] A subsequent Python script will read the LLM findings and prepare the final True/False labels based on the combination of (classification, severity).

Since each of the 12 abnormality classes follows a different threshold, 
this approach provides more tokens to arrive at the final conclusion (i.e., fewer computations per token). 
Also, if we ever need to tune only the thresholds between severity classes (e.g., consider only HIGH as True rather than MEDIUM/HIGH), 
we do not need to make an LLM call from scratch.

Ref: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343


LIMITATIONS: Since the LLM is not aware of arbitrary medical cutoffs, 
if a finding is described in terms of absolute numbers (e.g., a 10 mm cyst) without any qualifier word (e.g., mild, moderate, serious, etc.), 
severity is tagged as HIGH by default.


NOTE: The annotated reference set is <2% of the total training set; 
Any empirical estimate of the form 
    P(abnormality | category of report) 
derived from this reference set may not generalize to the full dataset, 
especially if the reference set was hand-picked to showcase all 12 label classes. 
Therefore, 
we only encode the ordinal information of the categories in an uncalibrated score between 0 - 1:
    NEGATIVE < NA < LOW < MEDIUM < HIGH

"""


import polars as pl, pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import roc_auc_score


logger = logging.getLogger(__name__)



BASE_DIR = Path(__file__).parent
IN_PATH = BASE_DIR.parent / "step-1 extract labels" / "out" / "report_labels_api.jsonl"
REF_PATH = BASE_DIR / "input" / "gold_labels.csv"
OUTPUT_CSV = BASE_DIR / "out" / "report_labels.csv"


ALL = ["LOW", "MEDIUM", "HIGH"]
EXCLUDE_LOW = ALL[1:]
HIGH_ONLY = ALL[-1:]

THRESHOLDS = {
    "ACL": EXCLUDE_LOW, "MCL": EXCLUDE_LOW, 
    "Medial_Meniscus": ALL, "Lateral_Meniscus": ALL,
    "Medial_OA": EXCLUDE_LOW, "Lateral_OA": EXCLUDE_LOW, "PF_OA": EXCLUDE_LOW,
    "Effusion": EXCLUDE_LOW, "Synovitis":  ALL,
    "Bakers": EXCLUDE_LOW,
    "Contusion": ALL, "Fracture": ALL,
}
SCORE = {
    "POSITIVE_HIGH": 1.-1e-6,
    "POSITIVE_MEDIUM": 0.8,
    "POSITIVE_LOW": 0.6,
    "NEGATIVE": 0.+1e-6,
    "NA": 0.2,
    # HIGH/MEDIUM/LOW has a natural semantic ordering;
    # when something is not mentioned, 
    # it is scored a bit higher than something explicitly discarded;
    # the ordering: NEGATIVE < NA < LOW < MEDIUM < HIGH
}


ABNORMALITY_CLASSES = {
    "ACL":'''ACL''', "MCL":'''MCL''',
    "Medial_Meniscus":'''Medial Meniscus''', "Lateral_Meniscus":'''Lateral Meniscus''',
    "Medial_OA":'''Medial OA''', "Lateral_OA":'''Lateral OA''', "PF_OA":'''PF OA''',
    "Effusion":'''Effusion''', "Synovitis":'''Synovitis''',
    "Bakers":'''Baker's''',
    "Contusion":'''Contusion''', "Fracture":'''Fracture''',
}
LABELS = {cls: i for i, cls in enumerate(ABNORMALITY_CLASSES)}


REPORT_SCHEMA = {
    "StudyInstanceUID": pl.String,
    "out": pl.Struct(
        {
            "ambiguous_laterality": pl.Boolean,
            "findings": pl.Struct(
                {
                    cls: pl.Struct(
                        {
                            "classification": pl.String, "severity": pl.String,
                            "hedged": pl.Boolean,
                            "evidence": pl.String,
                        }
                    )
                    for cls in ABNORMALITY_CLASSES
                }
            ),
        }
    ),
}


def load_report_labels(path: Path) -> pl.DataFrame:
    DATA = pl.scan_ndjson(
        path, 
        schema=REPORT_SCHEMA, infer_schema_length=None,
    )

    DATA = (
        DATA
        .unnest("out")
        .unnest("findings")
        .unpivot(
            on=ABNORMALITY_CLASSES,
            index=["StudyInstanceUID", "ambiguous_laterality"],
            variable_name="CLASS",
            value_name="temp"
        )
        .unnest("temp")  
    ).with_columns(
        pl.col.severity.is_in(
            pl.col.CLASS.replace_strict(THRESHOLDS), 
            nulls_equal=True
        ).alias("FINAL_LABEL"),
        # determined by the combination of (classification, severity)
        # Ref: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
        pl.concat_str(
            ["classification", "severity"],
            separator='_',
            ignore_nulls=True,
        ).replace_strict(SCORE).alias("SCORE"),
        # we only encode the ordinal information of the categories in an uncalibrated score between 0 - 1:
        #   NEGATIVE < NA < LOW < MEDIUM < HIGH
    )

    return DATA.sort(
        [
            "StudyInstanceUID",
            pl.col.CLASS.replace_strict(LABELS)
        ]
    ).collect()


# --------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    labels = load_report_labels(IN_PATH)
    labels.write_csv(OUTPUT_CSV)
    logger.info(f"""Wrote {labels.n_unique("StudyInstanceUID")} UIDs to {OUTPUT_CSV}""")
    logger.info(f"""ambiguous laterality: {labels.filter("ambiguous_laterality").n_unique("StudyInstanceUID")}""")
    PREVALENCE = labels.group_by(
        "CLASS", maintain_order=True
    ).agg(
        (100*pl.col.FINAL_LABEL.mean()).alias("prevalence")
    )
    logger.info(f"""\nprevalence\n{pd.Series(PREVALENCE["prevalence"], index=PREVALENCE["CLASS"]).round(2)}\n""")


    # ======= AUC =======================================
    REF = pd.read_csv(
        REF_PATH,
        index_col="StudyInstanceUID", 
        dtype={"StudyInstanceUID":'string'},
    )[list(ABNORMALITY_CLASSES.values())]

    PRED = labels.filter(
        pl.col.StudyInstanceUID.is_in(REF.index)
    ).pivot(
        on="CLASS",
        index="StudyInstanceUID",
        values="SCORE", aggregate_function=None,
    ).rename(
        ABNORMALITY_CLASSES
    ).to_pandas().set_index("StudyInstanceUID")

    AUC = roc_auc_score(
        REF, PRED.reindex_like(REF),
        average=None,
    )
    logger.info(f"""AUC\n{pd.Series(AUC, index=ABNORMALITY_CLASSES).round(4)}\n""")
    logger.info(f"""macro-AUC:{AUC.mean()}""")
    # ===================