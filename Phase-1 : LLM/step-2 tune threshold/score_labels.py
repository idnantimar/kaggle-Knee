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
Any empirical estimate of the form P(abnormality | category of report) derived from this reference set may not generalize to the full dataset, 
especially if the reference set was hand-picked to showcase all 12 label classes. 
Therefore, we only encode the ordinal information of the categories in an uncalibrated score between 0 - 1:
    NEGATIVE < NA < LOW < MEDIUM < HIGH

"""


import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import logging
from sklearn.metrics import roc_auc_score


logger = logging.getLogger(__name__)
RANDOM_SEED = 42


BASE_DIR = Path(__file__).parent
IN_PATHS = {
    "A": BASE_DIR.parent / "step-1 extract labels" / "out" / "report_labels_api-run1.jsonl",
    "B": BASE_DIR.parent / "step-1 extract labels" / "out" / "report_labels_api-run2.jsonl",
}
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
    "POSITIVE_HIGH": 1.,
    "POSITIVE_MEDIUM": 0.8,
    "POSITIVE_LOW": 0.6,
    "NEGATIVE": 0.,
    "NA": 0.2,
    # HIGH/MEDIUM/LOW or POSITIVE/NEGATIVE has a natural semantic ordering;
}
HEDGE_PENALTY = (
    pl.when(pl.col.classification.eq("POSITIVE")).then(0.05)
      .when(pl.col.classification.eq("NEGATIVE")).then(-0.05)
    .otherwise(0)
)

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


def load_report_labels(path: Path) -> pl.LazyFrame:
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
    ).with_columns(pl.col(pl.String).str.strip_chars())

    return DATA


# --------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    run_a, run_b = load_report_labels(IN_PATHS["A"]), load_report_labels(IN_PATHS["B"])
    labels = run_a.join(
        run_b,
        on=["StudyInstanceUID", "CLASS"],
        how="inner", validate="1:1", maintain_order="left",
        suffix="_",
    ).collect()
    run_a_size, run_b_size = (run.select(pl.len()).collect().item() for run in (run_a, run_b))
    assert labels.height == run_a_size == run_b_size, f"runs do not cover the same UIDs: A={run_a_size}, B={run_b_size}, matched={labels.height}"

    agree = pl.all_horizontal(
        pl.col(field).eq_missing(pl.col(f"{field}_"))
        for field in ["classification", "severity", "hedged"]
    )
    pick_ab = pl.Series(np.random.default_rng(RANDOM_SEED).random(labels.height) < 0.5)
    labels = labels.select(
        "StudyInstanceUID", "CLASS",
        *[
            pl.when(pick_ab).then(field).otherwise(f"{field}_").alias(field)
            for field in ["classification", "severity", "hedged"]
        ],
        # when (classification, severity, hedged) disagree, pick A/B randomly;
        # for matched cases, they produce the same value anyway.
        (pl.col.ambiguous_laterality | pl.col.ambiguous_laterality_).alias("ambiguous_laterality"),
        # if at least one finding is ambiguous_laterality, keep it.
        pl.format(
            "A| {}\n\nB| {}",
            pl.col.evidence.fill_null(""), pl.col.evidence_.fill_null(""),
        ).alias("evidence"),
        pl.when(agree).then(1.).otherwise(1 / 2**0.5).alias("weight"),
        # downweight samples where A/B disagree.
    ).with_columns(
        pl.col.severity.is_in(
            pl.col.CLASS.replace_strict(THRESHOLDS), 
            nulls_equal=True
        ).alias("FINAL_LABEL"),
        # determined by the combination of (classification, severity)
        # Ref: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
        (
            pl.concat_str(["classification", "severity"], separator='_', ignore_nulls=True).replace_strict(SCORE)
            - HEDGE_PENALTY * pl.col.hedged.fill_null(False)
        ).alias("SCORE"),
        # we only encode the ordinal information of the categories in an uncalibrated score between 0 - 1:
        #   NEGATIVE < NEGATIVE_hedged < NA < LOW_hedged < LOW < MEDIUM_hedged < MEDIUM < HIGH_hedged < HIGH
    ).sort(
        [
            "StudyInstanceUID",
            pl.col.CLASS.replace_strict(LABELS)
        ] 
    )

    labels.write_csv(OUTPUT_CSV)
    logger.info(f"""Wrote {labels.n_unique("StudyInstanceUID")} UIDs to {OUTPUT_CSV}""")
    agreement_count = labels.filter(pl.col.weight == 1).height
    logger.info(f"""run 1 & 2 agree [{agreement_count}/{labels.height}] ({100*agreement_count/labels.height:.2f}%)""")
    logger.info(f"""ambiguous laterality: {labels.filter("ambiguous_laterality").n_unique("StudyInstanceUID")}""")
    

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