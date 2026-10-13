"""
API-based knee MRI report classifier.

Inputs:
    Prompt (*.md)
    few_shot_examples (*.jsonl)
    train (*.csv)

Outputs:
    /out/report_labels_api.jsonl

Design: @idnantimar
-------
- OpenAI Responses API
- GPT-6 Luna
- moderate reasoning effort for semantic clinical interpretation
- strict Structured Outputs / JSON Schema
- mandatory Pydantic audit
- exact, contiguous evidence (upto minimal normalization) 
  audit against the original report
- atmost 20 whitespace-separated evidence words 
  (warning on length violation, rather than raising an error)
- prompt caching
- checkpoint/ resume run
- validation failures are flagged for manual review

NOTE: output file is in append mode.
when retrying failed UIDs, make sure to remove their old entries first.

"""

from __future__ import annotations

import argparse, os, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, re, unicodedata
from pathlib import Path
from typing import Literal, TextIO, Protocol

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG

# CRITICAL: `temperature` cannot be zero when `reasoning` is enabled.
# We accept this trade-off because:
# - reasoning itself provides a form of self-stabilization that a single-pass model does not have
# - our independent Pydantic validation provides the load-bearing audit justification

# PROMPT was prepared according to the official terminology definitions of the competition.
# (see https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
# =========================================================
BASE_DIR = Path(__file__).parent
PROMPT_PATH = BASE_DIR / "input" / "RSNA Knee Abnormality Detection — MULTILINGUAL REPORT CLASSIFICATION.md"
EXAMPLES_PATH = BASE_DIR / "input" / "few_shot_examples.jsonl"
OUTPUT_JSONL = BASE_DIR / "out" / "report_labels_api.jsonl"

MODEL_NAME = "gpt-6-luna"
# designed for cost-sensitive, high-volume workloads;
# cost-effective compared to old generations.
# (see https://developers.openai.com/api/docs/models/gpt-6-luna)

REASONING_EFFORT = "medium" 
# this is semantic clinical interpretation task, not fix-keyword classification.
# 'low' reasoning is the minimal baseline, rather than None. `medium` is default configuration.
MAX_OUTPUT_TOKENS = 8192
# generous budget for this structured classification task with `low/medium` reasoning,
# while remaining well below the 128K default;
# keeps the rate-limit TPM estimate reasonably bounded.
# (see https://developers.openai.com/api/docs/guides/rate-limits)

EVIDENCE_WORD_LIMIT = 20 # Evidence budget per finding


# =========================================================
# SCHEMA VALIDATION : pydantic
# =========================================================

class _Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # validate individual key-value pairs
    classification: Literal["POSITIVE", "NEGATIVE", "NA"]
    severity: Literal["LOW", "MEDIUM", "HIGH"] | None
    hedged: bool | None
    evidence: str | None

    # validate cross-keys consistency
    @model_validator(mode="after")
    def validate_state_contract(self):
        if self.classification == "NA":
            if self.severity is not None:
                raise ValueError("NA requires severity=null")
            if self.hedged is not None:
                raise ValueError("NA requires hedged=null")
            if self.evidence is not None:
                raise ValueError("NA requires evidence=null")
        else:
            if self.evidence is None:
                raise ValueError("POSITIVE/NEGATIVE requires evidence")
            if self.hedged is None:
                raise ValueError("POSITIVE/NEGATIVE requires hedged=true/false")
            if self.classification == "NEGATIVE" and self.severity is not None:
                raise ValueError("NEGATIVE requires severity=null")
            if self.classification == "POSITIVE" and self.severity is None:
                raise ValueError("POSITIVE requires severity")

        return self


class VALIDATE_FINDINGS(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=False,
    )
    # ensure all 12 labels are present
    ACL: _Finding
    MCL: _Finding
    Medial_Meniscus: _Finding = Field(alias="Medial Meniscus")
    Lateral_Meniscus: _Finding = Field(alias="Lateral Meniscus")
    Medial_OA: _Finding = Field(alias="Medial OA")
    Lateral_OA: _Finding = Field(alias="Lateral OA")
    PF_OA: _Finding = Field(alias="PF OA")
    Effusion: _Finding
    Synovitis: _Finding
    Bakers: _Finding = Field(alias="Baker's")
    Contusion: _Finding
    Fracture: _Finding


class LLM_OUTPUT(BaseModel):
    # top-level wrapper
    model_config = ConfigDict(extra="forbid")
    ambiguous_laterality: bool
    findings: VALIDATE_FINDINGS



# =========================================================
# PREFIX : Prompt.md + few_shot_examples.jsonl 
# =========================================================

def load_static_prefix() -> str:
    """
    Load the two static files exactly once.     
    """
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Missing {PROMPT_PATH}")
    if not EXAMPLES_PATH.is_file():
        raise FileNotFoundError(f"Missing {EXAMPLES_PATH}")

    # Do not modify, normalize, translate, or regenerate either source.
    static_prefix = (
        PROMPT_PATH.read_text(encoding="utf-8").strip()
        + "\n\nFEW-SHOT EXAMPLES (JSONL):\n"
        + EXAMPLES_PATH.read_text(encoding="utf-8").strip()
    )

    return static_prefix


# =========================================================
# VALIDATION

# While OpenAI `Structured Outputs` 
# handles the JSON structure and allowed keywords:
#   - the field values can still contain hallucinated text
# STRICT INDEPENDENT VALIDATION IS CRITICAL FOR AUDIT PURPOSES
# =========================================================

_WHITESPACE_RE = re.compile(r"\s+")
_CHAR_VARIANTS = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u02bc": "'", "\u2032": "'", "`": "'",  # ‘ ’ ‛ ʼ ′ ` -> '
        "\u201c": '"', "\u201d": '"', "\u201f": '"', "\u2033": '"',  # “ ” ‟ ″ -> "
        "\u2013": "-", "\u2014": "-", "\u2010": "-",  # – — ‐ -> -
    }
)
  
def _normalize_text(s: str) -> str:
    # - collapse newlines/repeated spaces to a single space
    # - normalize common visually-identical character variants
    # these are tokenization/detokenization artifacts
    s = unicodedata.normalize("NFKC", s).translate(_CHAR_VARIANTS)
    return _WHITESPACE_RE.sub(" ", s).casefold().strip()
    # though not strictly lossless, NFKC is commonly used for text matching in NLP tasks.
    # here it is used only as an on-the-fly preprocessing layer for evidence auditing;
    # no source data is mutated in place.


def strict_validate(
    parsed: LLM_OUTPUT,
    report: str, uid:str,
):
    """
    read-only validation; no input is mutated.
    """
    report_ = _normalize_text(report)

    # evidence hallucinations are treated as strict validation errors;
    # evidence-length violations only trigger a warning and are left for downstream handling.
    for key, val in parsed.findings:
        evidence = val.evidence
        if evidence is None: 
            # `classification`:"NA" cases 
            # already validated by `_Finding.validate_state_contract(...)`
            continue 
        evidence_ = _normalize_text(evidence)
        key = VALIDATE_FINDINGS.model_fields[key].alias or key

        if evidence_ not in report_:
            raise ValueError(f"""{key}: evidence is not an exact substring of the source report""")

        n_words = len(evidence_.split())
        if n_words > EVIDENCE_WORD_LIMIT:
            logger.warning(
                """evidence for %s/%r exceeds %d-word limit (%d words)""",
                uid, key, EVIDENCE_WORD_LIMIT, n_words,
            )



# =========================================================
# API request
# =========================================================

def classify_one(
    client: OpenAI,
    static_prefix: str, 
    row: ReportRow, 
) -> tuple[dict, bool]:
    """
    (base function for API call) : Classify one individual report.
    """
    study_uid = str(row.StudyInstanceUID)
    report = str(row.Report)
    record = {"StudyInstanceUID": study_uid, "out": None}

    input_items = [
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": static_prefix,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        # Reuse the KV cache for the identical PREFIX across requests;
        # Given our large system prompt and few-shot examples against each report,
        # this can significantly reduce redundant computation.
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f'Report:\n"""\n{report}\n"""',
                }
            ],
        },
        # PREFIX remains the same throughout the session,
        # only the current report changes per call.
    ]

    success, message = True, None
    try:
        # raw llm output + metadata
        response = client.responses.parse(
            model=MODEL_NAME,
            input=input_items,
            reasoning={"effort": REASONING_EFFORT}, 
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False, 
            prompt_cache_key="knee-report-classifier-v0",
            prompt_cache_options={"mode": "explicit"},
            # gpt-6 handles cache routing automatically 
            # (see https://developers.openai.com/api/docs/guides/prompt-caching)
            text_format=LLM_OUTPUT,
            metadata={"UID":study_uid},
        )
        # validated pydantic model
        parsed = response.output_parsed
        record["out"] = parsed.model_dump(by_alias=False) 
        # complete data validation
        strict_validate(
            parsed, 
            report, study_uid,
        )
    except (ValidationError, ValueError) as e:
        # these custom validation failures require human-in-the-loop review
        message = f"""validation FAILED: {e}"""
    except Exception as e:
        # many transient API errors (e.g. 408, 429, and 5xx responses) are automatically
        # retried up to 2 times by default, using a short exponential backoff.
        # (see https://pypi.org/project/openai/)
        message = f"""API call FAILED: {e}"""

    # Case-1: [validation SUCCESS] 
    #   - `out` contains the LLM output as a dict
    #   - no `error` in record

    # Case-2: [Pydantic FAIL (rare) / API request FAIL (usually not billed)]
    #   - `out` = None (nothing is returned by the OpenAI SDK)
    #   - `error` captures the root cause

    # Case-3: [Pydantic SUCCESS + custom strict_validate FAIL]
    #   - `out` contains the LLM output as a dict
    #   - `error` provides a reasonable hint for manual review
    # this combination helps mitigate the most common failure mode without API cost wastage

    if message:
        success = False
        record["error"] = message
        logger.error("""MANUAL REVIEW REQUIRED [%s]: %s""", study_uid, message)

    return record, success


def process_concurrent(
    client: OpenAI, 
    static_prefix: str, 
    rows: pd.DataFrame, out_f: TextIO, 
    workers: int = 8,
) -> int:
    # this function only returns the progress count;
    # LLM outputs are written directly to disk.
    success = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                classify_one, 
                client, static_prefix, row,
            )
            for row in rows.itertuples(index=False, name="ReportRow")
        ]

        try:
            for future in as_completed(futures):
                record, ok = future.result()
                _write_record(out_f, record)
 
                success += int(ok)
        except KeyboardInterrupt:
            # cancel anything not yet started;
            # let in-flight calls finish and flush, so already-billed work isn't wasted.
            executor.shutdown(wait=False, cancel_futures=True)
            still_running = [f for f in futures if not f.done()]
            logger.warning(
                """Interrupted — cancelling queued work; waiting on %d in-flight request(s) to finish and flush.""",
                len(still_running),
            )
            for future in still_running:
                record, ok = future.result()
                _write_record(out_f, record)
 
                success += int(ok)

            raise

    return success


# =========================================================
# MAIN LOOP : with checkpoint/resume
# =========================================================

def load_done_ids(output_path: Path) -> set[str]:
    """
    Return StudyInstanceUIDs that have already been processed.

    (both success/failure records are considered complete)
    """
    done = set()

    if not output_path.is_file():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try: 
                uid = json.loads(line).get("StudyInstanceUID")
                if uid:
                    done.add(uid)
            except json.JSONDecodeError:
                continue
            # data was validated before being saved to disk.
            # an error here should only indicate file corruption or unexpected modification.

    return done


def main():
    """
    THE MAIN LOOP
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv", type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only N pending studies.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=8,
        help="Concurrent API requests.",
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(
            args.input_csv,
            usecols=list(ReportRow.__annotations__),
            dtype='string',
        )

        # malformed CSV are skipped loudly before reaching the API
        # (no API spend wastage)
        missing_mask = df.isna().any(axis=1)
        if missing_mask.any():
            df = df[~missing_mask]
            logger.warning("""Skipping %d row(s) missing StudyInstanceUID/Report.""", missing_mask.sum())

        # output is keyed by StudyInstanceUID.
        # a duplicate UID in the input would otherwise reach the API twice and produce two records for one study;
        # removing it here enforces one-record-per-UID in the output.
        dup_mask = df["StudyInstanceUID"].duplicated(keep="first")
        if dup_mask.any():
            df = df[~dup_mask]
            logger.warning("""Skipping %d row(s) with a duplicate StudyInstanceUID (keeping first occurrence).""", dup_mask.sum())


        # [Prompt.md + few_shot_examples.jsonl] remains the same throughout the session,
        # only the current report changes per call.
        static_prefix = load_static_prefix()

        # the failed cases (if any) require manual review, not an automatic retry:
        # without a zero-temperature pin, 
        # a rerun could produce a different result in principle, 
        # but different isn't the same as fixed — human-in-the-loop 
        # is the right choice rather than statistical resampling.
        # [both success/failure records are considered complete]
        done_ids = load_done_ids(OUTPUT_JSONL)
        todo = df[
            ~df["StudyInstanceUID"].isin(done_ids)
        ]
        if args.limit is not None:
            todo = todo.head(args.limit)

    except Exception as e:
        logger.error(f"""failed preprocessing: {e}""")
        raise 

    logger.info(
        """Total (non-null) rows: %d | Already processed: %d | Pending: %d""",
        len(df), len(done_ids), len(todo),
    )
    if todo.empty: # no pending
        return

    logger.info(
        """Model=%s | reasoning=%s | evidence_limit=%d""",
        MODEL_NAME, REASONING_EFFORT, EVIDENCE_WORD_LIMIT,
    )
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            """OPENAI_API_KEY is not set. Set it in the environment before running."""
        )
    client = OpenAI()

    n_success = 0
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # process the first report serially,
    # so the shared static prefix is established in the prompt cache 
    # before the concurrent burst.
    # keep appending output to the given path (don't overwrite existing results).
    with OUTPUT_JSONL.open("a", encoding="utf-8") as out_f:

        first_row = todo.iloc[0]
        logger.info(
            """Establishing prompt cache with first request: %s""",
            first_row.StudyInstanceUID,
        )
        record, success = classify_one(
            client,
            static_prefix,
            first_row,
        )
        _write_record(out_f, record)
        n_success += int(success)

        remaining = todo.iloc[1:]
        workers = max(args.max_workers, 1)
        if not remaining.empty:
            logger.info(
                """Starting %d remaining reports with max_workers=%d""",
                len(remaining), workers,
            )
            n_success += process_concurrent(
                client,
                static_prefix,
                remaining, out_f,
                workers,
            )

    logger.info(
        """DONE | success this session: [%d/%d] | output:%s""",
        n_success, len(todo),
        OUTPUT_JSONL,
    )


class ReportRow(Protocol):
    # Type hint only; 
    # this class is never instantiated explicitly.
    StudyInstanceUID: str
    Report: str

def _write_record(out_f: TextIO, record: dict):
    # write results to disk
    out_f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    out_f.flush()



# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    main()