"""The shape of `report.json` — the publication unit (ARCH §7).

Five metric families side by side and **no composite score**: any weighted index
is left to the reader, computed from published columns.

Two labels a reader needs in order not to over-read a number: which judge produced
a quality score, and whether a cost figure was measured or derived.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Attribution = Literal["measured", "derived", "unreliable", "unavailable"]
Track = Literal["exact", "unavailable"]
EvidenceUnit = Literal["message", "session"]
JudgeSource = Literal["suite", "dataset"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Quality(_Section):
    """Task quality, comparable within a dataset — never across datasets (§6)."""

    scorer: str
    accuracy: float
    questions: int
    correct: int
    judge: str | None = None
    judge_source: JudgeSource | None = None
    unparseable_judgements: int = 0
    exact_match: float | None = None
    token_f1: float | None = None
    rouge_l: float | None = None


class RetrievalQuality(_Section):
    """Recall and ranking on the labelled track, in the dataset's evidence unit (§7.2)."""

    track: Track
    evidence_unit: EvidenceUnit
    measured_questions: int
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    context_tokens_per_correct: float | None = None


class Cost(_Section):
    """Harness cost is measured; system cost is derived at the gateway (§7.3)."""

    harness_attribution: Attribution
    harness_prompt_tokens: int
    harness_completion_tokens: int
    harness_llm_calls: int
    system_attribution: Attribution
    system_prompt_tokens: int | None = None
    system_completion_tokens: int | None = None
    system_total_tokens: int | None = None
    system_write_tokens_per_chunk: float | None = None
    system_read_tokens_per_query: float | None = None


class Latency(_Section):
    """Reported alongside quality, never blended into it (§7.4)."""

    ingest_p50_ms: float | None = None
    ingest_p95_ms: float | None = None
    ingest_p99_ms: float | None = None
    ingest_wall_clock_ms: int | None = None
    retrieve_p50_ms: float | None = None
    retrieve_p95_ms: float | None = None
    retrieve_p99_ms: float | None = None


class Robustness(_Section):
    """Derived from artifacts and verify (§7.5)."""

    contract_violations: int = 0
    ingest_retries: int = 0
    retrieve_retries: int = 0
    ingest_failures: int = 0
    canary_leaks: int = 0
    rank_stability_tau: float | None = None


class Completeness(_Section):
    """What was planned versus what actually got scored."""

    planned_chunks: int
    planned_questions: int
    ingested_chunks: int
    retrieved_questions: int
    answered_questions: int
    judged_questions: int


class ReportRow(_Section):
    """One system on one dataset under one suite."""

    system: str
    dataset: str
    suite: str
    run_id: str
    scorer: str
    top_k: int
    limit: int | None = None
    in_process: bool = False
    contended: bool = False
    unpinned_image: bool = False
    # Set when `verify` found a check whose severity withholds the row (ARCH §8). The
    # numbers stay in the artifact; this marks them as not publishable.
    row_withheld: bool = False
    quality: Quality
    retrieval: RetrievalQuality
    cost: Cost
    latency: Latency
    robustness: Robustness
    completeness: Completeness


class Report(_Section):
    """`report.json`: one row, plus the caveats a reader must see."""

    report_version: int = 1
    row: ReportRow
    notes: list[str] = []
