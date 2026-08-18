"""`verify` — adversarial checks published beside the report (ARCH §8, §3.3).

Severity is part of each finding, because the failures mean different things:

- **Cross-user leakage** invalidates the run. A leaking system is not solving the
  benchmark task, so scoring it zero would misrepresent what happened.
- **Unmetered model use** withholds the row. If gateway activity does not reconcile
  with phase activity, the cost columns are describing traffic we cannot see.
- **Rank instability** is reported as variance, never as a failure. A black box cannot
  be seeded; measuring non-determinism is honest, calling it a defect is not.

The phase-ordering check lives here rather than with the stages because this is where
violating it corrupts a published number: D6(a) derives system-side per-phase cost
from the ingest/retrieve boundary, which only means anything if the phases are disjoint.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from orchestrator.artifacts import JsonlArtifact, read_json, write_json
from orchestrator.metrics import Usage

Severity = Literal["invalidates_run", "withholds_row", "informational"]
# `derived` is the only value that claims a real number: the counter diff across the
# phase boundary. `unreliable` means the boundary was broken, `unavailable` means there
# was never a counter to diff. They are three different statements and stay distinct.
CostAttribution = Literal["derived", "unreliable", "unavailable"]


@dataclass(frozen=True)
class CheckOutcome:
    """One check's result. `detail` explains a failure in one line."""

    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class StabilityOutcome:
    """Rank stability never fails; `tau` is None when there was nothing to compare."""

    passed: bool
    tau: float | None
    samples: int = 0


class Finding(BaseModel):
    """One verification check as published."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    severity: Severity
    detail: str = ""


class Verification(BaseModel):
    """`verify.json` — the adversarial checks for one run (ARCH §8)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verify_version: int = 1
    run_id: str
    system: str
    run_valid: bool
    row_withheld: bool
    cost_attribution: CostAttribution
    rank_stability_tau: float | None = None
    flags: list[str] = []
    findings: list[Finding] = []


def _bounds(rows: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Return the earliest start and latest end across rows, or None if untimed."""
    starts = [int(r["started_at_ms"]) for r in rows if r.get("started_at_ms")]
    ends = [int(r["ended_at_ms"]) for r in rows if r.get("ended_at_ms")]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def check_phase_ordering(
    ingest_rows: list[dict[str, Any]], retrieve_rows: list[dict[str, Any]]
) -> CheckOutcome:
    """Assert ingest fully drained before retrieve opened — the D6(a) invariant."""
    ingest = _bounds(ingest_rows)
    retrieve = _bounds(retrieve_rows)
    if ingest is None or retrieve is None:
        # An in-process baseline has no ingest phase; nothing to order against.
        return CheckOutcome(passed=True, detail="no ingest phase to order against")

    _, ingest_end = ingest
    retrieve_start, _ = retrieve
    if retrieve_start < ingest_end:
        return CheckOutcome(
            passed=False,
            detail=(
                f"phases overlap: retrieve began at {retrieve_start} but ingest ran until "
                f"{ingest_end}, so system-side per-phase cost is not attributable"
            ),
        )
    return CheckOutcome(passed=True)


def check_unmetered_model_use(
    answered: int, judged: int, usage: dict[str, Usage], metered: bool = True
) -> CheckOutcome:
    """Reconcile gateway call counts against phase activity (ARCH §8)."""
    if not metered:
        # No gateway means no counters to reconcile against. Unmeasurable is not the
        # same finding as unmetered, and reporting it as one would cry wolf every run.
        return CheckOutcome(passed=True, detail="no gateway: model use is unmetered by design")

    silent = [
        phase
        for phase, records in (("answer", answered), ("judge", judged))
        if records and usage.get(phase, Usage()).requests <= 0
    ]
    if silent:
        return CheckOutcome(
            passed=False,
            detail=(
                f"{', '.join(silent)} produced records but the gateway metered no calls; "
                "a model was reached without passing through the gateway"
            ),
        )
    # More gateway calls than records is normal: retries spend calls without a record.
    return CheckOutcome(passed=True)


def kendall_tau(first: list[str], second: list[str]) -> float | None:
    """Return Kendall τ over the items present in both rankings, or None if too few."""
    shared = [item for item in first if item in set(second)]
    if len(shared) < 2:
        return None
    position = {item: index for index, item in enumerate(second)}

    concordant = discordant = 0
    for left in range(len(shared)):
        for right in range(left + 1, len(shared)):
            delta = position[shared[right]] - position[shared[left]]
            if delta > 0:
                concordant += 1
            elif delta < 0:
                discordant += 1

    pairs = concordant + discordant
    return (concordant - discordant) / pairs if pairs else None


def rank_stability(repeats: list[tuple[list[str], list[str]]]) -> StabilityOutcome:
    """Return mean τ across repeated retrievals. Variance, never a failure (ARCH §8)."""
    taus = [tau for first, second in repeats if (tau := kendall_tau(first, second)) is not None]
    if not taus:
        return StabilityOutcome(passed=True, tau=None)
    return StabilityOutcome(passed=True, tau=sum(taus) / len(taus), samples=len(taus))


def _rows(run_dir: Path, name: str) -> list[dict[str, Any]]:
    path = run_dir / name
    return list(JsonlArtifact(path).stream()) if path.is_file() else []


def run_verify(
    run_dir: Path,
    usage: dict[str, Usage],
    contended: bool = False,
    leaked: bool = False,
    repeats: list[tuple[list[str], list[str]]] | None = None,
    metered: bool = True,
) -> Verification:
    """Run the checks over a finished run's artifacts and write `verify.json`."""
    manifest = read_json(run_dir / "manifest.json")
    ingest_rows = _rows(run_dir, "ingest.jsonl")
    retrieve_rows = _rows(run_dir, "retrieve.jsonl")

    ordering = check_phase_ordering(ingest_rows, retrieve_rows)
    metering = check_unmetered_model_use(
        answered=len(_rows(run_dir, "answer.jsonl")),
        judged=len(_rows(run_dir, "judge.jsonl")),
        usage=usage,
        metered=metered,
    )
    stability = rank_stability(repeats or [])

    findings = [
        Finding(
            name="cross_user_leakage",
            passed=not leaked,
            severity="invalidates_run",
            detail="" if not leaked else "a sibling user's canary was retrievable",
        ),
        Finding(
            name="unmetered_model_use",
            passed=metering.passed,
            severity="withholds_row",
            detail=metering.detail,
        ),
        Finding(
            name="phase_ordering",
            passed=ordering.passed,
            severity="withholds_row",
            detail=ordering.detail,
        ),
        Finding(
            name="rank_stability",
            passed=True,
            severity="informational",
            detail="" if stability.tau is None else f"mean Kendall tau {stability.tau:.3f}",
        ),
    ]

    # The override in 3.5.1 breaks the boundary by construction, so a contended run
    # cannot claim derived cost even when its own phases happened not to overlap.
    attribution: CostAttribution = "derived"
    if not metered:
        attribution = "unavailable"
    elif contended or not ordering.passed:
        attribution = "unreliable"

    verification = Verification(
        run_id=str(manifest["run_id"]),
        system=str(manifest["system"]["name"]),
        run_valid=not leaked,
        row_withheld=any(
            not f.passed for f in findings if f.severity in ("withholds_row", "invalidates_run")
        ),
        cost_attribution=attribution,
        rank_stability_tau=stability.tau,
        flags=[f.name for f in findings if not f.passed],
        findings=findings,
    )
    write_json(run_dir / "verify.json", verification.model_dump(mode="json"))
    return verification
