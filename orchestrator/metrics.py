"""Reading per-phase cost out of the gateway's `/metrics` (§3.1, D6).

Bifrost exposes no endpoint returning per-key token counts as data — budgets report
spend, not the prompt/completion split §7.3 needs. So metering is a Prometheus scrape
snapshotted at each phase boundary, diffed on `virtual_key_id`.

The counters are cumulative and monotonic. A decrease means the gateway restarted
mid-run, which makes the delta meaningless rather than small: that raises instead of
reporting a number nobody could defend.
"""

import re
from dataclasses import dataclass

# `name{label="value",...} 123.0` — the label block is optional in the exposition
# format, and series without a virtual key belong to no phase.
_SERIES = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
_LABEL = re.compile(r'(\w+)="([^"]*)"')

_COUNTERS = {
    "bifrost_input_tokens_total": "prompt_tokens",
    "bifrost_output_tokens_total": "completion_tokens",
    "bifrost_upstream_requests_total": "requests",
}


class MetricsError(Exception):
    """The scrape could not be read, or the counters moved in a way that voids a diff."""


@dataclass(frozen=True)
class Usage:
    """What one virtual key spent. Zero fields are real zeros, not missing data."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def is_zero(self) -> bool:
        """True when this key recorded no activity at all."""
        return not (self.prompt_tokens or self.completion_tokens or self.requests)


Snapshot = dict[str, Usage]


def _value(name: str, raw: str) -> int:
    try:
        return int(float(raw))
    except ValueError as err:
        raise MetricsError(f"{name} has a non-numeric value {raw!r}") from err


def parse_metrics(text: str) -> Snapshot:
    """Return per-virtual-key usage from a Prometheus exposition scrape."""
    totals: dict[str, dict[str, int]] = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SERIES.match(stripped)
        if match is None:
            continue
        field = _COUNTERS.get(match.group("name"))
        if field is None:
            continue
        labels = dict(_LABEL.findall(match.group("labels") or ""))
        key = labels.get("virtual_key_id")
        if not key:
            continue
        # One key may span several models; a phase's cost is their sum.
        bucket = totals.setdefault(key, {})
        bucket[field] = bucket.get(field, 0) + _value(match.group("name"), match.group("value"))

    return {key: Usage(**fields) for key, fields in totals.items()}


def diff(before: Snapshot, after: Snapshot) -> Snapshot:
    """Return the usage accrued between two snapshots, omitting keys that did nothing."""
    delta: Snapshot = {}

    for key, end in after.items():
        start = before.get(key, Usage())
        for field in ("prompt_tokens", "completion_tokens", "requests"):
            if getattr(end, field) < getattr(start, field):
                raise MetricsError(
                    f"{field} for virtual key {key!r} decreased across the phase boundary; "
                    "the gateway restarted mid-run and cost is no longer attributable"
                )
        spent = Usage(
            prompt_tokens=end.prompt_tokens - start.prompt_tokens,
            completion_tokens=end.completion_tokens - start.completion_tokens,
            requests=end.requests - start.requests,
        )
        if not spent.is_zero():
            delta[key] = spent

    return delta
