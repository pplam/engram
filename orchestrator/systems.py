"""Listing what `bench run --system` accepts (§3.5, §10, §11).

A runnable name comes from one of three places, and they are not interchangeable:

- **registry** — a YAML entry naming an adapter, which is how a real system is benchmarked.
- **baseline** — `no_memory` and `bm25`, in-repo `MemoryAdapter`s reachable without a
  registry entry at all.
- **privileged** — `oracle_gold` and `long_context`, which bypass the interface and emit
  `retrieve.jsonl` directly (D2). They have no adapter to name, so they appear in neither
  of the above and were the hardest names to find.

The kind is on every row because it decides how a row's numbers may be read: a privileged
baseline sees gold evidence no real system is given, so its quality column is an upper
bound rather than a comparable result.

Nothing here imports an adapter. `load_system` reads YAML and stops, which is what lets
this describe an entry whose SDK is absent — the case the command is most useful for.
`bench doctor` is where importing and constructing is supposed to fail.
"""

from dataclasses import dataclass
from pathlib import Path

from orchestrator.registry import DEFAULT_REGISTRY, RegistryError, load_system
from orchestrator.run import IN_PROCESS
from reference.baselines import BASELINE_ADAPTERS

# What a row's numbers may be compared with, widest first: a registered system is the
# real measurement, and the two baseline kinds are reference points of different strength.
KINDS = ("registry", "baseline", "privileged")


@dataclass(frozen=True)
class SystemRow:
    """One line of `bench systems`, already resolved for display."""

    name: str
    kind: str
    # What the entry runs, as `module:Class`. Empty for both baseline kinds: one is built
    # by name from `BASELINE_ADAPTERS` and the other has no adapter at all, so inventing
    # a value here would imply a registry entry that does not exist.
    adapter: str = "-"
    # Why a registry entry could not be read, when it could not be. The entry is still
    # listed: one malformed file must not hide the others, the way `bench ps` keeps an
    # unreadable run visible rather than dropping it.
    detail: str = ""


def _registry_rows(root: Path) -> list[SystemRow]:
    """Return one row per `*.yaml` in `root`, sorted, reading each and importing none."""
    if not root.is_dir():
        return []
    rows: list[SystemRow] = []
    for path in sorted(root.glob("*.yaml")):
        name = path.stem
        try:
            spec = load_system(name, root)
        except RegistryError as err:
            # The message names the file and the field, which is what makes a bad entry
            # fixable from the listing alone.
            rows.append(SystemRow(name=name, kind="registry", adapter="-", detail=str(err)))
        else:
            rows.append(SystemRow(name=name, kind="registry", adapter=spec.adapter))
    return rows


def collect_systems(registry: Path = DEFAULT_REGISTRY) -> list[SystemRow]:
    """Return every name `bench run --system` accepts, grouped by kind."""
    return [
        *_registry_rows(registry),
        *(SystemRow(name=name, kind="baseline") for name in sorted(BASELINE_ADAPTERS)),
        *(SystemRow(name=name, kind="privileged") for name in sorted(IN_PROCESS)),
    ]


def render_table(rows: list[SystemRow]) -> str:
    """Render rows as an aligned plain-text table (stdlib only, no `rich`)."""
    if not rows:
        return "no systems"

    headers = ("SYSTEM", "KIND", "ADAPTER")
    body = [(row.name, row.kind, row.adapter) for row in rows]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *body, strict=True)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()]
    for row, entry in zip(rows, body, strict=True):
        lines.append(
            "  ".join(cell.ljust(w) for cell, w in zip(entry, widths, strict=True)).rstrip()
        )
        if row.detail:
            # Indented under its own row rather than in a column: a validation message is
            # a sentence, and padding every other row to its width would be unreadable.
            lines.append(f"    ! {row.detail}")
    return "\n".join(lines)


__all__ = ["KINDS", "SystemRow", "collect_systems", "render_table"]
