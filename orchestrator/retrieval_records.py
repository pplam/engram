"""Reading `retrieve.jsonl`, across the record-shape change (ARCH §5).

The refactor renamed `data` → `memories` and `x_source_ids` → `source_ids`. Both shapes
read, in one place, because a published row must stay recomputable from the artifacts
that produced it: `bench rescore` over a pre-refactor bundle is Phase 6's acceptance
criterion, and a reader that only understood the new shape would silently score every
old bundle as zero retrieval rather than failing.

The `x_` prefix meant "extension to the wire contract". There is no wire contract left,
so `source_ids` is now a plain field of `Memory`.
"""

from typing import Any

Row = dict[str, Any]
MemoryRow = dict[str, Any]


def memories_of(row: Row) -> list[MemoryRow]:
    """Return one retrieval's memories, in rank order, from either record shape."""
    for key in ("memories", "data"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def source_ids_of(memory: MemoryRow) -> list[str]:
    """Return a memory's source chunk ids from either record shape."""
    for key in ("source_ids", "x_source_ids"):
        value = memory.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


__all__ = ["MemoryRow", "Row", "memories_of", "source_ids_of"]
