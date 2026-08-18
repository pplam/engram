"""Append-only JSONL artifact IO (ARCH §5).

Artifacts are never rewritten in place. A stage resumes by reading the IDs already
present in its output and skipping them. Records are streamed, never slurped, and
keys are sorted so equal records are byte-identical across runs.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Record = dict[str, Any]


class ArtifactError(Exception):
    """An artifact is malformed — a truncated line, bad JSON, or a record with no id."""


def _dump(record: Record) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class JsonlArtifact:
    """One append-only JSONL file."""

    path: Path

    def append(self, record: Record) -> None:
        """Append one record. Raises if it carries no `id`."""
        if "id" not in record:
            raise ArtifactError(f"record has no id: cannot append to {self.path.name}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_dump(record) + "\n")

    def stream(self) -> Iterator[Record]:
        """Yield each record in file order. Empty for a missing file."""
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record: Record = json.loads(text)
                except json.JSONDecodeError as err:
                    raise ArtifactError(
                        f"{self.path.name} line {number} is not valid JSON "
                        f"(a crash mid-write leaves a truncated final line): {err}"
                    ) from err
                if "id" not in record:
                    raise ArtifactError(f"{self.path.name} line {number} has no id")
                yield record

    def read_ids(self) -> set[str]:
        """Return the set of record IDs already present. Empty for a missing file."""
        return {str(record["id"]) for record in self.stream()}

    def count(self) -> int:
        """Return the number of records present. Zero for a missing file."""
        return sum(1 for _ in self.stream())


def write_json(path: Path, payload: Record) -> None:
    """Write one JSON document, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Record:
    """Return the JSON document at `path`."""
    if not path.is_file():
        raise ArtifactError(f"no such file: {path.name} (looked at {path})")
    try:
        loaded: Record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ArtifactError(f"{path.name} is not valid JSON: {err}") from err
    return loaded
