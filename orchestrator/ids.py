"""ID construction and parsing for the isolation boundary (ARCH §4.4, §4.5).

`user_id` embeds `run_id`, so a new run is a new namespace. There is no
namespace lifecycle to manage — only these strings.

A run id is generated, never supplied: 12 hex chars, Docker-style. Hex keeps it
free of `:` by construction, so it can never break `parse`, and every command
accepts an unambiguous prefix of one.
"""

import secrets
from dataclasses import dataclass
from pathlib import Path

PREFIX = "eval"
SEP = ":"
_CHUNK = "chunk-"
_QUESTION = "q-"
_RUN_ID_BYTES = 6


class IdError(ValueError):
    """Raised when an ID component is invalid or an ID cannot be parsed."""


@dataclass(frozen=True)
class ParsedId:
    """Components recovered from any of the three ID forms."""

    run_id: str
    dataset: str
    ctx_id: str
    chunk_id: str | None
    q_id: str | None


def _check(name: str, value: str) -> str:
    if not value.strip():
        raise IdError(f"{name} must not be empty")
    if SEP in value:
        raise IdError(f"{name} must not contain '{SEP}': {value!r}")
    return value


def new_run_id() -> str:
    """Return a fresh 12-hex-char run id (§4.5)."""
    return secrets.token_hex(_RUN_ID_BYTES)


def resolve_prefix(prefix: str, artifacts_root: Path) -> str:
    """Return the single run id under `artifacts_root` starting with `prefix`.

    An exact match wins outright, so naming a run in full is never ambiguous even
    when its id prefixes another. Otherwise an ambiguous prefix raises listing every
    candidate rather than picking one — guessing here would operate on the wrong run.
    """
    if not artifacts_root.is_dir():
        raise IdError(f"no runs found: {artifacts_root} is not a directory")

    candidates = sorted(p.name for p in artifacts_root.iterdir() if p.is_dir())
    if prefix in candidates:
        return prefix

    matches = [name for name in candidates if name.startswith(prefix)]
    if not matches:
        raise IdError(f"no run matches {prefix!r} under {artifacts_root}")
    if len(matches) > 1:
        listed = ", ".join(matches)
        raise IdError(f"{prefix!r} is ambiguous; it matches {len(matches)} runs: {listed}")
    return matches[0]


def user_id(run_id: str, dataset: str, ctx_id: str) -> str:
    """Return the isolation-boundary user_id `eval:<run_id>:<dataset>:<ctx>`."""
    return SEP.join(
        (
            PREFIX,
            _check("run_id", run_id),
            _check("dataset", dataset),
            _check("ctx_id", ctx_id),
        )
    )


def request_id(uid: str, chunk_id: str) -> str:
    """Return the per-chunk write id `<user_id>:chunk-<n>`."""
    return f"{uid}{SEP}{_CHUNK}{_check('chunk_id', chunk_id)}"


def question_id(uid: str, q_id: str) -> str:
    """Return the per-question read id `<user_id>:q-<n>`."""
    return f"{uid}{SEP}{_QUESTION}{_check('q_id', q_id)}"


def parse(value: str) -> ParsedId:
    """Return the components of a user_id, request_id, or question_id."""
    parts = value.split(SEP)
    if parts[0] != PREFIX:
        raise IdError(f"id must start with '{PREFIX}{SEP}': {value!r}")
    if len(parts) not in (4, 5):
        raise IdError(f"id has {len(parts)} components, expected 4 or 5: {value!r}")

    _, run_id, dataset, ctx_id = parts[:4]
    chunk_id: str | None = None
    q_id: str | None = None
    if len(parts) == 5:
        suffix = parts[4]
        if suffix.startswith(_CHUNK):
            chunk_id = suffix.removeprefix(_CHUNK)
        elif suffix.startswith(_QUESTION):
            q_id = suffix.removeprefix(_QUESTION)
        else:
            raise IdError(f"unknown id suffix {suffix!r} in {value!r}")

    return ParsedId(run_id=run_id, dataset=dataset, ctx_id=ctx_id, chunk_id=chunk_id, q_id=q_id)
