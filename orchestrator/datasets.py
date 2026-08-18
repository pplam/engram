"""Dataset plugin loading (ARCH §9).

A dataset is a directory: `dataset.yaml` + `adapter.py`. The loader validates the
manifest, verifies the source hash, imports the adapter, and calls its `load`.
It contains no branch on dataset name — adding a dataset never touches this file.
"""

import hashlib
import importlib.util
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

TaskType = Literal["open_qa", "mcq", "rubric", "ordering"]
Granularity = Literal["message", "session"]
DEFAULT_DATASETS = Path("datasets")

Message = dict[str, Any]


class DatasetError(Exception):
    """A dataset is missing, malformed, or its pinned source hash does not match."""


@dataclass(frozen=True)
class Context:
    """One ingestible corpus. Many questions may share it, or exactly one may."""

    ctx_id: str
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class Question:
    """One scorable question, bound to the context whose memories should answer it."""

    q_id: str
    ctx_id: str
    question: str
    gold: tuple[str, ...]
    options: tuple[str, ...] | None = None
    evidence_chunk_ids: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    url: str | None = None


class OfficialJudge(BaseModel):
    """A judge model pinned by the dataset, overriding the suite default (D7).

    `provider` names the block in `gateway/bifrost.json` that serves this model (D11).
    It is optional only so that a dataset written against suite v1 still loads; a
    dataset pinning a model no OpenAI endpoint serves — LoCoMo-Refined's `qwen3-14b` —
    has to name it, or the gateway has nowhere to route the call.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider: str | None = None
    temperature: float = 0


class DatasetConfig(BaseModel):
    """Validated `dataset.yaml`, plus the resolved path of its pinned source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    source: _Source
    task_type: TaskType
    scorer: str
    supports_recall: bool
    evidence_granularity: Granularity
    official_prompt: str | None = None
    official_judge: OfficialJudge | None = None
    root: Path = Path()

    @property
    def source_path(self) -> Path:
        """Absolute path of the pinned source file."""
        return self.root / self.source.path


@dataclass(frozen=True)
class Dataset:
    """A loaded dataset: its config plus the records its adapter produced."""

    config: DatasetConfig
    contexts: tuple[Context, ...]
    questions: tuple[Question, ...]


class Adapter(Protocol):
    """The entire dataset plugin API."""

    def load(self, config: DatasetConfig) -> tuple[Iterable[Context], Iterable[Question]]:
        """Return this dataset's contexts and questions."""
        ...


def _verify_source(config: DatasetConfig) -> None:
    path = config.source_path
    if not path.is_file():
        raise DatasetError(
            f"dataset {config.name!r}: source {config.source.path} not found (looked at {path})"
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != config.source.sha256:
        raise DatasetError(
            f"dataset {config.name!r}: source {config.source.path} sha256 mismatch — "
            f"pinned {config.source.sha256[:12]}..., file is {actual[:12]}..."
        )


def _import_adapter(name: str, directory: Path) -> Adapter:
    path = directory / "adapter.py"
    if not path.is_file():
        raise DatasetError(f"dataset {name!r}: no adapter.py at {path}")

    spec = importlib.util.spec_from_file_location(f"engram_dataset_{name}", path)
    if spec is None or spec.loader is None:
        raise DatasetError(f"dataset {name!r}: adapter.py at {path} is not importable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as err:
        raise DatasetError(f"dataset {name!r}: adapter.py raised on import: {err}") from err

    if not callable(getattr(module, "load", None)):
        raise DatasetError(f"dataset {name!r}: adapter.py defines no callable load(config)")
    adapter: Adapter = module
    return adapter


def load_config(name: str, root: Path = DEFAULT_DATASETS) -> DatasetConfig:
    """Return the validated `dataset.yaml` for `name`, with its source verified."""
    directory = root / name
    path = directory / "dataset.yaml"
    if not path.is_file():
        raise DatasetError(f"no dataset {name!r}: expected a manifest at {path}")

    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise DatasetError(f"dataset {name!r}: dataset.yaml must contain a YAML mapping")

    try:
        config = DatasetConfig.model_validate({**raw, "root": directory})
    except ValidationError as err:
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in err.errors())
        raise DatasetError(f"dataset {name!r}: dataset.yaml is invalid ({fields}): {err}") from err

    if config.name != name:
        raise DatasetError(
            f"dataset directory {name!r} declares name {config.name!r}; they must match"
        )
    _verify_source(config)
    return config


def load_dataset(name: str, root: Path = DEFAULT_DATASETS) -> Dataset:
    """Return the contexts and questions for dataset `name`."""
    config = load_config(name, root)
    adapter = _import_adapter(name, root / name)
    contexts, questions = adapter.load(config)
    return Dataset(config=config, contexts=tuple(contexts), questions=tuple(questions))
