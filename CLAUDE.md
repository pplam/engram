## Project

Engram is a memory benchmark. It compares memory systems as **black boxes**: the only integration surface is the `MemoryAdapter` protocol — `add` and `search` — which the harness imports and calls directly. A run is a pipeline of seven stages that append JSONL artifacts; scoring is a pure offline function over those artifacts.

The integration surface was an HTTP contract (`/add`, `/search`) through Phase 5 and became a Python protocol in Phase 6; see ARCH §4 and IMPLEMENTATION D8–D10. HTTP survives as one adapter (`adapters.http:HttpAdapter`) for containers and hosted services. The wire contract is fully gone from the code — the one place it still shows is `score`, which reads both the old `data` key and the new `memories` key so archived bundles stay rescorable. Keep that dual read; do not "simplify" it.

Design docs in `docs/`:

- **`docs/IMPLEMENTATION.md`** — the primary guide. TDD build order, per-step test cases, and the open decisions that block each phase. **Read and follow this for all implementation work.**
- `docs/ARCHITECTURE.md` — system design: adapter interface, pipeline stages, artifact shapes, metrics, verification, dataset plugin API, repo layout. The *what* and *why*; read alongside the build order.

Three artifact shapes (`plan.jsonl`, `report.json`, `verify.json`) are deliberately defined in code as Pydantic models rather than in prose — see `docs/IMPLEMENTATION.md` steps 2.5, 2.10, and Phase 3. When you define one, update ARCHITECTURE §5 so doc and code agree.

Three planes, one job each:

- **Control plane** (`orchestrator/`) — sequences stages, retries, artifact IO. Contains zero method-specific code.
- **Model plane** (`gateway/`) — [Bifrost](https://github.com/maximhq/bifrost), the only route to any LLM or embedder, for the harness *and* the systems under test. Pins models via per-key `allowed_models`, meters tokens, blocks other egress. One virtual key per `(run_id, system, phase)`; per-phase cost comes from diffing `/metrics` counters labelled by `virtual_key_id`.
- **Data plane** (`artifacts/<run>/`) — append-only JSONL; the sole input to scoring and the unit of publication.

Pipeline: `plan → ingest → retrieve → answer → judge → score → verify`. Only `ingest` and `retrieve` touch the system under test. `score` and `verify` must run with **no network access** to any memory system.

Runs are independent by construction — `user_id` embeds `run_id`, artifacts are per-run directories, and `score` is pure — so many runs execute in parallel as subprocesses. Two rules: runs against the **same** system are serialized (concurrent load would contaminate the latency and rank-stability metrics), and progress is observable via `bench ps` reading `artifacts/*/state.json`. No daemon, no database.

Project constraints:

- **Layout** per `docs/ARCHITECTURE.md` §11: `orchestrator/`, `contract/`, `adapters/`, `suites/`, `datasets/<name>/`, `scorers/`, `registry/`, `reference/`, `gateway/`. Tests mirror the package tree under `tests/`.
- **Per-system knowledge lives only in `adapters/`.** One module per system, importing its SDK inside `__init__` so a missing dependency fails one adapter rather than the CLI. Note `datasets/<name>/adapter.py` is a *dataset* adapter (a `load()` function) and unrelated to `MemoryAdapter`.
- **Python ≥ 3.12**, managed with `uv`. Lint+format: `uv run ruff check .` and `uv run ruff format .` · Types: `uv run mypy .` · Test: `uv run pytest`.
- **Lean dependencies.** `httpx`, `pydantic`, `pyyaml`, `pytest`. Add anything else only when the stdlib genuinely can't do it, and say why. CLI tables are plain-text stdlib formatting — no `rich`.
- **First datasets:** LoCoMo-Refined (CC BY-NC 4.0 — non-commercial, check before publishing) and LongMemEval-S Cleaned (MIT). A dataset may pin its own `official_prompt` and `official_judge`, which override the suite; the **resolved** judge goes in the manifest and on the report row. An `official_judge` names its own `provider` — the override replaces both halves of the `(provider, id)` pair, never one.
- **Suites are immutable, so a new pin is a new file.** `suites/v4.yaml` is current (`CURRENT_SUITE`); `v1` predates providers and stays loadable so rows published under it remain reproducible. Every version after v1 must name a `provider` per model and must not carry `retry_on`/`no_retry_on`; `orchestrator/suite.py` enforces both. From v4 on, `models.chat` also pins `max_tokens`: left unpinned, a system under test applies its own default, and one below what a reasoning model spends on reasoning yields an empty completion that the system stores as nothing while reporting success. Never edit a published suite.
- **Datasets differ in shape, and the code must not assume either.** LoCoMo has ~138 questions per context; LongMemEval-S has one private haystack per question. Evidence granularity likewise differs (message vs session), so recall carries a unit label.

## Workflow: TDD

Every change follows **Red → Green → Refactor**:

1. **Red** — write a failing test that pins the behavior. Run it; watch it fail for the right reason.
2. **Green** — write the minimum code to pass. No more.
3. **Refactor** — clean up with tests green.

Rules:

- No production code without a failing test that demands it.
- One behavior per test. Name tests for the behavior: `test_load_rejects_missing_source_hash`.
- Use `pytest.mark.parametrize` for input variations, not loops inside a test.
- Mock memory systems and model calls with a local `httpx` transport or a `pytest-httpx`-style stub. **Never real network calls in tests.**
- Conformance tests live in `contract/conformance/` and run against any adapter, including a trivial in-memory fake. No socket: instantiate the class.
- A task isn't done until `uv run pytest`, `uv run ruff check .`, and `uv run mypy .` pass. Report failures with output; don't hide them.

## Think Before Coding

- State assumptions. If `docs/ARCHITECTURE.md` is ambiguous or interpretations conflict, ask — don't pick silently. Open questions are listed in §15; adding to them beats guessing.
- If a simpler approach exists, say so before implementing.
- Name what's confusing instead of guessing.

## Simplicity First

- Minimum code that solves the problem. Nothing speculative.
- No abstractions for single-use code, no unrequested configurability, no error handling for impossible cases.
- Decouple through small protocols (the dataset `load()` function, the scorer callable), not shared state or base classes.
- A dataset plugin is `dataset.yaml` + `adapter.py`. Adding one must never require editing the orchestrator.
- If 200 lines could be 50, rewrite it.

## Surgical Changes

- Touch only what the task requires. Every changed line should trace to the request.
- Match existing style; don't refactor working code or reformat adjacent lines.
- Remove imports/vars/functions *your* change orphaned. Flag pre-existing dead code, don't delete it.

## Python Conventions

- Type-annotate every function signature and dataclass field. `mypy` strict-ish; no bare `Any` without a comment saying why.
- Prefer `@dataclass(frozen=True)` or Pydantic models over dicts for records that cross module boundaries. Artifact records are validated at the boundary, not deep inside.
- Raise specific exceptions; never `except Exception: pass`. Chain with `raise X(...) from err`.
- Propagate cancellation properly: async stages take an explicit timeout and let `asyncio.CancelledError` bubble. Bound fan-out with a semaphore from the suite's `limits`, never unbounded `gather`.
- `pathlib.Path` over string paths. `json` per line for JSONL; never load a whole artifact file into memory when streaming works.
- Modules and functions in `snake_case`, classes in `PascalCase`. Public functions get a one-line docstring stating what they return.
- Config is data: suites, registry, and dataset manifests are YAML parsed into typed models. No config branching by system name anywhere in `orchestrator/`.

## Determinism & Artifact Discipline

- Artifacts are **append-only**. Resume by skipping IDs already present in the output file; never rewrite an artifact in place.
- Store retrieval responses **verbatim** so retrieval metrics stay recomputable offline.
- `score` is a pure function: same artifacts in, same `report.json` out. No clocks, no randomness, no network.
- Write `manifest.json` before touching anything, with suite/dataset/prompt hashes and the system image digest.
- **Contract violations fail the run loudly**; a broken integration is not a score of zero. A single retryable upstream error is retried per the suite policy, not swallowed.
- **Stages are strictly sequential within a run.** System-side per-phase cost is derived from the ingest/retrieve boundary, so overlapping phases would silently corrupt a published number. Assert the ordering; don't rely on it implicitly.
- **Label derived numbers as derived.** Harness cost is measured per phase; system cost is inferred from metric snapshots. Don't merge them into one column.
- Never log API keys, virtual keys, prompts, full messages, or retrieved content. Log IDs, statuses, counts, and latencies.
