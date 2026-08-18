# Engram — Implementation Guide

TDD build order for the architecture in `docs/ARCHITECTURE.md`. That doc is *what* and *why*; this one is *in what order*, and *which test comes first*. It adds no design decisions — where a decision is still open it says so and stops.

Read `../CLAUDE.md` for the workflow rules. Every step below is Red → Green → Refactor, and no step is done until:

```bash
uv run pytest && uv run ruff check . && uv run mypy .
```

Phases 1–5 map 1:1 onto the migration table in ARCHITECTURE §13; Phase 3.5 is an addition (parallel runs and run monitoring) that §13 doesn't cover. Ship a phase before starting the next; each has an exit criterion you can demonstrate.

---

## Step 0 — Decisions (resolved)

| # | Decision | Resolution |
| --- | --- | --- |
| D1 | First datasets | **LoCoMo-Refined** and **LongMemEval-S Cleaned** (`longmemeval_s_cleaned.json`) |
| D2 | `oracle_gold` / `long_context` | **Privileged baselines** that bypass the integration surface |
| D3 | Gateway phase attribution | **One virtual key per `(run_id, system, phase)`** |
| D4 | Chunking tie-break | **`max_messages_per_chunk` wins** |
| D5 | Gateway | **[Bifrost](https://github.com/maximhq/bifrost)** (`maximhq/bifrost`), not LiteLLM |
| D6 | Phase-scoped key for a black-box system | **(a)** One key per `(run_id, system)`; phase split from the metric-snapshot boundary |
| D7 | Dataset judge vs suite judge | **(a)** Dataset wins — `dataset.yaml` may pin `official_judge` |
| D8 | Integration surface | **Python `MemoryAdapter` protocol**, imported and called directly. HTTP becomes one adapter, not the mandatory surface |
| D9 | Agent Memory Leaderboard wire compatibility | **Dropped.** Keep its ideas (retrieve-only, `user_id` isolation), not its JSON |
| D10 | `run_id` | **Generated** — 12 hex chars, Docker-style, prefix-matchable. `--resume` replaces id reuse |
| D11 | Non-OpenAI pinned models | **A model is `(provider, id)`**; a provider is a base URL + credential in gateway config. Closes the D7 follow-up |

**D8** is the refactor the second revision turns on. The old design made HTTP mandatory so that foreign runtimes stayed out of our interpreter; the honest accounting is that this forced every in-process library to be wrapped in a server we had to write, run, and debug across a process boundary — roughly 200 lines per system, for a benchmark whose entire value is comparing systems. An adapter is now ~25 lines and needs no port.

The cost is real and is stated rather than hidden: dependency conflicts return, and an in-process system cannot be network-isolated, so its gateway metering is `unenforced` (ARCH §3, §14). Four things bound it — lazy imports inside `__init__`, one process per run, optional extras for conflicting SDKs, and `HttpAdapter` as a permanent escape hatch for anything that should stay in a container.

**D9** follows from D8: once the harness calls methods, an envelope with `success: true`, three echoed IDs, a `session_id` nobody may filter on, and an `x_` prefix namespace are all pure overhead. `Memory` drops to `content` + `score` + `source_ids`; `id` and `created_at` go because grep confirms no scorer or stage read either.

**D10** removes a footgun rather than adding convenience. `--run-id` was required, and because artifacts are append-only and resume by skipping present IDs, reusing an id *silently resumes* — so a typo could splice two configurations into one published row. Resuming becomes explicit (`--resume`), with a manifest match required on system, dataset, suite, and limit.

`--resume <id>` takes no other arguments: `resumed_config()` reads that configuration back out of the manifest, since the only value a retyped argument may hold is the one already recorded. `--suite` therefore defaults to `None` rather than `CURRENT_SUITE` — a flag that cannot distinguish "not passed" from "passed explicitly" would silently request the current suite when resuming a run pinned to an older one. `limit` is included in the match for the same reason it is inherited: it decides how much of the dataset was planned, and a resume that omitted it replanned the run at a different size than its artifacts were built for.

**D11** closes the item D7 left dangling: LoCoMo-Refined pins Qwen3-14B, the first `gateway/bifrost.json` hardcoded one OpenAI provider and three OpenAI model names, and nothing reconciled the two. A model becomes `(provider, id)`; a provider is a base URL plus a credential (ARCH §6.1). Verified against Bifrost's source while writing this:

- `vllm` and `ollama` are **first-class providers** whose per-key `url` is a `SecretVar`, so `env.*` resolves and the endpoint never sits in the config file. Prefer this route.
- Anything else OpenAI-compatible is a **custom provider**: `network_config.base_url` plus `custom_provider_config.base_provider_type: "openai"`.
- **`network_config.base_url` is a plain `string`, not a `SecretVar`** — `env.FOO` there is used literally and silently fails. Only `SecretVar` fields resolve `env.*`. This one is worth a test in 3.3.
- A virtual key's restriction is `provider_configs[].allowed_models`, and each entry carries the `provider` it applies to. The current `_create_key` posts a bare model list with no `provider`, which on a multi-provider gateway restricts nothing — so it is a pinning bug, not a cosmetic gap.

D2 means `orchestrator/` stays method-free but `reference/` gains two baselines that receive plan data directly instead of over HTTP. They skip the `ingest`/`retrieve` stages and inject a synthetic `retrieve.jsonl` — gold evidence chunks for `oracle_gold`, the whole corpus for `long_context`. Everything downstream is unchanged, which is what keeps them comparable. They are exempt from the §1.5 conformance suite; note that in the report so a reader knows those two rows came from a different path.

D4 means: fill until `max_messages_per_chunk`, and `max_words_per_chunk` is a **secondary cut** inside that window. A single message over the word limit is split at a sentence boundary (`boundary: message_or_sentence`) and its parts keep the parent `chunk_id` with an ordinal suffix, so provenance survives the split.

**D6(a)** keeps the contract untouched: a system gets one key for the whole run, and its per-phase cost comes from snapshotting `/metrics` at each phase boundary. This is sound **only because ingest and retrieve never overlap within a run**, which makes that non-overlap a load-bearing invariant rather than an implementation detail. Two consequences to enforce in code, not prose:

- Stages stay strictly sequential per run. If anything ever pipelines retrieve behind ingest, system-traffic cost attribution silently becomes wrong — assert the ordering (3.3) rather than trusting it.
- `--allow-concurrent-same-system` (3.5.1) breaks the invariant by construction. Those runs must record `cost_attribution: "unreliable"` in `verify.json` and omit the per-phase cost columns instead of printing numbers that can't be defended.

**D7(a)** makes the judge a per-dataset constant. `dataset.yaml` gains an optional `official_judge` (model id + params) beside `official_prompt`; when set, it overrides the suite's judge for that dataset. Requirements this creates:

- The **effective** judge, not the suite judge, goes in `manifest.json` and on every report row. A reader must never have to infer which judge produced a number.
- Cross-dataset quality comparison is now explicitly out of scope. Say so in the report rather than letting a reader average across datasets that used different judges.
- The suite judge remains the default for datasets that declare none, so ARCH §6 still holds for everything else.
- LoCoMo-Refined pins Qwen3-14B — a non-OpenAI model, so it must be a Bifrost-routable provider. **Resolved by D11:** a pinned model is `(provider, id)`, and `qwen3-14b` is served by a `vllm` provider block in `gateway/bifrost.json`. Confirm during 2.1 that the judge's provider exists in the gateway config; a missing provider must fail at run start, not in 2.9.

This resolves ARCH §15.1 for these two datasets. Open questions ARCH §15.2–15.3 (fuzzy-matcher calibration, per-dataset `top_k`) block *publishing numbers*, not building. Leave them open.

### Still open

Nothing blocking. Fold in when the adapters are written (2.3): whether LoCoMo's `is_multi_modality` questions are in or out, and whether LongMemEval-S's abstention questions (if `_s_cleaned` retains them) are in scope — abstention needs a scorer none of 2.10's three provide.

### Dataset notes — verify before 2.3

- **LoCoMo-Refined** — [`mem-eval-suite/LoCoMo_refined`](https://github.com/mem-eval-suite/LoCoMo_refined). 1,382 questions over 10 conversations; `data/public/questions.jsonl` + `conversations.jsonl`. Question fields map cleanly onto our `Question`: `qa_id`→`q_id`, `sample_id`→`ctx_id`, `answer` is a **list of acceptable candidates**, `evidence_messages`→`evidence_chunk_ids`. Categories: single-hop, temporal, multi-hop, open-domain. **License CC BY-NC 4.0 — non-commercial.** Flag before publishing a leaderboard. `is_multi_modality` questions need an explicit include/exclude decision (§9 has no stance on multimodal).
- **LongMemEval Cleaned — `longmemeval_s_cleaned.json`** (MIT) from [`zhangdw/Anchor-benchmarks`](https://huggingface.co/datasets/zhangdw/Anchor-benchmarks); "removes noisy sessions interfering with answer correctness" vs the [upstream release](https://github.com/xiaowu0162/LongMemEval). 500 questions. Fields: `question_id`, `question_type`, `question`, `answer`, `haystack_sessions`, `haystack_dates`, `answer_session_ids`. Types: single-session-user, single-session-assistant, single-session-preference, multi-session, temporal-reasoning, knowledge-update. `_m_cleaned` (~500 sessions/question) is out of scope for now; `oracle` is structurally our `oracle_gold` baseline, not a dataset.

Both are content-pinned by `source.sha256` per §9. Record release date and the contamination note for each.

#### The two datasets have opposite shapes — this drives 2.3 and 2.5

| | LoCoMo-Refined | LongMemEval-S Cleaned |
| --- | --- | --- |
| Contexts | 10 conversations | **500 — one private haystack per question** |
| Questions | 1,382 | 500 |
| Questions per context | ~138 | **1** |
| Ingest volume | 10 conversations | **~40 sessions × 500 ≈ 115k tokens × 500** |
| Evidence handle | `evidence_messages` | `answer_session_ids` (session-level, not message-level) |

Three consequences:

1. **`Context` must be a first-class many-to-one relation, not an afterthought.** A 1:1 question:context dataset and a 138:1 one both have to fall out of the same `load()` signature — which ARCH §9's tuple-of-iterables already supports, so no API change, but write the fixture dataset in 2.3 with *both* shapes or the loader will quietly bake in one assumption.
2. **Ingest cost is ~50M tokens for one LongMemEval-S run** — 500 haystacks, no sharing, each re-ingested per system. That is the dominant cost in the whole benchmark and the reason `--limit` (2.5) matters more than it looks. Smoke-test with `--limit 5` always; budget a full run deliberately. Also the strongest argument for Phase 3.5's parallelism.
3. **Evidence is session-level, so exact recall needs a mapping decision.** `answer_session_ids` names sessions; our chunks come from D4's message-window chunking, so one session spans ≥1 chunk. The adapter must map session → the `chunk_id`s derived from it, and a chunk is "gold" if its parent session is. Recall@k is then measured against a coarser unit than LoCoMo's, so **label the granularity in the report** — comparing LoCoMo's message-level recall to LongMemEval's session-level recall as if they were one metric would be wrong.

Verify while writing the adapter (2.3): whether `_s_cleaned` keeps upstream's abstention questions (an `_abs` question-id convention upstream — abstention needs a scorer that rewards *not* answering, which none of 2.10's three handle), and whether `haystack_dates` should populate the contract's per-message `timestamp` field. The latter is probably yes, and temporal-reasoning questions are unanswerable without it.

---

## Phase 1 — Adapter interface, conformance, `bench doctor`

**Exit:** the conformance suite passes against a trivial in-repo fake adapter, and `bench doctor <system>` reports pass/fail per check. Nothing here touches a real memory system or a model.

### 1.1 IDs — `orchestrator/ids.py`

The isolation boundary is a string (§4.4), so it gets a function and tests, not string-formatting scattered across stages.

- `new_run_id()` → 12 lowercase hex chars from `secrets.token_hex(6)` (§4.5)
- `user_id(run_id, dataset, ctx_id)` → `eval:<run_id>:<dataset>:<ctx>`
- `request_id(user_id, chunk_id)` → `<user_id>:chunk-<n>`
- `question_id(user_id, q_id)` → `<user_id>:q-<n>`
- `parse(...)` → round-trips every form above.
- `resolve_prefix(prefix, artifacts_root)` → the one run id starting with `prefix`.

Tests: round-trip property for each form; rejects a component containing `:`; a different `run_id` yields a different `user_id` for identical dataset+ctx (this *is* the namespace guarantee). For the generated id: matches `^[0-9a-f]{12}$`, never contains `:` (so it can never break `parse`), and 10k generated ids collide zero times. For prefix resolution: an exact match wins even when it prefixes another id, an ambiguous prefix raises listing every candidate, and an unmatched prefix raises naming the prefix — three separate tests, because "picks the first match" is the bug this function exists to prevent.

`session_id` is not implemented, and now never will be: it is removed from the interface (§4.2). It was grouping-only, never a filter, and nothing read it.

### 1.2 The interface — `contract/adapter.py`

The `MemoryAdapter` protocol plus the two frozen dataclasses it exchanges (§4.1–4.2):

- `Message` — `role`, `content`, `timestamp: int | None`
- `Memory` — `content`, `score: float | None`, `source_ids: Sequence[str]`
- `AdapterError` and `ContractViolation` (§4.3)

Tests: a `Memory` with only `content` is valid and reports `source_ids == ()`; both records are frozen (mutation raises); a class implementing both methods satisfies `isinstance(x, MemoryAdapter)` under `runtime_checkable`, and one missing `search` does not. Keep these dataclasses free of Pydantic — they are called once per chunk and once per question in the hot path, and there is no untrusted wire input left to validate. Validation now belongs at the *artifact* boundary, which 2.4 already owns.

There is no OpenAPI document to keep in sync, so the old 1.3 (spec-vs-model drift test) disappears along with `contract/openapi.yaml`.

### 1.3 Adapter loading — `orchestrator/registry.py`

`registry/<system>.yaml` is now `name`, `adapter` (an `module:Class` target), and an optional `config` mapping passed to `__init__` verbatim.

Resolving an adapter is `importlib.import_module` plus `getattr`, and every failure mode needs its own message: an unimportable module, a missing attribute, a target that is not a class, a class that does not satisfy the protocol, and a `__init__` that raises (a missing SDK, typically). Each becomes a `RegistryError` naming the system and the target.

Tests: each of those five failures produces a distinct message; a valid entry constructs; `config` reaches `__init__` unchanged. **Also test that the import happens lazily** — loading `registry/mem0.yaml` must not import `mem0` until the class is constructed, which is the guard against one broken SDK taking down the whole CLI (ARCH §14).

The `auth` block is gone: credentials are the adapter's own business now, read from its own environment exactly as the system's SDK would (§4.3). Nothing in the registry names a credential, so nothing in the registry can leak one.

### 1.4 Fake adapters — `reference/fakes.py`

A minimal conformant adapter: dict keyed by `user_id`, substring scoring, honours `top_k`, reports `source_ids`. No ASGI app, no transport, no socket — tests instantiate the class.

Also write the **deliberately misbehaving** variants now; they are the only way to test that conformance actually fails. One flag each: leaks across `user_id`, returns `top_k + 5` items, returns memories with empty `content`, drops `source_ids`, and indexes late (searchable only on the *second* search after `add` returns).

Two flags from the old mock are deleted with the fields they targeted: `drop_success` (no envelope to drop) and `mangle_request_id` (no echo to mangle).

### 1.5 Conformance suite — `contract/conformance/`

`pytest` over an adapter *factory* rather than a base URL, so it runs against the fake in CI and against any registered system on demand (`--system mem0`). One test per §8 row checkable without a gateway: cross-user leakage, searchable-on-return, `len(memories) ≤ top_k`, content present.

Dropped: `health` (no endpoint; a failing `add` is the signal) and `auth_accepted` (no harness-supplied credential to accept — an adapter that cannot authenticate fails to construct, which 1.3 already covers).

Tests-of-tests: each misbehaving fake from 1.4 must fail exactly the check it targets and pass the rest.

### 1.6 `bench doctor` — `orchestrator/cli.py`

Loads the adapter via 1.3, runs 1.5 against it, prints a per-check table, exits non-zero on any failure. Constructs the adapter with a generated `run_id` so probe writes land in a throwaway namespace, and always calls `close()`.

Tests: a registry naming an unimportable adapter fails with a message identifying the target; the good fake passes every check; a misbehaving fake exits non-zero. Add one warning-path test: an in-process adapter whose config does not point at the gateway prints a metering warning but does not fail (ARCH §3) — a warning, because we cannot prove where a library sends its traffic, only that it was not *told* to use the gateway.

---

## Phase 2 — Stages 1–6 with `no_memory` and `bm25`

**Exit:** `bench run` produces a `report.json` end-to-end for D1's dataset, with `no_memory` and `bm25` rows. Still no gateway — model calls go through one seam that Phase 3 replaces.

### 2.1 Suite loading — `orchestrator/suite.py`

Typed model of `suites/v1.yaml` (§6). Write `suites/v1.yaml` itself from the doc's block. The suite judge is now the *default* rather than the only judge (D7), so pick a reasonable one and note that both first datasets override it.

Each pinned model is a **`(provider, id)` pair**, not a bare id (D11, ARCH §6.1). `provider` names a block in `gateway/bifrost.json`, which is what lets a suite pin a non-OpenAI model — LoCoMo-Refined's `qwen3-14b` is exactly that case, and it is why the bare-id shape was never going to work. Add `provider` to `ChatModel`, `EmbeddingModel`, and `JudgeModel`, and test that a model missing it fails validation rather than defaulting to `openai`: a silent default is how a run ends up measuring a model nobody chose.

`resolve_judge` now returns a pair, so its four-combination test (2.9) covers the provider too. A dataset's `official_judge` must name its own provider for the same reason.

Prompt refs carry a `sha256`; loading verifies the file against it and aborts on mismatch. Test that first — it's the drift detector the whole reproducibility claim rests on.

### 2.2 Chunking — `orchestrator/chunking.py`

Pure function: messages in, chunks out. Per D4, `max_messages_per_chunk` is the primary bound and the word limit cuts inside it. Table-driven tests: exactly at the message limit, exactly at the word limit, both at once (message limit must win), one message longer than `max_words_per_chunk` (splits at a sentence boundary, parts keep the parent `chunk_id` plus ordinal), empty input.

Deterministic and side-effect free, so the tests are exhaustive rather than illustrative.

### 2.3 Dataset plugin loader — `orchestrator/datasets.py`

Discovers `datasets/<name>/`, validates `dataset.yaml`, verifies `source.sha256`, imports `adapter.py`, calls `load(config)`. `Context` and `Question` as frozen dataclasses.

Tests use fixture datasets under `tests/datasets/fixtures/` — tiny, committed, hash-pinned. Per the shape table in Step 0, commit **two**: one many-questions-per-context (LoCoMo-shaped) and one question-per-context (LongMemEval-shaped). A loader that passes only the first will bake in an assumption that breaks on the second.

A hash mismatch must abort. The loader must contain no branch on dataset name.

### 2.4 Artifact IO — `orchestrator/artifacts.py`

Append-only JSONL: `append(record)`, `read_ids()`, `stream()`. Resume is `read_ids()` then skip.

Tests: append-then-stream round-trips; `read_ids` on a missing file returns empty, not an error; a truncated final line (crash mid-write) is reported, not silently dropped; re-running a stage over existing output adds nothing.

### 2.5 `plan` — `orchestrator/stages/plan.py`

**Define `plan.jsonl` here as a Pydantic model** — ARCH §5 names the file and omits the shape. It needs at minimum: chunk records (`id`, `user_id`, `ctx_id`, `chunk_id`, messages) and question records (`id`, `user_id`, `ctx_id`, `question`, `options?`, `gold`, `evidence_chunk_ids?`). Write the model, round-trip it, and add the resulting shape to §5's table so doc and code agree.

Also writes `manifest.json` (§12) *before anything else runs*. Test that a run aborts if the manifest can't be written.

Deterministic: same dataset + suite → byte-identical `plan.jsonl`. Assert that.

`--limit N` belongs here — it truncates the plan, so every downstream stage inherits it for free. Given LongMemEval-S's ~50M-token full run (Step 0), a working `--limit` is the difference between a 2-minute smoke test and an expensive mistake. Test that `--limit` keeps questions and their contexts consistent: limiting questions must not orphan a context, and must not drag in haystacks for questions that were cut.

### 2.6 Adapter call policy — `orchestrator/client.py`

One place for the retry policy from `limits`, now wrapping an *adapter* rather than an HTTP client: attempt cap, timeout via `asyncio.timeout`, bounded concurrency via `asyncio.Semaphore` (never bare `gather`).

The retryable/non-retryable split no longer keys on status codes, since there are none. It keys on exception type: `AdapterError` is retryable, `ContractViolation` is not and fails the run loudly (§4.3). `suites/v1.yaml` loses `retry_on` and `no_retry_on`, which is a **suite version bump** — every published row names the suite it ran under, so this cannot be an in-place edit.

Response validation moves here too: `len(memories) <= top_k` and non-empty `content` are checked on every `search` return, raising `ContractViolation`. That is the same guard `check_search_response` applied to a JSON body, relocated to the call boundary.

Tests against a fake adapter: a transient `AdapterError` retries then succeeds; a `ContractViolation` fails immediately without a second call; attempts are capped; an adapter exceeding `top_k` raises; an adapter returning empty `content` raises; a hung `add` hits the timeout; `CancelledError` propagates rather than being swallowed as a retryable error.

### 2.7 `ingest` and `retrieve` — `orchestrator/stages/`

Thin: read `plan.jsonl`, skip done IDs, call the adapter, append the record shapes given in §5. Timing and attempt counts recorded per call.

`retrieve` writes `memories` (the projected `Memory` records) rather than `data` plus a duplicate `response` blob. Three fields are dropped as unread: `http_status` and `bytes` on the ingest record, and `response` on the retrieve record — verified unread by grep before removal, and the `score`/`answer` readers of `data` are updated in the same change.

The old "verbatim response" test goes away with the field it tested; the projection means an unknown vendor field no longer reaches the artifact (ARCH §5 states the cost). Replace it with a test that `source_ids` survives the round trip intact, since recall is what actually depends on that path.

### 2.8 Model seam — `orchestrator/models.py`

A narrow protocol (`chat(messages, model, temperature, seed)`) with one direct-provider implementation now and the gateway implementation in Phase 3. Tests stub the protocol; no test may make a real model call.

### 2.9 `answer` and `judge`

Pinned prompts from the suite, `temperature: 0`. Prompt files under `prompts/`, hashed into the manifest.

Per D7(a), resolution is dataset-then-suite for **both** prompt and judge model: `official_prompt` overrides the suite answer/judge prompt, and `official_judge` overrides the suite judge model. Put this in one small `resolve_judge(suite, dataset)` function so the precedence exists in exactly one place, and test all four combinations (neither set, prompt only, judge only, both).

The **resolved** judge id lands in `manifest.json` and on the report row — never the suite default when a dataset overrode it. Test that a dataset with `official_judge` produces a manifest naming that model; a row whose judge is ambiguous is a reporting bug.

LoCoMo-Refined's judge prompt is part of its scoring definition, so it is content-hashed like any other prompt. Its stricter rules (temporal granularity, list completeness, no unsupported detail) come from the dataset, not from us — don't reword them.

### 2.10 `score` — `scorers/` + `orchestrator/stages/score.py`

**Define `report.json` here as a Pydantic model** — it's the publication unit and ARCH doesn't specify it. Columns per §7: quality, retrieval, cost, latency, robustness, side by side, no composite score.

Build scorers in this order: `mcq` (deterministic, no judge), then `judge_binary`, then `exact_match_f1`. Defer `judge_rubric` and `event_ordering` until a dataset needs them. Recall metrics: exact track (`source_ids`) first — the fuzzy matcher is §15.2 and unresolved, so gate it behind a labelled flag and don't publish its numbers yet.

`mcq` keeps reading `options` from `plan.jsonl`. Only the *retrieval* call stops receiving them (§4.2) — passing the answer set into a retrieval query leaked the answer, and no shipped dataset used it.

Recall carries a **granularity label** per Step 0: LoCoMo's evidence is message-level, LongMemEval's is session-level. Same metric name, different unit. Put the unit in the report column, and add a test that the label follows the dataset rather than being hardcoded.

**`score` must be a pure function.** Enforce it with a test that runs the stage with network access monkeypatched to raise. Same artifacts in → byte-identical report out.

`bench rescore <run_id>` is then just this stage over existing artifacts.

### 2.11 Baselines — `reference/`

`no_memory` and `bm25` are genuine `MemoryAdapter` implementations, so the conformance suite from 1.5 applies to them unchanged — the cheapest possible check that the interface is implementable. Each is now a class rather than an ASGI app plus a client factory.

`oracle_gold` and `long_context` bypass the interface per D2: they read `plan.jsonl` and emit `retrieve.jsonl` directly, skipping 2.7. Give them a shared entry point so `answer`/`judge`/`score` cannot tell the difference, and mark their rows `privileged: true` (renamed from `in_process`, which now describes every adapter — ARCH §10). Test that `oracle_gold`'s synthetic retrieval contains exactly the gold evidence chunks and nothing else — it's the ceiling, and a leaky ceiling silently flatters every system beneath it.

---

## Phase 3 — Gateway (Bifrost) *(D5)*

**Exit:** `verify` catches a fake adapter that calls a provider directly.

### 3.1 Attribution — verified Bifrost capabilities

D3 is implementable, but not the way §7.3 implies. What Bifrost actually gives us:

- Virtual key CRUD: `POST/GET/PUT/DELETE /api/governance/virtual-keys`, `{vk_id}` addressable. So one key per `(run_id, system, phase)` is creatable at run start.
- Per-key model restriction via `provider_configs[].allowed_models`; a blocked model returns **403 `model_blocked`**. This is the pinning *mechanism* — stronger than policy text. Each entry also carries a `provider`, so provisioning derives its provider set from the suite's pinned pairs (D11) rather than posting a bare model list; `["*"]` allows everything and `[]` denies everything, which makes an omitted list the dangerous default.
- Keys travel as `x-bf-vk`, `Authorization: Bearer`, or `x-api-key`.
- `/metrics` (Prometheus) carries `virtual_key_id` and `virtual_key_name` as base labels on upstream metrics, with `bifrost_input_tokens_total`, `bifrost_output_tokens_total`, `bifrost_cost_total`, `bifrost_upstream_requests_total`.
- `x-bf-dim-*` request headers inject custom dimension labels, propagated to metrics, logs, and OTel spans.

**The gap:** there is no documented endpoint that returns per-key token counts as data. Budgets expose `current_usage`, but that's spend, not the prompt/completion split §7.3 requires. So metering readback is **scrape `/metrics` and diff the counters**, keyed on `virtual_key_id` — take a snapshot before and after each phase; the delta is that phase's cost. Verify these metric names against the running image before building on them; they're from docs, not from a live scrape.

This makes the phase-scoped key the attribution unit *and* the metric label, so D3 and §7.3 land on the same mechanism. Add `x-bf-dim-run: <run_id>` and `x-bf-dim-phase: <phase>` as belt-and-braces for parallel runs (Phase 6) — with many runs in flight, `virtual_key_id` alone is decodable but the dimension labels make Prometheus queries legible.

### 3.2 Two key scopes *(D6(a))*

Harness traffic carries a key per `(run_id, system, phase)` — we control those calls, so attribution is exact. A **black-box system cannot** take a phase-scoped key: a container gets one env var at start, and an in-process adapter is constructed once per run. Per D6(a) it gets one key per `(run_id, system)`, and its per-phase split comes from the `/metrics` snapshot boundary instead.

D8 adds a wrinkle here. For a **container**, the run-scoped key plus an internal network means metering is enforced. For an **in-process adapter**, we pass the key through its `config` and it may or may not honour it, so the same derived number is `unenforced` — a weaker claim from an identical mechanism. Carry that distinction into `verify.json` rather than letting two different guarantees share one column, and test both paths.

**Status: the key delivery is built, `unenforced` is not.** A system receives its run-scoped key through an optional `use_model_key(value)` on its adapter, called at run start between provisioning and the first snapshot — not through `config`, and not through the container's environment, since the key is named for a run that does not exist when the container starts. But `CostAttribution` in `orchestrator/verify.py` has only `derived` / `unreliable` / `unavailable`; there is no `unenforced` member, so an in-process adapter's cost is currently labelled `derived` exactly like a container's, and the distinction this section asks for is not carried. `GATEWAY_HINT` in `orchestrator/cli.py` already tells the user their column "will read `unenforced`", which no code can produce. Closing this means adding the member and setting it from `request.is_in_process`, plus report and score shape changes.

So `orchestrator/gateway.py` provisions two key families per run and the report must distinguish them. Harness-side cost is exact; system-side cost is boundary-derived. Label the columns accordingly — collapsing them into one number would overstate precision on the half that's inferred.

**The invariant this rests on:** ingest and retrieve never overlap within a run. Assert it in 3.3 and mark it in `verify.json` when 3.5.1's override breaks it. If a future change needs overlapping phases, the fallback is one gateway port per phase mapped to a phase-scoped key server-side — no contract change, more moving parts in `gateway/`. Don't build that until something needs it.

### 3.3 Build order

`gateway/bifrost.json` config → key provisioning at run start (`orchestrator/gateway.py`) → egress policy (docker network, for container deployments; systems reach only the gateway) → `/metrics` snapshot/diff → then `verify` (§8) with the checks Phases 1–2 can't do: model pinning reconciliation and rank stability. **Define `verify.json` here**, alongside the other two schemas.

Add a test that phases are strictly ordered within a run — the D6(a) invariant. It belongs with the gateway rather than with the stages, because that's where violating it corrupts a published number.

Confirm Bifrost serves `POST /v1/embeddings` before relying on it — the suite pins an embedding model and the `embedding_rag` baseline needs it. Docs confirm chat completions and OpenAI compatibility broadly; embeddings weren't explicitly listed. If it doesn't, `embedding_rag` needs another route and that changes 2.11's scope.

Provider routing (D11) gets its own tests here, because every failure mode is silent:

- A key provisioned for `(vllm, qwen3-14b)` reaches that model and gets **403** for `gpt-4o-mini` — pinning holds *per provider*, not just per model name.
- A suite naming a provider absent from `bifrost.json` fails at run start with a message naming the provider, not mid-`judge` after ingest has been paid for. This is the cheapest possible check and the one most worth having.
- `env.*` in `network_config.base_url` does **not** resolve. Assert the literal-string behaviour so nobody "fixes" the config into a form that silently points nowhere.
- The judge routes to a different provider than the answer model within one run, and `/metrics` attributes both correctly — the mixed-provider case is the whole point of D11 and is otherwise untested.

Test with fake adapters that violate on purpose: one bypassing the gateway (must flag `unmetered_model_use`), one with unstable ranking (must report variance, not fail), one requesting a non-pinned model (must get 403 and fail the run loudly).

---

## Phase 3.5 — Parallel runs and `bench ps`

**Not in ARCH §13** — this is an addition. It sits here because it needs the gateway's per-key attribution to be real, and because retrofitting concurrency into stages that assumed a single run is much more expensive than building it once one run works end to end. It can slip alongside Phase 5 if Phase 4 is more urgent.

Nothing here changes the integration surface, the artifact shapes, or scoring. A parallel run must produce byte-identical artifacts to the same run executed alone — that is the acceptance test for the whole phase.

One-process-per-run gains a second justification under D8: each run constructs its own adapter instance, so two runs cannot share a mutable client, a connection pool, or a module-level cache. A single-process batch would have to assume every third-party SDK is re-entrant, which is not a property we can verify for someone else's code.

### 3.5.1 Why this is mostly free, and where it isn't

Three properties from ARCH already make runs independent: `user_id` embeds `run_id` so namespaces can't collide (§4.4), artifacts are per-run directories, and `score` is pure and offline. So parallelism across *different* `(system, dataset)` pairs needs no new isolation.

What is genuinely shared and must be bounded:

| Resource | Contention | Handling |
| --- | --- | --- |
| Gateway | Rate limits, provider quota | Global concurrency ceiling across all runs, not per-run |
| A single system under test | One container, N runs hammering it | **Serialize runs against the same system** by default |
| `limits.workers` | Suite pins per-run workers; K runs = K× load | Ceiling is `min(sum(per-run workers), global cap)` |
| Local disk | Trivial | Ignore |

**The serialization rule is a fairness requirement, not a performance choice.** §7.4 reports latency and §7.5 reports rank stability; two concurrent runs against one container contaminate both. Same system → queued. Different systems → parallel. Make this the default and require an explicit `--allow-concurrent-same-system` to override, with the resulting runs marked `contended: true` in the report so the numbers can't be quoted as clean.

Note this bounds the headline speedup: with N systems on M datasets, useful parallelism is at most N (one run per system at a time), not N×M.

The override also breaks D6(a)'s attribution invariant, so runs made under it record `cost_attribution: "unreliable"` and omit per-phase system cost rather than reporting a derived number that is no longer derivable.

### 3.5.2 Run state — `orchestrator/runstate.py`

`bench ps` needs to read progress without touching the running process, and artifacts alone can't distinguish "crashed mid-ingest" from "still ingesting."

Each run directory gets a small state file, written atomically (temp + `os.replace`):

```jsonc
// artifacts/<run_id>/state.json
{"run_id":"...","system":"...","dataset":"...","suite":"v1",
 "status":"running","stage":"ingest","stage_index":2,"stage_total":7,
 "pid":12345,
 "started_at":"...","heartbeat_at":"...",
 "stage_progress":{"done":412,"total":1382},"progress_unit":"chunks",
 "error":null}
```

`stage_index`/`stage_total` place the stage in the pipeline, and `progress_unit` says what `stage_progress` counted. All three are optional: `queued` has no stage, and a state file written before these fields existed must stay readable — `read_state` degrades to `None` on a validation error, which would blank a live run out of the table rather than merely showing it without a position.

`stage_total` is per-run, not the constant 7: the in-process baselines skip `ingest` (D2) and must not read as stuck at "2/7".

Status is one of `queued | running | done | failed | stale`. `stale` is *derived*, never written: a run whose `heartbeat_at` is older than a threshold and whose `pid` is gone. This is what makes crash detection possible without a daemon.

Progress counts come from `read_ids()` on the current stage's artifact — the source of truth stays the artifacts; `state.json` is a cache plus liveness. Test that a hand-corrupted `state.json` degrades to "unknown" rather than crashing `bench ps`.

Counts are refreshed by a background ticker (`Progress.ticking`, every `TICK_INTERVAL_S`) for the duration of each long stage, not written once when the stage returns — otherwise the number is frozen for exactly the hours a user wants to watch it. The ticker is cancelled and awaited before its final write, so it can never outlive its stage and stamp a heartbeat onto the next one; test that, and test that it neither swallows a stage's exception nor fails a stage when it reads a torn final line mid-append.

Deliberately **no daemon and no database.** A directory scan of `artifacts/*/state.json` is enough for the scale here, and it keeps the "artifacts are the API" property (P3) intact.

### 3.5.3 Scheduler — `orchestrator/scheduler.py`

`bench run` accepts multiple systems and datasets and expands to the cross product:

```bash
bench run sys-a,sys-b --dataset locomo-refined,longmemeval-cleaned --parallel 4
```

Each `(system, dataset)` is one run with its own generated `run_id` (D10). A worker pool pulls from a queue, respecting the same-system lock from 6.1. Each run is a **subprocess**, not a task in one event loop — a segfault or OOM in one run must not take down the others, and `pid` in `state.json` only means something if runs are separate processes. `--parallel` sizes the pool and defaults to the number of distinct systems, since the same-system lock makes anything above that unspendable.

A cross product **detaches by default**, like a single run. One supervisor detaches, not each job: detaching them individually would release the 6.1 lock when a child was *launched* rather than when it finished, which is the whole point of holding it. `--foreground` blocks and prints the aggregated rows.

D10 simplifies the parent/child contract here. Previously the parent derived each child's id from a `--run-id` prefix, so the batch owned a naming scheme; now ids are generated opaquely, with no prefix arithmetic and no way for two children to derive the same id. An attached batch lets each child generate and print its own. A **detached** batch cannot: the supervisor's stdout goes nowhere, so the launcher generates every id, prints them, and passes them down with repeated `--run-id` flags. A batch is only resumable by ids the user was shown before it detached.

Tests: same-system runs never overlap (assert on state transitions, not timing); different-system runs do; `--parallel 1` is exactly today's sequential behaviour; killing one worker leaves siblings running and marks only that run `failed`; SIGINT drains gracefully — in-flight runs are resumable per §5, which they already are since resume is by ID. For the detached path: the launcher returns without waiting, prints one id per job before detaching, and the ids it printed are the ids that ran; `--foreground` still blocks; a usage error is reported before anything detaches.

Per D6(a), phase attribution for system traffic relies on ingest and retrieve not overlapping. With parallel runs of *different* systems that still holds — different keys. Same system is serialized, so it also holds. If 3.5.1's override is used, phase attribution for that system degrades; say so in `verify.json` rather than reporting a number you can't defend.

### 3.5.4 `bench ps` — `orchestrator/cli.py`

Docker-style table over `artifacts/*/state.json`:

```
RUN ID              SYSTEM      DATASET             STAGE         PROGRESS            ELAPSED  STATUS
2026-08-03T10-a1b2  mem0        locomo-refined      2/7 ingest    412/1382 chunks     4m12s    running
2026-08-03T10-c3d4  zep         locomo-refined      3/7 retrieve  1201/1382 searches  11m03s   running
2026-08-03T09-e5f6  bm25        longmemeval-cleaned -             -                   2m44s    done
2026-08-03T09-a7b8  letta       locomo-refined      2/7 ingest    88/1382 chunks      31m10s   stale
```

The position leads the stage cell so the column scans by how far along a run is. `--json` carries `stage_index`/`stage_total` as numbers, so a consumer never parses `"2/7 ingest"` back apart.

Flags: `--all` (include finished), `--watch` (redraw on an interval), `--json` (scriptable). Companions: `bench logs <run_id>`, `bench stop <run_id>`, `bench rm <run_id>`.

Keep it a pure function of directory contents — no state in the CLI process. Tests build fixture artifact trees and assert the rendered table, including the `stale` derivation (fake a dead pid and an old heartbeat).

Formatting is plain text via stdlib. No `rich` dependency for a table this simple; `--watch` is a clear-and-redraw loop.

---

## Phase 4 — Two contrasting real systems, both paths

**Exit:** discrepancies between the old vendored harness and this one are *explained*, not just observed.

Per ARCH §13 this is the phase that distinguishes "measuring correctly" from "producing different numbers." Run both paths on one dataset with shared dataset and suite pins. Budget real time for reconciliation — a mismatch here is a finding about the harness, and the reason to keep the old path alive until it's understood.

---

## Phase 5 — Remaining systems and datasets

Port systems; admit datasets under §9's criteria. Retire the old harness once its last row is reproduced. Archive superseded datasets rather than deleting them.

---

## Phase 6 — Adapter refactor *(D8, D9, D10, D11)* — **done**

**Exit:** a pre-refactor artifact bundle rescores to a **byte-identical `report.json`**. That is the whole acceptance criterion, and it is what proves the change touched integration rather than measurement.

**Met.** Five bundles were captured before any code changed — `bm25`, `no_memory`, `oracle_gold`, `long_context` on LoCoMo-Refined, plus `bm25` on LongMemEval-Cleaned — and all five rescore to the same bytes afterwards. A fresh run of each reproduces its golden report row field for field with one exception: latency is lower, which is the HTTP round-trip that is no longer there.

Two notes for anyone reading the steps below as history rather than as a plan:

- **The dual read in `score` is permanent, not transitional.** `orchestrator/retrieval_records.py` understands both `data`/`x_source_ids` and `memories`/`source_ids`, which is what keeps an archived bundle rescorable. A reader that only understood the new shape would score every old bundle as *zero retrieval* rather than failing — silently wrong beats loudly broken, so this stays.
- **Step 8 grew.** `provider` turned out not to be recordable-and-ignored: `_create_key` was posting a bare model list, so pinning was never enforced on a multi-provider gateway. Provisioning now groups pinned models by provider, `resolve_judge` carries the provider through, the manifest records all three, and `bench run` defaults to the current suite rather than to `v1`.

It lands after Phase 5 deliberately. It changes the integration surface, the retrieve record shape, and every `--run-id` call site at once; doing that mid-port would confuse "this port is wrong" with "this refactor is wrong."

Order matters, because the middle steps are where a silent regression would hide:

1. **`contract/adapter.py`** — the protocol and the two dataclasses (1.2). Additive; nothing uses them yet.
2. **`orchestrator/ids.py`** — `new_run_id()` and `resolve_prefix()` (1.1). Also additive.
3. **`adapters/`** — the package, `HttpAdapter`, and `Mem0Adapter`. `HttpAdapter` comes first and must reproduce the old client's behaviour exactly, since it is the compatibility bridge every currently-registered system falls back to.
4. **`orchestrator/client.py`** — retry keyed on exception type rather than status code (2.6), plus the relocated response checks.
5. **Stages** — `ingest`/`retrieve` call the adapter; `data` → `memories`; drop the three unread fields. **Update the two readers in `answer.py` and `score.py` in the same commit** — grep confirms `data` is read in exactly those two places, and missing one produces empty retrieval that scores as a wrong answer rather than an error.
6. **Registry** — `adapter` + `config`, `auth` deleted (1.3).
7. **CLI** — `--run-id` deleted, `--resume` and `--name` added, prefix matching on every id argument.
8. **Suite v2** — `retry_on`/`no_retry_on` removed, and every model gains a `provider` (D11). A new suite file, never an edit: published rows name their suite version, and both changes alter what a row means. The provider has to reach the gateway to be worth having, so this step also fixes `_create_key`'s bare model list and makes `v2` the default `--suite`.
9. **Delete** — `contract/openapi.yaml`, `reference/mock/` (ASGI), `reference/adapters/mem0.py`, the `wsgi`/`app` plumbing, and the conformance `--base-url` option.

Keep the rescore-identity test green at every step. Between steps 5 and 8 the artifact shape has changed, so the check becomes: rescoring an *old* bundle still works (the score stage must read both `data` and `memories` during the transition), and a *new* run over the same dataset produces the same quality and recall numbers as the pre-refactor run did.

Two things to test that no earlier phase covers: a registry entry whose adapter import fails must not prevent `bench ps` or `bench rescore` from working (lazy import, ARCH §14), and `bench doctor` on an in-process adapter not configured for the gateway warns without failing (§3).

---

## Conventions for every stage

- **Stage signature:** `run(ctx) -> None`, reads one artifact, appends another, skips IDs already present.
- **No stage knows a system's name.** If you need a branch on system identity in `orchestrator/`, the design is wrong — that's what the registry and the adapter interface are for. The two privileged baselines (D2) are the sole exception and live in `reference/`, never in a stage.
- **A parallel run is byte-identical to a solo run.** Any concurrency that changes artifacts is a bug, not a tradeoff.
- **Contract violations fail loudly; upstream errors retry.** Never conflate them; a broken integration is not a score of zero.
- **Never log** keys, prompts, messages, or retrieved content. IDs, statuses, counts, latencies only.
- **Fixtures over network, always.** A test that needs a socket is a test that's wrong.
