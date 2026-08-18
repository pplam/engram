# Memory Benchmark — Architecture Design

**Status:** proposal · **Supersedes:** the in-process vendored-runtime harness (`main.py` + `utils/agent.py`)

> **Amendments since first draft** — see `IMPLEMENTATION.md` Step 0 for rationale.
> - **Gateway is [Bifrost](https://github.com/maximhq/bifrost)**, not LiteLLM. Virtual key per `(run_id, system, phase)`; per-phase tokens come from diffing `/metrics` counters labelled `virtual_key_id`, since Bifrost has no usage-readback API (§7.3, §11).
> - **`oracle_gold` and `long_context` are privileged baselines** that bypass the integration surface (§10). They emit `retrieve.jsonl` directly.
> - **First datasets:** LoCoMo-Refined and LongMemEval-S Cleaned (§9). They have opposite context shapes (138 questions per context vs 1) and different evidence granularity (message vs session) — both are load-bearing for the plugin API and for recall reporting.
> - **Chunking:** `max_messages_per_chunk` is the primary bound; the word limit cuts inside it (§6).
> - **Parallel runs + `bench ps`** are in scope (§11); §13 gains a Phase 3.5.
> - **A dataset may pin its own judge model** via `official_judge`, overriding the suite (§6, §9). Resolves §15.1 for the first two datasets; cross-dataset quality comparison becomes out of scope.
> - **Two key scopes:** phase-scoped for harness traffic, run-scoped for systems, whose per-phase split is derived from metric snapshots at phase boundaries (§3, §7.3).
>
> **Amendments in the second revision** — the integration surface changed shape.
> - **The contract is a Python interface, not a wire protocol** (§4). A system is integrated by implementing `MemoryAdapter` — `add` and `search` — which the harness imports and calls directly. An adapter is a client, never a server. HTTP survives as one adapter (`adapters.http:HttpAdapter`) for containers and hosted services, which also keeps the strong egress guarantee (§3). `contract/openapi.yaml` is deleted; `adapters/` becomes top-level (§11).
> - **Agent Memory Leaderboard wire compatibility is dropped** (§4, Appendix). The envelope, the echoed ID fields, `session_id`, the `x_` prefix, and the `{"data": […]}` wrapper are gone; `Memory` is `content` + `score` + `source_ids`. Its *ideas* are still adopted.
> - **`run_id` is generated, not supplied** (§4.5) — 12 hex chars, Docker-style, with prefix matching. `--run-id` is replaced by `--resume <id>` for continuing a run and `--name <label>` for a human tag.
> - **Two consequences worth flagging.** In-process systems cannot be network-isolated, so their gateway metering is `unenforced` rather than enforced, and the report labels it (§3, §14). Retrieval artifacts now record the adapter's projection onto `Memory` rather than a system's raw bytes (§5). Foreign runtimes re-enter our import path — the cost the HTTP contract had bought off, accepted deliberately and bounded four ways (§14).

---

## 1. Purpose

Compare **any** memory system fairly, as a **black box**, across **multiple metrics**, on **current** datasets.

Three claims we want to be able to defend:

1. **Fair** — every system is measured under identical conditions. Differences in score come from memory architecture, not from model choice, prompt wording, or retrieval budget.
2. **Black box** — the only thing we require is two methods: store messages, return memories. No vendored source, no monkeypatching of private methods, no knowledge of how a system works inside.
3. **Reproducible** — a third party can recompute every published number from the artifact bundle alone, without re-running any memory system.

### Non-goals

- Benchmarking base LLMs. The answer model is a pinned constant, not a variable.
- Benchmarking long-context reading. If a task is solvable by pasting the corpus into a prompt, it is not measuring memory.
- Provisioning memory systems. Operators start them (Docker or native); the harness only talks to URLs.

---

## 2. Design principles

| # | Principle | Consequence |
| --- | --- | --- |
| P1 | **One variable at a time** | Answer model, judge model, `top_k`, prompts, chunking are pinned per suite version. Only the system under test changes. |
| P2 | **Black box or nothing** | The adapter interface is the whole integration surface. Anything we want measured must be observable from that interface or from the gateway — never from a system's internals. |
| P3 | **Stages are processes, artifacts are the API** | Each stage reads and appends JSONL. Rescoring never re-touches a memory system. |
| P4 | **Scoring is a pure function** | `score` has no network access. Same artifacts in, same report out. |
| P5 | **Datasets are plugins** | Adding a dataset is a directory. It never requires editing the orchestrator. |
| P6 | **Trust nothing, verify cheaply** | Isolation, searchability, and model pinning are asserted by tests, not by policy text. |
| P7 | **Minimal config** | A run is one command. Defaults come from the pinned suite; a system is ~6 lines of YAML. |

---

## 3. System overview

```
                    ┌─────────────────────────────────────────┐
                    │  orchestrator  (knows datasets+stages,  │
                    │                 knows no method)        │
                    └───────┬─────────────────────┬───────────┘
              add()/search() │                     │  append-only
             via MemoryAdapter                     ▼
                            ▼            ┌──────────────────┐
              ┌──────────────────────┐   │  artifacts/      │
              │  adapter (imported)  │   │  *.jsonl         │
              │  registry/*.yaml     │   └────────┬─────────┘
              │  names the class     │            │
              └──────────┬───────────┘            ▼
                         │              ┌──────────────────┐
             the system itself:         │  score (pure fn, │
             in-process library,        │  no network)     │
             or a container it          └────────┬─────────┘
             talks to over HTTP                  ▼
                         │                  report.json
                   model calls               + verify.json
                         ▼
              ┌──────────────────────┐
              │  gateway (Bifrost)   │
              │  pins (provider,     │
              │  model), meters      │
              │  tokens, blocks rest │
              └──────────┬───────────┘
                         ▼
         one provider block per base URL (§6.1)
         openai · local vLLM · any OpenAI-compatible
                         ▼
              pinned chat + embedding + judge models
```

Three planes, one job each:

- **Control plane** (orchestrator) — sequences stages, applies retry policy, writes artifacts. Contains zero method-specific code.
- **Model plane** (gateway) — the only route to any LLM or embedder, for both the harness and the systems under test. Pins identity, meters cost, enforces isolation.
- **Data plane** (artifacts) — append-only JSONL, the sole input to scoring and the unit of publication.

`bench run` reaches the model plane at the gateway, with nothing to configure: the gateway is the *only* route to a model, so it is a constant rather than a flag, and a run works as soon as `bench gateway start` is up. An endpoint flag would let a run be unmetered and unpinned while wearing the same report shape as a metered one, which is the confound this plane exists to close. The credential is read from one fixed variable, `ENGRAM_GATEWAY_KEY` — never the credential as an argument a child process carries, and unset is legitimate for a gateway with governance off. `--stub-models` substitutes a fixed reply for the whole plane. **Under a stub every judgement is `CORRECT`, so accuracy reads 1.000 for every system including the `no_memory` floor** — a plumbing signal, not a measurement. Only retrieval discriminates in a stubbed run.

### Why the gateway is the centerpiece

Fairness fails quietly when each system uses its own model. The gateway converts a *policy* ("please use gpt-4o-mini") into a *mechanism*:

- **Pinning** — one virtual key per `(run_id, system, phase)`; `provider_configs[].allowed_models` restricts it to exactly the suite's pinned `(provider, model)` pairs, and anything else is refused with `403 model_blocked`. Enforcement, not policy.
- **Routing** — a provider is a base URL plus a credential (§6.1), so a pinned model can live on any OpenAI-compatible endpoint: a hosted API, a local vLLM, a private deployment. The harness and every system still see one URL and one credential shape.
- **Metering** — exact prompt/completion tokens and call counts per system per phase, measured identically for everyone. This yields a cost axis (§7) that in-process harnesses can only estimate. Harness traffic carries the phase-scoped key directly; a black-box system gets one key for the whole run, so *its* per-phase split comes from the metric-snapshot boundary between non-overlapping phases (see `IMPLEMENTATION.md` D6). Both sides are handed their keys at run start — the harness through the chat client, the system through the optional `use_model_key` on its adapter (§7.3) — because a call made before its key arrives carries no `virtual_key_id` and no counter can attribute it.
- **Enforcement** — strength depends on where the system runs, and the report says which. A **containerized** system is egress-blocked except to the gateway, so a system calling its own provider produces *zero* gateway traffic and the violation is caught in `verify`. An **in-process** adapter shares our interpreter and cannot be network-isolated from it: pointing the library at the gateway is then configuration, not a mechanism.

  This is the real cost of the imported-adapter design (§4), and it is why the two deployments are not interchangeable for cost claims. In-process rows carry `deployment: in_process`, and `verify` reports their metering as `unenforced` — model use was *observed* through whatever the adapter was configured to use, not *constrained* to it. A published cost comparison should either be all-container, or state the difference. `bench doctor` warns when an in-process adapter's configuration does not point at the gateway, which catches the accident but cannot prevent the deliberate case.

---

## 4. The adapter interface

The integration surface is a **Python interface**, not a wire protocol. A system is benchmarked by implementing two methods; the harness imports the adapter and calls it directly. An adapter is a **client**, never a server — it needs no socket, no process, and no port.

We are deliberately **not** wire-compatible with the [Agent Memory Leaderboard API](https://agentmemories.ai/api-guide). Matching it cost us an envelope (`success: true`), three ID fields that existed only to be echoed back, an `x_` prefix namespace, and a response wrapper — none of which any metric reads. What we keep from it is the part that carries weight: retrieve-only discipline and `user_id` as the isolation boundary.

### 4.1 The interface

```python
class MemoryAdapter(Protocol):
    """Everything the harness needs from a memory system."""

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Store messages for one user. Return only once they are searchable."""

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """Return at most top_k memories for this user, best first."""

    async def close(self) -> None:
        """Release connections. Default no-op."""
```

Two methods and an optional teardown. No lifecycle, no capability negotiation, no health probe — a failing `add` is a failing system, which is what a health check was proxying for anyway.

**Retrieve-only, universally.** `search` returns memories; it never returns an answer. The harness generates the answer from returned `content` with a pinned prompt and pinned model. This is the single most important fairness decision in the design: it makes answer generation a constant instead of a per-system confound, and removes any need to split systems into "retrieval" vs "agentic" tiers that cannot be compared.

### 4.2 The two records

```python
@dataclass(frozen=True)
class Message:
    role: str
    content: str
    timestamp: int | None = None   # unix ms; temporal questions need it

@dataclass(frozen=True)
class Memory:
    content: str                   # what the answer stage reads
    score: float | None = None     # the system's own relevance, if it has one
    source_ids: Sequence[str] = ()  # which chunk_ids this came from
```

`Memory` is three fields. Gone from the old wire shape: `id` (nothing read it — rank came from list order, not the id), `created_at` (never read by any scorer or stage), and the `{"data": [...]}` envelope (a list is a list). `x_source_ids` loses its prefix and becomes `source_ids`, since there is no foreign spec left to avoid colliding with.

**`source_ids` is the one field worth care.** It is how the harness knows which chunk a memory came from, which is what makes recall measurable. A system that cannot report it still scores quality, but its retrieval reads `unavailable` rather than `exact`. Carry `chunk_id` into whatever metadata the system supports and echo it back.

**`options` is gone from the interface.** Multiple-choice options were passed to `search` for choice-type tasks, but no shipped dataset uses the `mcq` scorer, and passing answer options to a retrieval call is a questionable idea regardless: it leaks the answer set into the retrieval query. The plan still carries `options` for the `mcq` scorer, which reads them at scoring time. If a choice-type dataset is admitted later and genuinely needs them at retrieval, that is an interface change to argue for then.

**`session_id` is gone.** It was documented as "grouping only, never a filter" — a field the harness computed, sent, and required systems to echo, while forbidding them to use it for anything. It appeared in exactly one place in the harness: a constant string built per run.

**Ordering:** rank order is the sequence the adapter returns. The harness reads at most `top_k` and preserves that order.

### 4.3 Errors

An adapter raises. Two kinds, and they are never conflated:

- **`AdapterError`** — a transport or upstream failure. Retried per the suite's policy.
- **`ContractViolation`** — the adapter returned something unusable (more than `top_k` items, a `Memory` with empty `content`). Fails the run loudly rather than scoring zero, because a broken integration is not a research finding.

Anything else propagating out of an adapter is treated as `AdapterError` on the first occurrence and re-raised if the policy exhausts. Credentials are the adapter's own business: it reads its own environment, exactly as the system's own SDK would.

### 4.4 No namespace lifecycle

There is no create/delete namespace call. Isolation is a *string*: `user_id` embeds `run_id`, so **a new `run_id` is a new namespace**. This deletes an entire class of complexity — the current harness needs ~120 lines of per-method external-state resets (Letta agents, Zep cloud users, Neo4j graphs, MemoRAG caches) that this design does not need at all.

Operators may wipe storage between runs; correctness does not depend on it.

### 4.5 Run IDs are generated, not supplied

A `run_id` is generated by the harness, Docker-style: a 12-character lowercase hex string from a CSPRNG.

```
bench run mysystem --dataset locomo-refined
→ 3f9a2b7c1d04
```

It was previously a **required** `--run-id` argument, which was the wrong default three ways. It made the common case verbose; it invited collisions, and because artifacts are append-only and resume by skipping IDs already present, reusing an id silently *resumes* a run instead of starting one — a typo could append to an unrelated run's artifacts and publish a row spliced from two configurations. And hand-written ids drifted toward encoding metadata (`2026-08-04-mysystem-locomo`) that the manifest already records in typed fields.

Generated ids are collision-free in practice (48 bits of entropy), fixed-width so `bench ps` tables align, and free of the separator that `user_id` parsing reserves.

Three affordances keep it ergonomic:

- **Prefix matching.** Every command that takes a run id accepts any unambiguous prefix — `bench rescore 3f9a` — exactly like `docker`. An ambiguous prefix is an error listing the candidates, never a guess.
- **`--resume <id>`** is how you continue an interrupted run. Resuming is now something you *ask for*, rather than a side effect of typing an id that already exists. **The id is sufficient**: the target's `manifest.json` already records the system, dataset, suite, and limit, so `bench run --resume <id>` reads them back rather than making you retype a configuration whose only permitted value is the one already on disk. Anything passed explicitly is still checked against the manifest and refused loudly on mismatch.

  Reading `limit` back is a correctness fix, not just convenience: it decides how much of the dataset was planned, so a resume that *omitted* `--limit` used to replan the run at full size and rewrite the manifest — the exact splice `--resume` exists to prevent, leaking through the one argument that wasn't checked.
- **`--name <label>`** attaches a human label for `bench ps`. It is metadata: it is not the namespace, so it need not be unique and cannot corrupt anything.

A generated id never contains `:`, so `user_id` parsing is unaffected. `--run-id` is gone rather than deprecated: this is pre-1.0, and keeping a footgun alive for compatibility with no external users is not a trade worth making.

---

## 5. Pipeline

Seven stages. Exactly **two** touch the system under test.

| # | Stage | Touches SUT | Reads | Writes | Deterministic |
| --- | --- | --- | --- | --- | --- |
| 1 | `plan` | no | dataset + suite | `plan.jsonl`, `manifest.json` | yes |
| 2 | `ingest` | **yes** | `plan.jsonl` | `ingest.jsonl` | no |
| 3 | `retrieve` | **yes** | `plan.jsonl` | `retrieve.jsonl` | no |
| 4 | `answer` | no | `retrieve.jsonl` | `answer.jsonl` | pinned `temperature: 0` |
| 5 | `judge` | no | `answer.jsonl` | `judge.jsonl` | pinned `temperature: 0` |
| 6 | `score` | no | all above | `report.json` | **yes, offline** |
| 7 | `verify` | probes only | all above + gateway | `verify.json` | yes |

**Why this split matters.** Ingestion is the slow, expensive, non-deterministic part. Because it is isolated, you can fix a judge prompt, add a metric, or re-answer with a different model at zero ingestion cost. The current harness cannot do this — it rewrites one monolithic result JSON after every query and needs a dedicated backfill mode to recompute anything.

**Resume** is by ID: each stage skips IDs already present in its output file. Append-only, crash-safe, no rewrite-on-every-query.

**`score` has no network access** (P4). This is what lets a reviewer verify your table from a published artifact bundle without access to any memory system.

### Artifact record shapes

```jsonc
// plan.jsonl — two record kinds in one file, discriminated by `kind`.
// Defined in code as Pydantic models in orchestrator/plan_records.py.
{"kind":"chunk","id":"eval:<run>:<ds>:<ctx>:chunk-c7","user_id":"eval:<run>:<ds>:<ctx>",
 "ctx_id":"<ctx>","chunk_id":"c7",
 "messages":[{"role":"user","content":"...","timestamp":1704067200000}]}

{"kind":"question","id":"eval:<run>:<ds>:<ctx>:q-q3","user_id":"eval:<run>:<ds>:<ctx>",
 "ctx_id":"<ctx>","q_id":"q3","question":"...","gold":["..."],
 "options":["A. ...","B. ..."],          // omitted unless the task is choice-type
 "evidence_chunk_ids":["m0"],            // dataset's own handles, in its own unit
 "evidence_chunks":["c7"],               // those handles resolved onto chunk ids
 "meta":{"category":"single-hop"}}

// ingest.jsonl — one per chunk
{"id":"...:chunk-7","user_id":"...","status":"ok","attempts":1,
 "latency_ms":812,"started_at_ms":1704067200000,"ended_at_ms":1704067200812}

// retrieve.jsonl — the adapter's returned memories + timing
{"id":"...:q-3","user_id":"...","query":"...","top_k":100,
 "latency_ms":344,"attempts":1,
 "memories":[{"content":"...","score":0.87,"source_ids":["c7"]}]}

// answer.jsonl
{"id":"...:q-3","generated_answer":"...","prompt_tokens":1512,"completion_tokens":48}

// judge.jsonl
{"id":"...:q-3","label":"CORRECT","is_correct":true,"judge_response":"{...}"}

// verify.json — one object per run, not JSONL. Defined in orchestrator/verify.py.
{"verify_version":1,"run_id":"...","system":"...","run_valid":true,
 "row_withheld":false,"cost_attribution":"derived","rank_stability_tau":0.94,
 "flags":[],
 "findings":[{"name":"cross_user_leakage","passed":true,
              "severity":"invalidates_run","detail":""}]}
```

`verify.json` is defined in code as Pydantic models in `orchestrator/verify.py`: a
`Verification` carrying `run_valid`, `row_withheld`, `cost_attribution`,
`rank_stability_tau`, a `flags` list, and one `Finding` per check (§8). Each finding
carries its own `severity` — `invalidates_run`, `withholds_row`, or `informational` —
because the failures are not interchangeable: leakage invalidates a run, unmetered
model use withholds a row, and rank instability is variance rather than a defect.
`cost_attribution` is `derived` only when the ingest/retrieve boundary was clean and a
gateway was metering; `unreliable` when the boundary was broken (overlap, or 3.5.1's
`--allow-concurrent-same-system`), and `unavailable` when there was no gateway to
derive from. The report omits the per-phase system-cost figures unless it is `derived`.

Ingest and retrieve records carry `started_at_ms` / `ended_at_ms` as **wall-clock**
milliseconds alongside the monotonic `latency_ms`, because the phase-ordering check
compares them across phases and a monotonic epoch is not comparable.

`report.json` is defined in code as Pydantic models in `orchestrator/report.py`:
one `row` plus `notes`. The row carries `quality`, `retrieval`, `cost`, `latency`,
`robustness`, and `completeness` side by side, and deliberately **no composite
score** (§7). Three labels travel with the numbers: `quality.judge` +
`quality.judge_source` (which judge produced it, §6), `cost.harness_attribution` +
`cost.system_attribution` (measured vs derived vs unreliable, §7.3), and
`retrieval.track` + `retrieval.evidence_unit` (which recall track, in what unit,
§7.2). `in_process`, `contended`, and `unpinned_image` mark rows whose provenance
differs from a clean contract run.

Storing what retrieval returned is deliberate: it is what makes retrieval-quality metrics (§7) recomputable offline and auditable by a third party.

The retrieve record changed shape with the adapter interface. `data` becomes `memories`, holding the three-field `Memory` (§4.2) rather than a vendor's raw JSON. Three fields are dropped as unread: `http_status` and `bytes` on the ingest record (meaningless for a non-HTTP adapter, and no metric read either), and the duplicate `response` blob on the retrieve record, which stored the entire body a second time alongside `data`.

One property does weaken. Previously "verbatim" meant the literal bytes a system sent, so a third party could re-derive fields we never modelled. Now the adapter projects onto `Memory` first, and anything outside those three fields is gone before it reaches the artifact. That is the honest cost of dropping the wire format: the bundle stays sufficient for every published metric, but it is a record of what the harness *saw*, not of what the system *said*. An adapter doing something surprising in that projection is no longer detectable from artifacts alone — which is why adapters live in-repo and under review rather than being supplied per-vendor at runtime.

---

## 6. Variable control

Everything below is fixed by a **suite version**. A suite version is a single immutable YAML file; changing any value requires a new version number, and reports always carry the version they were produced under.

```yaml
# suites/v2.yaml — the pinned world. One file, one source of truth.
suite: v2

models:                      # identical for the harness AND every system's internal use
  chat:      {provider: openai, id: gpt-4o-mini, temperature: 0, seed: 7,
              max_tokens: 8000}
                             # max_tokens from v4 on. Unpinned, each system applies
                             # its own default; below what a reasoning model spends
                             # on reasoning, the completion comes back empty and a
                             # system stores nothing while still reporting success.
  embedding: {provider: openai, id: text-embedding-3-small, dimensions: 1536}
  judge:     {provider: openai, id: gpt-4.1-mini, temperature: 0}
                             # deliberately != chat; default — a dataset's
                             # official_judge overrides it, and names its own
                             # provider. `provider` names a gateway provider
                             # (§6.1), so a pinned model need not be an OpenAI
                             # one: LoCoMo-Refined's judge is {vllm, qwen3-14b}.

retrieval:
  top_k: 100                 # same budget for everyone

chunking:                    # harness-side; systems receive identical inputs
  max_messages_per_chunk: 20
  max_words_per_chunk: 2000
  boundary: message_or_sentence

prompts:                     # content-hashed; drift is detectable
  answer: {ref: prompts/answer/v1.txt,  sha256: "..."}
  judge:  {ref: prompts/judge/v1.txt,   sha256: "..."}

limits:                      # no status lists: retry is keyed on the exception
  ingest: {workers: 16, timeout_s: 1200, attempts: 6}
  retrieve: {workers: 8,  timeout_s: 1200, attempts: 6}
```

Two things follow from immutability, and both are enforced in `orchestrator/suite.py`
rather than left to prose:

- **`v1` stays loadable.** It predates providers and carries `retry_on` / `no_retry_on`,
  so rows published under it remain reproducible. It is the only exempt version, and the
  exemption is a closed set — a new suite is held to today's rules by default, because
  forgetting to list it would silently grant it v1's exemptions.
- **Every version after v1 must name a provider per model, and must not carry a status
  list.** A bare model id restricts nothing on a multi-provider gateway, so a suite
  without providers pins in name only; a status list cannot express the retry policy for
  an adapter that never speaks HTTP (§4.3). `bench run` defaults to the current version,
  never a legacy one, so the weaker shape is only ever a deliberate choice.

### 6.1 Providers — any base URL, any model

A pinned model is a `(provider, id)` pair, and a provider is a **base URL plus a credential**. Nothing in the design assumes OpenAI; that was an artifact of the first `gateway/bifrost.json`, which hardcoded one provider and three OpenAI model names. LoCoMo-Refined pins Qwen3-14B, so this is not hypothetical — the suite already names a model no OpenAI endpoint serves.

Two routes to a custom endpoint, and which one applies depends on the endpoint. Field names below are verified against [`core/schemas/provider.go`](https://github.com/maximhq/bifrost/blob/main/core/schemas/provider.go) and `core/schemas/account.go` rather than taken from the docs site:

```jsonc
// gateway/bifrost.json
{
  "providers": {
    "openai": {
      "keys": [{"value": "env.OPENAI_API_KEY",
                "models": ["gpt-4o-mini", "text-embedding-3-small"],
                "weight": 1.0}]
    },

    // (a) vLLM and Ollama are first-class providers: the endpoint is per-key,
    //     and its `url` is a SecretVar, so `env.*` resolves and the URL never
    //     sits in the file. One key per instance also load-balances replicas.
    "vllm": {
      "keys": [{"value": "env.VLLM_API_KEY", "models": ["qwen3-14b"], "weight": 1.0,
                "vllm_key_config": {"url": "env.VLLM_BASE_URL",
                                    "model_name": "qwen3-14b"}}],
      "network_config": {"default_request_timeout_in_seconds": 600}
    },

    // (b) Anything else OpenAI-compatible — TGI, Together, Groq, OpenRouter,
    //     a private deployment — is a custom provider.
    "internal-llm": {
      "keys": [{"value": "env.INTERNAL_API_KEY", "models": ["*"], "weight": 1.0}],
      "network_config": {
        "base_url": "https://llm.internal.example.com",   // plain string: NO env.* here
        "default_request_timeout_in_seconds": 600
      },
      "custom_provider_config": {
        "base_provider_type": "openai",   // speak the OpenAI dialect to it
        "is_key_less": false,             // true for an unauthenticated endpoint
        "allowed_requests": {"chat_completion": true, "chat_completion_stream": true}
        // "request_path_overrides": {"chat_completion": "/api/v2/chat"}
        //   only when it doesn't serve /v1/chat/completions
      }
    }
  },

  "governance": {"enable_governance": true, "enable_virtual_keys": true},
  "telemetry":  {"enable_prometheus": true, "prometheus_path": "/metrics",
                 "custom_dimensions": ["run", "phase"]}
}
```

**One asymmetry to watch:** `network_config.base_url` is a plain `string`, not a `SecretVar`, so `env.VLLM_BASE_URL` there is used **literally** and the request goes nowhere useful. Only fields typed `SecretVar` (`keys[].value`, `vllm_key_config.url`, `ollama_key_config.url`, proxy fields, `ca_cert_pem`) resolve `env.*`. Prefer route (a) when the endpoint is vLLM or Ollama; with route (b) the URL is config, not a secret, so keep private hostnames in mind before publishing a bundle.

Three consequences for the harness:

- **Credentials stay env-var references.** `env.VLLM_API_KEY` is resolved by the gateway, so a provider secret is never written into a config file or a manifest — the same rule the registry follows (§10).
- **Virtual keys must name the provider.** A key's model restriction is `provider_configs[].allowed_models`, and each entry carries the `provider` it applies to. Provisioning therefore groups the suite's pinned models by provider and posts one entry per provider, rather than a bare model list — a list without a `provider` silently restricts nothing on a multi-provider gateway. Pinning is only enforcement if the key names both halves of the pair. A legacy suite naming bare ids still provisions, and pins the model list alone, which is all that suite ever said.
- **The model plane stays one hop.** Systems and harness both see one base URL (the gateway) and one credential shape regardless of how many providers sit behind it. Adding a provider is a gateway-config change and never a harness change, which is what keeps `orchestrator/` free of provider branching (P1, P7).

**A provider change is a suite version bump.** The suite pins the *environment*, and which endpoint served a model is part of that environment: two runs naming `qwen3-14b` against different deployments are not directly comparable — quantization, serving stack, and sampling defaults all move the number. The provider name therefore travels into `manifest.json` and onto the report row beside the model id. What we deliberately do **not** claim is bit-identical behaviour across two endpoints serving the same weights; that is a `same-model, different-provider` caveat for a reader, not something the harness can verify.

### The four confounds this closes

| Confound | Mechanism |
| --- | --- |
| Different LLM per system | Gateway virtual key pins `(provider, model)`; egress block prevents bypass; `verify` reconciles traffic. |
| Different retrieval budget | `top_k` pinned; harness truncates to `top_k`; `verify` asserts `len(data) ≤ top_k`. |
| Different answer/judge prompt | Prompt files content-hashed into `manifest.json`. |
| Judge favouring its own family | Judge model is separate from the answer model, and separately pinned. |

**Judge resolution.** The suite judge is a default, not an absolute: a dataset whose published scoring names a specific judge declares it as `official_judge` (§9) and that wins. Reproducing a dataset's official metric matters more than judge uniformity across datasets — a number scored under a different judge than the paper's is not the paper's number. The **resolved** judge is recorded in `manifest.json` and on every report row, and quality columns are comparable within a dataset, never across datasets.

### What is *not* pinned, by design

Internal chunking, extraction prompts, index type, graph construction, summarization policy, and storage engine belong to the system under test. That is the thing being measured. We pin the *environment*, never the *architecture*.

### Fairness caveats we state rather than hide

- **Concurrency is capped identically, not equalized.** A system that is internally slower gets the same worker count, not the same throughput. Latency is reported, never used to adjust quality scores.
- **Systems differ in how much LLM work they do at write time.** That is a real architectural difference; §7 reports it as cost rather than normalizing it away.
- **Gateway metering covers model calls, not internal compute.** GPU or CPU used inside a system is out of scope and is declared as such.

---

## 7. Metrics

Five families. A single accuracy number hides the tradeoffs that actually distinguish memory architectures.

### 7.1 Task quality

Per-dataset scorer, declared by the dataset plugin — never inferred from a name.

| Scorer | Use |
| --- | --- |
| `judge_binary` | Open-ended QA → CORRECT/WRONG, structured JSON label |
| `judge_rubric` | Multi-requirement answers → per-requirement satisfaction ratio |
| `mcq` | Choice tasks → deterministic option mapping |
| `exact_match_f1` | Short-form extraction → EM, token F1, ROUGE-L |
| `event_ordering` | Temporal sequence tasks → Kendall τ-b |

### 7.2 Retrieval quality — recall@k without breaking the black box

Recall is worth keeping, and it does **not** require the system to expose provenance. Two tracks:

- **Fuzzy (always available).** Normalize returned `content`, match against gold evidence spans by anchor overlap. Works on any conformant system.
- **Exact (opt-in).** If the adapter reports `source_ids`, compare ID sets directly.

Where a system offers both, the exact track **calibrates the fuzzy matcher** — we publish the matcher's precision/recall against exact ground truth so readers know the error bar on fuzzy numbers. Every recall figure is labelled with the track that produced it.

**Evidence granularity is per-dataset and must be labelled too.** LoCoMo-Refined annotates evidence at message level; LongMemEval-S annotates it at session level, so a gold session expands to every chunk derived from it. Recall@k against a coarser unit is a systematically easier target — the numbers are comparable within a dataset, never across datasets with different granularity.

That expansion happens once, at plan time, in `orchestrator/evidence.py`: a dataset handle resolves onto the chunk ids a system can report in `source_ids`, and `plan.jsonl` records both — the original handles stay auditable in `evidence_chunk_ids`, the resolved ids go in `evidence_chunks`. Chunks carry their `source_indices` so the mapping survives the D4 oversized-message split, where two chunks share one source message. Scoring and the `oracle_gold` ceiling both read the resolved field, so the ceiling cannot drift from what recall measures.

A message-granular handle may be either a position (`m3`, as the fixtures annotate) or the dataset's own message id (LoCoMo's `dia_id`, `D1:3`); both resolve, so an adapter never has to renumber upstream evidence into our positions. A handle that resolves to no chunk drops that question to the `unavailable` track rather than scoring it zero — LoCoMo has five upstream evidence typos, and counting them as retrieval misses would understate every system.

Reported: `recall@k`, `precision@k`, `MRR`, `nDCG@k`, plus `context_efficiency` = tokens of retrieved content needed per correct answer.

### 7.3 Cost — from the gateway, not self-reported

`write_tokens_per_chunk`, `read_tokens_per_query`, `llm_calls_per_chunk`, `llm_calls_per_query`, `total_tokens`. Measured at the gateway, identically for all systems, so a system that spends 10× the tokens for +2 points is visible rather than flattered.

**Two attribution qualities, labelled as such.** Bifrost exposes per-key token counters on `/metrics` (labelled `virtual_key_id`) but no usage-readback API, so per-phase cost is a counter diff across phase boundaries. Harness calls carry a key per `(run_id, system, phase)`, so their split is exact. A black-box system gets one key per `(run_id, system)` — it holds a single credential for the whole run, not one per phase — so its split is *derived* from the boundary between non-overlapping phases.

**How that key reaches the system.** Not through the container's environment: the key is named for a run that does not exist when the container starts, so env delivery would force a restart per run. Instead an adapter may implement an optional `use_model_key(value)`, which the harness calls once at run start, after provisioning and before any stage runs. A system that cannot accept a credential at runtime omits it and its cost reads `unavailable` — never zero. `adapters/mem0.py` implements it as a partial `POST /configure`, which mem0 deep-merges without disturbing its pinned models or dropping its stored memories. That non-overlap is therefore a correctness invariant, asserted in tests; when it is deliberately broken (§11, concurrent runs against one system) the run records `cost_attribution: "unreliable"` and omits the per-phase columns.

### 7.4 Latency

p50 / p95 / p99 for ingest and retrieve, plus total ingest wall-clock. Reported alongside quality, never blended into it.

### 7.5 Robustness

Derived from artifacts and `verify`: contract violation count, retry rate, timeout rate, **rank stability** across repeated identical retrievals (§8), and canary leakage (must be zero).

### Reporting rule

No single composite score. The report is a table with quality, recall, cost, and latency side by side; any weighted index is left to the reader, computed from published columns.

Two labels the reader needs in order not to over-read a number: **which judge produced a quality score** (§6 — datasets may override the suite judge, so quality is comparable within a dataset, not across), and **whether a cost figure is measured or derived** (§7.3 — harness-side is exact per-phase, system-side is inferred from phase-boundary snapshots).

---

## 8. Verification

Black-box trust requires cheap adversarial checks. `verify` runs on every benchmark run and its output is published beside the report.

| Check | Method | On failure |
| --- | --- | --- |
| **Cross-user leakage** | Ingest canary memories under a sibling `user_id`; search the target user for them | **Run invalidated.** Not a score penalty. |
| **Searchable on return** | Immediately after `add` returns, probe-search a known chunk | Flag `late_indexing`; results marked provisional |
| **Model pinning** | Reconcile gateway call count against phase activity | Flag `unmetered_model_use`; row withheld |
| **Budget honoured** | Assert `len(memories) ≤ top_k` on every `search` | Contract violation |
| **Content present** | Every returned `Memory` has non-empty `content` | Contract violation |
| **Rank stability** | Repeat retrieval on a sample; report Kendall τ across runs | Reported as variance, not a failure |
| **Writes landed** | Between `ingest` and `retrieve`, probe a bounded sample of ingested contexts with one of their own planned questions | Contract violation |

Two rows changed with the interface. **Searchable on 200** becomes *searchable on return*: there is no status code, and the obligation now attaches to `add` returning rather than to an HTTP response — the same guarantee, expressed in the only terms the interface has. **ID echo** is deleted outright: it verified that a system copied `request_id`, `user_id`, and `session_id` back into its response, which was only ever a proxy for "the system read the fields we sent." A method call cannot fail to receive its arguments, so the check has no failure mode left to catch. It is replaced by **content present**, which catches a real adapter bug the old suite missed: a projection that maps the wrong field and returns memories with empty `content`, which would score as a confident wrong answer rather than as a broken integration.

**On determinism.** A black box cannot be seeded. Rather than claiming determinism we do not have, we *measure* non-determinism (rank stability) and pin everything downstream of retrieval (`temperature: 0`, fixed prompts) so that observed variance is attributable to the system, not to our harness.

Leakage is treated as invalidating rather than score-reducing because a leaking system is not solving the benchmark task at all.

---

## 9. Datasets as plugins

A dataset is a **directory**. Adding one never requires editing the orchestrator (P5).

```
datasets/<name>/
├── dataset.yaml        # identity, version, source hash, scorer, task type
├── adapter.py          # raw source → normalized records
├── data/               # the corpus itself: fetched, never committed
└── prompts/            # only if the dataset mandates its own official prompt
```

**Corpora are fetched, not vendored.** `dataset.yaml` pins a `url` and a `sha256`, which makes a fetched file verifiable without it living in git — LongMemEval-S Cleaned is 265 MB, and LoCoMo-Refined is CC BY-NC 4.0 and should not be redistributed. `uv run python -m scripts.fetch_datasets` reads the manifests and installs anything missing or stale; a download whose hash does not match its pin is refused rather than installed, since the pin *is* the integrity guarantee. `datasets/*/data/` is gitignored, and no test depends on a corpus being present — the adapter tests run against trimmed real-shape samples under `tests/datasets/real_shapes/`.

A *dataset* adapter normalizes into the harness's `Message` shape (`role`, `content`, `timestamp`), carrying whatever its evidence unit needs alongside: LoCoMo keeps `dia_id` per message, LongMemEval keeps its own `session_id`. Those are dataset fields used for evidence resolution at plan time, unrelated to the contract `session_id` that §4.2 removes. Upstream field names never reach the chunker.

```yaml
# datasets/<name>/dataset.yaml
name: <name>
version: "1.0"
source: {url: "...", sha256: "..."}    # content-pinned; hash mismatch aborts
task_type: open_qa | mcq | rubric | ordering
scorer: judge_binary
supports_recall: true                   # does gold evidence exist for this dataset?
official_prompt: prompts/answer.txt     # optional; overrides suite prompt when the
                                        # dataset's own paper mandates specific wording
official_judge:                         # optional; overrides the suite judge model when
  id: <model>                           # the dataset's scoring definition names one
  temperature: 0                        # (e.g. LoCoMo-Refined pins Qwen3-14B)
```

`adapter.py` implements one function returning two iterables:

```python
def load(config) -> tuple[Iterable[Context], Iterable[Question]]:
    """
    Context:  {ctx_id, messages: [{role, content, timestamp}], chunk_ids: [...]}
    Question: {q_id, ctx_id, question, options?, gold, evidence_chunk_ids?, meta}
    """
```

That is the entire plugin API. Everything downstream — chunking, ingestion, retrieval, answering, judging, scoring — is generic.

### Admission criteria (how "latest" is decided)

We do not maintain a static list of blessed datasets; we maintain **rules**, so the suite stays current without repeated re-litigation:

1. **Genuinely memory-bound** — not solvable by pasting the corpus into the pinned model's context. Verified empirically by a long-context control run (§10); if the control scores near the top, the dataset is measuring long-context reading, not memory.
2. **Official scoring exists** — a published prompt or metric we can reproduce, cited by URL and commit.
3. **Contamination-resistant** — released or refreshed recently enough to reduce pretraining overlap; we record release date and note the risk.
4. **Discriminative** — separates systems in practice. A dataset where everything scores 95%+ is retired to an archive tier.
5. **Content-pinned** — a stable hash. Silently mutating datasets are rejected.

**Superseded datasets are archived, not deleted.** They remain runnable under an `archive/` tier for historical comparison, and are excluded from the headline table. This matters because a lot of published numbers exist on older datasets; we want continuity without letting saturated benchmarks drive the ranking.

Each dataset directory records why it is in the current tier, so the selection is auditable rather than a maintainer's taste.

---

## 10. Baselines (controls, not competitors)

Every run includes reference systems. They are **experimental controls** and are always shown in the report.

`no_memory`, `bm25`, and `embedding_rag` implement `MemoryAdapter` and pass the conformance suite unchanged. `oracle_gold` and `long_context` **cannot** — they need gold evidence and the full corpus, which the interface deliberately does not carry — so they bypass it, reading `plan.jsonl` and emitting `retrieve.jsonl` directly. Everything downstream of retrieval is identical, which is what keeps them comparable; their rows are marked `privileged: true` so a reader knows the provenance differs.

That flag was called `in_process`, which the refactor makes ambiguous: *every* adapter is in-process now, so the old name would read as "ordinary" rather than "skipped the interface." `privileged` names the thing that actually differs — these two baselines see data no real system is given.

| Baseline | Answers the question |
| --- | --- |
| `no_memory` | What does the pinned model score with no memory at all? → the floor |
| `bm25` | Does this system beat lexical search? |
| `embedding_rag` | Does this system beat flat dense retrieval? |
| `long_context` | Is this dataset actually memory-bound? → the dataset validity check |
| `oracle_gold` | Ceiling: perfect retrieval of gold evidence → separates retrieval failure from answering failure |

`oracle_gold` and `no_memory` bracket every score, which converts a raw percentage into a meaningful position between floor and ceiling. `long_context` doubles as the admission test in §9.1 — a system that cannot beat `bm25` on a dataset, or a dataset where `long_context` wins, both deserve scrutiny.

These also serve as reference implementations of the interface for anyone integrating — `bm25` is the shortest complete example of an adapter that reports `source_ids` and therefore lands on the exact recall track.

### Adapters — where per-system knowledge is allowed to live

Real systems speak their own API. The self-hosted mem0 server stores at `POST /memories` with `{messages: [{role, content}], user_id, metadata}` and searches at `POST /search` with `{query, filters: {user_id}, top_k}`, returning `{"results": [{id, memory, score, metadata}]}`. Something must translate, and that something is an adapter class under `adapters/`, one module per system.

This is the *only* place per-system names may appear. `orchestrator/` stays method-free (P1), so a system is added by writing an adapter plus a registry entry, never by editing the harness.

```python
# adapters/mem0.py
class Mem0Adapter:
    """The self-hosted mem0 server → the MemoryAdapter interface (§4.1)."""

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        # `timestamp` is dropped: the server's Message model is {role, content}.
        await self._post("/memories", {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "user_id": user_id,
            "metadata": {"x_chunk_id": chunk_id},
        })

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        # The query goes through verbatim: rewriting it would measure our rewriting.
        payload = await self._post("/search", {
            "query": query, "filters": {"user_id": user_id}, "top_k": top_k,
        })
        return [
            Memory(
                content=item["memory"],
                score=item.get("score"),
                source_ids=_chunk_ids(item.get("metadata")),
            )
            for item in payload["results"]
        ]
```

That is the whole integration — roughly 80 lines with error classification, against ~200 for the ASGI shim the old wire contract required.

The adapter's job stays narrow: rename fields, pass `user_id` through as the isolation key, hand `top_k` to whatever the system calls its limit, and send the query **verbatim**.

Note this is *not* `HttpAdapter`, which speaks the harness's own retired `/add`+`/search` shape. A system with its own HTTP API needs its own adapter; `HttpAdapter` fits only a system built to the old contract. Sharing a path name is not sharing a payload.

Two obligations. Carry `chunk_id` into whatever metadata the system supports and echo it back as `source_ids`, or the row falls off the exact recall track. And for a system reached as a **library** rather than over HTTP, **import it inside `__init__`, never at module scope** — a foreign runtime in our import path is the failure mode this design otherwise re-admits (§14), so a broken or absent dependency must fail one adapter rather than the whole CLI.

An adapter is finished when `bench doctor <system>` passes every check — including `no_cross_user_leakage`, which catches an adapter that forgot the isolation filter.

### Systems that only speak HTTP

A hosted service or a container is reached by an adapter that makes HTTP calls, which is an ordinary adapter and not a special case. `HttpAdapter` ships as one, configured by URL:

```yaml
# registry/hosted.yaml
name: hosted
adapter: adapters.http:HttpAdapter
config:
  base_url: https://memory.example.com
  auth: {scheme: bearer, key_env: HOSTED_TOKEN}
```

So the HTTP path is preserved for the cases that need it — containerized systems keep the strong egress guarantee of §3, and a vendor can still be measured without shipping us a Python client. What changed is that HTTP is now one adapter among several rather than the mandatory surface every system must put on. Credentials remain env-var names, never literals and never in a URL.

---

## 11. Usability

### Running a benchmark

```bash
# 1. Write an adapter — one class, two methods (§10)
#    adapters/mysystem.py

# 2. Register it — the whole integration, once
cat > registry/mysystem.yaml <<'YAML'
name: mysystem
adapter: adapters.mysystem:MySystemAdapter
config: {}          # passed to __init__ verbatim; optional
YAML

# 3. Check the adapter before spending money on a full run
bench doctor mysystem

# 4. Run — the run id is generated and printed
bench run mysystem --dataset <name> --suite v1
→ 3f9a2b7c1d04
```

`bench run` executes all seven stages and prints the report. Everything else has a pinned default from the suite.

```bash
bench run mysystem --dataset <name> --limit 5   # smoke run, minutes
bench rescore 3f9a                              # new metric, zero SUT cost; prefix ok
bench verify 3f9a                               # re-run integrity checks
bench run --resume 3f9a2b7c1d04                             # continue an interrupted run
```

Step 1 is the only real work, and for a system whose SDK is close to the interface it is under 30 lines. A system that runs as a container or a hosted service skips it — `adapters.http:HttpAdapter` takes a URL.

### Parallel runs and monitoring

Runs are independent by construction: `user_id` embeds `run_id`, artifacts are per-run directories, and `score` is pure. So `bench run` takes a cross product and executes it as a pool of subprocesses.

```bash
bench run sys-a,sys-b --dataset locomo-refined,longmemeval-cleaned --parallel 4
```

Runs against the **same system** are serialized — concurrent load on one backing store contaminates the latency (§7.4) and rank-stability (§7.5) metrics, so this is a fairness rule rather than a tuning knob. Useful parallelism is therefore bounded by the number of distinct systems, not by the size of the cross product. `--allow-concurrent-same-system` overrides it and marks the affected rows `contended: true`.

That bound is also the **default for `--parallel`**: the number of distinct systems in the cross product. A fixed default of 1 left every system but one idle while buying no fairness the same-system lock was not already providing, and a default above the system count cannot be spent.

Note that one process per run also does the work an ASGI shim used to do incidentally: each run constructs its **own adapter instance** in its own interpreter, so two runs cannot share a mutable client, a connection pool, or a module-level cache. In-process adapters make that isolation load-bearing rather than convenient — a single-process batch would have to trust every adapter to be re-entrant, which is not a property we can verify for someone else's SDK.

One process per run is what makes the isolation real: a segfault or an OOM kill takes down one run rather than the batch, and `pid` in `state.json` — which `bench stop` signals and `stale` is derived from — only means something when runs are separate processes. Each child is the same `bench run` a user invokes by hand, so there is one code path rather than two. Contention is the exception to that symmetry: whether a run shares its system is a property of the batch, not something a child can observe, so the parent passes `--contended` down explicitly.

**Every run detaches by default, batches included.** A run is measured in hours, so holding the launching terminal for it makes a closed lid, a dropped SSH session, or a Ctrl-C meant to stop *watching* into the end of the run — and that argument applies harder to a cross product than to one run. `bench run` therefore starts the work in its own session with no inherited stdio, prints the run ids, and returns. `--foreground` keeps the attached behaviour — it streams in the launching process and prints the report row when each run finishes — and is what the batch supervisor passes to each child, since that supervisor awaits the child and reads its stdout for the row.

What detaches for a cross product is **one supervisor, not each job**. Detaching the jobs individually would release the per-system lock the instant each child was launched rather than when it finished, so two runs against one system could overlap and contaminate exactly the metrics the lock protects. A single detached supervisor keeps the lock meaningful while the launching terminal goes free.

Because the supervisor's stdout goes nowhere once detached, the **launcher** generates the run ids and hands them down rather than letting each child invent its own. A batch interrupted an hour later is only resumable by ids the user has already seen.

Validation happens before the spawn, so a misnamed system or a bad flag still fails on the command the user typed rather than dying unseen in a child.

Detaching makes the log the primary view of a run, so each run directory also carries **`run.log`**: one line per stage boundary and one per unit of work — every `add`, every `search`, every model call — with ids, statuses, counts, attempt numbers, and latencies.

Every model call names the **model that served it**, and a `run models` line records the run's resolved chat, judge, and embedding models up front. A model id is not content, and it is what makes a per-call line attributable: a dataset may pin its own judge and override the suite (D11), so `answer` and `judge` in one run can be served by different models — and the *resolved* judge is the one logged, since the suite's id would name a model that never ran. Ids are provider-prefixed, because a pinned model is `(provider, id)` and the id alone does not say which endpoint answered. `run models` is its own line rather than a field on `run start`: that one is written before the suite is loaded, so it cannot name a model the run has not resolved yet. A retry that would otherwise be invisible is logged at `WARNING`, since a silently retried call is what makes a flaky system look healthy. The prohibition from §12 is absolute here: no prompts, no messages, no retrieved content, no completions, no keys. `bench logs <id>` prints the status header and the tail of that file, and `-f` follows it until the run stops.

Progress is a directory scan — no daemon, no database. Each run maintains `artifacts/<run_id>/state.json` (status, stage, pid, heartbeat, counts); `bench ps` renders it Docker-style, deriving `stale` from a dead pid plus a cold heartbeat.

```
RUN ID        SYSTEM  DATASET              STAGE       PROGRESS         ELAPSED  STATUS
3f9a2b7c1d04  mem0    locomo-refined       2/7 ingest  412/1986 chunks  4m12s    running
8c1e04b7f2a9  bm25    longmemeval-cleaned  -           -                2m44s    done
b70d3e91a5c8  letta   locomo-refined       2/7 ingest  88/1986 chunks   31m10s   stale
```

The stage cell carries its **position in the pipeline**, because a stage name alone does not say how much of a run is left. The total is per-run, not a constant: the in-process baselines inject retrieval instead of ingesting (D2), so they run six stages and are numbered out of six. `PROGRESS` carries the **unit** it counted for the same reason a recall figure carries one — `412/1986` does not say whether those are chunks, searches, or answers.

In-stage counts are refreshed by a background ticker while the stage runs, not written once at the stage boundary. A count written after `run_ingest` returns is a count that never moved during the hours ingest was actually running, which is the whole thing `bench ps` is for. The ticker counts the artifact the stage is appending to, so no stage has to know it is being watched, and an unreadable count (a torn final line in a file being appended to) leaves the last good number in place rather than failing the stage over the thing observing it.

Fixed-width ids are why that table aligns; the previous hand-written ids were arbitrary-length and pushed every later column around.

Flags: `--all` includes finished runs, `--json` is scriptable, and `--watch` clears and redraws until no run is live. `--watch` and `--json` are mutually exclusive: a loop emitting one JSON document per pass is not parseable. Companions: `bench logs <id>` (`--tail`, `-f`), `bench stop <id>`, `bench rm <id>` — each taking a full id or an unambiguous prefix. Counts come from the artifacts themselves, so `state.json` is a liveness cache — losing it costs visibility, never data.

### Minimal config, in totals

| Artefact | Size | Frequency |
| --- | --- | --- |
| Adapter | ~25–30 lines Python | once per system |
| System registration | ~3 lines YAML | once per system |
| Suite | one pinned file | maintained centrally |
| Dataset plugin | `dataset.yaml` + `adapter.py` | once per dataset |
| Per-run flags | `--dataset`, `--suite` | per run |

Concurrency, timeouts, retries, `top_k`, models, and prompts are **not** per-run choices — that is what makes results comparable.

### Repository layout

```
orchestrator/     stages, retry policy, artifact IO      ← no method knowledge
contract/         MemoryAdapter protocol, Message/Memory,
                  conformance/ (checks against any adapter)
adapters/         one module per system: mem0, http, ... ← per-system knowledge
suites/           v1.yaml, ...                           ← the pinned world
datasets/<name>/  dataset.yaml + adapter.py              ← plugins
scorers/          judge_binary, judge_rubric, mcq,
                  exact_match_f1, event_ordering, recall_at_k
registry/         one YAML per system under test
reference/        baselines (§10)
gateway/          bifrost config, virtual keys, egress policy
artifacts/<run>/  manifest.json + *.jsonl + report.json + verify.json
                  + state.json (liveness for `bench ps`)
                  + run.log (what the run did; never what it sent)
```

`adapters/` is promoted to a top-level package from `reference/adapters/`: adapters are now the integration surface itself rather than shims sitting behind one, and burying them under `reference/` (which holds experimental controls) misfiled them. `contract/` keeps its name and its job — it still defines the one thing every system must satisfy — but that definition is now a Python protocol instead of an OpenAPI document, and `contract/openapi.yaml` is deleted.

Two dataset-plugin names are unfortunate and unchanged: `datasets/<name>/adapter.py` is a *dataset* adapter (a `load()` function) and has nothing to do with a `MemoryAdapter`. They have coexisted since the first draft; renaming the dataset one is a larger and separate change.

### Integrating a system with no Python client

Use `adapters.http:HttpAdapter` with a `base_url` (§10). A containerized or hosted system needs no code from you and keeps the stronger egress guarantee of §3.

---

## 12. Reproducibility

Every run writes `manifest.json` before touching anything:

```jsonc
{
  "run_id": "3f9a2b7c1d04",              // generated, 12 hex chars (§4.5)
  "name": "sweep-3",                     // optional --name label, if given
  "created_at": "2026-08-04T10:00:00Z",  // what the id used to encode
  "orchestrator_commit": "…",
  "suite": {"version": "v1", "sha256": "…"},
  "dataset": {"name": "…", "version": "1.0", "source_sha256": "…"},
  "system":  {"name": "…",
              "adapter": "adapters.mem0:Mem0Adapter",  // the exact class that ran
              "adapter_version": "…",                   // system's own version, if it reports one
              "deployment": "in_process|container",
              "image_digest": "sha256:…"},              // container deployments only
  "models":  {                                   // each is {provider, id} — which
                                                 // endpoint served it is part of
                                                 // the pinned environment (§6.1)
    "chat":      {"provider": "openai", "id": "gpt-4o-mini"},
    "embedding": {"provider": "openai", "id": "text-embedding-3-small"},
    "judge":     {"provider": "vllm",   "id": "qwen3-14b"},  // resolved
    "judge_source": "dataset|suite"                          // dataset official_judge, else suite
  },
  "providers": {                                 // resolved base URLs, never credentials
    "openai": {"base_url": "https://api.openai.com"},
    "vllm":   {"base_url": "https://vllm.internal.example.com",
               "served_model": "qwen3-14b"}      // as the endpoint reports it, if it does
  },
  "prompts": {"answer_sha256": "…", "judge_sha256": "…"},
  "top_k": 100
}
```

**Digests, not tags.** `system:latest` is not reproducible; `sha256:…` is. A run whose system cannot report a digest is marked `unpinned_image` in the report.

**Base URLs, not credentials.** The `providers` block records where each model was served so a reader can tell a hosted `qwen3-14b` from a locally-served one; the credential is an env-var reference in gateway config and never reaches the manifest. A self-hosted endpoint is the weakest link in this section: an image digest pins a system and a sha256 pins a prompt, but "whatever `vllm.internal` was serving that day" pins nothing. Record the served model's own reported version where the endpoint exposes one, and treat cross-provider comparisons of the same model id as a caveat rather than a controlled variable.

**Publishable bundle.** `manifest.json` + all JSONL + `report.json` + `verify.json`. Because `score` is offline and pure, anyone can recompute the table from the bundle — no memory system, no API keys, no GPU. This is the strongest reproducibility property in the design, and it is the direct payoff of P3 and P4.

**Data handling.** Evaluation data is used only to complete the run; systems must not train on it or retain it beyond an agreed window. Stated as a participation term, and partially checkable: a system that has memorized the dataset scores anomalously on the `no_memory` control.

---

## 13. Migration

Incremental. No flag day, and nothing published is invalidated mid-flight.

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 1 | `MemoryAdapter` protocol + conformance suite + `bench doctor` | Suite passes against a trivial in-memory adapter |
| 2 | Orchestrator stages 1–6 with `no_memory` and `bm25` baselines | Report produced end-to-end |
| 3 | Gateway with pinning, metering, egress block | `verify` catches a deliberately misbehaving mock |
| 4 | Port 2 contrasting real systems; run **both** old and new paths on one dataset | Discrepancies explained, not just observed |
| 3.5 | Parallel runs, run state, `bench ps` | Parallel run byte-identical to solo run |
| 5 | Port remaining systems; migrate datasets under §9 criteria | Old harness retired |

**Phase 4 is the important one.** Running the vendored path and the adapter path side by side on the same dataset is how you distinguish "the new architecture is measuring correctly" from "the new architecture produces different numbers." Old and new reports share the dataset and suite pins, so rows are directly comparable.

The adapter refactor (§4, §4.5) landed as a **Phase 6**, after Phase 5 rather than before it: it changes the integration surface, the retrieve record shape, and every `--run-id` call site at once, and doing that while systems are still being ported would confuse "the port is wrong" with "the refactor is wrong." Its exit criterion was that a pre-refactor artifact bundle still rescores to a byte-identical `report.json`, which is the check that the change touched integration and not measurement. That holds for every bundle captured before the refactor, and `score` reads both the old `data` key and the new `memories` key so an archived bundle stays rescorable indefinitely (§5).

---

## 14. Tradeoffs accepted

| Gain | Cost |
| --- | --- |
| Integration is ~25 lines of Python, no server to write or run | Foreign runtimes are back in our import path — see below |
| One process, one stack trace; a failing `add` raises where it broke | Dependency conflicts between systems become real again |
| No socket, port, or serialization between harness and system | In-process systems cannot be network-isolated, so gateway metering is `unenforced` for them (§3) |
| Uniform, gateway-measured cost metrics | Internal (non-LLM) compute is unmeasured and declared out of scope |
| Rescoring costs nothing | Retrieval artifacts record the adapter's projection, not the system's raw bytes (§5) |
| Anything measurable is measurable for everyone | We can no longer instrument internals — the metric list must be decided **up front**, since a gap discovered post-publication cannot be backfilled |
| Systems upgrade independently for container deployments | For in-process ones, reproducibility depends on a pinned dependency set rather than an image digest |

**The import-path cost is the one that was previously bought off, and it is worth being explicit.** An earlier draft made HTTP mandatory specifically to keep foreign runtimes out of our interpreter — that was the named failure mode of the harness this design replaced. Choosing simplicity re-admits it. Four things keep it bounded:

- Adapters import their system **inside `__init__`**, never at module scope, so a broken dependency fails one adapter rather than the CLI.
- Each run is its own process (§11), so a crash or OOM takes one run.
- Systems with incompatible dependencies are installed as optional extras, or run as containers behind `HttpAdapter` — the escape hatch stays open precisely because this cost is real.
- The gateway remains the only route to a model, so a system cannot quietly change the answer model even while sharing our interpreter.

**Mitigation for the reviewer-burden cost:** publish the artifact bundle. Reproducing the *numbers* requires only `bench rescore`; reproducing the *run* requires the systems installed. Most verification needs only the former.

---

## 15. Open questions

1. **Judge model choice.** A stronger judge is more accurate but costlier and may favour its own family. Proposal: pin one judge per suite version, publish agreement rates against human labels on a sample. **Now contested:** LoCoMo-Refined's scoring definition *is* a specific judge (Qwen3-14B) plus a stricter prompt — 86.33% human agreement vs 43.67% for the original setup. Either `dataset.yaml` may pin `official_judge` and override the suite, or our numbers are not the official metric and must be labelled so. Tracked as D7 in `IMPLEMENTATION.md`.
2. **Fuzzy recall matcher precision.** Needs calibration against systems that do report `source_ids` before fuzzy recall numbers are published as headline figures.
3. **`top_k = 100` for all task types.** Generous for single-hop, possibly tight for multi-hop synthesis. Consider a per-dataset pin declared in `dataset.yaml` rather than one global constant.
4. **Multi-turn / interactive tasks.** The current contract is write-then-read. Interleaved read-write sessions need a session-ordering extension.
5. **Retirement threshold.** What score triggers moving a dataset to `archive/`? Needs a concrete number to avoid case-by-case argument.

---

## Appendix — relationship to prior art

**[Agent Memory Leaderboard](https://agentmemories.ai/api-guide) (partly adopted):** retrieve-only discipline, `user_id`-as-namespace isolation, separated answer/judge model configuration, and stage-per-process JSONL pipelines. We extend it with a metering gateway, retrieval-quality metrics it omits, an explicit verification stage, and pluggable datasets.

We **dropped its wire format** (§4). Earlier drafts were deliberately wire-compatible so a system integrated there would work here unchanged; the compatibility bought less than it cost. It required an envelope, three echoed ID fields, and an `x_` prefix namespace that no metric read, and it forced every system — including in-process libraries — to be wrapped in a server. Its ideas were the valuable part, not its JSON.

**Current in-process harness (replaced):** its dataset normalization (context chunks + question records) and its own black-box text-anchor recall technique are carried forward. What is dropped: vendored runtimes in the import path, per-method configuration branching, private-method monkeypatching for provenance, external-state reset special cases, and monolithic rewrite-per-query result files.
