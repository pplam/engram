# Engram

A benchmark for memory systems, measured as **black boxes**.

The only integration surface is two methods: store messages, return memories. Write a
~25-line adapter class and your system can be benchmarked — no server to run, no vendored
source, no monkeypatching of private methods. A third party can recompute every published
number from the artifact bundle alone, without re-running your system.

Three claims the design exists to defend:

- **Fair** — answer model, judge model, `top_k`, prompts, and chunking are pinned per
  suite version. Differences in score come from memory architecture, not model choice.
- **Black box** — anything we measure is observable from the contract or from the
  model gateway.
- **Reproducible** — scoring is a pure function over append-only JSONL artifacts.

Explicit non-goals: benchmarking base LLMs (the answer model is a constant), benchmarking
long-context reading (if pasting the corpus into a prompt solves it, it isn't measuring
memory), and provisioning memory systems (you start it; the harness only talks to a URL).

## How a run works

Seven stages, strictly sequential, each appending JSONL to `artifacts/<run_id>/`:

```
plan → ingest → retrieve → answer → judge → score → verify
         └────────┬───────┘
       the only stages that touch your system
```

`score` and `verify` are offline by construction — no network, no clocks, no randomness.
That is what makes `bench rescore` free: a new metric costs zero system calls.

Stage ordering is load-bearing, not tidiness. System-side per-phase cost is derived from
the ingest/retrieve boundary, so overlapping phases would silently corrupt a published
number.

## Requirements

- Python ≥ 3.12, managed with [`uv`](https://docs.astral.sh/uv/)
- A memory system to benchmark: an importable library, a container, or a hosted service.
  Any of the three becomes an adapter; the in-repo baselines need nothing at all
- An OpenAI-compatible model endpoint for the answer and judge stages — optional for a
  first smoke run, which can stub the models out

```bash
uv sync
```

## Step-by-step: benchmarking a memory system

### 1. Write an adapter

An adapter is a **client, not a server**. Two methods, defined in
[`contract/adapter.py`](contract/adapter.py):

```python
# adapters/mysystem.py
from contract.adapter import Memory, Message

class MySystemAdapter:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        from mysystem import Client        # import here, never at module scope
        self._client = Client(**(config or {}))

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Store messages. Return only once they are searchable."""
        await self._client.write(
            namespace=user_id,
            items=[{"role": m.role, "text": m.content} for m in messages],
            tags={"chunk_id": chunk_id},
        )

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """Return at most top_k memories, best first. Never an answer."""
        hits = await self._client.query(namespace=user_id, q=query, limit=top_k)
        return [
            Memory(content=h.text, score=h.score, source_ids=[h.tags["chunk_id"]])
            for h in hits
        ]
```

Four rules worth calling out:

- **`user_id` is the isolation boundary.** A search must never return another user's
  content; `verify` invalidates the whole run if it does.
- **`add` returns only once the content is searchable.** Returning early gets flagged as
  `late_indexing` and marks the results provisional.
- **`search` returns memories, never an answer.** Generating the answer is the harness's
  job with a pinned model — that is what makes systems comparable.
- **Report `source_ids`.** Carry `chunk_id` into whatever metadata your system supports
  and echo it back. Without it you still score quality, but recall reads `unavailable`.

Import your SDK inside `__init__`, not at module scope, so a missing dependency fails one
adapter rather than the whole CLI. A synchronous SDK is fine — wrap calls in
`asyncio.to_thread`.

**If your system is a container or hosted service**, you still write an adapter — it just
makes HTTP calls instead of library calls. `adapters/mem0.py` is a worked example
against the self-hosted mem0 server.

The bundled `HttpAdapter` is narrower than it looks: it speaks the harness's own retired
`/add` + `/search` shape, so it fits a system built to that contract and nothing else. A
system with its own HTTP API needs its own adapter, because sharing a path name is not
sharing a payload.

### 2. Register it

One file, ~3 lines. This is the entire integration.

```bash
cat > registry/mysystem.yaml <<'YAML'
name: mysystem
adapter: adapters.mysystem:MySystemAdapter
config: {}          # passed to __init__ verbatim
YAML
```

For a system reached over HTTP, the entry is a URL and the name of the variable holding
its credential:

```yaml
name: mem0
adapter: adapters.mem0:Mem0Adapter
config:
  base_url: http://127.0.0.1:8888
  # api_key_env: MEM0_SERVER_API_KEY   # env var name, never the key
```

Credentials for your own adapter are your own business — read them from the environment
exactly as your SDK would. Nothing in the registry names a credential, so nothing in the
registry can leak one.

### 3. Start the system, if it is a service

An adapter is a client, so anything it talks to has to be up first. The harness never
provisions a memory system. For the shipped mem0 entry, that is mem0's own stack —
FastAPI plus Postgres/pgvector, from its repo:

```bash
uv run python -m scripts.setup_mem0 --clone
```

That clones mem0 into `systems/mem0/` (git-ignored, one directory per benchmarked system),
writes its `.env` and a compose override, starts the stack, and publishes mem0's own ports:
the API on 8888 — which is what `registry/mem0.yaml` points at — and its dashboard on 3000.
With auth enabled instead, add `api_key_env` to the registry entry and export the key.

Note what publishing those ports costs. The strong egress guarantee below applies to a
system whose every network is `internal`, and Docker will not publish a port for such a
container. Since the harness dials mem0 from the host, mem0 keeps ordinary egress and its
model pin is *detected rather than prevented*: a bypass shows up in `verify` as a phase
with zero gateway traffic, which is a weaker claim than one that cannot happen.

The script exists because of the one thing the registry cannot express: **mem0's own LLM
and embedder are configured against the server**, through its `.env` and `POST /configure`,
not through the entry. So the script renders those from `suites/<version>.yaml` — including
the two values that are silently wrong by hand, the `<provider>/<id>` model naming Bifrost
needs and the pgvector column width, which is fixed at `CREATE TABLE` and defaults to
1536 against a suite that pins 4096. See [Model pinning](#the-model-plane) and the comments
in that registry file. Use `--write-only` to inspect the generated files without starting
anything.

### 4. Check conformance before spending money

```bash
uv run bench doctor mem0
```

```
CHECK                    STATUS  DETAIL
adapter_constructs       PASS
searchable_on_return     PASS
top_k_honoured           PASS
content_present          PASS
no_cross_user_leakage    PASS

5/5 checks passed
```

A contract violation fails the run loudly rather than scoring zero — a broken
integration is not a memory result. Fix `doctor` first.

You can also run the conformance suite directly against a registered system:

```bash
uv run pytest contract/conformance --system=mysystem
```

### 5. Fetch the datasets

Corpora are fetched, not vendored — each `dataset.yaml` pins a `url` and a `sha256`, and
a hash mismatch is refused rather than installed.

```bash
make fetch
```

| Dataset | Questions | Shape | Licence |
| --- | --- | --- | --- |
| `locomo-refined` | 1,986 QA pairs over 10 conversations | ~138 questions per context | **CC BY-NC 4.0 — non-commercial** |
| `longmemeval-cleaned` | 500 | one private haystack per question | MIT |

Check the LoCoMo licence before publishing any row derived from it. A dataset may pin its
own `official_prompt` and `official_judge` that override the suite — LoCoMo-Refined does,
because its "refined" scoring definition *is* a specific judge plus a stricter prompt.
The **resolved** judge lands in the manifest and on the report row.

### 6. Smoke run first

`--limit` caps the questions planned. Do this every time: one full LongMemEval-S run
ingests roughly 50M tokens, and it is the dominant cost in the benchmark.

```bash
uv run bench run mysystem --dataset locomo-refined --limit 5 --stub-models
→ 8c1e04b7f2a9
```

The run id is generated and printed — you don't pass one. Every command that takes an id
accepts an unambiguous prefix, so `bench rescore 8c1e` works.

`--stub-models` skips the model provider entirely. Every judgement comes back CORRECT, so
**accuracy is not a measurement** in this mode — only retrieval discriminates. It exists
to exercise the pipeline and your integration, not to produce a number.

### 7. Real run

Point the harness at your model endpoint. Any OpenAI-compatible base URL works; the
metering gateway is described below.

```bash
uv run bench gateway start
uv run bench run mysystem --dataset locomo-refined --name locomo-baseline
```

There is nothing to configure about models. The gateway is the only route to one, so
`bench run` uses it by default; a run works as soon as `bench gateway start` is up. If your
gateway enforces virtual keys, export the key as `ENGRAM_GATEWAY_KEY` — unset is fine for
a local gateway that needs none, and a gateway that does need one answers 401.

`--name` is an optional human label for `bench ps`. It is metadata, not the namespace, so
it need not be unique.

**A run detaches.** `bench run` prints the run id and returns; the run continues in its own
session, so it survives a closed terminal and a Ctrl-C meant to stop watching it. This holds
for a cross product too — see [Parallel runs](#parallel-runs). Follow it with `bench ps`, or
with `bench logs <id> -f`:

```bash
uv run bench run mysystem --dataset locomo-refined   # prints the id, returns at once
uv run bench logs 8c1e -f                            # stream the run's log
uv run bench stop 8c1e                               # end the run itself
```

Pass `--foreground` to stay attached and have the report row print when the run finishes —
useful for a short `--limit` run or in CI. Either way, a bad flag or an unknown system
fails on the command you typed, not silently in a child.

`--suite` selects the pinned world and defaults to the current version, `v3`. Every suite
is immutable once published, so a new pin means a new file rather than an edit: `v2` made
each model a `(provider, id)` pair, and `v3` points all three at one OpenAI-compatible
endpoint. `v1` and `v2` stay loadable so runs published under them remain reproducible.

The report row prints on completion:

```
system          mysystem
dataset         locomo-refined
suite           v3
run_id          8c1e04b7f2a9
scorer          judge_binary
accuracy        0.734 (1458/1986)
judge           zai-org/glm-5.2
recall@k        0.812
recall track    exact / message
harness tokens  2841203
system cost     unavailable
in_process      False
```

Read `recall track` before comparing recall across datasets. LoCoMo's evidence is
message-granular and LongMemEval's is session-granular, so the two recall numbers are
measured against different units and are not one metric. Quality is likewise comparable
*within* a dataset only, since datasets may pin their own judge.

There is deliberately **no composite score**. Five metric families sit side by side in
`report.json`; any weighted index is the reader's to compute from published columns.

### 8. Inspect, rescore, verify

```bash
uv run bench ps --all         # progress table; --watch to redraw, --json to script
uv run bench logs 8c1e        # status plus the tail of run.log; --tail N, -f to follow
uv run bench rescore 8c1e     # recompute report.json, zero system cost
uv run bench verify 8c1e      # replay the integrity checks offline
uv run bench stop 8c1e        # SIGINT; the run drains and stays resumable
uv run bench rm 8c1e          # delete a finished run's artifacts
```

Each takes a full id or any unambiguous prefix. An ambiguous prefix lists the candidates
rather than guessing.

Artifacts are append-only, so an interrupted run can continue where it stopped:

```bash
uv run bench run --resume 8c1e04b7f2a9
```

The id is all it takes. The run's own `manifest.json` records the system, dataset, suite,
and limit, so there is nothing to retype — and passing one anyway is checked rather than
ignored.

Resuming is explicit. It skips the IDs already present in each output file, and refuses if
the target run's manifest names a different system, dataset, suite, or limit — which is
what stops two configurations from being spliced into one published row.

`artifacts/<run_id>/` holds `manifest.json` (written before anything else, with
suite/dataset/prompt hashes), the seven stages' `*.jsonl`, `report.json`, `verify.json`,
`state.json` (the liveness cache `bench ps` reads — losing it costs visibility, never
data), and `run.log`.

`run.log` carries a line per stage boundary and per unit of work: every `add` and `search`
with its attempt count and latency, every model call with the model that served it and its
token counts, and every retry at `WARNING`. A `run models` line up front names the run's
resolved chat, judge, and embedding models. It records ids, statuses, counts, and latencies
only — never a prompt, a message, retrieved content, a completion, or a key.

## Baselines

Four reference systems calibrate the range. `no_memory` and `bm25` are ordinary
`MemoryAdapter` implementations and pass conformance unchanged — they need no registry
entry:

```bash
uv run bench run no_memory --dataset locomo-refined --limit 5 --stub-models
uv run bench run bm25      --dataset locomo-refined --limit 5 --stub-models
```

`bm25` is also the shortest complete example of an adapter that reports `source_ids`, so
it is worth reading before writing your own.

`oracle_gold` and `long_context` are **privileged baselines** that bypass the interface:
they need gold evidence and the whole corpus, which the interface deliberately does not
carry. They emit `retrieve.jsonl` directly and are exempt from conformance — report rows
mark them `privileged: true` so a reader knows the path differed.

There are also deliberately misbehaving fakes for exercising the harness itself — one
leaks across users, one overshoots `top_k`, one returns empty content, one indexes late.
They are how the conformance suite is tested, and they need no socket:

```bash
uv run pytest tests/reference/test_misbehaving.py
```

## Parallel runs

Runs are independent by construction — `user_id` embeds `run_id`, artifacts are per-run
directories, and `score` is pure. So a cross product executes as a pool of subprocesses,
one process per run, with no daemon and no database.

```bash
uv run bench run sys-a,sys-b --dataset locomo-refined,longmemeval-cleaned --parallel 4
```

**A cross product detaches too**, and returns as soon as it has printed one id per run:

```
a3f1c9d20e84  sys-a  locomo-refined
b7e04c1183aa  sys-b  locomo-refined
detached 4 runs. `bench ps` for progress, `bench logs <id>` for one run's log.
```

Every id is printed up front, before any work starts, so each run is stoppable and
resumable by an id you have already seen. What detaches is a single supervisor rather than
each run separately — detaching them individually would release the same-system lock below
as soon as a child launched, letting two runs against one system overlap. Pass
`--foreground` to block instead and get the aggregated table of report rows.

Runs against the **same system** are serialized. That is a fairness rule, not a tuning
knob: concurrent load on one backing store contaminates the latency and rank-stability
metrics. Useful parallelism is bounded by the number of distinct systems, not the size of
the cross product — which is why `--parallel` defaults to the number of distinct systems in
the batch. `--allow-concurrent-same-system` overrides the rule and marks the affected rows
`contended`, withholding their system cost rather than publishing a number that can't be
defended.

One process per run also means each run constructs its **own adapter instance**, so two
runs can never share a mutable client or connection pool.

## Metering gateway (optional)

[Bifrost](https://github.com/maximhq/bifrost) is the only route to any LLM or embedder —
for the harness *and* for the system under test. It pins models per virtual key via
`provider_configs[].allowed_models` (a non-pinned model gets a 403, rather than trusting
anyone to honour the suite's text), meters tokens, and blocks other egress.

### Pointing it at your own models

The gateway config is per-deployment and gitignored, so start from the tracked template:

```bash
cp gateway/bifrost.example.json gateway/bifrost.json
```

It ships with `openai` (suites `v1`/`v2`) and `vllm` (LoCoMo-Refined's `qwen3-14b` judge).
`v3` and `v4` pin every model against a private endpoint, which the template omits — add
your own block for those. Credentials are referenced as `env.NAME` and never inlined; a
custom provider's `network_config.base_url` is a literal host, which is the other reason
your copy stays out of git.

A pinned model is a `(provider, id)` pair, and a provider is a base URL plus a credential.
Nothing assumes OpenAI. Add a block per endpoint in
[`gateway/bifrost.example.json`](gateway/bifrost.example.json):

```jsonc
"vllm": {                                  // first-class provider: url is per-key
  "keys": [{"value": "env.VLLM_API_KEY", "models": ["qwen3-14b"], "weight": 1.0,
            "vllm_key_config": {"url": "env.VLLM_BASE_URL",
                                "model_name": "qwen3-14b"}}]
},
"internal-llm": {                          // anything else OpenAI-compatible
  "keys": [{"value": "env.INTERNAL_API_KEY", "models": ["*"], "weight": 1.0}],
  "network_config": {"base_url": "https://llm.internal.example.com"},
  "custom_provider_config": {"base_provider_type": "openai", "is_key_less": false}
}
```

Then reference the provider by name in the suite:

```yaml
models:
  chat:  {provider: openai, id: gpt-4o-mini, temperature: 0, seed: 7}
  judge: {provider: vllm,   id: qwen3-14b,   temperature: 0}
```

Two gotchas, both verified against Bifrost's source rather than its docs. `vllm` and
`ollama` are first-class providers whose per-key `url` supports `env.*`, so the endpoint
stays out of the file — prefer them. And `network_config.base_url` is a plain string, **not**
a secret reference: `env.FOO` there is used literally and silently goes nowhere.

Changing a provider is a **suite version bump**. Which endpoint served a model is part of
the pinned environment: the same `qwen3-14b` on two deployments can differ in quantization
and sampling defaults, so the provider name travels into `manifest.json` and onto the
report row beside the model id.

```bash
export PROXY_API_KEY=...
uv run bench gateway start
uv run bench run mysystem --dataset locomo-refined
uv run bench verify <id> --metered
```

`bench gateway start` wraps `docker compose up -d gateway` and adds the two things that are
easy to get wrong by hand. It starts the gateway service alone — the compose file also
carries a system-under-test template — and it polls `/metrics` until the gateway answers, so
the next command in a script does not race the container's boot. It also reads
`bifrost.json` for the `env.*` credentials it references and names any that are unset, by
name only: a gateway missing its provider key comes up healthy and 401s every model call,
which otherwise surfaces much later as a failed run. Missing *some* is a warning, since the
config holds one block per provider and a dataset uses one of them; missing all of them is a
refusal. `--no-wait` returns as soon as compose does, and `--skip-env-check` starts a
gateway whose endpoints need no credential.

`bench gateway stop` stops that same service and leaves it in place, so a later `start`
brings the same container back. It removes nothing and touches no other service in the
file, including a system under test you may be benchmarking.

`bench gateway status` asks the same `/metrics` endpoint once and exits 0 when the gateway
answers, 1 when it does not — so `bench gateway status && bench run ...` cannot run against
nothing. It reads no compose file and starts nothing, which means it also works against a
gateway you did not start here: pass `--url` for a hosted one. A container that is up but
answers an error on `/metrics` reads as down, since that is the endpoint per-phase cost is
derived from.

**Enforcement depends on where your system runs, and the report says which.** For a
containerized system the egress policy is a property of the Docker network rather than a
promise in prose: the `models` network is `internal`, so a system attached to it can
resolve the gateway and no provider. Containerized is necessary but not sufficient — the
guarantee holds only while *every* network the container joins is internal, which rules out
publishing a port to the host, so a system the harness reaches on 127.0.0.1 (the shipped
mem0 entry) is metered but not egress-blocked. An **in-process adapter shares our interpreter and
cannot be network-isolated from it** — we pass the gateway through its `config`, but
whether it honours that is its own business. Those rows record metering as `unenforced`
rather than enforced. A published cost comparison should either be all-container or state
the difference; `bench doctor` warns when an in-process adapter isn't pointed at the
gateway, which catches the accident but not the deliberate case.

Two key scopes, because the two sides differ in what they can hold. The harness gets one
key per `(run_id, system, phase)`, so its per-phase cost is **measured**. A system gets one
run-scoped key; its per-phase split is **derived** from `/metrics` counter diffs at the
phase boundaries. The report keeps those labels separate — merging them would overstate
precision on the half that is inferred.

Note that `bench run` does not yet wire the gateway into the run: `execute_run` accepts a
`gateway` argument and `orchestrator/gateway.py` implements provisioning and metering, but
the CLI passes `None`, so runs today record cost attribution as `unavailable` rather than
`derived`. Harness traffic still routes through the gateway, so it is metered on the
gateway's own counters; only the automatic per-phase diffing and virtual-key provisioning
are pending.

## Adding a dataset

A plugin is a directory: `dataset.yaml` + `adapter.py`. Adding one never requires editing
the orchestrator.

```
datasets/<name>/
  dataset.yaml     name, version, source url + sha256, task_type, scorer,
                   evidence_granularity, optional official_prompt / official_judge
  adapter.py       load() → contexts and questions
  prompts/         optional, when the dataset pins its own answer prompt
```

The two shipped datasets have deliberately opposite shapes (138 questions per context vs
one) and different evidence granularity (message vs session), because both are
load-bearing for the plugin API. Code must not assume either.

## Development

```bash
make check     # lint + types + tests
make test      # uv run pytest
make lint      # ruff check + format --check
make types     # mypy, including each dataset adapter individually
make fetch     # populate datasets/*/data/ from the pinned urls
```

The workflow is TDD: no production code without a failing test that demands it. Tests
mirror the package tree under `tests/`, and never make real network calls — memory systems
and model calls are stubbed in-process. Dependencies stay lean (`httpx`, `pydantic`,
`pyyaml`, `pytest`); CLI tables are stdlib string formatting, not `rich`.

### Current state

667 tests passing, with lint and `mypy --strict` clean. Phases 1 through 3.5 are built —
the seven stages, conformance checks, baselines, parallel runs, `bench ps` — and Phase 6
has landed, so everything above describes the code rather than a target: the adapter
interface, the simplified retrieval records, and generated run ids are all in place.

Phase 6's acceptance criterion was that a pre-refactor artifact bundle rescores to a
byte-identical `report.json`, proving the change touched integration rather than
measurement. It holds for all five bundles captured before the refactor. `score` reads
both the old `data` key and the new `memories` key, so an archived bundle stays
rescorable; a fresh run reproduces the same quality and recall, with lower latency for
the round-trip that is no longer there.

Two known gaps remain, neither part of that refactor: `bench run` does not yet wire the
gateway into a run (`orchestrator/gateway.py` implements provisioning and metering, but
the CLI passes `None`, so cost attribution records `unavailable`), and
`bench report --compare` is described in the architecture but not implemented.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design: adapter interface,
  stages, artifact shapes, metrics, verification, dataset plugin API. The *what* and *why*.
- [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) — TDD build order, per-step test
  cases, and resolved design decisions.
- [`CLAUDE.md`](CLAUDE.md) — working conventions for this repo.

## Layout

```
orchestrator/     stages, retry policy, artifact IO      ← no method knowledge
contract/         MemoryAdapter protocol, Message/Memory, conformance/
adapters/         one module per system: mem0, http, ... ← per-system knowledge
suites/           v1.yaml, v2.yaml — the pinned world, one file per version
datasets/<name>/  dataset.yaml + adapter.py — plugins
scorers/          judge_binary, mcq, exact_match_f1, retrieval metrics
registry/         one YAML per system under test
reference/        baselines and fakes
gateway/          bifrost config, virtual keys, egress policy
artifacts/<run>/  manifest.json + *.jsonl + report.json + verify.json + state.json
```

Note `datasets/<name>/adapter.py` is a *dataset* adapter — a `load()` function — and has
nothing to do with `MemoryAdapter`. The collision is unfortunate and predates the refactor.
