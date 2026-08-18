"""`bench run` drives all six stages and produces report.json (Phase 2 exit)."""

import shutil
from pathlib import Path

import httpx
import pytest

from contract.adapter import MemoryAdapter
from orchestrator.artifacts import JsonlArtifact, read_json
from orchestrator.gateway import Gateway
from orchestrator.models import StubChat
from orchestrator.run import RunError, RunRequest, execute_run
from orchestrator.runstate import read_state
from reference.baselines import build_baseline

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


def request_for(datasets: Path, tmp_path: Path, system: str = "bm25", **kw: object) -> RunRequest:
    return RunRequest(
        run_id="r1",
        system=system,
        dataset="fixture_many",
        suite="v1",
        artifacts_root=tmp_path / "artifacts",
        datasets_root=datasets,
        repo_root=REPO,
        **kw,  # type: ignore[arg-type]
    )


def adapter_for(system: str) -> MemoryAdapter:
    """The baseline as an ordinary adapter — the same path a registered system takes."""
    return build_baseline(system)


def chat_for() -> StubChat:
    """Answers, then judges — the stub cycles, so give it both replies."""
    return StubChat(replies=["Mochi"], prompt_tokens=20, completion_tokens=5)


async def test_produces_a_report(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    report = await execute_run(
        request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub()
    )
    assert (request.run_dir / "report.json").is_file()
    assert report.row.system == "bm25"


async def test_writes_every_stage_artifact(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    await execute_run(request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub())
    for name in (
        "manifest.json",
        "plan.jsonl",
        "ingest.jsonl",
        "retrieve.jsonl",
        "answer.jsonl",
        "judge.jsonl",
        "report.json",
    ):
        assert (request.run_dir / name).is_file(), name


async def test_bm25_scores_above_no_memory_on_a_retrievable_question(
    datasets: Path, tmp_path: Path
) -> None:
    """The floor must be below a lexical baseline, or the pipeline is not measuring."""
    lexical = request_for(datasets, tmp_path)
    lexical_report = await execute_run(
        lexical, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub()
    )
    floor = RunRequest(
        run_id="r2",
        system="no_memory",
        dataset="fixture_many",
        suite="v1",
        artifacts_root=tmp_path / "artifacts",
        datasets_root=datasets,
        repo_root=REPO,
    )
    floor_report = await execute_run(
        floor, adapter_for("no_memory"), chat=chat_for(), judge_chat=judge_stub()
    )
    lexical_recall = lexical_report.row.retrieval.recall_at_k
    assert lexical_recall is not None
    assert lexical_recall > 0.0, "a lexical baseline that retrieves nothing is not a floor test"
    assert floor_report.row.retrieval.recall_at_k in (None, 0.0)


async def test_limit_reaches_every_downstream_stage(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path, limit=2)
    report = await execute_run(
        request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub()
    )
    assert report.row.completeness.planned_questions == 2
    assert report.row.completeness.judged_questions == 2


async def test_manifest_records_the_run(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    await execute_run(request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub())
    manifest = read_json(request.run_dir / "manifest.json")
    assert manifest["run_id"] == "r1"
    assert manifest["system"]["name"] == "bm25"


async def test_a_second_run_resumes_without_redoing_work(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    await execute_run(request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub())
    before = (request.run_dir / "ingest.jsonl").read_bytes()

    second_chat = chat_for()
    await execute_run(request, adapter_for("bm25"), chat=second_chat, judge_chat=judge_stub())
    assert (request.run_dir / "ingest.jsonl").read_bytes() == before
    assert second_chat.calls == []


async def test_run_dir_is_named_by_run_id(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    assert request.run_dir == tmp_path / "artifacts" / "r1"


async def test_report_notes_warn_against_cross_dataset_comparison(
    datasets: Path, tmp_path: Path
) -> None:
    request = request_for(datasets, tmp_path)
    report = await execute_run(
        request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub()
    )
    assert any("comparable within" in note for note in report.notes)


async def test_in_process_baseline_skips_the_contract_stages(
    datasets: Path, tmp_path: Path
) -> None:
    request = request_for(datasets, tmp_path, system="oracle_gold")
    report = await execute_run(request, None, chat=chat_for(), judge_chat=judge_stub())
    assert report.row.in_process is True
    assert not (request.run_dir / "ingest.jsonl").exists()
    assert (request.run_dir / "retrieve.jsonl").is_file()


async def test_oracle_gold_retrieval_is_the_ceiling(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path, system="oracle_gold")
    report = await execute_run(request, None, chat=chat_for(), judge_chat=judge_stub())
    assert report.row.retrieval.recall_at_k == 1.0


def judge_stub() -> StubChat:
    return StubChat(replies=['{"label": "CORRECT"}'], prompt_tokens=10, completion_tokens=2)


async def test_a_run_writes_verify_json(datasets: Path, tmp_path: Path) -> None:
    """`verify` is the seventh stage and is published beside the report (ARCH §8)."""
    request = request_for(datasets, tmp_path)
    await execute_run(request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub())
    assert (request.run_dir / "verify.json").is_file()


async def test_the_report_carries_the_verification_outcome(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    report = await execute_run(
        request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub()
    )
    assert report.row.row_withheld is False


async def test_a_run_without_a_gateway_derives_no_system_cost(
    datasets: Path, tmp_path: Path
) -> None:
    """Without metering there is nothing to derive from, and that is stated, not guessed."""
    request = request_for(datasets, tmp_path)
    report = await execute_run(
        request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub()
    )
    assert report.row.cost.system_attribution == "unavailable"


async def test_ingest_and_retrieve_never_overlap(datasets: Path, tmp_path: Path) -> None:
    """The D6(a) invariant, asserted rather than trusted (§3.3)."""
    request = request_for(datasets, tmp_path)
    await execute_run(request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub())
    ingest = list(JsonlArtifact(request.run_dir / "ingest.jsonl").stream())
    retrieve = list(JsonlArtifact(request.run_dir / "retrieve.jsonl").stream())
    assert max(int(r["ended_at_ms"]) for r in ingest) <= min(
        int(r["started_at_ms"]) for r in retrieve
    )


def metering_gateway(counts: dict[str, int]) -> Gateway:
    """A gateway whose counters advance on every scrape, as real traffic would."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/governance/virtual-keys":
            counts["keys"] = counts.get("keys", 0) + 1
            return httpx.Response(
                200,
                json={
                    "message": "Virtual key created successfully",
                    "virtual_key": {"id": f"vk_{counts['keys']}", "value": "sk-bf-x"},
                },
            )
        counts["scrapes"] = counts.get("scrapes", 0) + 1
        seen = counts["scrapes"]
        return httpx.Response(
            200,
            text="".join(
                f'bifrost_input_tokens_total{{virtual_key_id="vk_{n}"}} {10 * seen}\n'
                f'bifrost_upstream_requests_total{{virtual_key_id="vk_{n}"}} {seen}\n'
                for n in (1, 2)
            ),
        )

    return Gateway(
        httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://gw.invalid")
    )


async def test_a_metered_run_reconciles_and_is_not_withheld(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    report = await execute_run(
        request,
        adapter_for("bm25"),
        chat=chat_for(),
        judge_chat=judge_stub(),
        gateway=metering_gateway({}),
    )
    assert report.row.row_withheld is False


async def test_a_metered_run_pins_every_model_it_may_call(datasets: Path, tmp_path: Path) -> None:
    """The chat, embedding, and resolved judge models are the only ones allowed."""
    counts: dict[str, int] = {}
    request = request_for(datasets, tmp_path)
    await execute_run(
        request,
        adapter_for("bm25"),
        chat=chat_for(),
        judge_chat=judge_stub(),
        gateway=metering_gateway(counts),
    )
    assert counts["keys"] == 3  # answer, judge, and the system's run-scoped key


async def test_verify_records_derived_cost_when_metered(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    await execute_run(
        request,
        adapter_for("bm25"),
        chat=chat_for(),
        judge_chat=judge_stub(),
        gateway=metering_gateway({}),
    )
    assert read_json(request.run_dir / "verify.json")["cost_attribution"] == "derived"


async def test_a_metered_run_publishes_derived_system_cost(datasets: Path, tmp_path: Path) -> None:
    """The system's own traffic is metered off its run-scoped key across both phases."""
    request = request_for(datasets, tmp_path)
    report = await execute_run(
        request,
        adapter_for("bm25"),
        chat=chat_for(),
        judge_chat=judge_stub(),
        gateway=metering_gateway({}),
    )
    assert report.row.cost.system_attribution == "derived"
    assert report.row.cost.system_total_tokens is not None


class KeyedChat(StubChat):
    """A StubChat that records the virtual keys it was handed, and when."""

    def __init__(self) -> None:
        super().__init__(replies=["Mochi"], prompt_tokens=20, completion_tokens=5)
        self.keys: list[tuple[str, str, str]] = []

    async def use_key(self, value: str, run_id: str, phase: str) -> None:
        self.keys.append((value, run_id, phase))


class KeyedAdapter:
    """Wraps a baseline, recording the run key and the order it arrived in."""

    def __init__(self, inner: MemoryAdapter) -> None:
        self._inner = inner
        self.events: list[str] = []

    async def use_model_key(self, value: str) -> None:
        self.events.append(f"key:{value}")

    async def add(self, user_id: str, messages: object, chunk_id: str) -> None:
        self.events.append("add")
        await self._inner.add(user_id, messages, chunk_id)  # type: ignore[arg-type]

    async def search(self, user_id: str, query: str, top_k: int) -> object:
        return await self._inner.search(user_id, query, top_k)


async def test_each_harness_phase_is_handed_its_own_key(datasets: Path, tmp_path: Path) -> None:
    """Per-phase keys are what make harness attribution exact rather than derived (D3)."""
    chat, judge = KeyedChat(), KeyedChat()
    await execute_run(
        request_for(datasets, tmp_path),
        adapter_for("bm25"),
        chat=chat,
        judge_chat=judge,
        gateway=metering_gateway({}),
    )
    assert [phase for _, _, phase in chat.keys] == ["answer"]
    assert [phase for _, _, phase in judge.keys] == ["judge"]
    assert chat.keys[0][1] == "r1"


async def test_an_unmetered_run_hands_out_no_keys(datasets: Path, tmp_path: Path) -> None:
    """No gateway means no keys to present; the run still completes."""
    chat, judge = KeyedChat(), KeyedChat()
    await execute_run(
        request_for(datasets, tmp_path), adapter_for("bm25"), chat=chat, judge_chat=judge
    )
    assert chat.keys == [] and judge.keys == []


async def test_the_system_gets_its_run_key_before_the_first_write(
    datasets: Path, tmp_path: Path
) -> None:
    """A key handed over after ingest began would leave those writes unattributable."""
    adapter = KeyedAdapter(adapter_for("bm25"))
    await execute_run(
        request_for(datasets, tmp_path),
        adapter,  # type: ignore[arg-type]
        chat=chat_for(),
        judge_chat=judge_stub(),
        gateway=metering_gateway({}),
    )
    assert adapter.events[0].startswith("key:"), adapter.events[:3]
    assert "add" in adapter.events


async def test_an_unmetered_run_hands_the_system_no_key(datasets: Path, tmp_path: Path) -> None:
    """Without a gateway there is no key, and the system keeps whatever it was given."""
    adapter = KeyedAdapter(adapter_for("bm25"))
    await execute_run(
        request_for(datasets, tmp_path),
        adapter,  # type: ignore[arg-type]
        chat=chat_for(),
        judge_chat=judge_stub(),
    )
    assert not any(event.startswith("key:") for event in adapter.events)


async def test_harness_and_system_cost_stay_separate_columns(
    datasets: Path, tmp_path: Path
) -> None:
    """§7.3: measured and derived must never be collapsed into one number."""
    request = request_for(datasets, tmp_path)
    report = await execute_run(
        request,
        adapter_for("bm25"),
        chat=chat_for(),
        judge_chat=judge_stub(),
        gateway=metering_gateway({}),
    )
    assert report.row.cost.harness_attribution == "measured"
    assert report.row.cost.system_attribution == "derived"


async def test_a_run_records_its_state_for_bench_ps(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    await execute_run(request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub())
    state = read_state(request.run_dir)
    assert state is not None
    assert state.status == "done"
    assert state.system == "bm25"


async def test_a_failed_run_records_the_failure(datasets: Path, tmp_path: Path) -> None:
    """`bench ps` must distinguish a crash from a run still in progress."""
    request = request_for(datasets, tmp_path, system="oracle_gold")
    with pytest.raises(RunError):
        # An in-process baseline takes no HTTP client; passing one is a usage error.
        await execute_run(request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub())
    state = read_state(request.run_dir)
    assert state is not None
    assert state.status == "failed"
    assert state.error


async def test_state_names_the_stage_that_finished_last(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    await execute_run(request, adapter_for("bm25"), chat=chat_for(), judge_chat=judge_stub())
    state = read_state(request.run_dir)
    assert state is not None
    assert state.stage == "score"


async def test_a_metered_contended_run_withholds_system_cost(
    datasets: Path, tmp_path: Path
) -> None:
    """With a gateway, the override's broken boundary is `unreliable`, not a number."""
    request = request_for(datasets, tmp_path, contended=True)
    report = await execute_run(
        request,
        adapter_for("bm25"),
        chat=chat_for(),
        judge_chat=judge_stub(),
        gateway=metering_gateway({}),
    )
    assert report.row.cost.system_attribution == "unreliable"
    assert report.row.cost.system_total_tokens is None
