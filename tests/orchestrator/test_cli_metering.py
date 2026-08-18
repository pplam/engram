"""`bench run` meters against the running gateway, or says it could not (§3.1, D6).

The gateway is a constant, not a flag: there is no `--metered` to pass, so the CLI builds
a client to it and provisions this run's keys. Two things are load-bearing here.

A gateway that cannot be reached must not fail the run. Baselines legitimately need no
gateway, and a gateway with governance disabled has no keys to hand out; both are working
configurations whose cost column reads `unavailable`. Refusing to start would turn a
supported setup into an error.

And a metered run must actually present its keys. `verify` withholds any row whose answer
and judge stages produced records while the gateway metered no calls, so a run that wires
the gateway without keying the harness would publish nothing at all.

No socket: the gateway client's transport is a local `httpx.MockTransport`, patched through
the same kind of seam `model_transport` uses for model calls.
"""

import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest

from orchestrator.artifacts import read_json
from orchestrator.cli import main
from orchestrator.gateway import Gateway

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


def run_argv(datasets: Path, artifacts: Path, system: str = "bm25") -> list[str]:
    return [
        "run",
        system,
        "--foreground",
        "--dataset",
        "fixture_many",
        "--suite",
        "v1",
        "--artifacts",
        str(artifacts),
        "--datasets",
        str(datasets),
    ]


def sole_run(artifacts: Path) -> Path:
    return next(p for p in artifacts.iterdir() if p.is_dir())


def metering_client(counts: dict[str, int]) -> Gateway:
    """A gateway whose counters advance per scrape, keyed so every vk shows traffic."""

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
                f'bifrost_output_tokens_total{{virtual_key_id="vk_{n}"}} {4 * seen}\n'
                f'bifrost_upstream_requests_total{{virtual_key_id="vk_{n}"}} {seen}\n'
                for n in (1, 2, 3)
            ),
        )

    return Gateway(
        httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://gw.invalid")
    )


@pytest.fixture
def metered(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Point the CLI's gateway seam at a local transport and report what it saw."""
    counts: dict[str, int] = {}
    monkeypatch.setattr("orchestrator.cli.gateway_client", lambda: metering_client(counts))
    return counts


def test_a_metered_run_provisions_this_runs_keys(
    datasets: Path, tmp_path: Path, metered: Any
) -> None:
    """Three keys: one per harness phase, plus the system's run-scoped one (D3, D6(a))."""
    assert main(run_argv(datasets, tmp_path / "artifacts")) == 0
    assert metered["keys"] == 3


def test_a_metered_run_is_not_withheld(datasets: Path, tmp_path: Path, metered: Any) -> None:
    """The harness presents its per-phase keys, so metering reconciles against records."""
    artifacts = tmp_path / "artifacts"
    assert main(run_argv(datasets, artifacts)) == 0
    verify = read_json(sole_run(artifacts) / "verify.json")
    finding = next(f for f in verify["findings"] if f["name"] == "unmetered_model_use")
    assert finding["passed"] is True, finding["detail"]
    assert read_json(sole_run(artifacts) / "report.json")["row"]["row_withheld"] is False


def test_a_metered_run_publishes_derived_system_cost(
    datasets: Path, tmp_path: Path, metered: Any
) -> None:
    """The number this whole mechanism exists to produce (§7.3)."""
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts))
    cost = read_json(sole_run(artifacts) / "report.json")["row"]["cost"]
    assert cost["system_attribution"] == "derived"
    assert cost["system_total_tokens"] is not None


def test_an_unreachable_gateway_still_completes_the_run(
    datasets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A baseline needs no gateway, and governance may be off; neither is an error."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "orchestrator.cli.gateway_client",
        lambda: Gateway(
            httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://gw.invalid")
        ),
    )
    artifacts = tmp_path / "artifacts"
    assert main(run_argv(datasets, artifacts)) == 0
    assert read_json(sole_run(artifacts) / "report.json")["row"]["row_withheld"] is False


def test_an_unreachable_gateway_reports_cost_as_unavailable(
    datasets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unmeasured is not zero: the column says the number was never obtainable."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "orchestrator.cli.gateway_client",
        lambda: Gateway(
            httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://gw.invalid")
        ),
    )
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts))
    cost = read_json(sole_run(artifacts) / "report.json")["row"]["cost"]
    assert cost["system_attribution"] == "unavailable"
    assert cost["system_total_tokens"] is None


def test_an_unreachable_gateway_warns_rather_than_going_quiet(
    datasets: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A silently unmetered run looks identical to a metered one until someone reads cost."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "orchestrator.cli.gateway_client",
        lambda: Gateway(
            httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://gw.invalid")
        ),
    )
    main(run_argv(datasets, tmp_path / "artifacts"))
    assert "unmetered" in capsys.readouterr().err.lower()
