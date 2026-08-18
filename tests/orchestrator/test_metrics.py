"""Per-phase cost is a diff of Prometheus counters keyed by virtual key (§3.1, D6).

There is no endpoint returning per-key token counts as data, so metering readback is
a `/metrics` scrape snapshotted at each phase boundary. The counters are cumulative;
only the delta between two snapshots belongs to a phase.
"""

import pytest

from orchestrator.metrics import MetricsError, Usage, diff, parse_metrics

SCRAPE = """
# HELP bifrost_input_tokens_total Input tokens
# TYPE bifrost_input_tokens_total counter
bifrost_input_tokens_total{virtual_key_id="vk_a",model="gpt-4o-mini"} 100
bifrost_output_tokens_total{virtual_key_id="vk_a",model="gpt-4o-mini"} 20
bifrost_upstream_requests_total{virtual_key_id="vk_a",model="gpt-4o-mini"} 4
bifrost_input_tokens_total{virtual_key_id="vk_b",model="gpt-4o-mini"} 7
"""


def test_a_scrape_is_parsed_per_virtual_key() -> None:
    usage = parse_metrics(SCRAPE)
    assert usage["vk_a"] == Usage(prompt_tokens=100, completion_tokens=20, requests=4)


def test_keys_are_kept_apart() -> None:
    assert parse_metrics(SCRAPE)["vk_b"].prompt_tokens == 7


def test_a_missing_counter_reads_as_zero() -> None:
    """A key that made no completion calls has no output-token series at all."""
    assert parse_metrics(SCRAPE)["vk_b"].completion_tokens == 0


def test_series_without_a_virtual_key_label_are_ignored() -> None:
    """Bifrost's own process metrics carry no key and are not anyone's cost."""
    scrape = 'go_goroutines 12\nbifrost_input_tokens_total{model="x"} 5\n'
    assert parse_metrics(scrape) == {}


def test_comments_and_blank_lines_are_skipped() -> None:
    assert parse_metrics("# HELP x y\n\n") == {}


def test_the_same_key_is_summed_across_label_sets() -> None:
    """One key may span several models; the phase cost is their total."""
    scrape = (
        'bifrost_input_tokens_total{virtual_key_id="vk_a",model="m1"} 10\n'
        'bifrost_input_tokens_total{virtual_key_id="vk_a",model="m2"} 5\n'
    )
    assert parse_metrics(scrape)["vk_a"].prompt_tokens == 15


def test_float_counter_values_are_accepted() -> None:
    """Prometheus counters are floats on the wire even when they count whole tokens."""
    scrape = 'bifrost_input_tokens_total{virtual_key_id="vk_a"} 100.0\n'
    assert parse_metrics(scrape)["vk_a"].prompt_tokens == 100


def test_a_malformed_value_is_reported() -> None:
    scrape = 'bifrost_input_tokens_total{virtual_key_id="vk_a"} not_a_number\n'
    with pytest.raises(MetricsError, match="bifrost_input_tokens_total"):
        parse_metrics(scrape)


def test_a_phase_costs_the_delta_between_two_snapshots() -> None:
    before = {"vk_a": Usage(prompt_tokens=100, completion_tokens=20, requests=4)}
    after = {"vk_a": Usage(prompt_tokens=180, completion_tokens=35, requests=7)}
    assert diff(before, after)["vk_a"] == Usage(prompt_tokens=80, completion_tokens=15, requests=3)


def test_a_key_absent_before_the_phase_counts_from_zero() -> None:
    after = {"vk_a": Usage(prompt_tokens=9, completion_tokens=1, requests=1)}
    assert diff({}, after)["vk_a"].prompt_tokens == 9


def test_a_key_that_did_nothing_in_the_phase_is_omitted() -> None:
    """An all-zero delta is noise on a report, not a measurement."""
    same = {"vk_a": Usage(prompt_tokens=5, completion_tokens=1, requests=1)}
    assert diff(same, same) == {}


def test_a_counter_that_went_backwards_is_reported() -> None:
    """Counters only decrease if the gateway restarted, which voids attribution."""
    before = {"vk_a": Usage(prompt_tokens=100, completion_tokens=0, requests=1)}
    after = {"vk_a": Usage(prompt_tokens=40, completion_tokens=0, requests=1)}
    with pytest.raises(MetricsError, match="decreased"):
        diff(before, after)


def test_usage_totals_are_available_for_a_report_row() -> None:
    assert Usage(prompt_tokens=3, completion_tokens=4, requests=1).total_tokens == 7
