"""Tests for leaderboard generation functions."""

import pandas as pd
import pytest

from backend.leaderboard import (
    EntryComparison,
    EntryMetric,
    FinalLeaderboard,
    LeaderboardEntry,
    generate_dynamic_alphabet,
)


def test_generate_dynamic_alphabet():
    """Test the generate_dynamic_alphabet function."""
    size = 200
    alphabet = generate_dynamic_alphabet(size_needed=size)
    assert isinstance(alphabet, list)
    assert len(alphabet) >= size
    assert alphabet[0] == "a"
    assert alphabet[25] == "z"
    assert alphabet[26] == "aa"
    # Verify order: a, b... z, aa, ab...
    sorted_alphabet = sorted(alphabet, key=lambda x: (len(x), x))
    assert alphabet == sorted_alphabet


def test_EntryMetric():
    """Test the EntryMetric dataclass."""
    metric = EntryMetric(name="Test Metric", mean=0.95, std=0.01)
    assert metric.name == "Test Metric"
    assert metric.mean == 0.95
    assert metric.std == 0.01


def test_LeaderboardEntry_structure(example_averaged_results):
    """Check initialization and metric parsing."""
    entry = LeaderboardEntry(
        username="user1",
        anonymous=False,
        user_alias="Alias1",
        submitted_at=pd.Timestamp("2024-01-01"),
        model_report_link="http://report.com",
        used_proprietary_data=False,
        averaged_results=example_averaged_results,
    )

    assert entry.username == "user1"
    assert entry.bootstrap_data is None
    # Check metrics list (accuracy and f1 should be there, coverage should not)
    metric_names = [m.name for m in entry.metrics]
    assert "accuracy" in metric_names
    assert "f1" in metric_names
    assert "coverage" not in metric_names

    accuracy_metric = next(m for m in entry.metrics if m.name == "accuracy")
    assert accuracy_metric.mean == 0.90
    assert accuracy_metric.std == 0.02
    # open_source_code defaults to False when not passed.
    assert entry.open_source_code is False


def test_FinalLeaderboard_includes_open_source_code(example_bootstrap_data):
    """Check that open_source_code is surfaced as a leaderboard_df column."""
    averaged_results = pd.DataFrame({"accuracy_mean": [0.9], "accuracy_std": [0.01]})
    entry = LeaderboardEntry(
        username="user1",
        anonymous=False,
        user_alias="u1",
        submitted_at=pd.Timestamp("2024-01-01"),
        model_report_link="",
        used_proprietary_data=False,
        averaged_results=averaged_results,
        bootstrap_data=example_bootstrap_data,
        open_source_code=True,
    )
    leaderboard = FinalLeaderboard(entries=[entry], primary_metric="accuracy")
    assert bool(leaderboard.leaderboard_df.iloc[0]["open_source_code"]) is True


def test_LeaderboardEntry_with_bootstrap_data(
    example_averaged_results, example_bootstrap_data
):
    """Check that bootstrap data is stored correctly."""
    entry = LeaderboardEntry(
        username="user1",
        anonymous=False,
        user_alias="Alias1",
        submitted_at=pd.Timestamp("2024-01-01"),
        model_report_link="n/a",
        used_proprietary_data=True,
        averaged_results=example_averaged_results,
        bootstrap_data=example_bootstrap_data,
    )
    assert entry.bootstrap_data is not None
    assert len(entry.bootstrap_data) == 100
    assert "accuracy" in entry.bootstrap_data.columns


def test_EntryComparison(example_averaged_results):
    """Verify statistical calculations and significance logic."""
    # Create two entries where Entry A is consistently better than Entry B
    n = 100
    boot_a = pd.DataFrame({"Sample": range(n), "accuracy": [0.95] * n})
    boot_b = pd.DataFrame({"Sample": range(n), "accuracy": [0.80] * n})

    entry_a = LeaderboardEntry(
        "a", False, "a", pd.Timestamp.now(), "", False, example_averaged_results, boot_a
    )
    entry_b = LeaderboardEntry(
        "b", False, "b", pd.Timestamp.now(), "", False, example_averaged_results, boot_b
    )

    # abs_mean_diff calculation based on example_averaged_results (0.90 - 0.90 = 0.0)
    # Note: In real use, averaged_results would match bootstrap means
    comparison = EntryComparison(entry_a, entry_b, "accuracy", abs_mean_diff=0.0)

    # A is always > B, so r = 0 and p = 2 * (0 + 1) / (n + 1): the smallest
    # attainable p-value, never exactly 0.
    assert comparison.p_value == pytest.approx(2 / (n + 1))
    assert len(comparison.paired_bootstrap_data) == n

    # Test adjusted threshold (default correction: Benjamini-Hochberg)
    # alpha=0.05, total=1, rank=1 => 0.05 * 1 / 1 = 0.05
    comparison.determine_adjusted_threshold(total_comparisons=1, p_rank=1)
    assert comparison.adjusted_threshold == 0.05

    # Test Significance
    assert comparison.determine_significance() is True


def test_EntryComparison_tie():
    """Check that identical data results in p=1.0."""
    boot_data = pd.DataFrame({"Sample": range(10), "accuracy": [0.9] * 10})
    avg_data = pd.DataFrame({"accuracy_mean": [0.9], "accuracy_std": [0.0]})

    entry_a = LeaderboardEntry(
        "a", False, "a", pd.Timestamp.now(), "", False, avg_data, boot_data
    )
    entry_b = LeaderboardEntry(
        "b", False, "b", pd.Timestamp.now(), "", False, avg_data, boot_data
    )

    comparison = EntryComparison(entry_a, entry_b, "accuracy", abs_mean_diff=0.0)
    assert comparison.p_value == 1.0


def _comparison_from_diffs(diffs: list[float]) -> EntryComparison:
    """An EntryComparison whose paired bootstrap differences are ``diffs``."""
    n = len(diffs)
    boot_b = pd.DataFrame({"Sample": range(n), "accuracy": [0.5] * n})
    boot_a = boot_b.assign(accuracy=[0.5 + d for d in diffs])
    avg = pd.DataFrame({"accuracy_mean": [0.5], "accuracy_std": [0.0]})
    entry_a = LeaderboardEntry(
        "a", False, "a", pd.Timestamp.now(), "", False, avg, boot_a
    )
    entry_b = LeaderboardEntry(
        "b", False, "b", pd.Timestamp.now(), "", False, avg, boot_b
    )
    return EntryComparison(entry_a, entry_b, "accuracy")


@pytest.mark.parametrize(
    ("n_pos", "n_neg", "n_zero", "expected"),
    [
        # r = min(#>=0, #<=0); p = 2 * (r + 1) / (B + 1), capped at 1.
        (999, 0, 0, 2 * 1 / 1000),  # one side wins every sample: the floor, not 0
        (0, 999, 0, 2 * 1 / 1000),  # symmetric
        (950, 49, 0, 2 * 50 / 1000),
        (500, 499, 0, 1.0),  # 2 * 500 / 1000 = 1.0
        (600, 0, 399, 2 * 400 / 1000),  # zero differences count towards both sides
        (998, 0, 1, 2 * 2 / 1000),  # a single tie lifts the floor
    ],
)
def test_EntryComparison_p_value_plus_one_correction(n_pos, n_neg, n_zero, expected):
    """#11: the (r + 1) / (B + 1) form, so a p-value is never exactly 0."""
    diffs = [0.1] * n_pos + [-0.1] * n_neg + [0.0] * n_zero
    comparison = _comparison_from_diffs(diffs)
    assert comparison.p_value == pytest.approx(expected)
    assert 0 < comparison.p_value <= 1


def test_FinalLeaderboard(example_bootstrap_data):
    """Check sorting, ranking, and CLD generation integration."""
    # Entry 1: Mean 0.9
    entry1 = LeaderboardEntry(
        "user1",
        False,
        "u1",
        pd.Timestamp("2024-01-01"),
        "",
        False,
        pd.DataFrame({"accuracy_mean": [0.9], "accuracy_std": [0.01]}),
        example_bootstrap_data,
    )

    # Entry 2: Mean 0.95 (Better)
    entry2 = LeaderboardEntry(
        "user2",
        False,
        "u2",
        pd.Timestamp("2024-01-01"),
        "",
        False,
        pd.DataFrame({"accuracy_mean": [0.95], "accuracy_std": [0.01]}),
        example_bootstrap_data,
    )

    # Test sorting (descending: higher accuracy is better)
    lb = FinalLeaderboard(
        entries=[entry1, entry2],
        primary_metric="accuracy",
        metric_sort_ascending=False,
        significant_method="CLD",
    )

    # user2 should be rank 1
    assert lb.leaderboard_df.iloc[0]["username"] == "user2"
    assert lb.leaderboard_df["rank"].iloc[0] == 1  # Check rank index

    # Check that CLD column was inserted
    assert "Significance (CLD)" in lb.leaderboard_df.columns
    assert isinstance(lb.leaderboard_df.iloc[0]["Significance (CLD)"], str)


def test_FinalLeaderboard_includes_additional_columns(example_bootstrap_data):
    """Check that requested extra averaged-results columns are included."""
    averaged_results = pd.DataFrame(
        {
            "accuracy_mean": [0.91],
            "accuracy_std": [0.02],
            "coverage_mean": [0.87],
        }
    )
    entry = LeaderboardEntry(
        username="user1",
        anonymous=False,
        user_alias="u1",
        submitted_at=pd.Timestamp("2024-01-01"),
        model_report_link="",
        used_proprietary_data=False,
        averaged_results=averaged_results,
        bootstrap_data=example_bootstrap_data,
    )

    leaderboard = FinalLeaderboard(
        entries=[entry],
        primary_metric="accuracy",
        additional_columns=["coverage_mean"],
    )

    assert "coverage_mean" in leaderboard.leaderboard_df.columns
    assert leaderboard.leaderboard_df.iloc[0]["coverage_mean"] == pytest.approx(0.87)


def test_FinalLeaderboard_missing_additional_column_is_nan(example_bootstrap_data):
    """An entry missing an extra column gets NaN for it; the other entries and the
    rest of the leaderboard are unaffected (#26)."""

    def _entry(username: str, averaged_results: pd.DataFrame) -> LeaderboardEntry:
        return LeaderboardEntry(
            username=username,
            anonymous=False,
            user_alias="",
            submitted_at=pd.Timestamp("2024-01-01"),
            model_report_link="",
            used_proprietary_data=False,
            averaged_results=averaged_results,
            bootstrap_data=example_bootstrap_data,
        )

    new_entry = _entry(
        "new",
        pd.DataFrame(
            {"accuracy_mean": [0.91], "accuracy_std": [0.02], "coverage_mean": [0.87]}
        ),
    )
    old_entry = _entry(
        "old", pd.DataFrame({"accuracy_mean": [0.80], "accuracy_std": [0.02]})
    )

    leaderboard = FinalLeaderboard(
        entries=[new_entry, old_entry],
        primary_metric="accuracy",
        significant_method=None,
        additional_columns=["coverage_mean"],
    )

    df = leaderboard.leaderboard_df.set_index("username")
    assert len(df) == 2
    assert df.loc["new", "coverage_mean"] == pytest.approx(0.87)
    assert pd.isna(df.loc["old", "coverage_mean"])
