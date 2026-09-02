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


def test_LeaderboardEntry_structure(mock_averaged_results):
    """Check initialization and metric parsing."""
    entry = LeaderboardEntry(
        username="user1",
        anonymous=False,
        user_alias="Alias1",
        submitted_at=pd.Timestamp("2024-01-01"),
        model_report_link="http://report.com",
        used_proprietary_data=False,
        averaged_results=mock_averaged_results,
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


def test_FinalLeaderboard_includes_open_source_code(mock_bootstrap_data):
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
        bootstrap_data=mock_bootstrap_data,
        open_source_code=True,
    )
    leaderboard = FinalLeaderboard(entries=[entry], primary_metric="accuracy")
    assert bool(leaderboard.leaderboard_df.iloc[0]["open_source_code"]) is True


def test_LeaderboardEntry_with_bootstrap_data(
    mock_averaged_results, mock_bootstrap_data
):
    """Check that bootstrap data is stored correctly."""
    entry = LeaderboardEntry(
        username="user1",
        anonymous=False,
        user_alias="Alias1",
        submitted_at=pd.Timestamp("2024-01-01"),
        model_report_link="n/a",
        used_proprietary_data=True,
        averaged_results=mock_averaged_results,
        bootstrap_data=mock_bootstrap_data,
    )
    assert entry.bootstrap_data is not None
    assert len(entry.bootstrap_data) == 100
    assert "accuracy" in entry.bootstrap_data.columns


def test_EntryComparison(mock_averaged_results):
    """Verify statistical calculations and significance logic."""
    # Create two entries where Entry A is consistently better than Entry B
    boot_a = pd.DataFrame({"Sample": range(10), "accuracy": [0.95] * 10})
    boot_b = pd.DataFrame({"Sample": range(10), "accuracy": [0.80] * 10})

    entry_a = LeaderboardEntry(
        "a", False, "a", pd.Timestamp.now(), "", False, mock_averaged_results, boot_a
    )
    entry_b = LeaderboardEntry(
        "b", False, "b", pd.Timestamp.now(), "", False, mock_averaged_results, boot_b
    )

    # abs_mean_diff calculation based on mock_averaged_results (0.90 - 0.90 = 0.0)
    # Note: In real use, averaged_results would match bootstrap means
    comparison = EntryComparison(entry_a, entry_b, "accuracy", abs_mean_diff=0.0)

    # Since A is always > B, min(prop>0, prop<0) is min(1.0, 0.0) * 2 = 0.0
    assert comparison.p_value == 0.0
    assert len(comparison.paired_bootstrap_data) == 10

    # Test Holm-Bonferroni Threshold
    # alpha=0.05, total=1, rank=1 => 0.05 / (1 - 1 + 1) = 0.05
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


def test_FinalLeaderboard(mock_bootstrap_data):
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
        mock_bootstrap_data,
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
        mock_bootstrap_data,
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


def test_FinalLeaderboard_includes_additional_columns(mock_bootstrap_data):
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
        bootstrap_data=mock_bootstrap_data,
    )

    leaderboard = FinalLeaderboard(
        entries=[entry],
        primary_metric="accuracy",
        additional_columns=["coverage_mean"],
    )

    assert "coverage_mean" in leaderboard.leaderboard_df.columns
    assert leaderboard.leaderboard_df.iloc[0]["coverage_mean"] == pytest.approx(0.87)


def test_FinalLeaderboard_missing_additional_column_raises(mock_bootstrap_data):
    """Check that requesting a missing extra column raises a clear error."""
    averaged_results = pd.DataFrame({"accuracy_mean": [0.91], "accuracy_std": [0.02]})
    entry = LeaderboardEntry(
        username="user1",
        anonymous=False,
        user_alias="u1",
        submitted_at=pd.Timestamp("2024-01-01"),
        model_report_link="",
        used_proprietary_data=False,
        averaged_results=averaged_results,
        bootstrap_data=mock_bootstrap_data,
    )

    with pytest.raises(ValueError, match="coverage_mean"):
        FinalLeaderboard(
            entries=[entry],
            primary_metric="accuracy",
            additional_columns=["coverage_mean"],
        )
