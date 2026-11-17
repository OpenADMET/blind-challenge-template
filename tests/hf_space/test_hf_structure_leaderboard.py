"""Tests for the HF Space structure leaderboard preparation."""

import pandas as pd
from leaderboards import _prepare_structure_df


def _structure_leaderboard_csv(**overrides: list) -> pd.DataFrame:
    """A structure leaderboard as the backend writes it, in file order."""
    data = {
        "rank": [2, 1, 3],
        "username": ["bob", "alice", "carol"],
        "submitted_at": ["2026-11-01T00:00:00Z"] * 3,
        "model_report_link": ["", "", ""],
        "LDDT-PLI_mean": [0.6, 0.9, 0.4],
        "LDDT-PLI_std": [0.01, 0.01, 0.01],
        "BiSyRMSD_mean": [2.0, 1.0, 3.0],
        "BiSyRMSD_std": [0.1, 0.1, 0.1],
        "LDDT-LP_mean": [0.7, 0.8, 0.6],
        "LDDT-LP_std": [0.01, 0.01, 0.01],
        "coverage_mean": [1.0, 1.0, 0.9],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_prepare_structure_df_orders_by_rank():
    """Rows follow the backend's rank, not a metric the Space picks itself."""
    df = _prepare_structure_df(_structure_leaderboard_csv())
    assert list(df["username"]) == ["alice", "bob", "carol"]


def test_prepare_structure_df_tolerates_missing_metric_columns():
    """A CSV written before a metric existed still renders (#26)."""
    csv = _structure_leaderboard_csv().drop(
        columns=["LDDT-PLI_mean", "LDDT-PLI_std", "coverage_mean"]
    )
    df = _prepare_structure_df(csv)
    assert list(df["username"]) == ["alice", "bob", "carol"]
    assert "LDDT-PLI" not in df.columns
    assert "BiSyRMSD" in df.columns
