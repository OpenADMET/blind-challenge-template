"""Tests for loading, preparing and downloading the HF Space leaderboards."""

import pandas as pd
import pytest
from config import (
    ACTIVITY_METRIC_DISPLAY_NAMES,
    REGRESSION_TRACK,
    STRUCTURE_TRACK,
)

import leaderboards

USERS = ["bob", "alice", "carol"]


def _activity_csv() -> pd.DataFrame:
    """A regression leaderboard as the backend writes it, in file order."""
    data: dict[str, list] = {
        "rank": [2, 1, 3],
        "username": USERS,
        "user_alias": ["", "Wizard", ""],
        "anonymous": [False, True, False],
        "submitted_at": ["2026-11-01T09:30:00Z"] * 3,
        "model_report_link": ["https://example.org/b", None, "Not submitted"],
        "used_proprietary_data": [True, False, None],
        "open_source_code": [False, True, False],
    }
    # Written in reverse of the display order, which must be restored.
    for i, metric in enumerate(reversed(list(ACTIVITY_METRIC_DISPLAY_NAMES))):
        data[f"{metric}_mean"] = [0.123456 + i, 0.2, 0.3]
        data[f"{metric}_std"] = [0.01, 0.01, 0.01]
    return pd.DataFrame(data)


def _structure_csv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": [1, 2],
            "username": ["alice", "bob"],
            "user_alias": ["A&B", ""],
            "anonymous": [True, False],
            "submitted_at": ["2026-11-01T00:00:00Z"] * 2,
            "model_report_link": [None, "https://example.org/b"],
            "LDDT-PLI_mean": [0.9, 0.6],
            "LDDT-PLI_std": [0.01, 0.01],
            "coverage_mean": [1.0, 0.95],
        }
    )


@pytest.fixture
def s3_files(monkeypatch):
    """Serve leaderboard CSVs by S3 key; any other key raises like a missing object."""
    files: dict[str, pd.DataFrame] = {}
    requested: list[str] = []

    def _load(key, parquet=False):
        requested.append(key)
        if key not in files:
            raise FileNotFoundError(key)
        return files[key].copy()

    monkeypatch.setattr(leaderboards, "_load_csv_from_s3", _load)
    return files, requested


# --- format_leaderboard_uri ---------------------------------------------------


def test_format_leaderboard_uri_defaults_to_master_latest():
    assert (
        leaderboards.format_leaderboard_uri(REGRESSION_TRACK, "live")
        == "leaderboard/live/regression/MA_leaderboard_latest.csv"
    )


def test_format_leaderboard_uri_endpoint_and_version():
    assert (
        leaderboards.format_leaderboard_uri(
            STRUCTURE_TRACK, "final", endpoint_slug="structure", version="2026-11-01"
        )
        == "leaderboard/final/structure/structure_leaderboard_2026-11-01.csv"
    )


@pytest.mark.parametrize(
    ("track", "stage"), [("docking", "live"), (REGRESSION_TRACK, "phase_1")]
)
def test_format_leaderboard_uri_rejects_unknown_values(track, stage):
    with pytest.raises(ValueError):
        leaderboards.format_leaderboard_uri(track, stage)


# --- sort_leaderboard_columns -------------------------------------------------


def test_sort_leaderboard_columns_order_and_drops_unnamed():
    df = pd.DataFrame(
        columns=[
            "Open Code",
            "MAE",
            "Unnamed: 0",
            "username",
            "rank",
            "Proprietary Data",
        ]
    )
    assert list(leaderboards.sort_leaderboard_columns(df).columns) == [
        "rank",
        "username",
        "MAE",
        "Proprietary Data",
        "Open Code",
    ]


# --- load_activity_leaderboard ------------------------------------------------


def _load_regression(prefix: str = "MA-") -> pd.DataFrame:
    return leaderboards.load_activity_leaderboard(
        "live",
        REGRESSION_TRACK,
        ACTIVITY_METRIC_DISPLAY_NAMES,
        metric_prefix=prefix,
    )


def test_load_activity_leaderboard_display_table(s3_files):
    files, requested = s3_files
    files["leaderboard/live/regression/MA_leaderboard_latest.csv"] = _activity_csv()

    df = _load_regression()

    assert requested == ["leaderboard/live/regression/MA_leaderboard_latest.csv"]
    metric_cols = [f"MA-{d}" for d in ACTIVITY_METRIC_DISPLAY_NAMES.values()]
    assert list(df.columns) == [
        "Rank",
        "Username",
        "Submitted",
        "Model Report Link",
        *metric_cols,
        "Proprietary Data",
        "Open Code",
    ]
    assert list(df["Rank"]) == [1, 2, 3]
    assert df.loc[0, "Submitted"] == "2026-11-01 09:30 UTC"
    # Anonymous entries show the (escaped) alias; others link to the HF profile.
    assert df.loc[0, "Username"] == "Wizard"
    assert 'href="https://huggingface.co/bob"' in df.loc[1, "Username"]
    assert df.loc[0, "Model Report Link"] == "Not submitted"
    assert 'href="https://example.org/b"' in df.loc[1, "Model Report Link"]
    assert list(df["Proprietary Data"]) == ["No", "Yes", "No"]
    assert list(df["Open Code"]) == ["Yes", "No", "No"]
    # Means are rounded for display; std columns are not shown.
    assert df.loc[1, metric_cols[-1]] == pytest.approx(0.1235)
    assert not any(c.endswith(" std") for c in df.columns)


def test_load_activity_leaderboard_missing_file_shows_empty_table(s3_files):
    df = _load_regression(prefix="")
    assert df.empty
    assert list(df.columns) == [
        "Rank",
        "Username",
        "Submitted",
        "Model Report Link",
        "Proprietary Data",
        "Open Code",
        *ACTIVITY_METRIC_DISPLAY_NAMES.values(),
    ]


def test_download_activity_leaderboard_keeps_std_and_raw_alias(s3_files):
    files, _ = s3_files
    csv = _activity_csv()
    csv.loc[1, "user_alias"] = "Fish & Chips"
    files["leaderboard/final/regression/MA_leaderboard_latest.csv"] = csv

    path = leaderboards.download_activity_leaderboard(
        "final", REGRESSION_TRACK, ACTIVITY_METRIC_DISPLAY_NAMES, metric_prefix="MA-"
    )
    df = pd.read_csv(path)

    assert df.loc[0, "Username"] == "Fish & Chips"  # not HTML-escaped in the CSV
    assert "MA-MAE std" in df.columns
    assert df.loc[1, f"MA-{list(ACTIVITY_METRIC_DISPLAY_NAMES.values())[-1]}"] == (
        pytest.approx(0.123456)
    )


def test_download_activity_leaderboard_missing_file_gives_empty_csv(s3_files):
    path = leaderboards.download_activity_leaderboard(
        "final", REGRESSION_TRACK, ACTIVITY_METRIC_DISPLAY_NAMES
    )
    assert pd.read_csv(path).empty


# --- structure leaderboard -----------------------------------------------------


def test_load_structure_leaderboard_display_table(s3_files):
    files, _ = s3_files
    files["leaderboard/live/structure/structure_leaderboard_latest.csv"] = (
        _structure_csv()
    )

    df = leaderboards.load_structure_leaderboard()

    assert df.loc[0, "Username"] == "A&amp;B"
    assert df.loc[0, "Model Report Link"] == "Not submitted"
    assert list(df["LDDT-PLI"]) == [0.9, 0.6]
    assert "LDDT-PLI std" not in df.columns
    assert list(df["Proprietary Data"]) == ["No", "No"]


def test_load_structure_leaderboard_missing_file_shows_empty_table(s3_files):
    df = leaderboards.load_structure_leaderboard()
    assert df.empty
    assert "LDDT-PLI" in df.columns


def test_download_structure_leaderboard_keeps_std_and_raw_alias(s3_files):
    files, _ = s3_files
    files["leaderboard/live/structure/structure_leaderboard_latest.csv"] = (
        _structure_csv()
    )
    df = pd.read_csv(leaderboards.download_structure_leaderboard())
    assert df.loc[0, "Username"] == "A&B"
    assert "LDDT-PLI std" in df.columns
