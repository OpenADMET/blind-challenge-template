"""Tests for loading, preparing and downloading the HF Space leaderboards."""

import pandas as pd
import pytest
from config import (
    CLASSIFICATION_SPEC,
    MACRO_ENDPOINT_LABEL,
    REGRESSION_SPEC,
    REGRESSION_TRACK,
    STRUCTURE_SPEC,
    STRUCTURE_TRACK,
    metric_display_names,
)

import leaderboards

USERS = ["bob", "alice", "carol"]
REGRESSION_LABELS = list(metric_display_names(REGRESSION_SPEC).values())
MA_LABELS = [f"MA-{label}" for label in REGRESSION_LABELS]
MA_KEY = "leaderboard/live/regression/MA_leaderboard_latest.csv"
STRUCTURE_KEY = "leaderboard/live/structure/structure_leaderboard_latest.csv"


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
        "Significance (tiers)": [1, 1, 2],
    }
    # Written in reverse of the display order, which must be restored.
    for i, metric in enumerate(reversed(list(metric_display_names(REGRESSION_SPEC)))):
        data[f"{metric}_mean"] = [0.123456 + i, 0.2, 0.3]
        data[f"{metric}_std"] = [0.01, 0.02, 0.03]
    return pd.DataFrame(data)


def _structure_csv(**overrides: list) -> pd.DataFrame:
    """A structure leaderboard as the backend writes it, in file order."""
    data = {
        "rank": [2, 1, 3],
        "username": ["bob", "alice", "carol"],
        "user_alias": ["", "A&B", ""],
        "anonymous": [False, True, False],
        "submitted_at": ["2026-11-01T00:00:00Z"] * 3,
        "model_report_link": ["https://example.org/b", None, ""],
        "LDDT-PLI_mean": [0.6, 0.9, 0.4],
        "LDDT-PLI_std": [0.01, 0.02, 0.03],
        "BiSyRMSD_mean": [2.0, 1.0, 3.0],
        "BiSyRMSD_std": [0.1, 0.1, 0.1],
        "LDDT-LP_mean": [0.7, 0.8, 0.6],
        "LDDT-LP_std": [0.01, 0.01, 0.01],
        "coverage_mean": [1.0, 1.0, 0.9],
    }
    data.update(overrides)
    return pd.DataFrame(data)


@pytest.fixture
def s3_files(monkeypatch, tmp_path):
    """Serve leaderboard CSVs by S3 key; any other key raises like a missing object.

    Downloads are written under ``tmp_path``.
    """
    files: dict[str, pd.DataFrame] = {}
    requested: list[str] = []

    def _load(key, parquet=False):
        requested.append(key)
        if key not in files:
            raise FileNotFoundError(key)
        return files[key].copy()

    monkeypatch.setattr(leaderboards, "_load_csv_from_s3", _load)
    monkeypatch.setattr(leaderboards, "DOWNLOAD_DIR", tmp_path)
    return files, requested


# --- format_leaderboard_uri ---------------------------------------------------


def test_format_leaderboard_uri_defaults_to_master_latest():
    assert leaderboards.format_leaderboard_uri(REGRESSION_TRACK, "live") == MA_KEY


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


# --- score_labels ---------------------------------------------------------------


def test_score_labels_follow_the_spec_then_extra_columns():
    assert leaderboards.score_labels(STRUCTURE_SPEC) == {
        "LDDT-PLI": "LDDT-PLI",
        "BiSyRMSD": "BiSyRMSD",
        "LDDT-LP": "LDDT-LP",
        "coverage": "Coverage",
    }
    labels = leaderboards.score_labels(CLASSIFICATION_SPEC, "MA-")
    assert list(labels) == CLASSIFICATION_SPEC.metrics
    assert labels["F1"] == "MA-F1 Score"


# --- load_leaderboard -----------------------------------------------------------

FINAL_MA_KEY = "leaderboard/final/regression/MA_leaderboard_latest.csv"


def _load(stage, spec=REGRESSION_SPEC, slug=MACRO_ENDPOINT_LABEL):
    return leaderboards.load_leaderboard(stage, spec, slug)


def test_load_leaderboard_columns_and_order(s3_files):
    files, requested = s3_files
    files[MA_KEY] = _activity_csv()

    table, _ = _load("live")

    assert requested == [MA_KEY]  # one read for both the table and the download
    assert table.columns == [
        "Rank",
        "Username",
        "Submitted",
        "Model Report Link",
        *MA_LABELS,
        "Significance (tiers)",
        "Proprietary Data",
        "Open Code",
    ]
    assert list(table.values["Rank"]) == [1, 2, 3]
    assert table.display.loc[0, "Submitted"] == "2026-11-01 09:30 UTC"
    assert list(table.values["Proprietary Data"]) == ["No", "Yes", "No"]
    assert list(table.values["Open Code"]) == ["Yes", "No", "No"]


def test_load_leaderboard_reads_mixed_timestamp_formats(s3_files):
    """Older leaderboards mix timestamps with and without fractional seconds."""
    files, _ = s3_files
    csv = _activity_csv()
    csv["submitted_at"] = [
        "2026-11-01T09:30:00Z",
        "2026-11-02 10:15:30.123456+00:00",
        "2026-11-03T11:00:00.5Z",
    ]
    files[MA_KEY] = csv

    table, _ = _load("live")

    assert list(table.values["Submitted"]) == [
        "2026-11-02 10:15 UTC",  # alice, rank 1
        "2026-11-01 09:30 UTC",
        "2026-11-03 11:00 UTC",
    ]


def test_final_scores_sort_by_full_mean_and_show_3_sig_figs(s3_files):
    files, _ = s3_files
    files[FINAL_MA_KEY] = _activity_csv()

    table, _ = _load("final")
    last = MA_LABELS[-1]

    assert table.datatypes[last] == "number"
    # bob (rank 2) sorts by the full-precision mean...
    assert table.values.loc[1, last] == pytest.approx(0.123456)
    # ...and shows "mean ± std" to 3 significant figures.
    assert table.display.loc[1, last] == "0.123 ± 0.0100"
    assert not any(c.endswith(" std") for c in table.columns)


def test_live_scores_are_rounded_to_2_sig_figs_everywhere(s3_files):
    """The live board never sends full-precision scores to the browser."""
    files, _ = s3_files
    files[MA_KEY] = _activity_csv()

    table, csv_path = _load("live")
    last = MA_LABELS[-1]

    assert table.display.loc[1, last] == "0.12 ± 0.010"
    assert table.values.loc[1, last] == 0.12
    download = pd.read_csv(csv_path)
    assert download.loc[1, last] == 0.12
    assert download.loc[1, f"{last} std"] == 0.01


def test_load_leaderboard_username_cells(s3_files):
    files, _ = s3_files
    csv = _activity_csv()
    csv.loc[1, "user_alias"] = "<b>Wiz</b>"
    files[MA_KEY] = csv

    table, _ = _load("live")

    # alice is anonymous: her alias is the value (sorted and searched on, ignoring
    # case) and is shown escaped, with no link to her profile.
    assert table.values.loc[0, "Username"] == "<b>wiz</b>"
    assert table.display.loc[0, "Username"] == "&lt;b&gt;Wiz&lt;/b&gt;"
    assert "alice" not in table.display["Username"].str.cat()
    assert table.filter("alice", {}).values.empty
    # bob sorts by his name and links to his profile.
    assert table.values.loc[1, "Username"] == "bob"
    assert 'href="https://huggingface.co/bob"' in table.display.loc[1, "Username"]


def test_long_names_are_shortened_with_the_full_name_on_hover(s3_files):
    files, _ = s3_files
    csv = _activity_csv()
    long_alias = "The Extremely Long Team Name & Co"
    long_user = "a-very-long-hugging-face-username"
    csv.loc[1, "user_alias"] = long_alias
    csv.loc[0, "username"] = long_user
    files[MA_KEY] = csv

    table, _ = _load("live")

    alias_cell = table.display.loc[0, "Username"]
    assert alias_cell == (
        '<span title="The Extremely Long Team Name &amp; Co">'
        "The Extremely Long Team…</span>"
    )
    user_cell = table.display.loc[1, "Username"]
    assert f'title="{long_user}"' in user_cell
    assert ">a-very-long-hugging-fac…</a>" in user_cell
    # Sorting and search still use the full name.
    assert table.values.loc[0, "Username"] == long_alias.casefold()
    assert list(table.filter("name & co", {}).values["Rank"]) == [1]


def test_load_leaderboard_model_report_cells(s3_files):
    files, _ = s3_files
    files[MA_KEY] = _activity_csv()

    table, _ = _load("live")

    assert table.display.loc[0, "Model Report Link"] == "Not submitted"
    assert 'href="https://example.org/b"' in table.display.loc[1, "Model Report Link"]
    assert table.display.loc[2, "Model Report Link"] == "Not submitted"


def test_load_leaderboard_missing_file_shows_empty_table(s3_files):
    table, csv_path = _load("live", slug="ENDPOINT_1")
    assert table.values.empty
    assert table.columns == [
        "Rank",
        "Username",
        "Submitted",
        "Model Report Link",
        *REGRESSION_LABELS,
        "Proprietary Data",
        "Open Code",
    ]
    assert table.datatypes["Username"] == "html"
    assert table.datatypes[REGRESSION_LABELS[0]] == "number"
    download = pd.read_csv(csv_path)
    assert download.empty
    assert "MAE" in download.columns


def test_load_structure_leaderboard(s3_files):
    files, _ = s3_files
    files["leaderboard/final/structure/structure_leaderboard_latest.csv"] = (
        _structure_csv()
    )

    table, _ = _load("final", STRUCTURE_SPEC, "structure")

    # Rows follow the backend's rank, not a metric the Space picks itself.
    assert list(table.values["Username"]) == ["a&b", "bob", "carol"]
    assert table.display.loc[0, "Username"] == "A&amp;B"
    assert list(table.values["LDDT-PLI"]) == [0.9, 0.6, 0.4]
    assert table.display.loc[0, "LDDT-PLI"] == "0.900 ± 0.0200"
    # Coverage has no std, so it shows the mean alone.
    assert table.display.loc[2, "Coverage"] == "0.900"
    assert list(table.values["Proprietary Data"]) == ["No", "No", "No"]


def test_load_structure_leaderboard_tolerates_missing_metric_columns(s3_files):
    """A CSV written before a metric existed still renders (#26)."""
    files, _ = s3_files
    files[STRUCTURE_KEY] = _structure_csv().drop(
        columns=["LDDT-PLI_mean", "LDDT-PLI_std", "coverage_mean"]
    )
    table, _ = _load("live", STRUCTURE_SPEC, "structure")
    assert list(table.values["Username"]) == ["a&b", "bob", "carol"]
    assert "LDDT-PLI" not in table.columns
    assert "BiSyRMSD" in table.columns


# --- the CSV download -----------------------------------------------------------


def test_download_keeps_separate_full_precision_mean_and_std(s3_files):
    files, _ = s3_files
    csv = _activity_csv()
    csv.loc[1, "user_alias"] = "Fish & Chips"
    files[FINAL_MA_KEY] = csv

    _, csv_path = _load("final")
    df = pd.read_csv(csv_path)

    assert csv_path.endswith("regression_MA_final_leaderboard.csv")
    assert df.loc[0, "Username"] == "Fish & Chips"  # not HTML-escaped in the CSV
    # Each score is a mean column then its std column, in the spec's order.
    score_columns = [c for c in df.columns if c.startswith("MA-")]
    assert score_columns == [
        col for label in MA_LABELS for col in (label, f"{label} std")
    ]
    assert df.loc[1, MA_LABELS[-1]] == pytest.approx(0.123456)
    assert df.loc[1, f"{MA_LABELS[-1]} std"] == pytest.approx(0.01)
    assert list(df.columns[:4]) == [
        "Rank",
        "Username",
        "Submitted",
        "Model Report Link",
    ]
    assert list(df.columns[-2:]) == ["Proprietary Data", "Open Code"]


def test_download_is_overwritten_on_each_load(s3_files, tmp_path):
    """One file per leaderboard, so refreshes don't pile files up."""
    files, _ = s3_files
    files[MA_KEY] = _activity_csv()
    _, first = _load("live")
    files[MA_KEY] = _activity_csv().iloc[:1]
    _, second = _load("live")
    assert first == second
    assert len(pd.read_csv(second)) == 1
    assert [p.name for p in tmp_path.iterdir()] == [
        "regression_MA_live_leaderboard.csv"
    ]


def test_download_structure_leaderboard(s3_files):
    files, _ = s3_files
    files[STRUCTURE_KEY] = _structure_csv()
    _, csv_path = _load("live", STRUCTURE_SPEC, "structure")
    df = pd.read_csv(csv_path)
    assert df.loc[0, "Username"] == "A&B"
    assert "LDDT-PLI std" in df.columns
    assert "Coverage" in df.columns
