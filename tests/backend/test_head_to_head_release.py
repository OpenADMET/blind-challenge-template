"""Tests for the head-to-head release files (backend/head_to_head_release.py)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import backend.head_to_head_release as h2h
from backend.aws_leaderboards import TrackLeaderboards, _build_final_leaderboard
from backend.config import REGRESSION_PATHS, STRUCTURE_PATHS

USERS = ["alice", "bob", "carol", "dave"]
ENDPOINTS = REGRESSION_PATHS.endpoints
COMPOUNDS = [f"MOL-{i}" for i in range(6)]


@pytest.fixture
def identifiers():
    return pd.DataFrame(
        {
            "SMILES": ["C"] * 6,
            "Molecule_Name": COMPOUNDS,
            "set": ["live", "hidden"] * 3,
        }
    )


@pytest.fixture
def ground_truth():
    data = {"Molecule_Name": COMPOUNDS[::-1]}  # a different order to the identifiers
    for i, endpoint in enumerate(ENDPOINTS):
        data[endpoint] = np.arange(6.0) + i
        data[f"{endpoint}_conf_low"] = np.arange(6.0) + i - 0.5
        data[f"{endpoint}_conf_high"] = np.arange(6.0) + i + 0.5
    data["NOT_THIS_TRACK"] = 1.0
    return pd.DataFrame(data)


def _uri(user: str, kind: str) -> str:
    if kind == "predictions":
        return f"s3://b/submissions/regression/{user}/sid-{user}/predictions.csv"
    return f"s3://b/scores/all/regression/{user}/sid-{user}/{kind}"


@pytest.fixture
def manifest():
    return pd.DataFrame(
        {
            "username": USERS,
            "anonymous": [False, True, False, False],
            "user_alias": ["", "Secret Squirrel", "", ""],
            "submitted_at": pd.date_range("2026-10-01", periods=4, tz="UTC"),
            "model_report_link": [""] * 4,
            "used_proprietary_data": [False] * 4,
            "open_source_code": [False] * 4,
            "original_data_uri": [_uri(u, "predictions") for u in USERS],
            "scores_uri_for_leaderboard": [
                _uri(u, "averaged-results.parquet") for u in USERS
            ],
        }
    )


@pytest.fixture
def track_leaderboards(manifest):
    """A real FinalLeaderboard, with significance testing, over synthetic scores."""
    rng = np.random.default_rng(0)
    averaged, bootstrap = {}, {}
    for idx, mean in zip(manifest.index, [0.5, 0.6, 0.7, 0.9]):
        averaged[idx] = pd.DataFrame(
            {"MA_ST-RAE_mean": [mean], "MA_ST-RAE_std": [0.01]}
        )
        bootstrap[idx] = pd.DataFrame(
            {
                "Sample": range(100),
                "Endpoint": "MA",
                "ST-RAE": rng.normal(mean, 0.05, 100),
            }
        )
    board = _build_final_leaderboard(
        manifest=manifest,
        averaged_scores=averaged,
        bootstrap_scores=bootstrap,
        primary_metric="ST-RAE",
        metric_sort_ascending=True,
        comparisons="tiers",
        endpoint="MA",
        additional_columns=[],
    )
    return TrackLeaderboards(
        leaderboards={"MA": board.leaderboard_df},
        master_slug="MA",
        manifest=manifest,
        comparisons=board.comparison_df,
    )


def _reader(asked: list[str]):
    """Fake S3: a small predictions or bootstrap file for whichever URI is asked."""

    def _read(uri: str) -> pd.DataFrame:
        asked.append(uri)
        if uri.endswith("predictions.csv"):
            data = {"SMILES": "C", "Molecule_Name": COMPOUNDS}
            data.update({e: 1.0 for e in ENDPOINTS})
            return pd.DataFrame(data)
        if uri.endswith("bootstrap-results.parquet"):
            return pd.DataFrame(
                {"Sample": [0, 1], "Endpoint": "MA", "ST-RAE": [0.5, 0.6]}
            )
        return pd.DataFrame({"Molecule_Name": COMPOUNDS, "LDDT-PLI": 0.8})

    return _read


# --- ground truth ---------------------------------------------------------------


def test_ground_truth_tabular(identifiers, ground_truth):
    gt = h2h.build_ground_truth(REGRESSION_PATHS, identifiers, ground_truth)
    assert list(gt["Molecule_Name"]) == COMPOUNDS  # in identifier order
    assert list(gt["live"]) == [True, False] * 3
    assert f"{ENDPOINTS[0]}_conf_low" in gt.columns
    # Only this track's endpoints are released.
    assert "NOT_THIS_TRACK" not in gt.columns
    assert gt.loc[gt["Molecule_Name"] == "MOL-0", ENDPOINTS[0]].item() == 5.0


def test_ground_truth_structure_has_no_values(identifiers):
    gt = h2h.build_ground_truth(STRUCTURE_PATHS, identifiers)
    assert list(gt.columns) == ["Molecule_Name", "SMILES", "live"]


def test_ground_truth_needs_every_compound(identifiers, ground_truth):
    with pytest.raises(ValueError, match="aren't in the ground truth"):
        h2h.build_ground_truth(REGRESSION_PATHS, identifiers, ground_truth.iloc[1:])


def test_ground_truth_needs_the_data_for_tabular_tracks(identifiers):
    with pytest.raises(ValueError, match="required"):
        h2h.build_ground_truth(REGRESSION_PATHS, identifiers)


# --- the whole release -------------------------------------------------------------


def test_release_identifies_entrants_only_by_rank(
    track_leaderboards, identifiers, ground_truth
):
    asked: list[str] = []
    files = h2h.build_head_to_head_release(
        REGRESSION_PATHS, track_leaderboards, identifiers, ground_truth, _reader(asked)
    )
    assert set(files) == set(h2h.TABULAR_FILES)
    entries = files["entries"]
    assert list(entries["rank"]) == [1, 2, 3, 4]
    # Anonymous entrants appear only under their alias.
    assert list(entries["display_name"]) == [
        "alice",
        "Secret Squirrel",
        "carol",
        "dave",
    ]
    # No username anywhere in the release.
    for name, df in files.items():
        assert "username" not in df.columns, name
        assert not df.isin(["bob"]).any().any(), name


def test_release_pairwise_by_rank(track_leaderboards, identifiers, ground_truth):
    files = h2h.build_head_to_head_release(
        REGRESSION_PATHS, track_leaderboards, identifiers, ground_truth, _reader([])
    )
    pairwise = files["pairwise"]
    assert list(pairwise.columns) == h2h.PAIRWISE_COLUMNS
    assert len(pairwise) == 6  # every pair of 4 entrants
    assert set(pairwise["rank_a"]) | set(pairwise["rank_b"]) == {1, 2, 3, 4}


def test_release_reads_each_entrants_files(
    track_leaderboards, identifiers, ground_truth
):
    asked: list[str] = []
    files = h2h.build_head_to_head_release(
        REGRESSION_PATHS, track_leaderboards, identifiers, ground_truth, _reader(asked)
    )
    assert sorted(asked) == sorted(
        [_uri(u, "predictions") for u in USERS]
        + [_uri(u, "bootstrap-results.parquet") for u in USERS]
    )
    predictions = files["predictions"]
    assert list(predictions.columns) == ["rank", "Molecule_Name", *ENDPOINTS]
    assert len(predictions) == 4 * len(COMPOUNDS)
    assert set(files["bootstrap"]["rank"]) == {1, 2, 3, 4}


def test_structure_release_has_per_compound_scores(track_leaderboards, identifiers):
    asked: list[str] = []
    files = h2h.build_head_to_head_release(
        STRUCTURE_PATHS, track_leaderboards, identifiers, read=_reader(asked)
    )
    assert set(files) == set(h2h.STRUCTURE_FILES)
    assert any(uri.endswith("per-compound-results.parquet") for uri in asked)
    assert not any(uri.endswith("predictions.csv") for uri in asked)


def test_release_needs_comparisons(track_leaderboards, identifiers, ground_truth):
    track_leaderboards.comparisons = None
    with pytest.raises(ValueError, match="pairwise comparisons"):
        h2h.build_head_to_head_release(
            REGRESSION_PATHS, track_leaderboards, identifiers, ground_truth, _reader([])
        )


def test_release_needs_a_master_board(track_leaderboards, identifiers, ground_truth):
    track_leaderboards.leaderboards = {"MA": pd.DataFrame()}
    with pytest.raises(ValueError, match="no master leaderboard"):
        h2h.build_head_to_head_release(
            REGRESSION_PATHS, track_leaderboards, identifiers, ground_truth, _reader([])
        )


# --- saving ---------------------------------------------------------------------


def _files():
    return {name: pd.DataFrame({"x": [1]}) for name in h2h.TABULAR_FILES}


def test_save_refuses_before_the_final_deadline(monkeypatch):
    monkeypatch.setattr(h2h, "FINAL_LEADERBOARD_DEADLINE", "2026-11-01T23:59:59Z")
    monkeypatch.setattr(h2h.wr.s3, "to_parquet", lambda **k: pytest.fail("no write"))
    before = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
    with pytest.raises(PermissionError, match="before"):
        h2h.save_head_to_head_release(REGRESSION_PATHS, _files(), now=before)


def test_save_after_the_deadline(monkeypatch):
    monkeypatch.setattr(h2h, "FINAL_LEADERBOARD_DEADLINE", "2026-11-01T23:59:59Z")
    monkeypatch.setattr(h2h, "S3_BUCKET", "bucket")
    written = []
    monkeypatch.setattr(
        h2h.wr.s3, "to_parquet", lambda df, path, index: written.append(path)
    )
    after = datetime(2026, 11, 2, tzinfo=timezone.utc)
    paths = h2h.save_head_to_head_release(REGRESSION_PATHS, _files(), now=after)
    assert paths == written
    assert written[0] == (
        "s3://bucket/leaderboard/final/regression/head_to_head/ground_truth.parquet"
    )


def test_save_needs_exactly_the_tracks_files(monkeypatch):
    monkeypatch.setattr(h2h, "FINAL_LEADERBOARD_DEADLINE", "2026-11-01T23:59:59Z")
    after = datetime(2026, 11, 2, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="expected files"):
        h2h.save_head_to_head_release(STRUCTURE_PATHS, _files(), now=after)
