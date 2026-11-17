"""Tests for AWS leaderboard/manifest generation functions."""

import dataclasses
import numpy as np
import pandas as pd
import pytest

import backend.aws_submission_processing as asp
from backend.aws_leaderboards import (
    _build_leaderboard,
    _leaderboard_file_path,
    create_track_leaderboards,
    _narrow_averaged_results_to_endpoint,
)
from backend.config import (
    CLASSIFICATION_PATHS,
    CLASSIFICATION_SPEC,
    REGRESSION_PATHS,
    STRUCTURE_PATHS,
    STRUCTURE_SPEC,
    TrackPaths,
)


@pytest.mark.parametrize(
    "track_paths", [REGRESSION_PATHS, CLASSIFICATION_PATHS, STRUCTURE_PATHS]
)
def test_leaderboard_file_path_known_tracks(track_paths):
    """Every real track resolves to a leaderboard path without raising."""
    path = _leaderboard_file_path(track_paths.track, "live", "MA", "latest")
    assert path == f"{track_paths.leaderboard_live}/MA_leaderboard_latest.csv"


def test_create_track_leaderboards_empty_endpoints_is_a_noop():
    """A track with no endpoints (e.g. CLASSIFICATION_SPEC.endpoints = []) returns {}
    rather than raising IndexError on track_paths.endpoints[0]."""
    empty_track = TrackPaths(
        spec=dataclasses.replace(CLASSIFICATION_SPEC, endpoints=[]),
        valid_filenames=frozenset(),
    )
    result = create_track_leaderboards(track_paths=empty_track, stage="live")
    assert result == {}


def test_narrow_averaged_results_to_endpoint():
    """Narrowing selects one endpoint's columns and strips the prefix."""
    wide = pd.DataFrame(
        {
            "ENDPOINT_1_RAE_mean": [0.1],
            "ENDPOINT_1_RAE_std": [0.01],
            "MA_RAE_mean": [0.2],
            "MA_RAE_std": [0.02],
        }
    )
    narrowed = _narrow_averaged_results_to_endpoint(wide, "ENDPOINT_1")
    assert set(narrowed.columns) == {"RAE_mean", "RAE_std"}
    assert narrowed["RAE_mean"].iloc[0] == 0.1

    narrowed_macro = _narrow_averaged_results_to_endpoint(wide, "MA")
    assert set(narrowed_macro.columns) == {"RAE_mean", "RAE_std"}
    assert narrowed_macro["RAE_mean"].iloc[0] == 0.2


def test_narrow_averaged_results_to_endpoint_is_case_insensitive():
    """Structure score files saved before v0.1.1 use a "Structure_" prefix while
    STRUCTURE_SPEC.endpoints is ["structure"]; narrowing must still find them."""
    wide = pd.DataFrame(
        {"Structure_LDDT-PLI_mean": [0.8], "Structure_LDDT-PLI_std": [0.1]}
    )
    narrowed = _narrow_averaged_results_to_endpoint(wide, STRUCTURE_SPEC.endpoints[0])
    assert set(narrowed.columns) == {"LDDT-PLI_mean", "LDDT-PLI_std"}
    assert narrowed["LDDT-PLI_mean"].iloc[0] == 0.8


def _structure_scores(monkeypatch, lddt_pli: float, endpoint_label: str | None = None):
    """Run the real structure aggregation on a synthetic per-compound table.

    Monkeypatches out OST scoring only; bootstrapping, averaging, coverage and the
    wide pivot are the production code path in ``score_structure_submission``.
    ``endpoint_label`` relabels the bootstrap rows, to mimic files scored before
    v0.1.1 with the hard-coded "Structure" label.
    """
    rng = np.random.default_rng(0)
    n = 20
    per_compound = pd.DataFrame(
        {
            "Molecule_Name": [f"STRUCT-{i:04d}" for i in range(n)],
            "LDDT-PLI": np.clip(rng.normal(lddt_pli, 0.05, n), 0, 1),
            "BiSyRMSD": np.abs(rng.normal(2.0, 0.5, n)),
            "LDDT-LP": np.clip(rng.normal(0.7, 0.05, n), 0, 1),
            "coverage": 1.0,
        }
    )
    monkeypatch.setattr(
        asp,
        "score_structure_predictions",
        lambda predicted, gt, smiles_map=None: (per_compound, {}),
    )
    if endpoint_label is not None:
        real_bootstrap = asp.bootstrap_structure_metrics
        monkeypatch.setattr(
            asp,
            "bootstrap_structure_metrics",
            lambda df, b: real_bootstrap(df, b).assign(Endpoint=endpoint_label),
        )
    _, bootstrap_df, averaged_df, _ = asp.score_structure_submission(
        predicted={}, phase=0, ground_truth={}
    )
    monkeypatch.undo()
    return averaged_df, bootstrap_df


def test_structure_leaderboard_builds_end_to_end(monkeypatch):
    """Regression test for #25: a structure leaderboard builds from real aggregation
    output, including an entry scored under the pre-v0.1.1 "Structure" label."""
    scores = [
        _structure_scores(monkeypatch, 0.9),
        _structure_scores(monkeypatch, 0.5),
        _structure_scores(monkeypatch, 0.7, endpoint_label="Structure"),
    ]
    manifest = pd.DataFrame(
        {
            "username": ["alice", "bob", "carol"],
            "anonymous": [False, False, False],
            "user_alias": ["", "", ""],
            "submitted_at": ["2026-11-01T00:00:00Z"] * 3,
            "model_report_link": [""] * 3,
        }
    )
    leaderboard = _build_leaderboard(
        manifest=manifest,
        averaged_scores={i: avg for i, (avg, _) in enumerate(scores)},
        bootstrap_scores={i: boot for i, (_, boot) in enumerate(scores)},
        primary_metric=STRUCTURE_SPEC.primary_metric,
        metric_sort_ascending=False,
        comparisons="tiers",
        endpoint=STRUCTURE_PATHS.endpoints[0],
        additional_columns=["coverage_mean"],
    )
    assert list(leaderboard["username"]) == ["alice", "carol", "bob"]
    assert leaderboard["coverage_mean"].tolist() == [1.0, 1.0, 1.0]
    for metric in STRUCTURE_SPEC.metrics:
        assert leaderboard[f"{metric}_mean"].notna().all()


@pytest.mark.parametrize(
    ("track_paths", "ascending", "extra"),
    [
        (REGRESSION_PATHS, True, []),
        (CLASSIFICATION_PATHS, False, []),
        (STRUCTURE_PATHS, False, ["coverage_mean"]),
    ],
)
def test_create_track_leaderboards_ranks_by_the_track_spec(
    monkeypatch, track_paths, ascending, extra
):
    """#16: ranking metric, direction and extra columns come from the TrackSpec."""
    import backend.aws_leaderboards as al

    monkeypatch.setattr(
        al,
        "_fetch_leaderboard_manifest_and_scores",
        lambda **kwargs: (pd.DataFrame({"username": ["a"]}), {}, {}),
    )
    calls = []
    monkeypatch.setattr(
        al,
        "_build_leaderboard",
        lambda **kwargs: calls.append(kwargs) or pd.DataFrame(),
    )

    create_track_leaderboards(track_paths=track_paths, stage="live")

    assert calls
    for kwargs in calls:
        assert kwargs["primary_metric"] == track_paths.spec.primary_metric
        assert kwargs["metric_sort_ascending"] is ascending
        assert kwargs["additional_columns"] == extra
