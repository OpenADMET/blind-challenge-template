"""Tests for AWS leaderboard/manifest generation functions."""

import pandas as pd
import pytest

from backend.aws_leaderboards import (
    _leaderboard_file_path,
    create_track_leaderboards,
    _narrow_averaged_results_to_endpoint,
)
from backend.config import CLASSIFICATION_PATHS, REGRESSION_PATHS, STRUCTURE_PATHS, TrackPaths


@pytest.mark.parametrize(
    "track_paths", [REGRESSION_PATHS, CLASSIFICATION_PATHS, STRUCTURE_PATHS]
)
def test_leaderboard_file_path_known_tracks(track_paths):
    """Every real track resolves to a leaderboard path without raising."""
    path = _leaderboard_file_path(
        track_paths.track, "live", "MA", "latest"
    )
    assert path == f"{track_paths.leaderboard_live}/MA_leaderboard_latest.csv"


def test_create_track_leaderboards_empty_endpoints_is_a_noop():
    """A track with no endpoints (e.g. CLASSIFICATION_ENDPOINTS = []) returns {}
    rather than raising IndexError on track_paths.endpoints[0]."""
    empty_track = TrackPaths(track="classification", valid_filenames=frozenset(), endpoints=[])
    result = create_track_leaderboards(
        track_paths=empty_track,
        stage="live",
        primary_metric="MCC",
        metric_sort_ascending=False,
    )
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
