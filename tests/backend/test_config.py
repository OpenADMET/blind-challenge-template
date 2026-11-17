"""Consistency checks on the track definitions in ``backend/config.py``."""

import dataclasses

import pytest

from backend.config import (
    METRIC_FUNCTIONS,
    REGRESSION_PATHS,
    REGRESSION_SPEC,
    TRACK_SPECS,
    TrackSpec,
)


@pytest.mark.parametrize("spec", TRACK_SPECS.values(), ids=list(TRACK_SPECS))
def test_specs_are_keyed_by_track(spec: TrackSpec):
    assert TRACK_SPECS[spec.track] is spec


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"primary_metric": "AUC"}, "primary_metric 'AUC' is not one of"),
        (
            {"metrics": ["MAE", "R2", "MAE"], "primary_metric": "MAE"},
            "duplicate metrics",
        ),
        ({"endpoints": ["E1", "E1"]}, "duplicate endpoints"),
        ({"dataset_size": -1}, "dataset_size must not be negative"),
    ],
)
def test_invalid_spec_is_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(REGRESSION_SPEC, **change)


@pytest.mark.parametrize(
    "spec",
    [s for s in TRACK_SPECS.values() if s.dataset == "activity"],
    ids=lambda s: s.track,
)
def test_activity_metrics_have_scoring_functions(spec: TrackSpec):
    assert [name for name, _ in spec.metric_functions] == spec.metrics
    assert all(callable(fn) for _, fn in spec.metric_functions)
    assert set(spec.metrics) <= set(METRIC_FUNCTIONS)


def test_tracks_sharing_a_dataset_share_its_size():
    sizes: dict[str, int] = {}
    for spec in TRACK_SPECS.values():
        assert sizes.setdefault(spec.dataset, spec.dataset_size) == spec.dataset_size


def test_track_paths_read_from_the_spec():
    assert REGRESSION_PATHS.track == "regression"
    assert REGRESSION_PATHS.endpoints is REGRESSION_PATHS.spec.endpoints
    assert REGRESSION_PATHS.identifiers_file == (
        "ground_truth/activity-identifiers.parquet"
    )
