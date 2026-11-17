"""``backend/config.py`` and ``hf_space/config.py`` must describe the same tracks.

The backend and the Space are separate deployables, and the Space never imports the
backend, so each track's ``TrackSpec`` and a few shared constants are defined twice.
These tests fail when the two copies disagree.
"""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from backend import config as backend_config
from backend.aws_leaderboards import _leaderboard_file_path

_HF_CONFIG_PATH = Path(__file__).resolve().parents[2] / "hf_space" / "config.py"


def _load_hf_config():
    # Loaded under its own name: the Space imports it as a top-level "config".
    spec = importlib.util.spec_from_file_location("hf_space_config", _HF_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hf_config = _load_hf_config()


def test_same_tracks():
    assert set(hf_config.TRACK_SPECS) == set(backend_config.TRACK_SPECS)
    assert set(hf_config.TRACK_VISIBILITY) == set(backend_config.TRACK_SPECS)


def test_spec_fields_match_the_backend():
    backend_fields = {f.name for f in dataclasses.fields(backend_config.TrackSpec)}
    hf_fields = {f.name for f in dataclasses.fields(hf_config.TrackSpec)}
    assert hf_fields == backend_fields


@pytest.mark.parametrize("track", sorted(backend_config.TRACK_SPECS))
def test_track_spec_matches(track):
    assert dataclasses.asdict(hf_config.TRACK_SPECS[track]) == dataclasses.asdict(
        backend_config.TRACK_SPECS[track]
    )


@pytest.mark.parametrize(
    "name",
    [
        "MACRO_ENDPOINT_LABEL",
        "IDENTIFIER_COLUMNS",
        "ACTIVITY_DATASET_SIZE",
        "STRUCTURE_DATASET_SIZE",
        "REQUIRE_MODEL_REPORT",
    ],
)
def test_shared_constant_matches(name):
    assert getattr(hf_config, name) == getattr(backend_config, name)


@pytest.mark.parametrize("stage", ["live", "interim", "final"])
@pytest.mark.parametrize("track", sorted(backend_config.TRACK_SPECS))
def test_space_reads_leaderboards_where_the_backend_writes_them(track, stage):
    slug = backend_config.TRACK_SPECS[track].endpoints[0]
    assert hf_config.LEADERBOARD_URI_FORMAT.format(
        track=track, leaderboard_type=stage, endpoint_slug=slug, version="latest"
    ) == _leaderboard_file_path(track, stage, slug, "latest")
