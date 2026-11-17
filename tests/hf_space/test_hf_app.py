"""Smoke tests: the Space's pages build for every challenge phase.

These build the Gradio component tree without launching a server, with S3 replaced by
a stub. They catch import errors, broken component arguments and wrong leaderboard
paths, which matter most when Gradio or the leaderboard component is upgraded.
"""

import gradio as gr
import pytest
from config import (
    CLASSIFICATION_ENDPOINTS,
    MACRO_ENDPOINT_LABEL,
    REGRESSION_ENDPOINTS,
    STRUCTURE_ENDPOINTS,
)

import leaderboards
import submission


@pytest.fixture
def requested_keys(monkeypatch):
    """Record the leaderboard keys read while building, and serve nothing."""
    keys: list[str] = []

    def _missing(key, parquet=False):
        keys.append(key)
        raise FileNotFoundError(key)

    monkeypatch.setattr(leaderboards, "_load_csv_from_s3", _missing)
    return keys


def _expected_keys(stage: str) -> set[str]:
    keys = set()
    for track, endpoints in [
        ("regression", REGRESSION_ENDPOINTS),
        ("classification", CLASSIFICATION_ENDPOINTS),
    ]:
        slugs = [MACRO_ENDPOINT_LABEL, *endpoints] if len(endpoints) > 1 else endpoints
        keys |= {
            f"leaderboard/{stage}/{track}/{s}_leaderboard_latest.csv" for s in slugs
        }
    keys |= {
        f"leaderboard/{stage}/structure/{s}_leaderboard_latest.csv"
        for s in STRUCTURE_ENDPOINTS
    }
    return keys


def test_app_module_builds():
    import app

    assert isinstance(app.demo, gr.Blocks)


@pytest.mark.parametrize("stage", ["live", "interim", "final"])
def test_leaderboard_tabs_build_and_read_every_board(requested_keys, stage):
    with gr.Blocks() as demo:
        if stage == "live":
            leaderboards.render_live_leaderboards(phase=1, demo=demo)
        elif stage == "interim":
            leaderboards.render_interim_leaderboards()
        else:
            leaderboards.render_final_leaderboards(phase=4)
    assert set(requested_keys) == _expected_keys(stage)


@pytest.mark.parametrize(
    ("render", "kwargs"),
    [
        (leaderboards.render_live_leaderboards, {"phase": 0}),
        (leaderboards.render_final_leaderboards, {"phase": 3}),
    ],
)
def test_leaderboards_not_yet_available_read_nothing(requested_keys, render, kwargs):
    with gr.Blocks() as demo:
        if render is leaderboards.render_live_leaderboards:
            kwargs = {**kwargs, "demo": demo}
        render(**kwargs)
    assert requested_keys == []


@pytest.mark.parametrize("phase", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("structure_live", [False, True])
def test_submission_tab_builds_for_every_phase(phase, structure_live):
    with gr.Blocks():
        submission.render_submission_tab(
            phase,
            regression_active=True,
            classification_active=True,
            structure_track_live=structure_live,
        )
