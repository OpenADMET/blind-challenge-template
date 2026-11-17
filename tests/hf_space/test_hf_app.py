"""Smoke tests: the Space's pages build for every challenge phase.

These build the Gradio component tree without launching a server, with S3 replaced by
a stub. They catch import errors, broken component arguments and wrong leaderboard
paths, which matter most when Gradio or the leaderboard component is upgraded.
"""

import gradio as gr
import pytest
from config import (
    CLASSIFICATION_NAME,
    CLASSIFICATION_SPEC,
    MACRO_ENDPOINT_LABEL,
    REGRESSION_NAME,
    REGRESSION_SPEC,
    STRUCTURE_NAME,
    STRUCTURE_SPEC,
    TRACK_VISIBILITY,
    TrackVisibility,
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


ALL_VISIBLE = {
    track: TrackVisibility(submissions_open=True)
    for track in ("regression", "classification", "structure")
}


def _expected_keys(stage: str, visibility: dict[str, TrackVisibility]) -> set[str]:
    keys = set()
    for track, endpoints in [
        ("regression", REGRESSION_SPEC.endpoints),
        ("classification", CLASSIFICATION_SPEC.endpoints),
        ("structure", STRUCTURE_SPEC.endpoints),
    ]:
        if not endpoints or not visibility[track].on_leaderboard(stage):
            continue
        multi = len(endpoints) > 1
        slugs = [MACRO_ENDPOINT_LABEL, *endpoints] if multi else endpoints
        keys |= {
            f"leaderboard/{stage}/{track}/{s}_leaderboard_latest.csv" for s in slugs
        }
    return keys


def _render_stage(stage: str) -> None:
    with gr.Blocks() as demo:
        if stage == "live":
            leaderboards.render_live_leaderboards(phase=1, demo=demo)
        elif stage == "interim":
            leaderboards.render_interim_leaderboards()
        else:
            leaderboards.render_final_leaderboards(phase=4)


def test_app_module_builds():
    import app

    assert isinstance(app.demo, gr.Blocks)


@pytest.mark.parametrize("stage", ["live", "interim", "final"])
@pytest.mark.parametrize("visibility", ["template default", "all visible"])
def test_leaderboard_tabs_read_every_visible_board(
    monkeypatch, requested_keys, stage, visibility
):
    visible = TRACK_VISIBILITY if visibility == "template default" else ALL_VISIBLE
    monkeypatch.setattr(leaderboards, "TRACK_VISIBILITY", visible)
    _render_stage(stage)
    assert set(requested_keys) == _expected_keys(stage, visible)


def test_track_can_be_hidden_on_one_stage(monkeypatch, requested_keys):
    """E.g. a structure track that launched after the interim snapshot."""
    visible = {
        **ALL_VISIBLE,
        "structure": TrackVisibility(submissions_open=True, interim=False),
    }
    monkeypatch.setattr(leaderboards, "TRACK_VISIBILITY", visible)
    _render_stage("interim")
    assert not any("/structure/" in k for k in requested_keys)
    requested_keys.clear()
    _render_stage("final")
    assert any("/structure/" in k for k in requested_keys)


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
def test_submission_tab_builds_for_every_phase(phase):
    with gr.Blocks():
        submission.render_submission_tab(phase, ALL_VISIBLE)


@pytest.mark.parametrize(
    "open_tracks",
    [
        {"regression", "classification", "structure"},
        {"regression"},
        {"classification", "structure"},
    ],
)
def test_submission_tab_offers_only_open_tracks(open_tracks):
    visibility = {
        track: TrackVisibility(submissions_open=track in open_tracks)
        for track in ("regression", "classification", "structure")
    }
    with gr.Blocks() as demo:
        submission.render_submission_tab(1, visibility)
    [track_radio] = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Radio) and block.label == "Track *"
    ]
    names = {"regression": REGRESSION_NAME, "classification": CLASSIFICATION_NAME}
    names["structure"] = STRUCTURE_NAME
    assert {choice[0] for choice in track_radio.choices} == {
        names[t] for t in open_tracks
    }
