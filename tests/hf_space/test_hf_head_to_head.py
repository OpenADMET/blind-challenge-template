"""Tests for the Head-to-Head tab (``hf_space/head_to_head.py``)."""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from config import CLASSIFICATION_SPEC, REGRESSION_SPEC, STRUCTURE_SPEC

import head_to_head as h2h

COMPOUNDS = [f"MOL-{i}" for i in range(8)]
NAMES = ["Alice", "<img src=x onerror=alert(1)>", "Carol"]


def _common(spec, metric: str, higher_is_better: bool) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    endpoints = [*spec.endpoints, "MA"] if len(spec.endpoints) > 1 else spec.endpoints
    boot = []
    for rank in (1, 2, 3):
        centre = 0.8 - 0.1 * rank if higher_is_better else 0.2 + 0.1 * rank
        for endpoint in endpoints:
            boot.append(
                pd.DataFrame(
                    {
                        "rank": rank,
                        "Sample": range(50),
                        "Endpoint": endpoint,
                        metric: rng.normal(centre, 0.02, 50),
                    }
                )
            )
    return {
        "entries": pd.DataFrame({"rank": [1, 2, 3], "display_name": NAMES}),
        "pairwise": pd.DataFrame(
            {
                "rank_a": [1, 1, 2],
                "rank_b": [2, 3, 3],
                "abs_mean_diff": [0.1, 0.2, 0.1],
                "p_value": [0.01, 0.001, 0.2],
                "adjustment_rank": [2, 1, 3],
                "adjusted_threshold": [0.03, 0.02, 0.05],
                "significant_difference": [True, True, False],
            }
        ),
        "bootstrap": pd.concat(boot, ignore_index=True),
    }


def _tabular(spec, metric, higher_is_better, values) -> dict[str, pd.DataFrame]:
    files = _common(spec, metric, higher_is_better)
    truth = {"Molecule_Name": COMPOUNDS, "live": [True, False] * 4}
    predictions = []
    for endpoint in spec.endpoints:
        truth[endpoint] = values(0)
        truth[f"{endpoint}_conf_low"] = values(0) - 0.1
        truth[f"{endpoint}_conf_high"] = values(0) + 0.1
    for rank in (1, 2, 3):
        rows = {"rank": rank, "Molecule_Name": COMPOUNDS}
        rows.update({e: values(rank) for e in spec.endpoints})
        predictions.append(pd.DataFrame(rows))
    files["ground_truth"] = pd.DataFrame(truth)
    files["predictions"] = pd.concat(predictions, ignore_index=True)
    return files


@pytest.fixture
def released(tmp_path, monkeypatch):
    """Release files for every track, read from a local folder."""
    rng = np.random.default_rng(1)
    releases = {
        "regression": _tabular(
            REGRESSION_SPEC,
            "ST-RAE",
            False,
            lambda r: np.arange(8.0) + rng.normal(0, r * 0.3, 8),
        ),
        "classification": _tabular(
            CLASSIFICATION_SPEC,
            "MCC",
            True,
            lambda r: np.array([True, False] * 4) ^ (rng.random(8) < r * 0.1),
        ),
    }
    structure = _common(STRUCTURE_SPEC, "LDDT-PLI", True)
    structure["ground_truth"] = pd.DataFrame(
        {"Molecule_Name": COMPOUNDS, "live": [True, False] * 4}
    )
    structure["per_compound"] = pd.concat(
        [
            pd.DataFrame(
                {"rank": r, "Molecule_Name": COMPOUNDS, "LDDT-PLI": rng.random(8)}
            )
            for r in (1, 2, 3)
        ],
        ignore_index=True,
    )
    releases["structure"] = structure
    for track, files in releases.items():
        (tmp_path / track).mkdir()
        for name, df in files.items():
            df.to_parquet(tmp_path / track / f"{name}.parquet")
    monkeypatch.setattr(h2h, "H2H_LOCAL_DIR", str(tmp_path))
    monkeypatch.setattr(h2h, "_RELEASE_CACHE", {})
    return tmp_path


@pytest.fixture
def nothing_released(tmp_path, monkeypatch):
    monkeypatch.setattr(h2h, "H2H_LOCAL_DIR", str(tmp_path))
    monkeypatch.setattr(h2h, "_RELEASE_CACHE", {})


# --- data ------------------------------------------------------------------------


def test_not_released_yet(nothing_released):
    assert h2h.load_release("regression") is None
    _, _, status, _, _ = h2h.on_track_change("regression")
    assert status["visible"] is True
    assert "once the final leaderboard" in status["value"]


def test_release_appears_once_published(nothing_released, released):
    assert h2h.load_release("regression") is not None


def test_entrant_choices_are_valued_by_rank(released):
    choices = h2h.entrant_choices("regression")
    assert [value for _, value in choices] == ["1", "2", "3"]
    assert choices[0][0] == "#1 · Alice"


def test_entrant_b_excludes_a(released):
    update = h2h.on_user_a_change("regression", "1", "1")
    assert [value for _, value in update["choices"]] == ["2", "3"]
    assert update["value"] is None


def test_unknown_track_or_rank_is_handled(released):
    assert h2h.entrant_choices("docking") == []
    result = h2h.run_comparison("regression", "1", "1")
    assert "two different entrants" in result[2]["value"]
    result = h2h.run_comparison("regression", "1", "99")
    assert "from the list" in result[2]["value"]


# --- statistics ---------------------------------------------------------------------


def test_advantage_follows_the_metric_direction():
    boot_a = pd.DataFrame({"Sample": [0, 1], "ST-RAE": [0.2, 0.3], "MCC": [0.5, 0.6]})
    boot_b = pd.DataFrame({"Sample": [0, 1], "ST-RAE": [0.4, 0.3], "MCC": [0.4, 0.6]})
    # ST-RAE: lower is better, so A's lower error is an advantage.
    assert list(
        h2h.advantage(REGRESSION_SPEC, boot_a, boot_b, "ST-RAE")
    ) == pytest.approx([0.2, 0.0])
    assert list(
        h2h.advantage(CLASSIFICATION_SPEC, boot_a, boot_b, "MCC")
    ) == pytest.approx([0.1, 0.0])
    assert h2h.win_share(pd.Series([1.0, 0.0, -1.0, 1.0])) == 0.625


def test_mcnemar_exact_p():
    assert h2h.mcnemar_exact_p(0, 0) == 1.0
    assert h2h.mcnemar_exact_p(10, 0) == pytest.approx(2 / 2**10)
    assert h2h.mcnemar_exact_p(5, 5) == 1.0


# --- comparisons ----------------------------------------------------------------


def _figures(result) -> list[go.Figure]:
    """Every figure in the outputs, bare or inside an update."""
    found = []
    for item in result:
        value = item.get("value") if isinstance(item, dict) else item
        if isinstance(value, go.Figure):
            found.append(value)
    return found


@pytest.mark.parametrize(
    ("track", "per_endpoint"),
    [("regression", 2), ("classification", 3), ("structure", None)],
)
def test_comparison_builds_every_view(released, track, per_endpoint):
    result = h2h.run_comparison(track, "1", "2")
    n_plots = sum(h2h._plot_count(spec) for spec in h2h.TRACKS.values())
    assert len(result) == 7 + len(h2h.TRACKS) + n_plots
    assert result[3]["visible"] is True  # results shown
    spec = h2h.TRACKS[track]
    expected = 1 if per_endpoint is None else per_endpoint * len(spec.endpoints)
    # The delta plot, then this track's per-compound figures.
    assert len(_figures(result)) == 1 + expected


def test_names_are_escaped_in_html(released):
    result = h2h.run_comparison("regression", "1", "2")
    summary, table = result[4], result[5]
    assert "<img src=x" not in summary and "<img src=x" not in table
    assert "&lt;img src=x onerror=alert(1)&gt;" in summary


def test_summary_reports_significance(released):
    summary = h2h.run_comparison("regression", "2", "3")[4]
    assert "Not significantly different" in summary
    summary = h2h.run_comparison("regression", "1", "3")[4]
    assert "Significantly different" in summary
    assert "p = 0.0010" in summary


def test_tab_builds():
    with gr.Blocks() as demo:
        h2h.render_head_to_head()
    track = next(
        b
        for b in demo.blocks.values()
        if isinstance(b, gr.Dropdown) and b.label == "Track"
    )
    assert [value for _, value in track.choices] == list(h2h.TRACKS)
