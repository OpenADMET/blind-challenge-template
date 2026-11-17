"""Tests for ``scripts/leaderboard_checks.py``."""

from datetime import timedelta

import pandas as pd
import pytest

from backend.config import (
    CLASSIFICATION_SPEC,
    MACRO_ENDPOINT_LABEL,
    REGRESSION_SPEC,
    STRUCTURE_SPEC,
)
from scripts import leaderboard_checks as checks

DEADLINE = pd.Timestamp("2026-11-01T23:59:59Z")


def _board(n: int = 4, spec=REGRESSION_SPEC, **overrides) -> pd.DataFrame:
    """A master leaderboard as create_track_leaderboards builds it."""
    step = -0.1 if spec.higher_is_better else 0.1
    data = {
        "rank": list(range(1, n + 1)),
        "username": [f"user{i}" for i in range(n)],
        "submitted_at": [DEADLINE - timedelta(hours=i + 1) for i in range(n)],
        checks.TIER_COLUMN: [f"Tier {i // 2 + 1}" for i in range(n)],
    }
    for metric in spec.metrics:
        data[f"{metric}_mean"] = [0.5 + step * i for i in range(n)]
        data[f"{metric}_std"] = [0.05] * n
    data.update(overrides)
    return pd.DataFrame(data)


def test_master_slug():
    assert checks.master_slug(REGRESSION_SPEC) == MACRO_ENDPOINT_LABEL
    assert checks.master_slug(STRUCTURE_SPEC) == "structure"


def test_a_good_board_passes():
    result = checks.check_board(REGRESSION_SPEC, _board(), DEADLINE, tiers=True)
    assert result.ok
    assert result.review == []


def test_board_with_rank_as_index_is_accepted():
    board = _board().set_index("rank")
    assert checks.check_board(REGRESSION_SPEC, board, DEADLINE, tiers=True).ok


def test_build_failures_block():
    boards = {MACRO_ENDPOINT_LABEL: _board(), "ENDPOINT_1": None}
    result = checks.check_build(REGRESSION_SPEC, boards)
    assert result.blockers == ["regression: failed to build ['ENDPOINT_1']"]


def test_empty_master_board_is_for_review():
    result = checks.check_build(REGRESSION_SPEC, {MACRO_ENDPOINT_LABEL: pd.DataFrame()})
    assert result.ok
    assert "no eligible entries" in result.review[0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda b: b.assign(submitted_at=DEADLINE + timedelta(hours=1)),
            "after the deadline",
        ),
        (lambda b: b.assign(username="same"), "more than one entry"),
        (lambda b: b.assign(rank=[1, 2, 2, 4]), "ranks aren't 1 to 4"),
        (
            lambda b: b.assign(**{"ST-RAE_mean": [0.5, None, 0.7, 0.8]}),
            "have no ST-RAE_mean",
        ),
        (
            lambda b: b.assign(**{"ST-RAE_mean": [0.9, 0.5, 0.7, 0.8]}),
            "aren't in ST-RAE_mean order",
        ),
        (lambda b: b.drop(columns=checks.TIER_COLUMN), "no 'Significance (tiers)'"),
        (
            lambda b: b.assign(**{checks.TIER_COLUMN: ["", "", "", ""]}),
            "top entry has no tier",
        ),
        (
            lambda b: b.assign(MAE_mean=[-0.1, 0.5, 0.6, 0.7]),
            "MAE_mean outside its range",
        ),
        (
            lambda b: b.assign(Spearman_R_mean=[1.2, 0.5, 0.6, 0.7]),
            "Spearman_R_mean outside",
        ),
    ],
)
def test_board_problems_block(change, message):
    result = checks.check_board(REGRESSION_SPEC, change(_board()), DEADLINE, tiers=True)
    assert any(message in b for b in result.blockers), result.blockers


def test_higher_is_better_order():
    board = _board(spec=CLASSIFICATION_SPEC)
    assert checks.check_board(CLASSIFICATION_SPEC, board, DEADLINE, tiers=True).ok
    flipped = board.assign(MCC_mean=board["MCC_mean"][::-1].to_numpy())
    result = checks.check_board(CLASSIFICATION_SPEC, flipped, DEADLINE, tiers=True)
    assert any("order" in b for b in result.blockers)


def test_live_boards_need_no_tiers():
    board = _board().drop(columns=checks.TIER_COLUMN)
    assert checks.check_board(REGRESSION_SPEC, board, DEADLINE, tiers=False).ok


def test_stale_data_is_for_review():
    board = _board().assign(submitted_at=DEADLINE - timedelta(days=10))
    result = checks.check_board(REGRESSION_SPEC, board, DEADLINE, tiers=True)
    assert result.ok
    assert "newest entry" in result.review[0]


# --- comparison with live ---------------------------------------------------------


def test_matching_live_board_passes():
    result, top = checks.check_against_live(
        REGRESSION_SPEC, _board(), _board(), DEADLINE
    )
    assert result.ok and result.review == []
    assert list(top["rank_change"]) == [0, 0, 0, 0]


def test_no_live_board_is_for_review():
    result, top = checks.check_against_live(REGRESSION_SPEC, _board(), None, DEADLINE)
    assert "no live leaderboard" in result.review[0]
    assert top.empty


def test_entrants_missing_from_live_block():
    live = _board(3)
    result, _ = checks.check_against_live(REGRESSION_SPEC, _board(4), live, DEADLINE)
    assert "aren't on the live board" in result.blockers[0]


def test_entrants_missing_from_the_board_are_listed():
    """On the final board, usually a missing model report."""
    board = _board(4).iloc[:3]
    result, _ = checks.check_against_live(REGRESSION_SPEC, board, _board(4), DEADLINE)
    assert result.ok
    assert any("user3" in item for item in result.review)


def test_big_moves_near_the_top_are_listed():
    n = 8
    live = _board(n)
    board = _board(n)
    # user7 jumps from 8th on live to 1st here.
    order = [7, 0, 1, 2, 3, 4, 5, 6]
    board["username"] = [f"user{i}" for i in order]
    result, top = checks.check_against_live(REGRESSION_SPEC, board, live, DEADLINE)
    assert any("big moves" in item and "user7" in item for item in result.review)
    assert top.loc[top["username"] == "user7", "rank_change"].item() == -7


def test_check_track_runs_everything():
    boards = {MACRO_ENDPOINT_LABEL: _board(), "ENDPOINT_1": _board()}
    result, top = checks.check_track(REGRESSION_SPEC, boards, DEADLINE, _board())
    assert result.ok
    assert len(top) == 4
