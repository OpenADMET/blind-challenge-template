"""Tests for ``scripts/leaderboard_release.py`` (no S3: builds and uploads are faked)."""

import pandas as pd
import pytest

from backend.aws_leaderboards import TrackLeaderboards
from backend.config import MACRO_ENDPOINT_LABEL
from scripts import leaderboard_release as release
from scripts.leaderboard_checks import CheckResult

from test_leaderboard_checks import _board


def _built():
    return {
        "regression": TrackLeaderboards(
            {MACRO_ENDPOINT_LABEL: _board(), "ENDPOINT_1": _board()},
            master_slug=MACRO_ENDPOINT_LABEL,
        ),
        "structure": TrackLeaderboards(
            {"structure": pd.DataFrame()}, master_slug="structure"
        ),
    }


def test_unknown_stage_is_refused():
    with pytest.raises(ValueError, match="stage"):
        release.build("live")


def test_build_skips_tracks_without_endpoints(monkeypatch):
    calls = []
    monkeypatch.setattr(
        release,
        "build_track_leaderboards",
        lambda paths, stage: (
            calls.append((paths.spec.track, stage)) or TrackLeaderboards({})
        ),
    )
    built = release.build("interim")
    expected = [t for t, p in release.TRACK_PATHS.items() if p.spec.endpoints]
    assert list(built) == expected
    assert calls == [(t, "interim") for t in expected]


def test_check_reads_each_live_master_board(monkeypatch, capsys):
    monkeypatch.setattr(release, "STAGE_DEADLINES", {"interim": "2026-11-01T23:59:59Z"})
    asked = []

    def _live(track, stage, slug):
        asked.append((track, stage, slug))
        return _board()

    result = release.check("interim", {"regression": _built()["regression"]}, _live)
    assert asked == [("regression", "live", MACRO_ENDPOINT_LABEL)]
    assert result.ok
    assert "0 blocking" in capsys.readouterr().out


def test_save_local_writes_non_empty_boards(tmp_path):
    written = release.save_local("final", _built(), tmp_path)
    assert sorted(p.name for p in written) == [
        "regression_ENDPOINT_1.csv",
        "regression_MA.csv",
    ]
    assert all(p.parent == tmp_path / "final" for p in written)


def test_upload_writes_each_board_and_skips_empty_ones():
    saved = []

    def _save(board, track, stage, endpoint_slug):
        saved.append((track, stage, endpoint_slug, board.index.name))
        return f"s3://bucket/{track}/{endpoint_slug}"

    paths = release.upload("interim", _built(), CheckResult(), save=_save)
    assert saved == [
        ("regression", "interim", MACRO_ENDPOINT_LABEL, "rank"),
        ("regression", "interim", "ENDPOINT_1", "rank"),
    ]
    assert len(paths) == 2


def test_upload_refuses_after_a_blocker():
    with pytest.raises(RuntimeError, match="blocking"):
        release.upload(
            "interim",
            _built(),
            CheckResult(blockers=["x"]),
            save=lambda *a, **k: pytest.fail("nothing may be uploaded"),
        )


def test_upload_refuses_a_partial_set():
    built = _built()
    built["regression"].leaderboards["ENDPOINT_1"] = None
    with pytest.raises(RuntimeError, match="failed to build"):
        release.upload(
            "final",
            built,
            CheckResult(),
            save=lambda *a, **k: pytest.fail("nothing may be uploaded"),
        )


def test_release_head_to_head_skips_empty_tracks(monkeypatch, capsys):
    released = []
    monkeypatch.setattr(release, "load_activity_ground_truth", lambda: pd.DataFrame())
    monkeypatch.setattr(release, "load_track_identifiers", lambda paths: pd.DataFrame())
    monkeypatch.setattr(
        release,
        "build_head_to_head_release",
        lambda paths, boards, identifiers, ground_truth: {"track": paths.track},
    )

    def _save(paths, files, force):
        released.append((paths.track, force))
        return [f"s3://b/{paths.track}"]

    monkeypatch.setattr(release, "save_head_to_head_release", _save)
    assert release.release_head_to_head(_built(), force=True) == ["s3://b/regression"]
    assert released == [("regression", True)]
    assert "skip" in capsys.readouterr().out
