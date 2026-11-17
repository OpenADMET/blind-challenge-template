"""Tests for the Space's status line and model report reminder."""

from datetime import datetime, timedelta, timezone

import pytest
from banners import format_deadline, model_report_banner, parse_deadline, status_text

INTERIM = "2026-10-01T23:59:59Z"
FINAL = "2026-11-03T23:59:59Z"
FINAL_AT = datetime(2026, 11, 3, 23, 59, 59, tzinfo=timezone.utc)


def test_parse_and_format_deadline():
    assert parse_deadline(FINAL) == FINAL_AT
    assert format_deadline(FINAL_AT) == "3 November 2026, 23:59 UTC"


@pytest.mark.parametrize(
    ("phase", "now", "expected"),
    [
        (0, FINAL_AT, "The challenge hasn't opened yet"),
        (
            1,
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            "Submissions are open · Interim leaderboard deadline 1 October 2026, "
            "23:59 UTC · Submissions close 3 November 2026, 23:59 UTC",
        ),
        (
            2,
            datetime(2026, 10, 15, tzinfo=timezone.utc),
            "Submissions are open · Submissions close 3 November 2026, 23:59 UTC",
        ),
        (
            2,
            FINAL_AT + timedelta(seconds=1),
            "Submissions are open · Submissions have closed",
        ),
        (
            3,
            FINAL_AT,
            "Submissions have closed · The final leaderboard is being prepared",
        ),
        (4, FINAL_AT, "The challenge has closed · The final leaderboard is out"),
    ],
)
def test_status_text(phase, now, expected):
    assert status_text(phase, now, INTERIM, FINAL) == expected


def test_status_text_drops_a_passed_interim_deadline():
    after_interim = datetime(2026, 10, 15, tzinfo=timezone.utc)
    assert "Interim" not in status_text(1, after_interim, INTERIM, FINAL)


def _banner(now, phase=2, required=True, days=7):
    return model_report_banner(phase, now, FINAL, required, days)


def test_model_report_banner_in_the_final_week():
    html = _banner(FINAL_AT - timedelta(days=2))
    assert "model report is required" in html
    assert "3 November 2026, 23:59 UTC" in html
    assert "banner-warning" in html


@pytest.mark.parametrize(
    "now",
    [
        FINAL_AT - timedelta(days=7, seconds=1),  # before the window
        FINAL_AT,  # submissions have closed
        FINAL_AT + timedelta(days=1),
    ],
)
def test_model_report_banner_outside_the_final_week(now):
    assert _banner(now) == ""


def test_model_report_banner_starts_exactly_days_before():
    assert _banner(FINAL_AT - timedelta(days=7)) != ""
    assert _banner(FINAL_AT - timedelta(days=10), days=14) != ""


def test_model_report_banner_needs_the_requirement():
    assert _banner(FINAL_AT - timedelta(days=1), required=False) == ""


@pytest.mark.parametrize("phase", [0, 3, 4])
def test_model_report_banner_only_while_submissions_are_open(phase):
    assert _banner(FINAL_AT - timedelta(days=1), phase=phase) == ""
