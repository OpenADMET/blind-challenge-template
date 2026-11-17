"""The banners at the top of the Space: challenge status and the model report reminder.

Both depend on the date, so the app rebuilds them on every page load rather than once
when the Space starts.
"""

import html
from datetime import datetime, timedelta, timezone

DEADLINE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_deadline(deadline: str) -> datetime:
    """Parse a deadline setting (UTC, ``DEADLINE_FORMAT``).

    Parameters
    ----------
    deadline : str
        E.g. "2026-11-03T23:59:59Z".

    Returns
    -------
    datetime
        The deadline, timezone-aware in UTC.

    """
    return datetime.strptime(deadline, DEADLINE_FORMAT).replace(tzinfo=timezone.utc)


def format_deadline(deadline: datetime) -> str:
    """Format a deadline for the page, e.g. "3 November 2026, 23:59 UTC".

    Parameters
    ----------
    deadline : datetime
        The deadline, in UTC.

    Returns
    -------
    str
        The formatted deadline.

    """
    return f"{deadline.day} {deadline:%B %Y, %H:%M} UTC"


def status_text(phase: int, now: datetime, interim: str, final: str) -> str:
    """Describe where the challenge is, for the status line under the header.

    Parameters
    ----------
    phase : int
        ``CURRENT_PHASE``.
    now : datetime
        The current time, timezone-aware.
    interim : str
        ``INTERIM_LEADERBOARD_DEADLINE``.
    final : str
        ``FINAL_LEADERBOARD_DEADLINE``.

    Returns
    -------
    str
        One line of plain text. Empty if there's nothing to say.

    """
    interim_at, final_at = parse_deadline(interim), parse_deadline(final)
    closes = f"Submissions close {format_deadline(final_at)}"
    if phase == 0:
        return "The challenge hasn't opened yet"
    if phase in (1, 2):
        parts = ["Submissions are open"]
        if phase == 1 and now < interim_at:
            parts.append(f"Interim leaderboard deadline {format_deadline(interim_at)}")
        parts.append(closes if now < final_at else "Submissions have closed")
        return " · ".join(parts)
    if phase == 3:
        return "Submissions have closed · The final leaderboard is being prepared"
    if phase == 4:
        return "The challenge has closed · The final leaderboard is out"
    return ""


def model_report_banner(
    phase: int,
    now: datetime,
    final: str,
    require_model_report: bool,
    days_before: int,
) -> str:
    """Build the model report reminder, shown in the last days before submissions close.

    Parameters
    ----------
    phase : int
        ``CURRENT_PHASE``. Only shown while submissions are open (1 or 2).
    now : datetime
        The current time, timezone-aware.
    final : str
        ``FINAL_LEADERBOARD_DEADLINE``.
    require_model_report : bool
        ``REQUIRE_MODEL_REPORT``. Without it there's nothing to remind people of.
    days_before : int
        How many days before the deadline the reminder appears.

    Returns
    -------
    str
        The banner's HTML, or "" when it shouldn't be shown.

    """
    final_at = parse_deadline(final)
    window_opens = final_at - timedelta(days=days_before)
    if not (
        require_model_report and phase in (1, 2) and window_opens <= now < final_at
    ):
        return ""
    deadline = html.escape(format_deadline(final_at))
    return (
        '<div class="banner banner-warning" role="status">'
        "<strong>A model report is required to appear on the final leaderboard</strong>"
        "<p>Add a link to your model report in the <b>Model Report Link</b> field of "
        f"the Submit tab before submissions close on <b>{deadline}</b>. Only your "
        "latest submission counts for the final leaderboard, so it must include the "
        "link.</p>"
        "</div>"
    )
