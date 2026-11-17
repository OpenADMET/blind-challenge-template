"""The participant ban list: its file format, lookups and the Space's cached copy.

The list is one CSV in the challenge bucket (``BAN_LIST_KEY``), one row per ban.
Lifting a ban fills in ``reinstated_at`` rather than deleting the row, so the file
is also the history. Organisers edit it with ``scripts/manage_bans.py``, which uses
the functions here, so the Space and the script read the file the same way.

The Space can read the file but not write it. It keeps a copy for
``BAN_LIST_REFRESH_SECONDS``, and fails closed: with no copy at all (the file has
never loaded), submissions are refused. If a refresh fails, the last copy is used.
"""

import io
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from loguru import logger
from utils import _safeify_username

BAN_LIST_KEY = "admin/banned_users.csv"
BAN_LIST_REFRESH_SECONDS = 300

COLUMNS = [
    "username",  # stored normalised (_safeify_username)
    "hf_user_id",  # the account's permanent ID (OAuth sub), if known
    "reason_code",  # picks the message shown to the participant
    "note",  # internal only, never shown
    "banned_at",
    "banned_by",
    "group",  # optional label tying related accounts together
    "reinstated_at",  # empty while the ban is active
]

# What a banned participant is told, by reason code. ``note`` is never shown.
REASON_MESSAGES = {
    "multiple_accounts": "Using more than one account, for example to get around "
    "the submission limit",
    "leaderboard_probing": "A pattern of submissions that probes the test set",
    "shared_predictions": "Sharing or copying predictions between entrants",
    "rule_violation": "Breaking the challenge rules",
}
DEFAULT_REASON_MESSAGE = REASON_MESSAGES["rule_violation"]


class BanListError(ValueError):
    """The ban list file is malformed."""


class BanListUnavailable(RuntimeError):
    """The ban list can't be read and there's no earlier copy to fall back on."""


def parse_ban_list(csv_text: str) -> pd.DataFrame:
    """Read and check the ban list file.

    Parameters
    ----------
    csv_text : str
        The file's contents.

    Returns
    -------
    pd.DataFrame
        One row per ban, every column as text ("" when empty).

    Raises
    ------
    BanListError
        If a column is missing or unexpected, a row has no username, or a username
        isn't stored normalised.

    """
    try:
        df = pd.read_csv(
            io.StringIO(csv_text), dtype=str, keep_default_na=False, na_filter=False
        )
    except Exception as exc:
        raise BanListError(f"can't read the ban list: {exc}") from exc
    if list(df.columns) != COLUMNS:
        raise BanListError(
            f"ban list columns must be {COLUMNS}, not {list(df.columns)}"
        )
    usernames = df["username"].str.strip()
    if (usernames == "").any():
        raise BanListError("every ban list row needs a username")
    not_normalised = df.loc[
        usernames.map(_safeify_username) != df["username"], "username"
    ]
    if not not_normalised.empty:
        raise BanListError(
            f"usernames must be stored normalised: {list(not_normalised)}"
        )
    return df


def to_csv(df: pd.DataFrame) -> str:
    """Write the ban list back out, after checking it still parses.

    Parameters
    ----------
    df : pd.DataFrame
        The ban list.

    Returns
    -------
    str
        The file's contents.

    """
    text = df[COLUMNS].to_csv(index=False)
    parse_ban_list(text)
    return text


def empty_ban_list() -> pd.DataFrame:
    """Return a ban list with no bans.

    Returns
    -------
    pd.DataFrame
        The empty list, with the right columns.

    """
    return pd.DataFrame(columns=COLUMNS, dtype=str)


def active_bans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the bans that haven't been lifted.

    Parameters
    ----------
    df : pd.DataFrame
        The ban list.

    Returns
    -------
    pd.DataFrame
        The rows with no ``reinstated_at``.

    """
    return df[df["reinstated_at"].str.strip() == ""]


def find_ban(df: pd.DataFrame, username: str, hf_user_id: str = "") -> pd.Series | None:
    """Find the active ban for an account, by username or permanent account ID.

    Matching on the ID too means renaming the account doesn't lift the ban.

    Parameters
    ----------
    df : pd.DataFrame
        The ban list.
    username : str
        The account's username (any case or form; it's normalised here).
    hf_user_id : str
        The account's permanent ID, if known.

    Returns
    -------
    pd.Series | None
        The ban's row, or None if the account isn't banned.

    """
    active = active_bans(df)
    match = active["username"] == _safeify_username(username)
    if hf_user_id:
        match |= active["hf_user_id"] == hf_user_id
    rows = active[match]
    return None if rows.empty else rows.iloc[0]


def reason_message(reason_code: str) -> str:
    """Return the message a banned participant sees for a reason code.

    Parameters
    ----------
    reason_code : str
        The ban's reason code.

    Returns
    -------
    str
        Its message, or a generic one for an unknown code.

    """
    return REASON_MESSAGES.get(reason_code.strip(), DEFAULT_REASON_MESSAGE)


def add_ban(
    df: pd.DataFrame,
    username: str,
    reason_code: str,
    note: str,
    banned_by: str,
    hf_user_id: str = "",
    group: str = "",
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return the ban list with a new active ban.

    Parameters
    ----------
    df : pd.DataFrame
        The ban list.
    username : str
        The account to ban (normalised here).
    reason_code : str
        One of ``REASON_MESSAGES``.
    note : str
        Internal note: evidence, links. Never shown to the participant.
    banned_by : str
        Who is adding the ban.
    hf_user_id : str
        The account's permanent ID, if known (from its submissions' metadata).
    group : str
        Optional label tying related accounts together.
    now : datetime | None
        The time to record (for tests); defaults to now.

    Returns
    -------
    pd.DataFrame
        The updated list.

    Raises
    ------
    ValueError
        If the reason code is unknown, or the account is already banned.

    """
    if reason_code not in REASON_MESSAGES:
        raise ValueError(
            f"unknown reason code {reason_code!r}; use one of {list(REASON_MESSAGES)}"
        )
    if find_ban(df, username, hf_user_id) is not None:
        raise ValueError(f"{username!r} is already banned")
    row = {
        "username": _safeify_username(username),
        "hf_user_id": hf_user_id.strip(),
        "reason_code": reason_code,
        "note": note.strip(),
        "banned_at": _timestamp(now),
        "banned_by": banned_by.strip(),
        "group": group.strip(),
        "reinstated_at": "",
    }
    return pd.concat([df, pd.DataFrame([row], dtype=str)], ignore_index=True)


def reinstate(
    df: pd.DataFrame, username: str, note: str, now: datetime | None = None
) -> pd.DataFrame:
    """Return the ban list with an account's active ban lifted.

    The row stays, with ``reinstated_at`` set and the note added, as history.

    Parameters
    ----------
    df : pd.DataFrame
        The ban list.
    username : str
        The banned account.
    note : str
        Why the ban is lifted; appended to the ban's note.
    now : datetime | None
        The time to record (for tests); defaults to now.

    Returns
    -------
    pd.DataFrame
        The updated list.

    Raises
    ------
    ValueError
        If the account has no active ban.

    """
    ban = find_ban(df, username)
    if ban is None:
        raise ValueError(f"{username!r} has no active ban")
    df = df.copy()
    when = _timestamp(now)
    df.loc[ban.name, "reinstated_at"] = when
    df.loc[ban.name, "note"] = f"{ban['note']} | reinstated {when}: {note.strip()}"
    return df


def _timestamp(now: datetime | None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class BanListCache:
    """The Space's copy of the ban list, refreshed every ``refresh_seconds``.

    Attributes
    ----------
    fetch : Callable[[], str]
        Reads the ban list file's contents.
    refresh_seconds : float
        How long a copy is used before reading the file again.

    """

    fetch: Callable[[], str]
    refresh_seconds: float = BAN_LIST_REFRESH_SECONDS

    def __post_init__(self) -> None:
        """Start with no copy."""
        self._bans: pd.DataFrame | None = None
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> pd.DataFrame:
        """Return the ban list, reading the file if the copy is out of date.

        Returns
        -------
        pd.DataFrame
            The ban list (possibly an older copy, if a refresh failed).

        Raises
        ------
        BanListUnavailable
            If the file can't be read or parsed and there's no earlier copy.

        """
        with self._lock:
            if (
                self._bans is not None
                and time.monotonic() - self._loaded_at < self.refresh_seconds
            ):
                return self._bans
            try:
                self._bans = parse_ban_list(self.fetch())
                self._loaded_at = time.monotonic()
            except Exception as exc:
                if self._bans is None:
                    logger.error(f"Ban list unavailable, refusing submissions: {exc}")
                    raise BanListUnavailable(str(exc)) from exc
                logger.error(f"Ban list refresh failed, using the last copy: {exc}")
            return self._bans
