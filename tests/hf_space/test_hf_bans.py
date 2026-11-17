"""Tests for the participant ban list (``hf_space/bans.py``)."""

import re
from datetime import datetime, timezone
from pathlib import Path

import bans
import pandas as pd
import pytest

HEADER = ",".join(bans.COLUMNS) + "\n"
NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def _list(*rows: dict) -> pd.DataFrame:
    df = bans.empty_ban_list()
    for row in rows:
        df = pd.concat(
            [df, pd.DataFrame([{c: row.get(c, "") for c in bans.COLUMNS}], dtype=str)],
            ignore_index=True,
        )
    return df


# --- the file format ------------------------------------------------------------


def test_tofu_creates_the_file_with_these_columns():
    """opentofu/s3.tf writes the header-only file the Space first reads."""
    s3_tf = (Path(__file__).resolve().parents[2] / "opentofu" / "s3.tf").read_text()
    [header] = re.findall(
        r'key\s*=\s*"admin/banned_users.csv".*?content\s*=\s*"([^"]*)\\n"', s3_tf, re.S
    )
    assert header.split(",") == bans.COLUMNS
    assert bans.BAN_LIST_KEY == "admin/banned_users.csv"


def test_parse_header_only_file_is_an_empty_list():
    df = bans.parse_ban_list(HEADER)
    assert df.empty
    assert list(df.columns) == bans.COLUMNS


def test_parse_keeps_empty_cells_as_text():
    text = HEADER + "mallory,,multiple_accounts,evidence,2026-11-01T00:00:00Z,org,,\n"
    df = bans.parse_ban_list(text)
    assert df.loc[0, "hf_user_id"] == ""
    assert df.loc[0, "reinstated_at"] == ""


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("username,reason_code\nmallory,x\n", "columns"),
        (HEADER + ",,multiple_accounts,,,,,\n", "needs a username"),
        (HEADER + "Mallory,,multiple_accounts,,,,,\n", "normalised"),
        ("", "can't read"),
    ],
)
def test_parse_rejects_malformed_files(text, message):
    with pytest.raises(bans.BanListError, match=message):
        bans.parse_ban_list(text)


def test_to_csv_round_trips():
    df = bans.add_ban(
        bans.empty_ban_list(), "Mallory", "multiple_accounts", "a, b", "org", now=NOW
    )
    assert bans.parse_ban_list(bans.to_csv(df)).equals(df)


# --- finding bans ---------------------------------------------------------------


@pytest.mark.parametrize("username", ["mallory", "Mallory", " MALLORY "])
def test_find_ban_by_username_in_any_case(username):
    df = _list({"username": "mallory", "reason_code": "multiple_accounts"})
    assert bans.find_ban(df, username) is not None


def test_find_ban_normalises_org_style_names():
    df = _list({"username": "org_alice", "reason_code": "multiple_accounts"})
    assert bans.find_ban(df, "Org/Alice") is not None


def test_find_ban_by_account_id_after_a_rename():
    df = _list({"username": "mallory", "hf_user_id": "id-123"})
    assert bans.find_ban(df, "new-name", "id-123") is not None
    assert bans.find_ban(df, "new-name", "id-999") is None
    assert bans.find_ban(df, "new-name") is None


def test_lifted_bans_dont_block():
    df = _list({"username": "mallory", "reinstated_at": "2026-11-02T00:00:00Z"})
    assert bans.find_ban(df, "mallory") is None


def test_reason_messages_never_show_the_note():
    assert (
        bans.reason_message("multiple_accounts")
        == bans.REASON_MESSAGES["multiple_accounts"]
    )
    assert bans.reason_message("made-up") == bans.DEFAULT_REASON_MESSAGE


# --- editing --------------------------------------------------------------------


def test_add_ban_normalises_and_records():
    df = bans.add_ban(
        bans.empty_ban_list(),
        "Org/Mallory",
        "multiple_accounts",
        " evidence ",
        "an organiser",
        hf_user_id="id-1",
        group="ring-a",
        now=NOW,
    )
    row = df.iloc[0]
    assert row["username"] == "org_mallory"
    assert row["note"] == "evidence"
    assert row["banned_at"] == "2026-11-01T12:00:00Z"
    assert (row["hf_user_id"], row["group"], row["reinstated_at"]) == (
        "id-1",
        "ring-a",
        "",
    )


def test_add_ban_rejects_unknown_codes_and_duplicates():
    df = bans.add_ban(bans.empty_ban_list(), "mallory", "rule_violation", "", "org")
    with pytest.raises(ValueError, match="unknown reason code"):
        bans.add_ban(df, "eve", "nope", "", "org")
    with pytest.raises(ValueError, match="already banned"):
        bans.add_ban(df, "Mallory", "rule_violation", "", "org")


def test_reinstate_keeps_the_row_as_history():
    df = bans.add_ban(
        bans.empty_ban_list(), "mallory", "multiple_accounts", "evidence", "org"
    )
    later = datetime(2026, 11, 3, tzinfo=timezone.utc)
    df = bans.reinstate(df, "Mallory", "apologised", now=later)
    assert len(df) == 1
    assert df.loc[0, "reinstated_at"] == "2026-11-03T00:00:00Z"
    assert df.loc[0, "note"] == (
        "evidence | reinstated 2026-11-03T00:00:00Z: apologised"
    )
    assert bans.find_ban(df, "mallory") is None
    # Banned again later: a new row, the old one stays.
    df = bans.add_ban(df, "mallory", "rule_violation", "again", "org")
    assert len(df) == 2
    assert bans.find_ban(df, "mallory")["note"] == "again"


def test_reinstate_without_a_ban_fails():
    with pytest.raises(ValueError, match="no active ban"):
        bans.reinstate(bans.empty_ban_list(), "mallory", "")


# --- the Space's cached copy ------------------------------------------------------


class _Source:
    """A ban list file that can be changed or broken between reads."""

    def __init__(self, text: str = HEADER) -> None:
        self.text = text
        self.reads = 0
        self.fail = False

    def __call__(self) -> str:
        self.reads += 1
        if self.fail:
            raise ConnectionError("S3 down")
        return self.text


def test_cache_reads_once_until_refresh(monkeypatch):
    source = _Source()
    clock = [100.0]
    monkeypatch.setattr(bans.time, "monotonic", lambda: clock[0])
    cache = bans.BanListCache(fetch=source, refresh_seconds=300)
    cache.get()
    cache.get()
    assert source.reads == 1
    clock[0] += 301
    source.text = HEADER + "mallory,,multiple_accounts,,,,,\n"
    assert bans.find_ban(cache.get(), "mallory") is not None
    assert source.reads == 2


def test_cache_fails_closed_with_no_copy():
    source = _Source()
    source.fail = True
    with pytest.raises(bans.BanListUnavailable):
        bans.BanListCache(fetch=source).get()


def test_cache_fails_closed_on_a_malformed_first_read():
    with pytest.raises(bans.BanListUnavailable):
        bans.BanListCache(fetch=_Source("not,the,right,columns\n")).get()


def test_cache_keeps_the_last_copy_when_a_refresh_fails(monkeypatch):
    source = _Source(HEADER + "mallory,,multiple_accounts,,,,,\n")
    clock = [100.0]
    monkeypatch.setattr(bans.time, "monotonic", lambda: clock[0])
    cache = bans.BanListCache(fetch=source, refresh_seconds=300)
    cache.get()
    clock[0] += 301
    source.fail = True
    assert bans.find_ban(cache.get(), "mallory") is not None
