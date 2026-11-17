"""Tests for the HF Space leaderboard rendering helpers."""

import pandas as pd
from leaderboards import (
    hide_username_for_anonymous_entries,
    make_tag_clickable,
    make_user_clickable,
)

XSS_ALIAS = "<img src=x onerror=alert(1)>"


def _leaderboard(alias: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "username": ["alice", "bob"],
            "user_alias": ["", alias],
            "anonymous": [False, True],
        }
    )


def test_anonymous_alias_is_escaped_on_the_leaderboard():
    """An alias containing markup is shown as text, not rendered as HTML."""
    df = hide_username_for_anonymous_entries(_leaderboard(XSS_ALIAS))
    assert df.loc[1, "username"] == "&lt;img src=x onerror=alert(1)&gt;"
    assert "<" not in df.loc[1, "username"]
    assert list(df.columns) == ["username"]


def test_anonymous_alias_is_not_escaped_in_the_csv_download():
    """The downloadable CSV isn't HTML, so it keeps the alias as entered."""
    df = hide_username_for_anonymous_entries(
        _leaderboard("Fish & Chips"), escape_html=False
    )
    assert df.loc[1, "username"] == "Fish & Chips"


def test_make_user_clickable_escapes_name():
    cell = make_user_clickable('a"b<c>')
    assert 'a"b' not in cell
    assert "<c>" not in cell
    assert "a&quot;b&lt;c&gt;" in cell


def test_make_tag_clickable_quote_cannot_break_out_of_href():
    """A quote in the URL can't close the href attribute and add an event handler."""
    cell = make_tag_clickable('https://example.org/" onmouseover="alert(1)')
    assert '" onmouseover="' not in cell
    assert 'href="https://example.org/&quot; onmouseover=&quot;alert(1)"' in cell


def test_make_tag_clickable_plain_link_unchanged():
    cell = make_tag_clickable("https://example.org/report?a=1")
    assert 'href="https://example.org/report?a=1"' in cell
    assert make_tag_clickable("") == "Not submitted"
    assert make_tag_clickable("ftp://example.org") == "Invalid link"
