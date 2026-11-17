"""Tests for the HF Space's sortable leaderboard table."""

import math

import gradio as gr
import pandas as pd
import pytest
from leaderboard_table import (
    LeaderboardTable,
    format_mean_std,
    format_sig,
    render_leaderboard_table,
    round_sig,
    shown_columns,
)

# --- significant figures --------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "sig_figs", "expected"),
    [
        (0.71234, 3, "0.712"),
        (0.71234, 2, "0.71"),
        (12.345, 3, "12.3"),
        (1234.5, 2, "1200"),
        (100.0, 2, "100"),
        (0.0311, 2, "0.031"),
        (0.125, 2, "0.13"),  # halves round away from zero
        (-0.125, 2, "-0.13"),
        (0.0999, 2, "0.10"),  # rounding up adds no extra digit
        (9.96, 2, "10"),
        (0.0, 3, "0.0"),
        (None, 3, ""),
        (float("nan"), 3, ""),
    ],
)
def test_format_sig(value, sig_figs, expected):
    assert format_sig(value, sig_figs) == expected


def test_round_sig():
    assert round_sig(0.123456, 2) == 0.12
    assert round_sig(1234.5, 2) == 1200.0
    assert math.isnan(round_sig(None, 2))
    assert math.isnan(round_sig(float("nan"), 2))


@pytest.mark.parametrize(
    ("mean", "std", "expected"),
    [
        (0.71234, 0.03111, "0.712 ± 0.0311"),
        (-1.5, 0.25, "-1.50 ± 0.250"),
        (2.0, None, "2.00"),
        (2.0, float("nan"), "2.00"),
        (None, 0.1, ""),
        (float("nan"), 0.1, ""),
    ],
)
def test_format_mean_std(mean, std, expected):
    assert format_mean_std(mean, std) == expected


def test_format_mean_std_sig_figs():
    assert format_mean_std(0.71234, 0.03111, sig_figs=2) == "0.71 ± 0.031"


# --- LeaderboardTable ----------------------------------------------------------


def _table() -> LeaderboardTable:
    values = pd.DataFrame(
        {
            "Rank": [1, 2, 3],
            "Username": ["alice", "Team Wizard", "carol"],
            "MAE": [0.5, 0.25, float("nan")],
            "Open Code": ["Yes", "No", "Yes"],
        }
    )
    display = pd.DataFrame(
        {
            "Rank": ["1", "2", "3"],
            "Username": ['<a href="alice">alice</a>', "Team Wizard", "<a>carol</a>"],
            "MAE": ["0.500 ± 0.010", "0.250 ± 0.020", ""],
            "Open Code": ["Yes", "No", "Yes"],
        }
    )
    datatypes = {"Rank": "number", "Username": "html", "MAE": "number"}
    return LeaderboardTable(values, display, {**datatypes, "Open Code": "str"})


def test_table_rejects_mismatched_columns():
    table = _table()
    with pytest.raises(ValueError, match="same columns"):
        LeaderboardTable(
            table.values, table.display.drop(columns="MAE"), table.datatypes
        )


def test_table_rejects_mismatched_index():
    table = _table()
    with pytest.raises(ValueError, match="same index"):
        LeaderboardTable(table.values, table.display.iloc[::-1], table.datatypes)


def test_table_requires_a_datatype_per_column():
    table = _table()
    datatypes = {k: v for k, v in table.datatypes.items() if k != "MAE"}
    with pytest.raises(ValueError, match="MAE"):
        LeaderboardTable(table.values, table.display, datatypes)


def test_dataframe_value_sorts_by_value_and_shows_text():
    value = _table().to_dataframe_value(["Rank", "Username", "MAE"])
    assert value["headers"] == ["Rank", "Username", "MAE"]
    # The data is what the browser sorts by: the mean, a number; missing is None.
    assert value["data"] == [
        [1, "alice", 0.5],
        [2, "Team Wizard", 0.25],
        [3, "carol", None],
    ]
    # The display text, row for row.
    assert value["metadata"]["display_value"][1] == [
        "2",
        "Team Wizard",
        "0.250 ± 0.020",
    ]


def test_dataframe_value_follows_the_requested_column_order():
    value = _table().to_dataframe_value(["MAE", "Rank"])
    assert value["headers"] == ["MAE", "Rank"]
    assert value["data"][0] == [0.5, 1]


def test_dataframe_value_survives_gradio_postprocess():
    """gr.Dataframe keeps the numbers as data and the text as display values."""
    table = _table()
    columns = ["Rank", "Username", "MAE"]
    component = gr.Dataframe(datatype=table.datatypes_for(columns), interactive=False)
    out = component.postprocess(table.to_dataframe_value(columns))
    assert out.data[0] == [1, "alice", 0.5]
    assert out.metadata["display_value"][0][2] == "0.500 ± 0.010"


def test_filter_search_matches_the_shown_name_case_insensitively():
    table = _table().filter("wizard", {})
    assert list(table.values["Username"]) == ["Team Wizard"]
    # The display text follows the same rows.
    assert list(table.display["Username"]) == ["Team Wizard"]


def test_filter_search_is_literal_text_not_a_pattern():
    assert _table().filter("a.i", {}).values.empty


def test_filter_blank_search_keeps_every_row():
    assert len(_table().filter("  ", {}).values) == 3


def test_filter_on_column_values():
    table = _table().filter("", {"Open Code": ["Yes"]})
    assert list(table.values["Rank"]) == [1, 3]
    assert _table().filter("", {"Open Code": []}).values.empty


def test_filter_combines_search_and_filters():
    table = _table().filter("a", {"Open Code": ["Yes"]})
    assert list(table.values["Username"]) == ["alice", "carol"]


def test_filter_ignores_columns_the_table_lacks():
    assert len(_table().filter("", {"Proprietary Data": ["No"]}).values) == 3


def test_filter_on_an_empty_table():
    table = _table()
    empty = LeaderboardTable(
        table.values.iloc[0:0], table.display.iloc[0:0], table.datatypes
    )
    filtered = empty.filter("alice", {"Open Code": ["Yes"]})
    assert filtered.values.empty
    assert filtered.to_dataframe_value(["Rank"])["data"] == []


def test_shown_columns_keeps_table_order():
    table = _table()
    assert shown_columns(table, ["Rank", "Username"], ["Open Code", "MAE"]) == [
        "Rank",
        "Username",
        "MAE",
        "Open Code",
    ]
    assert shown_columns(table, ["Rank"], ["Not a column"]) == ["Rank"]


# --- render_leaderboard_table ---------------------------------------------------


def _render(table: LeaderboardTable):
    with gr.Blocks():
        return render_leaderboard_table(
            table,
            filters={"Open Code": ["Yes", "No"]},
            default_columns=["MAE", "Open Code"],
        )


def test_render_shows_fixed_and_default_columns():
    view = _render(_table())
    assert view.table.value["headers"] == ["Rank", "Username", "MAE", "Open Code"]
    assert view.table.datatype == ["number", "html", "number", "str"]
    # Rank and Username can't be unticked, so the picker doesn't offer them.
    assert [c[0] for c in view.columns.choices] == ["MAE", "Open Code"]
    assert view.table.pinned_columns == 2


def test_render_starts_with_every_filter_value_ticked():
    view = _render(_table())
    assert view.filters["Open Code"].value == ["Yes", "No"]


def test_apply_filters_and_picks_columns():
    view = _render(_table())
    update = view._apply(_table(), "a", ["MAE"], ["Yes"])
    assert update["value"]["headers"] == ["Rank", "Username", "MAE"]
    assert [row[1] for row in update["value"]["data"]] == ["alice", "carol"]
    assert update["datatype"] == ["number", "html", "number"]


def test_apply_with_nothing_chosen_shows_the_fixed_columns():
    view = _render(_table())
    update = view._apply(_table(), None, None, ["Yes", "No"])
    assert update["value"]["headers"] == ["Rank", "Username"]
    assert len(update["value"]["data"]) == 3


def test_missing_values_are_none_not_nan():
    data = _table().to_dataframe_value(["MAE"])["data"]
    assert data[2] == [None]
    assert not any(isinstance(v, float) and math.isnan(v) for row in data for v in row)
