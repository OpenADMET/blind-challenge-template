"""A sortable, searchable leaderboard table built on ``gr.Dataframe``.

Every cell has a value, which its column sorts by, and display text, which is what
the visitor sees. ``gr.Dataframe`` supports this when it isn't editable: the value goes
in ``data`` and the text in ``metadata["display_value"]``. So a metric column sorts by
the mean, numerically, while showing "mean ± std", and the Username column sorts by
name while showing a link.

This module knows nothing about S3, tracks or stages; ``leaderboards.py`` builds each
``LeaderboardTable`` and decides which columns and filters it gets.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

import gradio as gr
import pandas as pd

DataType = Literal["str", "number", "html"]

SEARCH_COLUMN = "Username"
DEFAULT_SIG_FIGS = 3


def _is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _round_decimal(value: float, sig_figs: int) -> Decimal:
    """Round to ``sig_figs`` significant figures, halves away from zero."""
    exact = Decimal(repr(value))
    if value == 0 or not math.isfinite(value):
        return exact
    magnitude = math.floor(math.log10(abs(value)))
    rounded = exact.quantize(Decimal(1).scaleb(magnitude - sig_figs + 1), ROUND_HALF_UP)
    # Rounding up can add a digit (0.0999 -> 0.100); keep sig_figs digits.
    if rounded != 0 and math.floor(math.log10(abs(rounded))) != magnitude:
        rounded = rounded.quantize(
            Decimal(1).scaleb(magnitude - sig_figs + 2), ROUND_HALF_UP
        )
    return rounded


def round_sig(value: Any, sig_figs: int) -> float:
    """Round a number to some significant figures.

    Parameters
    ----------
    value : Any
        The number. Missing (None or NaN) gives NaN.
    sig_figs : int
        Significant figures to keep.

    Returns
    -------
    float
        The rounded number.

    """
    if _is_missing(value):
        return float("nan")
    return float(_round_decimal(float(value), sig_figs))


def format_sig(value: Any, sig_figs: int = DEFAULT_SIG_FIGS) -> str:
    """Format a number to some significant figures, without an exponent.

    Parameters
    ----------
    value : Any
        The number. Missing (None or NaN) gives an empty string.
    sig_figs : int
        Significant figures to show, e.g. 3 gives "0.712", "12.3" and "1230".

    Returns
    -------
    str
        The formatted number.

    """
    if _is_missing(value):
        return ""
    return f"{_round_decimal(float(value), sig_figs):f}"


def format_mean_std(mean: Any, std: Any, sig_figs: int = DEFAULT_SIG_FIGS) -> str:
    """Format a score as "mean ± std".

    Parameters
    ----------
    mean : Any
        The mean. Missing (None or NaN) gives an empty string.
    std : Any
        The standard deviation. Missing gives the mean alone.
    sig_figs : int
        Significant figures for each number.

    Returns
    -------
    str
        The formatted score.

    """
    if _is_missing(mean):
        return ""
    if _is_missing(std):
        return format_sig(mean, sig_figs)
    return f"{format_sig(mean, sig_figs)} ± {format_sig(std, sig_figs)}"


@dataclass(frozen=True)
class LeaderboardTable:
    """A leaderboard's cells: what each column sorts by, and what each cell shows.

    Attributes
    ----------
    values : pd.DataFrame
        The value each cell sorts and filters by.
    display : pd.DataFrame
        The text each cell shows (HTML for ``"html"`` columns). Same index and
        columns as ``values``.
    datatypes : dict[str, DataType]
        Every column's ``gr.Dataframe`` datatype.
    headers : dict[str, str]
        Header text for columns whose header differs from their name, e.g. with a
        ↑/↓ for the better direction. The column picker and CSV use the name.
    styling : pd.DataFrame | None
        CSS for each cell, e.g. a background colour, or None for no styling. Same
        index and columns as ``values``.

    """

    values: pd.DataFrame
    display: pd.DataFrame
    datatypes: dict[str, DataType]
    headers: dict[str, str] = field(default_factory=dict)
    styling: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Check that values, display text and datatypes describe the same cells.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the columns or index differ, or a column has no datatype.

        """
        if list(self.values.columns) != list(self.display.columns):
            raise ValueError("values and display must have the same columns")
        if not self.values.index.equals(self.display.index):
            raise ValueError("values and display must have the same index")
        missing = [c for c in self.values.columns if c not in self.datatypes]
        if missing:
            raise ValueError(f"no datatype for columns {missing}")
        if self.styling is not None and (
            list(self.styling.columns) != list(self.values.columns)
            or not self.styling.index.equals(self.values.index)
        ):
            raise ValueError("styling must have the same columns and index as values")

    @property
    def columns(self) -> list[str]:
        """The table's columns, in order."""
        return list(self.values.columns)

    def filter(
        self, search: str, allowed: dict[str, Sequence[str]]
    ) -> "LeaderboardTable":
        """Keep the rows that match a name search and every column filter.

        Parameters
        ----------
        search : str
            Case-insensitive text to find in the ``SEARCH_COLUMN`` value, which is the
            name shown on the leaderboard (an anonymous entry's alias, never its
            username). Blank keeps every row.
        allowed : dict[str, Sequence[str]]
            Column to the values a row may have in it. Columns the table doesn't
            have are ignored.

        Returns
        -------
        LeaderboardTable
            The matching rows, in the same order.

        """
        keep = pd.Series(True, index=self.values.index)
        search = search.strip()
        if search and SEARCH_COLUMN in self.values.columns:
            keep &= (
                self.values[SEARCH_COLUMN]
                .astype(str)
                .str.contains(search, case=False, regex=False)
            )
        for column, values in allowed.items():
            if column in self.values.columns:
                keep &= self.values[column].isin(list(values))
        return LeaderboardTable(
            self.values[keep],
            self.display[keep],
            self.datatypes,
            self.headers,
            None if self.styling is None else self.styling[keep],
        )

    def header(self, column: str) -> str:
        """Return a column's header text.

        Parameters
        ----------
        column : str
            The column's name.

        Returns
        -------
        str
            Its header text.

        """
        return self.headers.get(column, column)

    def to_dataframe_value(self, columns: Sequence[str]) -> dict[str, Any]:
        """Build the ``gr.Dataframe`` value for some of the table's columns.

        Parameters
        ----------
        columns : Sequence[str]
            The columns to show, in order.

        Returns
        -------
        dict[str, Any]
            ``headers`` (their header text), ``data`` (the values, missing ones as
            None), ``metadata.display_value`` (the text) and, with styling,
            ``metadata.styling``.

        """
        columns = list(columns)
        values = self.values[columns].astype(object)
        values = values.where(values.notna(), None)
        metadata: dict[str, Any] = {
            "display_value": self.display[columns].to_numpy().tolist()
        }
        if self.styling is not None:
            metadata["styling"] = self.styling[columns].to_numpy().tolist()
        return {
            "headers": [self.header(c) for c in columns],
            "data": values.to_numpy().tolist(),
            "metadata": metadata,
        }

    def datatypes_for(self, columns: Sequence[str]) -> list[DataType]:
        """Return the datatypes of some of the table's columns, in order.

        Parameters
        ----------
        columns : Sequence[str]
            The columns.

        Returns
        -------
        list[DataType]
            Their datatypes.

        """
        return [self.datatypes[c] for c in columns]


def shown_columns(
    table: LeaderboardTable, fixed: Sequence[str], chosen: Sequence[str]
) -> list[str]:
    """Return the columns to show: the fixed ones and the chosen ones, in table order.

    Parameters
    ----------
    table : LeaderboardTable
        The table.
    fixed : Sequence[str]
        Columns that are always shown.
    chosen : Sequence[str]
        Columns the visitor has picked.

    Returns
    -------
    list[str]
        The columns to show.

    """
    wanted = set(fixed) | set(chosen)
    return [c for c in table.columns if c in wanted]


@dataclass(frozen=True)
class LeaderboardTableView:
    """The components of one rendered leaderboard table.

    Attributes
    ----------
    state : gr.State
        The unfiltered ``LeaderboardTable``.
    search : gr.Textbox
        The name search.
    filters : dict[str, gr.CheckboxGroup]
        A checkbox group per filtered column.
    columns : gr.CheckboxGroup
        The optional columns that are shown.
    table : gr.Dataframe
        The table itself.
    fixed_columns : tuple[str, ...]
        Columns that are always shown.
    download : gr.DownloadButton | None
        The CSV download, if the table has one.

    """

    state: gr.State
    search: gr.Textbox
    filters: dict[str, gr.CheckboxGroup]
    columns: gr.CheckboxGroup
    table: gr.Dataframe
    fixed_columns: tuple[str, ...]
    download: gr.DownloadButton | None = None

    @property
    def _inputs(self) -> list[Any]:
        return [self.state, self.search, self.columns, *self.filters.values()]

    def _apply(
        self,
        table: LeaderboardTable,
        search: str,
        chosen: list[str],
        *allowed: list[str],
    ) -> Any:
        filtered = table.filter(search or "", dict(zip(self.filters, allowed)))
        columns = shown_columns(table, self.fixed_columns, chosen or [])
        return gr.update(
            value=filtered.to_dataframe_value(columns),
            datatype=filtered.datatypes_for(columns),
        )

    def refresh_on(
        self,
        triggers: Sequence[Any],
        load: Callable[[], Any],
        also: Sequence[Any] = (),
    ) -> None:
        """Reload the table on some events, keeping the visitor's search and filters.

        Parameters
        ----------
        triggers : Sequence[Any]
            Events to reload on, e.g. ``[demo.load, timer.tick]``.
        load : Callable[[], Any]
            Loads the latest table. Returns a tuple: the table, then the CSV path if
            the view has a download button, then a value for each of ``also``. With
            neither, just the table.
        also : Sequence[Any]
            Other components ``load`` updates at the same time, e.g. a "last
            checked" note.

        Returns
        -------
        None

        """
        outputs: list[Any] = [self.state]
        if self.download is not None:
            outputs.append(self.download)
        gr.on(triggers=list(triggers), fn=load, outputs=[*outputs, *also]).then(
            fn=self._apply, inputs=self._inputs, outputs=self.table
        )


def render_leaderboard_table(
    table: LeaderboardTable,
    filters: dict[str, list[str]],
    default_columns: Sequence[str],
    fixed_columns: Sequence[str] = ("Rank", SEARCH_COLUMN),
    download: gr.DownloadButton | None = None,
) -> LeaderboardTableView:
    """Render a leaderboard table with a toolbar (a name search and column filters)
    and a collapsed column picker.

    Parameters
    ----------
    table : LeaderboardTable
        The table to show first.
    filters : dict[str, list[str]]
        Column to the values offered as checkboxes; all start ticked.
    default_columns : Sequence[str]
        Optional columns shown before the visitor picks. The picker offers every
        column of ``table`` except the fixed ones.
    fixed_columns : Sequence[str]
        Columns that are always shown, and can't be unticked. Keep them first in
        ``table``: they're pinned while the table scrolls sideways.
    download : gr.DownloadButton | None
        The table's CSV download button, already rendered wherever the page wants
        it, so a refresh can update it; None if it has none.

    Returns
    -------
    LeaderboardTableView
        The rendered components, for wiring a refresh.

    """
    fixed = tuple(c for c in fixed_columns if c in table.columns)
    optional = [c for c in table.columns if c not in fixed]
    with gr.Row(elem_classes="leaderboard-toolbar"):
        search = gr.Textbox(
            label="Search",
            placeholder="Search by name",
            scale=2,
            min_width=160,
            max_lines=1,
        )
        filter_boxes = {
            column: gr.CheckboxGroup(
                choices=choices,
                value=list(choices),
                label=column,
                scale=1,
                min_width=180,
            )
            for column, choices in filters.items()
        }
    with gr.Accordion("Columns", open=False):
        columns = gr.CheckboxGroup(
            choices=[(table.header(c), c) for c in optional],
            value=[c for c in default_columns if c in optional],
            show_label=False,
        )
    shown = shown_columns(table, fixed, default_columns)
    dataframe = gr.Dataframe(
        value=table.to_dataframe_value(shown),
        datatype=table.datatypes_for(shown),
        interactive=False,
        show_label=False,
        pinned_columns=len(fixed),
        max_height=600,
        elem_classes="leaderboard-table",
    )
    view = LeaderboardTableView(
        state=gr.State(table),
        search=search,
        filters=filter_boxes,
        columns=columns,
        table=dataframe,
        fixed_columns=fixed,
        download=download,
    )
    gr.on(
        triggers=[
            search.change,
            columns.change,
            *(box.change for box in filter_boxes.values()),
        ],
        fn=view._apply,
        inputs=view._inputs,
        outputs=view.table,
    )
    return view
