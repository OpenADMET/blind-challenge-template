"""Load the leaderboards from S3 and show them in the Space.

Each leaderboard CSV the backend writes has the same layout for every track: rank,
user and submission details, and a ``<metric>_mean`` and ``<metric>_std`` column per
metric. Which metrics, and in what order, comes from the track's ``TrackSpec``. The
table itself (sorting, search, filters, column picker) is in ``leaderboard_table``.
"""

import html
import os
import tempfile
from functools import partial
from pathlib import Path

import gradio as gr
import pandas as pd
from config import (
    LEADERBOARD_SIG_FIGS,
    LEADERBOARD_URI_FORMAT,
    MACRO_ENDPOINT_LABEL,
    METRIC_DISPLAY_NAMES,
    ROUNDED_STAGES,
    TRACK_NAMES,
    TRACK_SPECS,
    TRACK_VISIBILITY,
    LeaderboardStage,
    TrackSpec,
    TrackVisibility,
    metric_display_names,
)
from leaderboard_table import (
    DataType,
    LeaderboardTable,
    LeaderboardTableView,
    format_mean_std,
    render_leaderboard_table,
    round_sig,
)
from loguru import logger
from utils import _load_csv_from_s3

INFO_COLUMNS = ["Rank", "Username", "Submitted", "Model Report Link"]
FILTERS = {"Proprietary Data": ["Yes", "No"], "Open Code": ["Yes", "No"]}
REFRESH_SECONDS = 30
# Longer names are cut short in the table, with the full name on hover, so one long
# alias can't make the Username column very wide.
MAX_NAME_CHARS = 24
# Where the CSV downloads are written. One file per leaderboard, overwritten on
# each refresh, so they don't pile up.
DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "leaderboard_downloads"


def score_labels(spec: TrackSpec, metric_prefix: str = "") -> dict[str, str]:
    """Map a track's score columns to their leaderboard labels, in display order.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    metric_prefix : str
        Prefix for every label, e.g. "MA-" on a track's macro-averaged leaderboard.

    Returns
    -------
    dict[str, str]
        Column name without ``_mean``/``_std`` to label: the track's metrics, then
        its extra columns.

    """
    labels = {m: f"{metric_prefix}{d}" for m, d in metric_display_names(spec).items()}
    for column in spec.extra_columns:
        base = column.removesuffix("_mean")
        labels[base] = f"{metric_prefix}{METRIC_DISPLAY_NAMES.get(base, base)}"
    return labels


def _metric_prefix(endpoint_slug: str) -> str:
    """Return "MA-" for a track's macro-averaged leaderboard, else ""."""
    return "MA-" if endpoint_slug == MACRO_ENDPOINT_LABEL else ""


def _default_columns(spec: TrackSpec, metric_prefix: str) -> list[str]:
    """Return the columns a leaderboard shows before the visitor picks any."""
    return [
        *INFO_COLUMNS,
        *score_labels(spec, metric_prefix).values(),
        *FILTERS,
    ]


def format_leaderboard_uri(
    track: str,
    leaderboard_type: str,
    endpoint_slug: str = MACRO_ENDPOINT_LABEL,
    version: str = "latest",
) -> str:
    """Format the leaderboard URI based on track, phase, and endpoint.

    Parameters
    ----------
    track : str
        A key of ``TRACK_SPECS``.
    leaderboard_type : str
        "live", "interim", or "final".
    endpoint_slug : str
        For regression/classification, a REGRESSION_SPEC.endpoints /
        CLASSIFICATION_SPEC.endpoints entry (e.g. "ENDPOINT_1") to load that endpoint's
        own leaderboard instead of the track's master (macro-ranked) leaderboard.
        Defaults to ``MACRO_ENDPOINT_LABEL`` ("MA"), which loads the master.
    version : str
        "latest" (default), or an ISO-format timestamp for a dated snapshot.

    Returns
    -------
    str
        The formatted S3 URI for the requested leaderboard.

    Raises
    ------
    ValueError
        If ``track`` or ``leaderboard_type`` is not one of the recognised values.

    """
    if track not in TRACK_SPECS:
        raise ValueError(f"Invalid track: {track}")
    if leaderboard_type not in ["live", "interim", "final"]:
        raise ValueError(f"Invalid phase: {leaderboard_type}")
    uri = LEADERBOARD_URI_FORMAT.format(
        track=track,
        leaderboard_type=leaderboard_type,
        endpoint_slug=endpoint_slug,
        version=version,
    )
    logger.info(f"Formatted leaderboard URI: {uri}")
    return uri


def sort_leaderboard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort leaderboard columns in a consistent order."""
    start_cols = ["rank", "username", "Submitted", "model_report_link"]
    end_cols = ["Proprietary Data", "Open Code"]
    start_cols = [c for c in start_cols if c in df.columns]
    end_cols = [c for c in end_cols if c in df.columns]
    middle_cols = [c for c in df.columns if c not in start_cols + end_cols]
    ordered_cols = start_cols + middle_cols + end_cols
    ordered_cols = [c for c in ordered_cols if "Unnamed" not in c]
    return df[ordered_cols]


def _rename_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename internal column names to their display-friendly titles.

    Applied last, after every step that still relies on the internal names (sorting,
    filtering usernames for anonymous entries, making links clickable) — those all
    reference "rank"/"username"/"model_report_link" directly.
    """
    return df.rename(
        columns={
            "rank": "Rank",
            "username": "Username",
            "model_report_link": "Model Report Link",
        }
    )


def _shorten(name: str) -> str:
    """Cut a name to ``MAX_NAME_CHARS``, ending in "…" if it was longer."""
    if len(name) <= MAX_NAME_CHARS:
        return name
    return name[: MAX_NAME_CHARS - 1] + "…"


def name_cell(name: str) -> str:
    """Show a name that isn't a link (an anonymous entry's alias) as HTML.

    The name is escaped. A long name is shortened, with the full name on hover.
    """
    if len(name) <= MAX_NAME_CHARS:
        return html.escape(name)
    return f'<span title="{html.escape(name)}">{html.escape(_shorten(name))}</span>'


def make_user_clickable(name: str) -> str:
    """Make a username clickable, linking to the user's HuggingFace profile.

    Leaderboard cells are rendered as raw HTML, so the link and the name are both
    escaped. A long name is shortened, with the full name on hover.
    """
    link = f"https://huggingface.co/{name}" if not name.startswith("https://") else name
    return f'<a target="_blank" href="{html.escape(link)}" title="{html.escape(name)}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{html.escape(_shorten(name))}</a>'


def make_tag_clickable(tag: str) -> str:
    """Make a model report link clickable, if it is a valid URL.

    Leaderboard cells are rendered as raw HTML, so the URL is escaped; a quote in it
    can't break out of the ``href`` attribute.
    """
    if not tag or (tag == "") or (tag == "Not submitted"):
        return "Not submitted"
    if not tag.startswith(("http://", "https://")):
        return "Invalid link"
    return f'<a target="_blank" href="{html.escape(tag)}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">link</a>'


def hide_username_for_anonymous_entries(
    df: pd.DataFrame, escape_html: bool = True
) -> pd.DataFrame:
    """Replace usernames with aliases for anonymous entries.

    Aliases are free text, and on the leaderboard the username column is rendered as
    raw HTML, so by default they are escaped. Aliases are also restricted at
    submission (see ``submission.alias_error``); escaping covers entries submitted
    before that rule existed.

    Parameters
    ----------
    df : pd.DataFrame
        Leaderboard with ``username``, ``user_alias`` and ``anonymous`` columns.
    escape_html : bool
        HTML-escape the aliases. Pass False only for output that is never rendered
        as HTML, such as the downloadable CSV.

    Returns
    -------
    pd.DataFrame
        The leaderboard with aliases in place of anonymous usernames, and the
        ``user_alias`` and ``anonymous`` columns dropped.

    """
    aliases = df.loc[df["anonymous"], "user_alias"].astype(str)
    df.loc[df["anonymous"], "username"] = (
        aliases.map(html.escape) if escape_html else aliases
    )
    return df.drop(columns=["user_alias", "anonymous"], errors="ignore")


def _prepare_leaderboard_df(
    df: pd.DataFrame,
    spec: TrackSpec,
    metric_prefix: str = "",
    round_to: int | None = None,
) -> pd.DataFrame:
    """Order a leaderboard CSV by rank and give its scores their labels.

    Each score keeps its mean (``"<label>"``) and std (``"<label> std"``), at full
    precision unless ``round_to`` is given; the table and the download both start
    from this. ``rank``,
    ``username``, ``model_report_link`` and the anonymity columns keep their CSV
    names, for ``hide_username_for_anonymous_entries`` and ``_rename_display_columns``.
    A CSV written before a metric existed still works; that metric is just absent.

    Parameters
    ----------
    df : pd.DataFrame
        The leaderboard CSV as the backend writes it.
    spec : TrackSpec
        The leaderboard's track.
    metric_prefix : str
        Prefix for the score labels, e.g. "MA-".
    round_to : int | None
        Round every mean and std to this many significant figures.

    Returns
    -------
    pd.DataFrame
        The prepared leaderboard.

    """
    df = df.sort_values("rank", ascending=True).reset_index(drop=True)
    labels = score_labels(spec, metric_prefix)
    rename_map = {"submitted_at": "Submitted"}
    for base, label in labels.items():
        rename_map[f"{base}_mean"] = label
        rename_map[f"{base}_std"] = f"{label} std"
    df = df.rename(columns=rename_map)
    # Scores in the spec's order (primary metric first), each followed by its std,
    # whatever order the CSV has them in.
    score_cols = [
        col
        for label in labels.values()
        for col in (label, f"{label} std")
        if col in df.columns
    ]
    df = df[[c for c in df.columns if c not in score_cols] + score_cols]
    if round_to is not None:
        for col in score_cols:
            df[col] = df[col].map(lambda v: round_sig(v, round_to))
    df["Submitted"] = pd.to_datetime(
        df["Submitted"], utc=True, format="mixed"
    ).dt.strftime("%Y-%m-%d %H:%M UTC")
    df["model_report_link"] = (
        df["model_report_link"].astype("string").fillna("")
        if "model_report_link" in df.columns
        else ""
    )
    for flag, column in [
        ("used_proprietary_data", "Proprietary Data"),
        ("open_source_code", "Open Code"),
    ]:
        df[column] = (
            df[flag].fillna(False).map({True: "Yes", False: "No"})
            if flag in df.columns
            else "No"
        )
    df = df.drop(columns=["used_proprietary_data", "open_source_code"], errors="ignore")
    return sort_leaderboard_columns(df)


def _text(column: pd.Series) -> pd.Series:
    """Show each value as plain text, and a missing one as an empty cell."""
    return column.map(lambda v: "" if pd.isna(v) else str(v)).astype(object)


def _build_table(
    df: pd.DataFrame,
    spec: TrackSpec,
    metric_prefix: str = "",
    sig_figs: int = 3,
) -> LeaderboardTable:
    """Build the table a leaderboard shows from its prepared CSV.

    Scores sort by their mean and show "mean ± std". Username sorts, ignoring case,
    and is searched by the name shown (an anonymous entry's alias); it links to the
    HF profile unless the entry is anonymous; a long name is shortened, with the full
    name on hover. Columns the Space doesn't know about, such as the significance
    tiers, are kept for the column picker.

    Parameters
    ----------
    df : pd.DataFrame
        The leaderboard, from ``_prepare_leaderboard_df``.
    spec : TrackSpec
        The leaderboard's track.
    metric_prefix : str
        Prefix for the score labels, e.g. "MA-".
    sig_figs : int
        Significant figures for the scores shown.

    Returns
    -------
    LeaderboardTable
        The table.

    """
    values: dict[str, pd.Series] = {}
    display: dict[str, pd.Series] = {}
    datatypes: dict[str, DataType] = {}

    def add(name: str, value: pd.Series, text: pd.Series, datatype: DataType) -> None:
        values[name] = value
        display[name] = text
        datatypes[name] = datatype

    anonymous = df["anonymous"].fillna(False).astype(bool)
    alias = df["user_alias"].fillna("").astype(str)
    username = df["username"].astype(str)
    add("Rank", df["rank"], _text(df["rank"]), "number")
    add(
        "Username",
        username.where(~anonymous, alias).str.casefold(),
        username.map(make_user_clickable).where(~anonymous, alias.map(name_cell)),
        "html",
    )
    add("Submitted", df["Submitted"], df["Submitted"], "str")
    links = df["model_report_link"].astype(str)
    add("Model Report Link", links, links.map(make_tag_clickable), "html")

    labels = list(score_labels(spec, metric_prefix).values())
    for label in labels:
        if label not in df.columns:
            continue
        std = df.get(f"{label} std", pd.Series(None, index=df.index))
        text = pd.Series(
            [format_mean_std(m, s, sig_figs) for m, s in zip(df[label], std)],
            index=df.index,
            dtype=object,
        )
        add(label, df[label], text, "number")

    known = {
        "rank",
        "username",
        "user_alias",
        "anonymous",
        "model_report_link",
        "Submitted",
        *labels,
        *(f"{label} std" for label in labels),
        *FILTERS,
    }
    for column in df.columns:
        if column not in known:
            numeric = pd.api.types.is_numeric_dtype(df[column])
            add(column, df[column], _text(df[column]), "number" if numeric else "str")

    for column in FILTERS:
        add(column, df[column], df[column], "str")
    return LeaderboardTable(pd.DataFrame(values), pd.DataFrame(display), datatypes)


def _empty_table(spec: TrackSpec, metric_prefix: str = "") -> LeaderboardTable:
    """Return the empty table shown when a leaderboard can't be loaded."""
    labels = set(score_labels(spec, metric_prefix).values())
    columns = _default_columns(spec, metric_prefix)
    datatypes: dict[str, DataType] = {
        c: "number" if c == "Rank" or c in labels else "str" for c in columns
    }
    datatypes["Username"] = datatypes["Model Report Link"] = "html"
    empty = pd.DataFrame(columns=columns)
    return LeaderboardTable(empty, empty.copy(), datatypes)


def _load_leaderboard_csv(
    stage: str, spec: TrackSpec, endpoint_slug: str
) -> pd.DataFrame | None:
    """Read a leaderboard CSV from S3, or return None if it can't be read."""
    try:
        return _load_csv_from_s3(
            format_leaderboard_uri(
                track=spec.track, leaderboard_type=stage, endpoint_slug=endpoint_slug
            )
        )
    except Exception as exc:
        logger.warning(
            "Could not load {} {} leaderboard ({}): {}",
            stage,
            spec.track,
            endpoint_slug,
            exc,
        )
        return None


def load_leaderboard(
    stage: LeaderboardStage, spec: TrackSpec, endpoint_slug: str = MACRO_ENDPOINT_LABEL
) -> tuple[LeaderboardTable, str]:
    """Load one leaderboard from S3: a track's macro-averaged leaderboard, or a
    single endpoint's.

    Reads S3 once for both the table and the CSV download. Scores show to
    ``LEADERBOARD_SIG_FIGS[stage]`` significant figures; on ``ROUNDED_STAGES`` the
    values the table sorts by and the download are rounded too.

    Parameters
    ----------
    stage : LeaderboardStage
        "live", "interim" or "final".
    spec : TrackSpec
        The track.
    endpoint_slug : str
        One of ``spec.endpoints``, or ``MACRO_ENDPOINT_LABEL`` for the track's
        macro-averaged leaderboard.

    Returns
    -------
    tuple[LeaderboardTable, str]
        The table, and the path of the CSV download. Both empty if the leaderboard
        can't be read.

    """
    logger.info("Refreshing {} leaderboard ({})...", spec.track, endpoint_slug)
    metric_prefix = _metric_prefix(endpoint_slug)
    sig_figs = LEADERBOARD_SIG_FIGS[stage]
    df = _load_leaderboard_csv(stage, spec, endpoint_slug)
    if df is None:
        table = _empty_table(spec, metric_prefix)
        download = pd.DataFrame(columns=_default_columns(spec, metric_prefix))
    else:
        round_to = sig_figs if stage in ROUNDED_STAGES else None
        df = _prepare_leaderboard_df(df, spec, metric_prefix, round_to)
        table = _build_table(df, spec, metric_prefix, sig_figs)
        download = _download_df(df)
        logger.info("{} leaderboard loaded: {} entries.", spec.track, len(df))
    path = DOWNLOAD_DIR / f"{spec.track}_{endpoint_slug}_{stage}_leaderboard.csv"
    _write_csv(download, path)
    return table, str(path)


def _download_df(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a prepared leaderboard into its CSV download.

    Each score's mean and std stay in separate columns, so the file can be sorted
    by either in a spreadsheet. Anonymous entries show their alias, as text rather
    than HTML.
    """
    df = hide_username_for_anonymous_entries(df.copy(), escape_html=False)
    return _rename_display_columns(df)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV in one step, so a download never gets a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    ) as f:
        df.to_csv(f, index=False)
    os.replace(f.name, path)


def render_interim_leaderboards():
    """Render the interim leaderboards for the Gradio app."""
    gr.Markdown("# Interim Leaderboards")
    _add_leaderboard_tabs(leaderboard_type="interim")


def render_final_leaderboards(phase: int):
    """Render the final leaderboards for the Gradio app."""
    gr.Markdown("# Final Leaderboards")
    if phase == 3:
        gr.Markdown(
            "The final leaderboard is being prepared and will be available shortly. "
            "Thank you for your patience!"
        )
    else:
        _add_leaderboard_tabs(leaderboard_type="final")


def render_live_leaderboards(phase: int, demo: gr.Blocks):
    """Render the live leaderboards for the Gradio app."""
    if phase == 0:
        gr.Markdown("# Live Leaderboards (Coming Soon)")
        gr.Markdown(
            "The challenge has not yet started. Leaderboards will be available once the"
            " challenge begins. Please check back once the challenge opens."
        )
    else:
        gr.Markdown("# Live Leaderboards")
        _add_leaderboard_tabs(leaderboard_type="live", demo=demo)


def _add_leaderboard(
    stage: LeaderboardStage, spec: TrackSpec, endpoint_slug: str
) -> tuple[LeaderboardTableView, gr.DownloadButton]:
    """Render one leaderboard: its download button and table.

    The button holds the CSV already, so one click downloads it.
    """
    table, csv_path = load_leaderboard(stage, spec, endpoint_slug)
    download = gr.DownloadButton(label="Download CSV", value=csv_path, size="sm")
    view = render_leaderboard_table(
        table,
        filters=FILTERS,
        default_columns=_default_columns(spec, _metric_prefix(endpoint_slug)),
    )
    return view, download


def _add_track_tabs(
    stage: LeaderboardStage, spec: TrackSpec
) -> dict[str, tuple[LeaderboardTableView, gr.DownloadButton]]:
    """Render a track's leaderboards.

    With more than one endpoint, a "🏆 Overall" (macro-averaged) tab plus one tab per
    endpoint. With one endpoint, just its leaderboard: an Overall tab would be the
    same.

    Parameters
    ----------
    stage : LeaderboardStage
        "live", "interim" or "final".
    spec : TrackSpec
        The track.

    Returns
    -------
    dict[str, tuple[LeaderboardTableView, gr.DownloadButton]]
        Endpoint slug (or ``MACRO_ENDPOINT_LABEL``) to its table and download
        button.

    """
    if len(spec.endpoints) == 1:
        slug = spec.endpoints[0]
        return {slug: _add_leaderboard(stage, spec, slug)}
    views: dict[str, tuple[LeaderboardTableView, gr.DownloadButton]] = {}
    with gr.Tabs():
        for slug in [MACRO_ENDPOINT_LABEL, *spec.endpoints]:
            title = (
                "🏆 Overall" if slug == MACRO_ENDPOINT_LABEL else slug.replace("_", " ")
            )
            with gr.TabItem(title):
                views[slug] = _add_leaderboard(stage, spec, slug)
    return views


def _add_leaderboard_tabs(
    leaderboard_type: LeaderboardStage,
    demo: gr.Blocks | None = None,
    visibility: dict[str, TrackVisibility] | None = None,
):
    """Add a tab per visible track to a leaderboard stage.

    A track gets a tab if its visibility shows it on ``leaderboard_type`` and it
    has at least one endpoint.

    Parameters
    ----------
    leaderboard_type : LeaderboardStage
        "live", "interim" or "final". Interim and final are static; live refreshes
        every ``REFRESH_SECONDS`` and on every page load.
    demo : gr.Blocks | None
        The app's Blocks instance, needed to wire the page-load refresh. Required
        when ``leaderboard_type == "live"``.
    visibility : dict[str, TrackVisibility] | None
        Where each track is shown. Defaults to ``config.TRACK_VISIBILITY``.

    Raises
    ------
    ValueError
        If ``leaderboard_type`` is "live" and ``demo`` is None.

    """
    visibility = TRACK_VISIBILITY if visibility is None else visibility
    live = leaderboard_type == "live"
    if live and demo is None:
        raise ValueError("demo is required to wire live leaderboards")
    timer = gr.Timer(value=REFRESH_SECONDS) if live else None

    with gr.Tabs():
        for spec in TRACK_SPECS.values():
            if not spec.endpoints or not visibility[spec.track].on_leaderboard(
                leaderboard_type
            ):
                continue
            with gr.TabItem(TRACK_NAMES[spec.track]):
                views = _add_track_tabs(leaderboard_type, spec)
            if timer is not None and demo is not None:
                for slug, (view, download) in views.items():
                    view.refresh_on(
                        [demo.load, timer.tick],
                        partial(load_leaderboard, leaderboard_type, spec, slug),
                        also=[download],
                    )
