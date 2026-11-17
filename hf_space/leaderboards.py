"""Load the leaderboards from S3 and show them in the Space.

Each leaderboard CSV the backend writes has the same layout for every track: rank,
user and submission details, and a ``<metric>_mean`` and ``<metric>_std`` column per
metric. Which metrics, and in what order, comes from the track's ``TrackSpec``. The
table itself (sorting, search, filters, column picker) is in ``leaderboard_table``.
"""

import html
import os
import tempfile
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import gradio as gr
import pandas as pd
from config import (
    LEADERBOARD_SIG_FIGS,
    LEADERBOARD_URI_FORMAT,
    MACRO_ENDPOINT_LABEL,
    METRIC_DISPLAY_NAMES,
    METRIC_HIGHER_IS_BETTER,
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
# Tier labels are "Tier 1", "Tier 2", ...; the column sorts by the number.
TIER_COLUMN = "Significance (tiers)"
# Row shading: every other significance tier on interim and final leaderboards, and
# the top three on the live leaderboard (which has no tiers). Mixed from theme
# colours, so they suit light and dark mode.
TIER_SHADE = "background: color-mix(in srgb, var(--color-accent) 7%, transparent);"
TOP_THREE_SHADES = {
    1: "background: color-mix(in srgb, #f5b301 22%, transparent);",
    2: "background: color-mix(in srgb, #8b9bb4 38%, transparent);",
    3: "background: color-mix(in srgb, #c2773a 20%, transparent);",
}
PRIMARY_METRIC_STYLE = "font-weight: 700;"
FLAG_TEXT = {"Yes": "✓", "No": "–"}
STAGE_NOTES = {
    "live": (
        "Scores are mean ± std over bootstrap samples, to 2 significant figures. "
        "Ranked by the **bold** column; ↑/↓ shows which direction is better. Sort "
        "from the ⋮ menu in a column header. Refreshes every {refresh} seconds."
    ),
    "interim": (
        "Scores are mean ± std over bootstrap samples. Ranked by the **bold** "
        "column; ↑/↓ shows which direction is better. Shading alternates between "
        "significance tiers. Sort from the ⋮ menu in a column header."
    ),
}
STAGE_NOTES["final"] = STAGE_NOTES["interim"]
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

    label_of = score_labels(spec, metric_prefix)
    labels = list(label_of.values())
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
    if TIER_COLUMN in df.columns:
        add(
            TIER_COLUMN,
            _tier_numbers(df[TIER_COLUMN]),
            _text(df[TIER_COLUMN]),
            "number",
        )
        known.add(TIER_COLUMN)
    for column in df.columns:
        if column not in known:
            numeric = pd.api.types.is_numeric_dtype(df[column])
            add(column, df[column], _text(df[column]), "number" if numeric else "str")

    for column in FILTERS:
        add(column, df[column], df[column].map(FLAG_TEXT).fillna(df[column]), "str")

    headers = {}
    for base, label in label_of.items():
        higher = (
            spec.higher_is_better
            if base == spec.primary_metric
            else METRIC_HIGHER_IS_BETTER.get(base)
        )
        if higher is not None and label in values:
            headers[label] = f"{label} {'↑' if higher else '↓'}"

    styling = pd.DataFrame("", index=df.index, columns=list(values))
    row_styles = _row_styles(df)
    for column in styling.columns:
        styling[column] = row_styles
    primary = label_of.get(spec.primary_metric)
    if primary in styling.columns:
        styling[primary] = styling[primary] + PRIMARY_METRIC_STYLE
    return LeaderboardTable(
        pd.DataFrame(values), pd.DataFrame(display), datatypes, headers, styling
    )


def _tier_numbers(labels: pd.Series) -> pd.Series:
    """Read the number from each tier label ("Tier 2" -> 2); NaN if there isn't one."""
    return pd.to_numeric(
        labels.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )


def _row_styles(df: pd.DataFrame) -> pd.Series:
    """Shade every other significance tier, or the top three rows if there are none."""
    if TIER_COLUMN in df.columns:
        tiers = _tier_numbers(df[TIER_COLUMN])
        return tiers.map(lambda t: TIER_SHADE if pd.notna(t) and t % 2 == 1 else "")
    return df["rank"].map(lambda r: TOP_THREE_SHADES.get(r, ""))


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
    gr.Markdown("## Interim Leaderboards")
    _add_leaderboard_tabs(leaderboard_type="interim")


def render_final_leaderboards(phase: int):
    """Render the final leaderboards for the Gradio app."""
    gr.Markdown("## Final Leaderboards")
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
        gr.Markdown("## Live Leaderboards (Coming Soon)")
        gr.Markdown(
            "The challenge has not yet started. Leaderboards will be available once the"
            " challenge begins. Please check back once the challenge opens."
        )
    else:
        gr.Markdown("## Live Leaderboards")
        _add_leaderboard_tabs(leaderboard_type="live", demo=demo)


def _last_checked() -> str:
    """Return the "last checked" note under a live leaderboard."""
    return f"Last checked {datetime.now(timezone.utc):%H:%M} UTC"


def _load_live(
    stage: LeaderboardStage, spec: TrackSpec, endpoint_slug: str
) -> tuple[LeaderboardTable, str, str]:
    """Load a live leaderboard with its download and "last checked" note."""
    table, csv_path = load_leaderboard(stage, spec, endpoint_slug)
    return table, csv_path, _last_checked()


def _add_leaderboard(
    stage: LeaderboardStage,
    spec: TrackSpec,
    endpoint_slug: str,
    table: LeaderboardTable,
    download: gr.DownloadButton,
    refresh_on: list | None = None,
) -> LeaderboardTableView:
    """Render one leaderboard's toolbar and table.

    Parameters
    ----------
    stage : LeaderboardStage
        "live", "interim" or "final".
    spec : TrackSpec
        The track.
    endpoint_slug : str
        One of ``spec.endpoints``, or ``MACRO_ENDPOINT_LABEL``.
    table : LeaderboardTable
        The leaderboard, already loaded.
    download : gr.DownloadButton
        Its CSV download button, rendered above with the endpoint picker.
    refresh_on : list | None
        Events to reload the leaderboard on, with a "last checked" note under it,
        or None for a leaderboard loaded once.

    Returns
    -------
    LeaderboardTableView
        The rendered table.

    """
    view = render_leaderboard_table(
        table,
        filters=FILTERS,
        # The tier column shows by default where the leaderboard has one (interim
        # and final), so the shaded bands have their labels.
        default_columns=[
            *_default_columns(spec, _metric_prefix(endpoint_slug)),
            TIER_COLUMN,
        ],
        download=download,
    )
    if refresh_on is not None:
        note = gr.Markdown(_last_checked(), elem_classes="leaderboard-note")
        view.refresh_on(
            refresh_on, partial(_load_live, stage, spec, endpoint_slug), also=[note]
        )
    return view


def _endpoint_label(slug: str) -> str:
    """Name an endpoint in the endpoint picker."""
    if slug == MACRO_ENDPOINT_LABEL:
        return "Overall (macro average)"
    return slug.replace("_", " ")


def _add_track(
    stage: LeaderboardStage, spec: TrackSpec, refresh_on: list | None = None
) -> dict[str, LeaderboardTableView]:
    """Render a track's leaderboards.

    A row of controls sits above the table: with more than one endpoint, an
    endpoint picker choosing between the macro-averaged leaderboard ("Overall") and
    each endpoint's own, then the chosen leaderboard's Download CSV button. With
    one endpoint, just its leaderboard and button: an Overall one would be the same.

    Parameters
    ----------
    stage : LeaderboardStage
        "live", "interim" or "final".
    spec : TrackSpec
        The track.
    refresh_on : list | None
        Events to reload the leaderboards on, or None to load them once.

    Returns
    -------
    dict[str, LeaderboardTableView]
        Endpoint slug (or ``MACRO_ENDPOINT_LABEL``) to its rendered table.

    """
    slugs = (
        [MACRO_ENDPOINT_LABEL, *spec.endpoints]
        if len(spec.endpoints) > 1
        else list(spec.endpoints)
    )
    loaded = {slug: load_leaderboard(stage, spec, slug) for slug in slugs}
    first = slugs[0]
    with gr.Row(equal_height=True, elem_classes="track-controls"):
        picker = (
            gr.Dropdown(
                choices=[(_endpoint_label(s), s) for s in slugs],
                value=first,
                label="Endpoint",
                scale=0,
                min_width=280,
            )
            if len(slugs) > 1
            else None
        )
        downloads = [
            gr.DownloadButton(
                "⬇ Download CSV",
                value=loaded[slug][1],
                visible=slug == first,
                variant="secondary",
                scale=0,
                min_width=220,
                elem_classes="download-button",
            )
            for slug in slugs
        ]
    views: dict[str, LeaderboardTableView] = {}
    groups: list[gr.Column] = []
    for slug, download in zip(slugs, downloads):
        with gr.Column(visible=slug == first) as group:
            views[slug] = _add_leaderboard(
                stage, spec, slug, loaded[slug][0], download, refresh_on
            )
        groups.append(group)
    if picker is not None:
        picker.change(
            lambda chosen: [gr.update(visible=s == chosen) for s in slugs] * 2,
            inputs=picker,
            outputs=[*groups, *downloads],
        )
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
    refresh_on = None
    if leaderboard_type == "live":
        if demo is None:
            raise ValueError("demo is required to wire live leaderboards")
        refresh_on = [demo.load, gr.Timer(value=REFRESH_SECONDS).tick]

    gr.Markdown(
        STAGE_NOTES[leaderboard_type].format(refresh=REFRESH_SECONDS),
        elem_classes="leaderboard-note",
    )
    with gr.Tabs():
        for spec in TRACK_SPECS.values():
            if not spec.endpoints or not visibility[spec.track].on_leaderboard(
                leaderboard_type
            ):
                continue
            with gr.TabItem(TRACK_NAMES[spec.track]):
                _add_track(leaderboard_type, spec, refresh_on)
