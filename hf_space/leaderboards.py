"""Functions to load and display leaderboards for the blind challenge."""

import tempfile
from functools import partial

import gradio as gr
import pandas as pd
from config import (
    ACTIVITY_METRIC_DISPLAY_NAMES,
    CLASSIFICATION_ENDPOINTS,
    CLASSIFICATION_METRIC_DISPLAY_NAMES,
    CLASSIFICATION_NAME,
    CLASSIFICATION_TRACK,
    LEADERBOARD_URI_FORMAT,
    MACRO_ENDPOINT_LABEL,
    REGRESSION_ENDPOINTS,
    REGRESSION_NAME,
    REGRESSION_TRACK,
    STRUCTURE_ENDPOINTS,
    STRUCTURE_NAME,
    STRUCTURE_TRACK,
    STRUCTURE_TRACK_LIVE,
)
from gradio_leaderboard import ColumnFilter, Leaderboard
from loguru import logger
from utils import _load_csv_from_s3

_STRUCTURE_EMPTY = pd.DataFrame(
    columns=[
        "Rank",
        "Username",
        "Submitted",
        "Model Report Link",
        "Proprietary Data",
        "Open Code",
        "LDDT-PLI",
        "BiSyRMSD",
        "LDDT-LP",
        "Coverage",
    ]
)


def _activity_empty_df(
    metric_display_names: dict[str, str], metric_prefix: str = ""
) -> pd.DataFrame:
    """Create a placeholder activity-style leaderboard shown when S3 loading fails."""
    return pd.DataFrame(
        columns=[
            "Rank",
            "Username",
            "Submitted",
            "Model Report Link",
            "Proprietary Data",
            "Open Code",
            *[f"{metric_prefix}{m}" for m in metric_display_names.values()],
        ]
    )


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
        REGRESSION_TRACK, CLASSIFICATION_TRACK, or STRUCTURE_TRACK.
    leaderboard_type : str
        "live", "interim", or "final".
    endpoint_slug : str
        For regression/classification, a REGRESSION_ENDPOINTS /
        CLASSIFICATION_ENDPOINTS entry (e.g. "ENDPOINT_1") to load that endpoint's
        own leaderboard instead of the track's master (macro-ranked) leaderboard.
        Defaults to ``MACRO_ENDPOINT_LABEL`` ("MA"), which loads the master.
    version : str
        "latest" (default), or an ISO-format timestamp for a dated snapshot.

    """
    if track not in [REGRESSION_TRACK, CLASSIFICATION_TRACK, STRUCTURE_TRACK]:
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


def make_user_clickable(name: str) -> str:
    """Make a username clickable, linking to the user's HuggingFace profile."""
    link = f"https://huggingface.co/{name}" if not name.startswith("https://") else name
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{name}</a>'


def make_tag_clickable(tag: str) -> str:
    """Make a model report link clickable, if it is a valid URL."""
    if not tag or (tag == "") or (tag == "Not submitted"):
        return "Not submitted"
    if not tag.startswith(("http://", "https://")):
        return "Invalid link"
    return f'<a target="_blank" href="{tag}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">link</a>'


def hide_username_for_anonymous_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Replace usernames with aliases for anonymous entries."""
    df.loc[df["anonymous"], "username"] = df.loc[df["anonymous"], "user_alias"]
    return df.drop(columns=["user_alias", "anonymous"], errors="ignore")


def _prepare_activity_df(
    df: pd.DataFrame,
    metric_display_names: dict[str, str],
    metric_prefix: str = "",
    for_download: bool = False,
) -> pd.DataFrame:
    """Sort and rename an activity-style (regression or classification) leaderboard's
    columns (no HTML).

    Every activity-style leaderboard CSV — a track's master (macro-ranked)
    leaderboard, or any single endpoint's own leaderboard — has the same bare
    metric-column schema, since the backend narrows down to one endpoint's columns
    before saving (see ``aws_leaderboards._narrow_averaged_results_to_endpoint``). So
    this one function renders all of them for both tracks; ``metric_display_names``
    (regression's ST-RAE/MAE/R2/Spearman/Kendall or classification's
    MCC/Accuracy/Precision/Recall/F1) and ``metric_prefix`` ("MA-" for a track's
    master, "" for endpoint tabs whose own tab title already identifies the endpoint)
    are the only things that differ between calls.

    Parameters
    ----------
    df : pd.DataFrame
        Raw leaderboard DataFrame from S3.
    metric_display_names : dict[str, str]
        Raw metric name -> display label, in display order (primary/sort metric
        first).
    metric_prefix : str
        Prefix for the metric column headers, e.g. "MA-".
    for_download : bool
        If True, keep mean and std as separate columns with full precision for the
        downloadable CSV. If False (default), drop std columns and round means to
        4 dp for the live leaderboard.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame.

    """
    df = df.sort_values("rank", ascending=True).reset_index(drop=True)
    rename_map = {"submitted_at": "Submitted"}
    for raw_name, display_name in metric_display_names.items():
        rename_map[f"{raw_name}_mean"] = f"{metric_prefix}{display_name}"
        rename_map[f"{raw_name}_std"] = f"{metric_prefix}{display_name} std"
    df = df.rename(columns=rename_map)
    metric_cols = [f"{metric_prefix}{d}" for d in metric_display_names.values()]
    if for_download:
        pass  # keep mean and std columns as-is with full precision
    else:
        std_cols = [c for c in df.columns if c.endswith(" std")]
        df = df.drop(columns=std_cols)
        for col in metric_cols:
            if col in df.columns:
                df[col] = df[col].round(4)
    # The source file's physical column order doesn't necessarily match
    # metric_display_names' canonical order (primary/sort metric first) — force it,
    # since sort_leaderboard_columns/gradio_leaderboard's select_columns both preserve
    # whatever order the DataFrame's columns already have rather than reordering them.
    ordered_metric_cols = []
    for col in metric_cols:
        if col in df.columns:
            ordered_metric_cols.append(col)
        std_col = f"{col} std"
        if std_col in df.columns:
            ordered_metric_cols.append(std_col)
    other_cols = [c for c in df.columns if c not in ordered_metric_cols]
    df = df[other_cols + ordered_metric_cols]
    df["Submitted"] = pd.to_datetime(df["Submitted"], utc=True).dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    df["Proprietary Data"] = (
        df["used_proprietary_data"].fillna(False).map({True: "Yes", False: "No"})
        if "used_proprietary_data" in df.columns
        else "No"
    )
    df["Open Code"] = (
        df["open_source_code"].fillna(False).map({True: "Yes", False: "No"})
        if "open_source_code" in df.columns
        else "No"
    )
    df = df.drop(columns=["used_proprietary_data", "open_source_code"], errors="ignore")
    return sort_leaderboard_columns(df)


def _prepare_structure_df(df: pd.DataFrame, for_download: bool = False) -> pd.DataFrame:
    """Sort and rename structure leaderboard columns (no HTML).

    Parameters
    ----------
    df : pd.DataFrame
        Raw leaderboard DataFrame from S3.
    for_download : bool
        If True, collapse mean/std pairs into 'XX±YY' strings for the downloadable
        CSV. If False (default), keep mean values as plain floats for numeric
        sorting in the live leaderboard.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame.

    """
    df = df.sort_values("LDDT-PLI_mean", ascending=False).reset_index(drop=True)
    rename_map = {
        "LDDT-PLI_mean": "LDDT-PLI",
        "BiSyRMSD_mean": "BiSyRMSD",
        "LDDT-LP_mean": "LDDT-LP",
        "Ligand_RMSD_mean": "Ligand RMSD",
        "coverage_mean": "Coverage",
        "LDDT-PLI_std": "LDDT-PLI std",
        "BiSyRMSD_std": "BiSyRMSD std",
        "LDDT-LP_std": "LDDT-LP std",
        "Ligand_RMSD_std": "Ligand RMSD std",
        "coverage_std": "Coverage std",
    }
    df = df.rename(columns=rename_map)
    if for_download:
        pass  # keep mean and std columns as-is with full precision
    else:
        std_cols = [c for c in df.columns if c.endswith(" std")]
        df = df.drop(columns=std_cols)
        for col in ["LDDT-PLI", "BiSyRMSD", "LDDT-LP", "Ligand RMSD", "Coverage"]:
            if col in df.columns:
                df[col] = df[col].round(4)
    df = df.rename(columns={"submitted_at": "Submitted"})
    df["Submitted"] = pd.to_datetime(df["Submitted"], utc=True).dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    df["model_report_link"] = df["model_report_link"].fillna("")
    df["Proprietary Data"] = (
        df["used_proprietary_data"].fillna(False).map({True: "Yes", False: "No"})
        if "used_proprietary_data" in df.columns
        else "No"
    )
    df["Open Code"] = (
        df["open_source_code"].fillna(False).map({True: "Yes", False: "No"})
        if "open_source_code" in df.columns
        else "No"
    )
    df = df.drop(columns=["used_proprietary_data", "open_source_code"], errors="ignore")
    return sort_leaderboard_columns(df)


def load_activity_leaderboard(
    leaderboard_type: str,
    track: str,
    metric_display_names: dict[str, str],
    endpoint_slug: str = MACRO_ENDPOINT_LABEL,
    metric_prefix: str = "",
) -> pd.DataFrame:
    """Load a regression or classification leaderboard from S3 — a track's master, or
    a single endpoint's own.
    """
    logger.info("Refreshing {} leaderboard ({})...", track, endpoint_slug)
    try:
        df = _load_csv_from_s3(
            format_leaderboard_uri(
                track=track,
                leaderboard_type=leaderboard_type,
                endpoint_slug=endpoint_slug,
            )
        )
    except Exception as exc:
        logger.warning("Could not load {} leaderboard: {}", track, exc)
        return _activity_empty_df(metric_display_names, metric_prefix)
    df = _prepare_activity_df(
        df, metric_display_names=metric_display_names, metric_prefix=metric_prefix
    )
    df["username"] = df["username"].map(make_user_clickable)
    df = hide_username_for_anonymous_entries(df)
    # Column was being loaded as a float, convert to str first
    df["model_report_link"] = df["model_report_link"].astype("string").fillna("")
    df["model_report_link"] = df["model_report_link"].map(make_tag_clickable)
    df = _rename_display_columns(df)
    logger.info("{} leaderboard loaded: {} entries.", track, len(df))
    return df


def load_structure_leaderboard(
    leaderboard_type: str = "live",
    track: str = STRUCTURE_TRACK,
    endpoint_slug: str = STRUCTURE_ENDPOINTS[0],
) -> pd.DataFrame:
    """Load a structure leaderboard from S3 — the track's master, or a single
    endpoint's own (today, STRUCTURE_ENDPOINTS has exactly one entry, so there is no
    master leaderboard file yet — see _add_structure_track_tabs).
    """
    logger.info("Refreshing structure leaderboard ({})...", endpoint_slug)
    try:
        df = _load_csv_from_s3(
            format_leaderboard_uri(
                track=track,
                leaderboard_type=leaderboard_type,
                endpoint_slug=endpoint_slug,
            )
        )
    except Exception as exc:
        logger.warning("Could not load structure leaderboard: {}", exc)
        return _STRUCTURE_EMPTY
    df = _prepare_structure_df(df)
    df["username"] = df["username"].map(make_user_clickable)
    df = hide_username_for_anonymous_entries(df)
    df["model_report_link"] = df["model_report_link"].map(make_tag_clickable)
    df = _rename_display_columns(df)
    logger.info("Structure leaderboard loaded: {} entries.", len(df))
    return df


def download_activity_leaderboard(
    leaderboard_type: str,
    track: str,
    metric_display_names: dict[str, str],
    endpoint_slug: str = MACRO_ENDPOINT_LABEL,
    metric_prefix: str = "",
) -> str:
    """Write a regression or classification leaderboard to a temp CSV and return the
    file path.
    """
    try:
        df = _load_csv_from_s3(
            format_leaderboard_uri(
                track=track,
                leaderboard_type=leaderboard_type,
                endpoint_slug=endpoint_slug,
            )
        )
    except Exception as exc:
        logger.warning("Could not load {} leaderboard for download: {}", track, exc)
        df = _activity_empty_df(metric_display_names, metric_prefix)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f"{track}_leaderboard_", delete=False
        ) as f:
            df.to_csv(f, index=False)
            return f.name
    df = _prepare_activity_df(
        df,
        metric_display_names=metric_display_names,
        metric_prefix=metric_prefix,
        for_download=True,
    )
    df = hide_username_for_anonymous_entries(df)
    df = _rename_display_columns(df)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f"{track}_leaderboard_", delete=False
    ) as f:
        df.to_csv(f, index=False)
        return f.name


def download_structure_leaderboard(
    leaderboard_type: str = "live",
    track: str = STRUCTURE_TRACK,
    endpoint_slug: str = STRUCTURE_ENDPOINTS[0],
) -> str:
    """Write the structure leaderboard to a temp CSV and return the file path."""
    try:
        df = _load_csv_from_s3(
            format_leaderboard_uri(
                track=track,
                leaderboard_type=leaderboard_type,
                endpoint_slug=endpoint_slug,
            )
        )
    except Exception as exc:
        logger.warning("Could not load structure leaderboard for download: {}", exc)
        df = _STRUCTURE_EMPTY
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="structure_leaderboard_", delete=False
        ) as f:
            df.to_csv(f, index=False)
            return f.name
    df = _prepare_structure_df(df, for_download=True)
    df = hide_username_for_anonymous_entries(df)
    df = _rename_display_columns(df)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="structure_leaderboard_", delete=False
    ) as f:
        df.to_csv(f, index=False)
        return f.name


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


def _add_activity_leaderboard(
    leaderboard_type: str,
    track: str,
    endpoint_slug: str,
    metric_display_names: dict[str, str],
    metric_prefix: str,
) -> Leaderboard:
    """Render one regression/classification leaderboard variant: download button,
    table, and filters.

    Identical for a track's master (macro-ranked) leaderboard and every endpoint's own
    leaderboard — only which S3 file is loaded (``endpoint_slug``) and how the metric
    columns are labelled (``metric_prefix``) differ.
    """
    download_btn = gr.DownloadButton(label="Download CSV", size="sm")
    download_btn.click(
        fn=partial(
            download_activity_leaderboard,
            leaderboard_type=leaderboard_type,
            track=track,
            metric_display_names=metric_display_names,
            endpoint_slug=endpoint_slug,
            metric_prefix=metric_prefix,
        ),
        outputs=download_btn,
    )
    metric_cols = [f"{metric_prefix}{d}" for d in metric_display_names.values()]
    return Leaderboard(
        value=load_activity_leaderboard(
            leaderboard_type=leaderboard_type,
            track=track,
            metric_display_names=metric_display_names,
            endpoint_slug=endpoint_slug,
            metric_prefix=metric_prefix,
        ),
        select_columns=[
            "Rank",
            "Username",
            "Submitted",
            "Model Report Link",
            "Proprietary Data",
            "Open Code",
            *metric_cols,
        ],
        search_columns=["Username"],
        filter_columns=[
            ColumnFilter(
                "Proprietary Data",
                type="checkboxgroup",
                label="Proprietary Data",
                choices=[("Yes", "Yes"), ("No", "No")],
                default=[("Yes", "Yes"), ("No", "No")],
            ),
            ColumnFilter(
                "Open Code",
                type="checkboxgroup",
                label="Open Code",
                choices=[("Yes", "Yes"), ("No", "No")],
                default=[("Yes", "Yes"), ("No", "No")],
            ),
        ],
        datatype=["number", "html", "str", "html", "str", "str"]
        + ["number"] * len(metric_cols),
    )


def _add_structure_leaderboard(
    leaderboard_type: str, track: str, endpoint_slug: str
) -> Leaderboard:
    """Render one structure leaderboard variant: download button, table, and filters.

    Same layout/columns regardless of ``endpoint_slug`` — mirrors
    ``_add_activity_leaderboard`` for the structure track's fixed metric set.
    """
    download_btn = gr.DownloadButton(label="Download CSV", size="sm")
    download_btn.click(
        fn=partial(
            download_structure_leaderboard,
            leaderboard_type=leaderboard_type,
            track=track,
            endpoint_slug=endpoint_slug,
        ),
        outputs=download_btn,
    )
    return Leaderboard(
        value=load_structure_leaderboard(
            leaderboard_type=leaderboard_type, track=track, endpoint_slug=endpoint_slug
        ),
        select_columns=[
            "Rank",
            "Username",
            "Submitted",
            "Model Report Link",
            "Proprietary Data",
            "Open Code",
            "LDDT-PLI",
            "BiSyRMSD",
            "LDDT-LP",
            "Coverage",
        ],
        search_columns=["Username"],
        filter_columns=[
            ColumnFilter(
                "Proprietary Data",
                type="checkboxgroup",
                label="Proprietary Data",
                choices=[("Yes", "Yes"), ("No", "No")],
                default=[("Yes", "Yes"), ("No", "No")],
            ),
            ColumnFilter(
                "Open Code",
                type="checkboxgroup",
                label="Open Code",
                choices=[("Yes", "Yes"), ("No", "No")],
                default=[("Yes", "Yes"), ("No", "No")],
            ),
        ],
        datatype=["number", "html", "str", "html", "str", "str"] + ["number"] * 4,
    )


def _add_activity_track_tabs(
    leaderboard_type: str,
    track: str,
    endpoints: list[str],
    metric_display_names: dict[str, str],
) -> dict[str, Leaderboard]:
    """Render a regression/classification track's leaderboard(s).

    With more than one endpoint, renders a "🏆 Overall" (macro-averaged, "MA-"
    prefixed) tab plus one tab per endpoint. With exactly one endpoint, renders just
    that endpoint's leaderboard directly — no wrapper tabs, no separate "Overall"
    tab (it would be identical to the only endpoint's own leaderboard).
    """
    lbs: dict[str, Leaderboard] = {}
    if len(endpoints) > 1:
        with gr.Tabs():
            with gr.TabItem("🏆 Overall"):
                lbs[MACRO_ENDPOINT_LABEL] = _add_activity_leaderboard(
                    leaderboard_type,
                    track,
                    MACRO_ENDPOINT_LABEL,
                    metric_display_names,
                    metric_prefix="MA-",
                )
            for endpoint in endpoints:
                with gr.TabItem(endpoint.replace("_", " ")):
                    lbs[endpoint] = _add_activity_leaderboard(
                        leaderboard_type,
                        track,
                        endpoint,
                        metric_display_names,
                        metric_prefix="",
                    )
    else:
        lbs[endpoints[0]] = _add_activity_leaderboard(
            leaderboard_type,
            track,
            endpoints[0],
            metric_display_names,
            metric_prefix="",
        )
    return lbs


def _add_structure_track_tabs(
    leaderboard_type: str, track: str, endpoints: list[str]
) -> dict[str, Leaderboard]:
    """Render the structure track's leaderboard(s) — same Overall/per-endpoint split
    as ``_add_activity_track_tabs``, for consistency if structure ever gains a second
    endpoint. Today STRUCTURE_ENDPOINTS has exactly one entry, so this always takes
    the single-leaderboard branch.
    """
    lbs: dict[str, Leaderboard] = {}
    if len(endpoints) > 1:
        with gr.Tabs():
            with gr.TabItem("🏆 Overall"):
                lbs[MACRO_ENDPOINT_LABEL] = _add_structure_leaderboard(
                    leaderboard_type, track, MACRO_ENDPOINT_LABEL
                )
            for endpoint in endpoints:
                with gr.TabItem(endpoint.replace("_", " ")):
                    lbs[endpoint] = _add_structure_leaderboard(
                        leaderboard_type, track, endpoint
                    )
    else:
        lbs[endpoints[0]] = _add_structure_leaderboard(
            leaderboard_type, track, endpoints[0]
        )
    return lbs


def _wire_activity_timer(
    timer: gr.Timer,
    demo: gr.Blocks,
    leaderboard_type: str,
    track: str,
    lbs: dict[str, Leaderboard],
    metric_display_names: dict[str, str],
) -> None:
    """Wire a live-refresh timer to every leaderboard component of one
    regression/classification track.

    Also fires on ``demo.load`` (i.e. whenever a browser session opens the page), so a
    visitor sees fresh data immediately rather than the snapshot baked in when the
    Blocks graph was built, and doesn't have to wait for the first timer tick.
    """
    for endpoint_slug, lb in lbs.items():
        metric_prefix = "MA-" if endpoint_slug == MACRO_ENDPOINT_LABEL else ""
        gr.on(
            triggers=[demo.load, timer.tick],
            fn=partial(
                load_activity_leaderboard,
                leaderboard_type=leaderboard_type,
                track=track,
                metric_display_names=metric_display_names,
                endpoint_slug=endpoint_slug,
                metric_prefix=metric_prefix,
            ),
            outputs=[lb],
        )


def _wire_structure_timer(
    timer: gr.Timer,
    demo: gr.Blocks,
    leaderboard_type: str,
    track: str,
    lbs: dict[str, Leaderboard],
) -> None:
    """Wire a live-refresh timer to every leaderboard component of the structure track.

    Also fires on ``demo.load`` — see ``_wire_activity_timer``.
    """
    for endpoint_slug, lb in lbs.items():
        gr.on(
            triggers=[demo.load, timer.tick],
            fn=partial(
                load_structure_leaderboard,
                leaderboard_type=leaderboard_type,
                track=track,
                endpoint_slug=endpoint_slug,
            ),
            outputs=[lb],
        )


def _add_leaderboard_tabs(leaderboard_type: str, demo: gr.Blocks | None = None):
    """Add the regression, classification, and structure leaderboard tabs to the
    Gradio app.

    Parameters
    ----------
    leaderboard_type : str
        The leaderboard_type of the leaderboard to display. One of "live",
        "interim", or "final". Both "interim" and "final" are static leaderboards,
        while "live" is updated every 30 seconds and on every page load.
    demo : gr.Blocks | None
        The app's Blocks instance, needed to wire the page-load refresh. Required
        when ``leaderboard_type == "live"``.

    """
    if leaderboard_type == "live":
        regression_timer = gr.Timer(value=30)
        classification_timer = gr.Timer(value=30)
        structure_timer = gr.Timer(value=30)

    with gr.Tabs():
        regression_lbs: dict[str, Leaderboard] = {}
        if len(REGRESSION_ENDPOINTS) > 0:
            with gr.TabItem(REGRESSION_NAME):
                regression_lbs = _add_activity_track_tabs(
                    leaderboard_type,
                    REGRESSION_TRACK,
                    REGRESSION_ENDPOINTS,
                    ACTIVITY_METRIC_DISPLAY_NAMES,
                )
        classification_lbs: dict[str, Leaderboard] = {}
        if len(CLASSIFICATION_ENDPOINTS) > 0:
            with gr.TabItem(CLASSIFICATION_NAME):
                classification_lbs = _add_activity_track_tabs(
                    leaderboard_type,
                    CLASSIFICATION_TRACK,
                    CLASSIFICATION_ENDPOINTS,
                    CLASSIFICATION_METRIC_DISPLAY_NAMES,
                )
        structure_lbs: dict[str, Leaderboard] = {}
        if len(STRUCTURE_ENDPOINTS) > 0:
            with gr.TabItem(STRUCTURE_NAME, visible=STRUCTURE_TRACK_LIVE):
                structure_lbs = _add_structure_track_tabs(
                    leaderboard_type, STRUCTURE_TRACK, STRUCTURE_ENDPOINTS
                )

        if leaderboard_type == "live":
            assert demo is not None, "demo is required to wire live leaderboards"
            _wire_activity_timer(
                regression_timer,
                demo,
                leaderboard_type,
                REGRESSION_TRACK,
                regression_lbs,
                ACTIVITY_METRIC_DISPLAY_NAMES,
            )
            _wire_activity_timer(
                classification_timer,
                demo,
                leaderboard_type,
                CLASSIFICATION_TRACK,
                classification_lbs,
                CLASSIFICATION_METRIC_DISPLAY_NAMES,
            )
            _wire_structure_timer(
                structure_timer, demo, leaderboard_type, STRUCTURE_TRACK, structure_lbs
            )
