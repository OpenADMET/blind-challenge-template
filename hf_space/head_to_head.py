"""The Head-to-Head tab: compare two final-leaderboard entrants on every test compound.

Everything comes from the release files ``scripts/final_leaderboard.ipynb`` writes
after the challenge, one set per track (see ``backend/head_to_head_release.py``):
``ground_truth``, ``entries``, ``pairwise``, ``bootstrap``, and ``predictions``
(tabular tracks) or ``per_compound`` (structure). The Space can't read ground truth or
anyone's submissions directly, so these files are all it has.

Entrants are identified only by their final rank: the files hold no usernames, and
anonymous entrants appear under their alias. Dropdown values are ranks too, since
Gradio sends choice values to the browser.
"""

import html
import math
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import (
    H2H_LOCAL_DIR,
    HEAD_TO_HEAD_KINDS,
    HEAD_TO_HEAD_URI_FORMAT,
    MACRO_ENDPOINT_LABEL,
    METRIC_DISPLAY_NAMES,
    METRIC_HIGHER_IS_BETTER,
    TRACK_NAMES,
    TRACK_SPECS,
    TrackSpec,
)
from loguru import logger
from utils import _load_csv_from_s3

# Tracks with a head-to-head view, in TRACK_SPECS order.
TRACKS: dict[str, TrackSpec] = {
    track: spec
    for track, spec in TRACK_SPECS.items()
    if spec.endpoints and track in HEAD_TO_HEAD_KINDS
}
COMMON_FILES = ("ground_truth", "entries", "pairwise", "bootstrap")
ALPHA_LEVEL = 0.05  # Must mirror backend EntryComparison.alpha_threshold
NOT_RELEASED_MD = (
    "ℹ️ Head-to-head data for this track will be available once the final "
    "leaderboard has been released."
)
LOADING_HTML = (
    '<div class="h2h-loading"><div class="h2h-spinner"></div>'
    "Loading both entrants' predictions and scores…</div>"
)

# Entrant colours, live-leaderboard membership and correctness.
COLOR_A = "#2F6DB5"
COLOR_B = "#E07A2E"
COLOR_A_LIGHT = "#D6E4F5"
COLOR_B_LIGHT = "#FBE3D0"
COLOR_LIVE = "#7A4FD6"
COLOR_HELD_OUT = "#A3ACB9"
COLOR_CORRECT = "#D9EEE3"
COLOR_INCORRECT_LIGHT = "#F3C5CC"
COLOR_GRID = "#EEEEEE"
PLOT_FONT = dict(family="system-ui, sans-serif", size=13, color="#222")
# Side-by-side per-compound plots share a height and top margin so they line up.
PER_COMPOUND_HEIGHT = 500
REGRESSION_TOP_MARGIN = 125
CLASSIFICATION_TOP_MARGIN = 150
# Per-compound plots per endpoint, and their relative widths.
VIEW_SCALES = {"regression": (3, 2), "classification": (1, 1, 1)}


def _kind(spec: TrackSpec) -> str:
    return HEAD_TO_HEAD_KINDS[spec.track]


def _release_files(spec: TrackSpec) -> tuple[str, ...]:
    extra = "per_compound" if _kind(spec) == "structure" else "predictions"
    return (*COMMON_FILES, extra)


# ==============================================================================
# 1. DATA
# ==============================================================================

_RELEASE_CACHE: dict[str, dict[str, pd.DataFrame]] = {}


def _read_release_file(track: str, name: str) -> pd.DataFrame:
    """Read one release file from S3, or from ``H2H_LOCAL_DIR`` in development."""
    if H2H_LOCAL_DIR:
        return pd.read_parquet(Path(H2H_LOCAL_DIR) / track / f"{name}.parquet")
    return _load_csv_from_s3(
        HEAD_TO_HEAD_URI_FORMAT.format(track=track, file=name), parquet=True
    )


def load_release(track: str) -> dict[str, pd.DataFrame] | None:
    """Return a track's release files, or None if they haven't been published.

    Only successful loads are cached, so files published while the Space is running
    appear the next time the track is picked.

    Parameters
    ----------
    track : str
        A key of ``TRACKS``.

    Returns
    -------
    dict[str, pd.DataFrame] | None
        The files by name, or None.

    """
    if track not in TRACKS:
        return None
    if track not in _RELEASE_CACHE:
        try:
            _RELEASE_CACHE[track] = {
                name: _read_release_file(track, name)
                for name in _release_files(TRACKS[track])
            }
        except Exception as exc:
            logger.warning(f"Head-to-head files unavailable for {track}: {exc}")
            return None
    return _RELEASE_CACHE[track]


def _rank(value: object) -> int | None:
    """Parse a dropdown value (a rank) safely."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _entry(track: str, rank: int) -> pd.Series | None:
    """An entrant's ``entries`` row, with ``display_name`` HTML-escaped.

    Aliases are free text and go into gr.HTML (rendered raw) and Plotly text (which
    decodes entities), so they're escaped here, once.
    """
    release = load_release(track)
    if release is None:
        return None
    entries = release["entries"]
    hits = entries[entries["rank"] == rank]
    if hits.empty:
        return None
    entry = hits.iloc[0].copy()
    entry["display_name"] = html.escape(str(entry["display_name"]))
    return entry


def _for_rank(track: str, name: str, rank: int) -> pd.DataFrame:
    """One entrant's rows of a per-entrant release file."""
    df = load_release(track)[name]  # type: ignore[index]
    return df[df["rank"] == rank].drop(columns="rank")


def _pairwise_row(track: str, rank_a: int, rank_b: int) -> pd.Series | None:
    """The final leaderboard's significance test for this pair, in either order."""
    pairwise = load_release(track)["pairwise"]  # type: ignore[index]
    mask = ((pairwise["rank_a"] == rank_a) & (pairwise["rank_b"] == rank_b)) | (
        (pairwise["rank_a"] == rank_b) & (pairwise["rank_b"] == rank_a)
    )
    hits = pairwise[mask]
    return hits.iloc[0] if not hits.empty else None


# ==============================================================================
# 2. HELPERS
# ==============================================================================


def _pretty_endpoint(endpoint: str) -> str:
    if endpoint == MACRO_ENDPOINT_LABEL:
        return "Overall (macro average)"
    return endpoint.replace("_", " ")


def _short(name: str, width: int = 22) -> str:
    """Truncate an HTML-escaped name to ``width`` visible characters."""
    text = html.unescape(name)
    return name if len(text) <= width else html.escape(text[: width - 1] + "…")


def _metric_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def _lower_is_better(spec: TrackSpec, metric: str) -> bool:
    if metric == spec.primary_metric:
        return not spec.higher_is_better
    return METRIC_HIGHER_IS_BETTER.get(metric) is False


def _summary_endpoint(spec: TrackSpec) -> str:
    """The macro-average for a multi-endpoint track, else its one endpoint."""
    return MACRO_ENDPOINT_LABEL if len(spec.endpoints) > 1 else spec.endpoints[0]


def _bootstrap_for_endpoint(boot: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Narrow bootstrap scores to one endpoint."""
    if "Endpoint" not in boot.columns:
        return boot
    mask = boot["Endpoint"].astype(str) == endpoint
    return boot[mask].drop(columns="Endpoint")


def _metric_columns(boot: pd.DataFrame, primary_metric: str) -> list[str]:
    """Numeric metric columns, primary metric first."""
    metrics = [
        c
        for c in boot.columns
        if c != "Sample" and pd.api.types.is_numeric_dtype(boot[c])
    ]
    return sorted(metrics, key=lambda m: m != primary_metric)


def advantage(
    spec: TrackSpec, boot_a: pd.DataFrame, boot_b: pd.DataFrame, metric: str
) -> pd.Series:
    """A's advantage over B in each paired bootstrap sample: positive means A is better.

    Parameters
    ----------
    spec : TrackSpec
        The track (for the primary metric's direction).
    boot_a : pd.DataFrame
        A's bootstrap scores for one endpoint.
    boot_b : pd.DataFrame
        B's bootstrap scores for the same endpoint.
    metric : str
        The metric.

    Returns
    -------
    pd.Series
        One value per sample both have.

    """
    merged = boot_a[["Sample", metric]].merge(
        boot_b[["Sample", metric]], on="Sample", suffixes=("_a", "_b")
    )
    delta = merged[f"{metric}_a"] - merged[f"{metric}_b"]
    return -delta if _lower_is_better(spec, metric) else delta


def win_share(values: pd.Series) -> float:
    """Share of bootstrap samples A wins, counting ties as half.

    Parameters
    ----------
    values : pd.Series
        From ``advantage``.

    Returns
    -------
    float
        Between 0 and 1, or NaN with no samples.

    """
    if values.empty:
        return float("nan")
    return float((values > 0).mean() + 0.5 * (values == 0).mean())


def _as_bool(value: object) -> bool | None:
    """Parse a boolean that may have round-tripped through text."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def mcnemar_exact_p(only_a_right: int, only_b_right: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    Parameters
    ----------
    only_a_right : int
        Compounds only A labelled correctly.
    only_b_right : int
        Compounds only B labelled correctly.

    Returns
    -------
    float
        The p-value.

    """
    n = only_a_right + only_b_right
    if n == 0:
        return 1.0
    k = min(only_a_right, only_b_right)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def merge_activity(
    ground_truth: pd.DataFrame,
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    endpoint: str,
) -> pd.DataFrame:
    """Join the measured values with both entrants' predictions for one endpoint.

    Compounds with no measured value or a missing prediction are dropped.

    Parameters
    ----------
    ground_truth : pd.DataFrame
        The ``ground_truth`` release file.
    pred_a : pd.DataFrame
        A's predictions.
    pred_b : pd.DataFrame
        B's predictions.
    endpoint : str
        The endpoint.

    Returns
    -------
    pd.DataFrame
        ``Molecule_Name``, ``live``, ``truth`` (and ``conf_low``/``conf_high`` if
        there are credible intervals), ``a`` and ``b``.

    """
    truth_cols = [
        c
        for c in (endpoint, f"{endpoint}_conf_low", f"{endpoint}_conf_high")
        if c in ground_truth.columns
    ]
    df = (
        ground_truth[["Molecule_Name", "live", *truth_cols]]
        .rename(
            columns={
                endpoint: "truth",
                f"{endpoint}_conf_low": "conf_low",
                f"{endpoint}_conf_high": "conf_high",
            }
        )
        .merge(
            pred_a[["Molecule_Name", endpoint]].rename(columns={endpoint: "a"}),
            on="Molecule_Name",
            how="left",
        )
        .merge(
            pred_b[["Molecule_Name", endpoint]].rename(columns={endpoint: "b"}),
            on="Molecule_Name",
            how="left",
        )
    )
    df["live"] = df["live"].astype(bool)
    return df.dropna(subset=["truth", "a", "b"]).reset_index(drop=True)


def _compound_hover(df: pd.DataFrame, extra: str = "") -> tuple[np.ndarray, str]:
    """Customdata and hovertemplate showing the compound and its set."""
    customdata = np.stack(
        [df["Molecule_Name"], np.where(df["live"], "Live leaderboard", "Held out")],
        axis=-1,
    )
    template = "<b>%{customdata[0]}</b><br>%{customdata[1]}" + extra + "<extra></extra>"
    return customdata, template


def _style(
    fig: go.Figure,
    title: str,
    height: int = PER_COMPOUND_HEIGHT,
    grid: bool = True,
    top: int = 105,
) -> go.Figure:
    """The shared look for every head-to-head figure."""
    fig.update_layout(
        template="simple_white",
        title=dict(
            text=title,
            font_size=15,
            x=0,
            xref="paper",
            xanchor="left",
            y=1,
            yref="container",
            yanchor="top",
            pad=dict(t=28),
        ),
        font=PLOT_FONT,
        height=height,
        margin=dict(l=75, r=35, t=top, b=65),
        legend=dict(
            orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", font_size=12
        ),
        hoverlabel=dict(font_family=PLOT_FONT["family"]),
    )
    fig.update_xaxes(showgrid=grid, gridcolor=COLOR_GRID, automargin=True)
    fig.update_yaxes(showgrid=grid, gridcolor=COLOR_GRID, automargin=True)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _discrete_colorscale(colors: list[str]) -> list[list]:
    """A colorscale mapping integer z values 0..n-1 to ``colors`` exactly."""
    n = len(colors)
    scale: list[list] = []
    for i, color in enumerate(colors):
        scale += [[i / n, color], [(i + 1) / n, color]]
    return scale


# ==============================================================================
# 3. SUMMARY: SCORECARDS, METRICS TABLE, BOOTSTRAP DIFFERENCE
# ==============================================================================


def build_summary_html(
    spec: TrackSpec,
    endpoint: str,
    entry_a: pd.Series,
    entry_b: pd.Series,
    boot_a: pd.DataFrame,
    boot_b: pd.DataFrame,
    pw_row: pd.Series | None,
) -> str:
    """Build the two entrant scorecards either side of the win rate and significance.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    endpoint : str
        The endpoint summarised.
    entry_a : pd.Series
        A's ``entries`` row (``display_name`` escaped).
    entry_b : pd.Series
        B's ``entries`` row.
    boot_a : pd.DataFrame
        A's bootstrap scores for ``endpoint``.
    boot_b : pd.DataFrame
        B's bootstrap scores for ``endpoint``.
    pw_row : pd.Series | None
        The pair's final-leaderboard significance test, if any.

    Returns
    -------
    str
        HTML.

    """
    metric = spec.primary_metric
    metric_name = _metric_name(metric)
    win_a = win_share(advantage(spec, boot_a, boot_b, metric))

    def card(entry: pd.Series, boot: pd.DataFrame, side: str) -> str:
        return (
            f'<div class="h2h-card {side}">'
            f'<div class="h2h-rank">Final rank #{int(entry["rank"])}</div>'
            f'<div class="h2h-name">{entry["display_name"]}</div>'
            f'<div class="h2h-metric">{boot[metric].mean():.3f} '
            f"<small>± {boot[metric].std():.3f} {metric_name}</small></div>"
            "</div>"
        )

    is_sig = _as_bool(pw_row["significant_difference"]) if pw_row is not None else None
    if is_sig is None:
        badge = '<span class="h2h-badge nsig">Significance unavailable</span>'
        sig_note = ""
    else:
        badge = (
            '<span class="h2h-badge sig">Significantly different</span>'
            if is_sig
            else '<span class="h2h-badge nsig">Not significantly different</span>'
        )
        assert pw_row is not None
        sig_note = (
            f"Overall ranking, p = {float(pw_row['p_value']):.4f}, "
            f"Benjamini–Hochberg at α = {ALPHA_LEVEL}"
        )

    leader = entry_a if win_a >= 0.5 else entry_b
    share = max(win_a, 1 - win_a)
    vs = (
        '<div class="h2h-vs">'
        f'<div class="win"><b>{leader["display_name"]}</b> is better in '
        f"<b>{share:.0%}</b> of bootstrap samples</div>"
        f'<div class="h2h-note">{metric_name} · {_pretty_endpoint(endpoint)}</div>'
        f"{badge}"
        f'<div class="h2h-note">{sig_note}</div>'
        "</div>"
    )
    return (
        f'<div class="h2h-cards">{card(entry_a, boot_a, "a")}{vs}'
        f"{card(entry_b, boot_b, 'b')}</div>"
    )


def build_metrics_table(
    spec: TrackSpec,
    name_a: str,
    name_b: str,
    boot_a: pd.DataFrame,
    boot_b: pd.DataFrame,
) -> str:
    """Build every metric's bootstrap mean ± std for both entrants, with A's win rate.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    name_a : str
        A's escaped display name.
    name_b : str
        B's escaped display name.
    boot_a : pd.DataFrame
        A's bootstrap scores for one endpoint.
    boot_b : pd.DataFrame
        B's bootstrap scores for the same endpoint.

    Returns
    -------
    str
        HTML.

    """
    rows = []
    for metric in _metric_columns(boot_a, spec.primary_metric):
        if metric not in boot_b.columns:
            continue
        mean_a, mean_b = boot_a[metric].mean(), boot_b[metric].mean()
        lower = _lower_is_better(spec, metric)
        a_better = mean_a < mean_b if lower else mean_a > mean_b
        b_better = mean_b < mean_a if lower else mean_b > mean_a
        win_a = win_share(advantage(spec, boot_a, boot_b, metric))
        rows.append(
            "<tr>"
            f"<td>{_metric_name(metric)} {'↓' if lower else '↑'}</td>"
            f'<td class="{"better" if a_better else ""}">'
            f"{mean_a:.3f} ± {boot_a[metric].std():.3f}</td>"
            f'<td class="{"better" if b_better else ""}">'
            f"{mean_b:.3f} ± {boot_b[metric].std():.3f}</td>"
            f"<td>{win_a:.0%}</td>"
            "</tr>"
        )
    return (
        '<div class="h2h-panel"><table class="h2h-table"><thead><tr><th>Metric</th>'
        f'<th style="color:{COLOR_A}">{name_a}</th>'
        f'<th style="color:{COLOR_B}">{name_b}</th>'
        f"<th>{_short(name_a, 14)} wins</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        '<p class="h2h-note">Bootstrap mean ± std over all test compounds. '
        "↑ higher is better, ↓ lower is better. "
        '"Wins" is the share of paired bootstrap samples in which '
        f"{name_a} scores better.</p></div>"
    )


def build_delta_plot(
    spec: TrackSpec,
    endpoint: str,
    name_a: str,
    name_b: str,
    boot_a: pd.DataFrame,
    boot_b: pd.DataFrame,
) -> go.Figure:
    """Build a histogram of A's per-sample advantage in the primary metric.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    endpoint : str
        The endpoint.
    name_a : str
        A's escaped display name.
    name_b : str
        B's escaped display name.
    boot_a : pd.DataFrame
        A's bootstrap scores.
    boot_b : pd.DataFrame
        B's bootstrap scores.

    Returns
    -------
    go.Figure
        The figure.

    """
    metric = spec.primary_metric
    metric_name = _metric_name(metric)
    values = advantage(spec, boot_a, boot_b, metric)
    spread = float(values.max() - values.min()) if len(values) else 0.0
    bins = dict(size=spread / 50 if spread > 0 else 0.01)

    fig = go.Figure()
    for mask, name, color in [
        (values > 0, f"{_short(name_a)} better", COLOR_A),
        (values <= 0, f"{_short(name_b)} better", COLOR_B),
    ]:
        fig.add_trace(
            go.Histogram(
                x=values[mask],
                name=f"{name} ({int(mask.sum())}/{len(values)})",
                marker_color=color,
                opacity=0.85,
                xbins=bins,
                hovertemplate="Δ %{x}<br>%{y} samples<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_dash="dot", line_color="#333", line_width=1.5)
    fig.update_layout(barmode="overlay", bargap=0.03, showlegend=True)
    fig.update_xaxes(
        title=f"{metric_name} advantage to {_short(name_a)} (positive = A better)"
    )
    fig.update_yaxes(title="Bootstrap samples")
    return _style(
        fig, f"Bootstrap {metric_name} difference · {_pretty_endpoint(endpoint)}", 400
    )


# ==============================================================================
# 4. PER-COMPOUND VIEWS
# ==============================================================================


def _split_live(df: pd.DataFrame):
    """Yield (subset, label, colour) for held-out then live compounds (live on top)."""
    yield df[~df["live"]], "Held out", COLOR_HELD_OUT
    yield df[df["live"]], "Live leaderboard", COLOR_LIVE


def build_regression_sorted(
    df: pd.DataFrame, name_a: str, name_b: str, endpoint: str
) -> go.Figure:
    """Build every compound ordered by measured value, with both entrants' predictions.

    Parameters
    ----------
    df : pd.DataFrame
        From ``merge_activity``.
    name_a : str
        A's escaped display name.
    name_b : str
        B's escaped display name.
    endpoint : str
        The endpoint.

    Returns
    -------
    go.Figure
        The figure.

    """
    df = df.sort_values("truth", kind="stable").reset_index(drop=True)
    x = np.arange(1, len(df) + 1)
    has_ci = {"conf_low", "conf_high"} <= set(df.columns)
    measured = df["truth"].map("{:.2f}".format)

    fig = go.Figure()
    for col, name, color in [("a", name_a, COLOR_A), ("b", name_b, COLOR_B)]:
        mae = (df[col] - df["truth"]).abs().mean()
        customdata, template = _compound_hover(
            df, f"<br>Measured %{{text}}<br>{_short(name)} predicted %{{y:.2f}}"
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                mode="markers",
                name=f"{_short(name)} · MAE {mae:.2f}",
                marker=dict(color=color, size=5, opacity=0.55),
                text=measured,
                customdata=customdata,
                hovertemplate=template,
            )
        )
    # The measured values last, on top of the predictions.
    live_mask = df["live"].to_numpy()
    for mask, label, color in [
        (~live_mask, "Measured · held out", COLOR_HELD_OUT),
        (live_mask, "Measured · live leaderboard", COLOR_LIVE),
    ]:
        subset = df[mask]
        customdata, template = _compound_hover(subset, "<br>Measured %{y:.2f}")
        error_y = (
            dict(
                type="data",
                array=subset["conf_high"] - subset["truth"],
                arrayminus=subset["truth"] - subset["conf_low"],
                color=_rgba(color, 0.45),
                thickness=1,
                width=0,
            )
            if has_ci
            else None
        )
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=subset["truth"],
                mode="markers",
                name=label,
                legend="legend2",
                marker=dict(color=color, size=6),
                error_y=error_y,
                customdata=customdata,
                hovertemplate=template,
            )
        )
    fig.update_xaxes(
        title="Compounds, ordered by measured value (lowest → highest)",
        range=[0, len(df) + 1],
    )
    fig.update_yaxes(title="Value")
    note = "Bars show the measured credible interval" if has_ci else ""
    title = (
        f"Predicted vs measured · {_pretty_endpoint(endpoint)}"
        f"<br><span style='font-size:12px'>{note}</span>"
    )
    fig = _style(fig, title, top=REGRESSION_TOP_MARGIN)
    # The sorted values leave the top-left corner empty: the measured legend goes
    # there, leaving the row above the plot for the two entrants.
    fig.update_layout(
        legend2=dict(
            x=0.01,
            xanchor="left",
            y=0.99,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#DDD",
            borderwidth=1,
            font_size=12,
        )
    )
    return fig


def build_regression_errors(
    df: pd.DataFrame, name_a: str, name_b: str, endpoint: str
) -> go.Figure:
    """Build A's vs B's absolute error per compound (above the diagonal: A is closer).

    Parameters
    ----------
    df : pd.DataFrame
        From ``merge_activity``.
    name_a : str
        A's escaped display name.
    name_b : str
        B's escaped display name.
    endpoint : str
        The endpoint.

    Returns
    -------
    go.Figure
        The figure.

    """
    df = df.assign(
        err_a=(df["a"] - df["truth"]).abs(), err_b=(df["b"] - df["truth"]).abs()
    )
    top = float(df[["err_a", "err_b"]].to_numpy().max()) * 1.05 or 1.0
    fig = go.Figure()
    for subset, label, color in _split_live(df):
        customdata, template = _compound_hover(
            subset,
            f"<br>|error| {_short(name_a)}: %{{x:.2f}}"
            f"<br>|error| {_short(name_b)}: %{{y:.2f}}",
        )
        fig.add_trace(
            go.Scatter(
                x=subset["err_a"],
                y=subset["err_b"],
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, opacity=0.75),
                customdata=customdata,
                hovertemplate=template,
            )
        )
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=0,
        y0=0,
        x1=top,
        y1=top,
        line=dict(color="#555", dash="dot", width=1),
    )
    n_a = int((df["err_a"] < df["err_b"]).sum())
    n_b = int((df["err_b"] < df["err_a"]).sum())
    for text, x, y, anchor, color in [
        (f"{_short(name_a)} closer: {n_a}", 0.02, 0.98, "left", COLOR_A),
        (f"{_short(name_b)} closer: {n_b}", 0.98, 0.02, "right", COLOR_B),
    ]:
        fig.add_annotation(
            text=f"<b>{text}</b>",
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            xanchor=anchor,
            yanchor="top" if y > 0.5 else "bottom",
            showarrow=False,
            font=dict(color=color, size=13),
        )
    fig.update_xaxes(title=f"|error| {_short(name_a)}", range=[0, top])
    fig.update_yaxes(title=f"|error| {_short(name_b)}", range=[0, top])
    title = (
        f"Who is closer? · {_pretty_endpoint(endpoint)}<br><span style='font-size:12px'>"
        "Each point is a compound's absolute error</span>"
    )
    return _style(fig, title, top=REGRESSION_TOP_MARGIN)


def _grid_2x2(
    counts: list[list[int]],
    lives: list[list[int]],
    labels: list[list[str]],
    x: list[str],
    y: list[str],
    z: list[list[int]],
    colors: list[str],
) -> go.Figure:
    """A 2×2 grid of compound counts (live counts underneath)."""
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=_discrete_colorscale(colors),
            zmin=-0.5,
            zmax=len(colors) - 0.5,
            showscale=False,
            xgap=4,
            ygap=4,
            customdata=[
                [f"{labels[r][c]}: {counts[r][c]} compounds" for c in range(2)]
                for r in range(2)
            ],
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    for r in range(2):
        for c in range(2):
            fig.add_annotation(
                x=c,
                y=r,
                text=(
                    f"<b style='font-size:22px'>{counts[r][c]}</b><br>"
                    f"<span style='font-size:11px'>{lives[r][c]} live</span>"
                ),
                showarrow=False,
            )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return fig


def build_classification_agreement(
    df: pd.DataFrame, name_a: str, name_b: str, endpoint: str
) -> go.Figure:
    """Build a 2×2 grid of who labelled each compound correctly, with McNemar's test.

    Parameters
    ----------
    df : pd.DataFrame
        From ``merge_activity``.
    name_a : str
        A's escaped display name.
    name_b : str
        B's escaped display name.
    endpoint : str
        The endpoint.

    Returns
    -------
    go.Figure
        The figure.

    """
    a_ok = df["a"].astype(bool) == df["truth"].astype(bool)
    b_ok = df["b"].astype(bool) == df["truth"].astype(bool)
    cells = [[a_ok & b_ok, a_ok & ~b_ok], [~a_ok & b_ok, ~a_ok & ~b_ok]]
    counts = [[int(m.sum()) for m in row] for row in cells]
    lives = [[int((m & df["live"]).sum()) for m in row] for row in cells]
    p_value = mcnemar_exact_p(counts[0][1], counts[1][0])
    fig = _grid_2x2(
        counts,
        lives,
        labels=[["Both right", "Only A right"], ["Only B right", "Both wrong"]],
        x=[f"{_short(name_b, 16)} right", f"{_short(name_b, 16)} wrong"],
        y=[f"{_short(name_a, 16)} right", f"{_short(name_a, 16)} wrong"],
        z=[[0, 1], [2, 3]],
        colors=[COLOR_CORRECT, COLOR_A_LIGHT, COLOR_B_LIGHT, COLOR_INCORRECT_LIGHT],
    )
    title = (
        f"Agreement · {_pretty_endpoint(endpoint)}<br>"
        f"<span style='font-size:12px'>McNemar exact p = {p_value:.3g} "
        f"({len(df)} labelled compounds)</span>"
    )
    fig = _style(fig, title, grid=False, top=CLASSIFICATION_TOP_MARGIN)
    fig.update_layout(margin=dict(b=30))
    return fig


def build_classification_confusion(
    df: pd.DataFrame, col: str, name: str, endpoint: str
) -> go.Figure:
    """Build one entrant's confusion matrix, with sensitivity and specificity.

    Parameters
    ----------
    df : pd.DataFrame
        From ``merge_activity``.
    col : str
        "a" or "b".
    name : str
        The entrant's escaped display name.
    endpoint : str
        The endpoint.

    Returns
    -------
    go.Figure
        The figure.

    """
    truth = df["truth"].astype(bool)
    pred = df[col].astype(bool)
    cells = [[truth & pred, truth & ~pred], [~truth & pred, ~truth & ~pred]]
    counts = [[int(m.sum()) for m in row] for row in cells]
    lives = [[int((m & df["live"]).sum()) for m in row] for row in cells]
    (tp, fn), (fp, tn) = counts
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    color, light = (COLOR_A, COLOR_A_LIGHT) if col == "a" else (COLOR_B, COLOR_B_LIGHT)
    fig = _grid_2x2(
        counts,
        lives,
        labels=[
            ["True positive", "False negative"],
            ["False positive", "True negative"],
        ],
        x=["Predicted +", "Predicted −"],
        y=["Measured +", "Measured −"],
        z=[[0, 1], [1, 0]],
        colors=[light, COLOR_INCORRECT_LIGHT],
    )
    title = (
        f"<span style='color:{color}'><b>{_short(name)}</b></span> · confusion matrix"
        f" · {_pretty_endpoint(endpoint)}"
        f"<br><span style='font-size:12px'>Sensitivity {sensitivity:.2f} · "
        f"Specificity {specificity:.2f}</span>"
    )
    fig = _style(fig, title, grid=False, top=CLASSIFICATION_TOP_MARGIN)
    fig.update_layout(margin=dict(b=30))
    return fig


def build_structure_bars(
    ground_truth: pd.DataFrame,
    per_a: pd.DataFrame,
    per_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    metric: str,
) -> go.Figure:
    """Build per-compound primary-metric bars for both entrants, live compounds shaded.

    Parameters
    ----------
    ground_truth : pd.DataFrame
        The ``ground_truth`` release file.
    per_a : pd.DataFrame
        A's per-compound scores.
    per_b : pd.DataFrame
        B's per-compound scores.
    name_a : str
        A's escaped display name.
    name_b : str
        B's escaped display name.
    metric : str
        The metric to plot.

    Returns
    -------
    go.Figure
        The figure.

    """
    df = (
        ground_truth[["Molecule_Name", "live"]]
        .merge(
            per_a[["Molecule_Name", metric]].rename(columns={metric: "a"}),
            on="Molecule_Name",
            how="left",
        )
        .merge(
            per_b[["Molecule_Name", metric]].rename(columns={metric: "b"}),
            on="Molecule_Name",
            how="left",
        )
    )
    df["live"] = df["live"].astype(bool)
    df = df.sort_values("live", ascending=False, kind="stable").reset_index(drop=True)
    metric_name = _metric_name(metric)
    fig = go.Figure()
    for col, name, color in [("a", name_a, COLOR_A), ("b", name_b, COLOR_B)]:
        customdata, template = _compound_hover(
            df, f"<br>{_short(name)} {metric_name}: %{{y:.3f}}"
        )
        fig.add_trace(
            go.Bar(
                x=df["Molecule_Name"],
                y=df[col],
                name=_short(name, 30),
                marker_color=color,
                customdata=customdata,
                hovertemplate=template,
            )
        )
    n_live = int(df["live"].sum())
    if n_live:
        fig.add_vrect(
            x0=-0.5,
            x1=n_live - 0.5,
            fillcolor=COLOR_LIVE,
            opacity=0.08,
            line_width=0,
            layer="below",
            annotation_text="Live leaderboard compounds",
            annotation_position="inside top",
            annotation_font=dict(color=COLOR_LIVE, size=12),
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_xaxes(title=None, tickangle=-45, showgrid=False)
    upper = 1.05 if metric.startswith("LDDT") else None
    fig.update_yaxes(title=metric_name, range=[0, upper] if upper else None)
    return _style(fig, f"Per-compound {metric_name} · all {len(df)} compounds")


# ==============================================================================
# 5. INTERFACE
# ==============================================================================


def _plot_count(spec: TrackSpec) -> int:
    if _kind(spec) == "structure":
        return 1
    return len(VIEW_SCALES[_kind(spec)]) * len(spec.endpoints)


def render_head_to_head() -> None:
    """Render the Head-to-Head tab. Data loads when a track is picked."""
    gr.Markdown("## Head-to-Head")
    gr.Markdown(
        "Pick a track and two final-leaderboard entrants to compare them on every "
        "test compound. Compounds used for the live leaderboard are highlighted in "
        "purple."
    )
    # allow_custom_value: when the track changes, events can briefly see the old
    # track's entrant values against the new choices; without it Gradio rejects them
    # with an error. Handlers ignore values they don't know.
    with gr.Row(equal_height=True):
        track_dd = gr.Dropdown(
            choices=[(TRACK_NAMES[t], t) for t in TRACKS],
            value=None,
            label="Track",
            scale=3,
            min_width=240,
        )
        user_a_dd = gr.Dropdown(
            choices=[],
            label="Entrant A",
            scale=3,
            min_width=240,
            allow_custom_value=True,
        )
        user_b_dd = gr.Dropdown(
            choices=[],
            label="Entrant B",
            scale=3,
            min_width=240,
            allow_custom_value=True,
        )
    compare_btn = gr.Button("Compare", variant="primary", interactive=False)

    status_md = gr.Markdown(visible=False)
    loading_html = gr.HTML(visible=False)

    with gr.Column(visible=False) as results_col:
        gr.Markdown("### Bootstrap statistics")
        summary_html = gr.HTML()
        with gr.Row():
            with gr.Column(scale=1):
                metrics_html = gr.HTML()
            with gr.Column(scale=1):
                delta_plot = gr.Plot(show_label=False)

        gr.Markdown("### Per-compound comparison")
        view_columns: list[gr.Column] = []
        view_plots: list[gr.Plot] = []
        for spec in TRACKS.values():
            with gr.Column(visible=False) as column:
                view_columns.append(column)
                if _kind(spec) == "structure":
                    view_plots.append(gr.Plot(show_label=False))
                    continue
                with gr.Tabs():
                    for endpoint in spec.endpoints:
                        with gr.TabItem(_pretty_endpoint(endpoint)):
                            with gr.Row(equal_height=True):
                                for scale in VIEW_SCALES[_kind(spec)]:
                                    with gr.Column(scale=scale, min_width=300):
                                        view_plots.append(gr.Plot(show_label=False))

    all_outputs = [
        loading_html,
        compare_btn,
        status_md,
        results_col,
        summary_html,
        metrics_html,
        delta_plot,
        *view_columns,
        *view_plots,
    ]
    track_dd.change(
        fn=on_track_change,
        inputs=[track_dd],
        outputs=[user_a_dd, user_b_dd, status_md, results_col, compare_btn],
        show_progress="full",
        show_progress_on=[user_a_dd, user_b_dd],
    )
    user_a_dd.change(
        fn=on_user_a_change,
        inputs=[track_dd, user_a_dd, user_b_dd],
        outputs=[user_b_dd],
    )
    gr.on(
        triggers=[user_a_dd.change, user_b_dd.change],
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=[results_col],
    )
    gr.on(
        triggers=[user_a_dd.change, user_b_dd.change],
        fn=lambda track, a, b: gr.update(
            interactive=bool(track and a and b and a != b)
        ),
        inputs=[track_dd, user_a_dd, user_b_dd],
        outputs=[compare_btn],
    )
    # Show the spinner straight away, then build the comparison.
    compare_btn.click(
        fn=show_loading,
        inputs=None,
        outputs=[loading_html, compare_btn, status_md, results_col],
        queue=False,
    ).then(
        fn=run_comparison,
        inputs=[track_dd, user_a_dd, user_b_dd],
        outputs=all_outputs,
        show_progress="full",
        show_progress_on=[loading_html],
    )


def show_loading() -> tuple:
    """Show the spinner while a comparison loads, hiding any previous results.

    Returns
    -------
    tuple
        Updates for the spinner, button, status and results.

    """
    return (
        gr.update(visible=True, value=LOADING_HTML),
        gr.update(interactive=False),
        gr.update(visible=False, value=""),
        gr.update(visible=False),
    )


def entrant_choices(track: str) -> list[tuple[str, str]]:
    """List a track's entrants for the dropdowns: "#rank · name", valued by rank.

    Parameters
    ----------
    track : str
        The track.

    Returns
    -------
    list[tuple[str, str]]
        (label, rank) pairs, best first; empty if the files aren't released.

    """
    release = load_release(track)
    if release is None:
        return []
    entries = release["entries"].sort_values("rank")
    return [
        (f"#{int(row['rank'])} · {row['display_name']}", str(int(row["rank"])))
        for _, row in entries.iterrows()
    ]


def on_track_change(track: str | None) -> tuple:
    """Refill the entrant dropdowns for the chosen track.

    Parameters
    ----------
    track : str | None
        The chosen track.

    Returns
    -------
    tuple
        Updates for both dropdowns, the status, the results and the button.

    """
    hidden = gr.update(visible=False)
    disabled = gr.update(interactive=False)
    if track not in TRACKS:
        empty = gr.update(choices=[], value=None)
        return empty, empty, gr.update(visible=False, value=""), hidden, disabled
    choices = entrant_choices(track)
    status = (
        gr.update(visible=False, value="")
        if choices
        else gr.update(visible=True, value=NOT_RELEASED_MD)
    )
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        status,
        hidden,
        disabled,
    )


def on_user_a_change(track: str | None, user_a: str | None, user_b: str | None) -> dict:
    """Offer every entrant except A as B, keeping B if still valid.

    Parameters
    ----------
    track : str | None
        The chosen track.
    user_a : str | None
        A's rank.
    user_b : str | None
        B's rank.

    Returns
    -------
    dict
        An update for B's dropdown.

    """
    if track not in TRACKS:
        return gr.update(choices=[], value=None)
    choices = [c for c in entrant_choices(track) if c[1] != user_a]
    keep = user_b if user_b in {value for _, value in choices} else None
    return gr.update(choices=choices, value=keep)


def _summary(spec: TrackSpec, entry_a: pd.Series, entry_b: pd.Series) -> list:
    track = spec.track
    endpoint = _summary_endpoint(spec)
    rank_a, rank_b = int(entry_a["rank"]), int(entry_b["rank"])
    boot_a = _bootstrap_for_endpoint(_for_rank(track, "bootstrap", rank_a), endpoint)
    boot_b = _bootstrap_for_endpoint(_for_rank(track, "bootstrap", rank_b), endpoint)
    if boot_a.empty or boot_b.empty:
        raise ValueError(f"no bootstrap scores for {endpoint}")
    name_a, name_b = entry_a["display_name"], entry_b["display_name"]
    return [
        build_summary_html(
            spec,
            endpoint,
            entry_a,
            entry_b,
            boot_a,
            boot_b,
            _pairwise_row(track, rank_a, rank_b),
        ),
        build_metrics_table(spec, name_a, name_b, boot_a, boot_b),
        build_delta_plot(spec, endpoint, name_a, name_b, boot_a, boot_b),
    ]


def per_compound_figures(
    spec: TrackSpec, entry_a: pd.Series, entry_b: pd.Series
) -> list[go.Figure]:
    """Build the track's per-compound figures, in the order the tab lays them out.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    entry_a : pd.Series
        A's ``entries`` row.
    entry_b : pd.Series
        B's ``entries`` row.

    Returns
    -------
    list[go.Figure]
        The figures.

    """
    track = spec.track
    ground_truth = load_release(track)["ground_truth"]  # type: ignore[index]
    rank_a, rank_b = int(entry_a["rank"]), int(entry_b["rank"])
    name_a, name_b = entry_a["display_name"], entry_b["display_name"]
    if _kind(spec) == "structure":
        return [
            build_structure_bars(
                ground_truth,
                _for_rank(track, "per_compound", rank_a),
                _for_rank(track, "per_compound", rank_b),
                name_a,
                name_b,
                spec.primary_metric,
            )
        ]
    pred_a = _for_rank(track, "predictions", rank_a)
    pred_b = _for_rank(track, "predictions", rank_b)
    figures = []
    for endpoint in spec.endpoints:
        df = merge_activity(ground_truth, pred_a, pred_b, endpoint)
        if _kind(spec) == "regression":
            figures += [
                build_regression_sorted(df, name_a, name_b, endpoint),
                build_regression_errors(df, name_a, name_b, endpoint),
            ]
        else:
            figures += [
                build_classification_agreement(df, name_a, name_b, endpoint),
                build_classification_confusion(df, "a", name_a, endpoint),
                build_classification_confusion(df, "b", name_b, endpoint),
            ]
    return figures


def run_comparison(
    track: str | None,
    user_a: str | None,
    user_b: str | None,
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> tuple:
    """Build every output for the chosen pair.

    Parameters
    ----------
    track : str | None
        The chosen track.
    user_a : str | None
        A's rank (the dropdown value).
    user_b : str | None
        B's rank.
    progress : gr.Progress
        Gradio's progress bar.

    Returns
    -------
    tuple
        Updates for every output of the tab.

    """
    n_plots = sum(_plot_count(spec) for spec in TRACKS.values())
    done = (gr.update(visible=False, value=""), gr.update(interactive=True))

    def failed(message: str) -> tuple:
        return (
            *done,
            gr.update(visible=True, value=f"⚠️ {message}"),
            gr.update(visible=False),
            *[gr.update()] * (3 + len(TRACKS) + n_plots),
        )

    rank_a, rank_b = _rank(user_a), _rank(user_b)
    if track not in TRACKS or rank_a is None or rank_b is None or rank_a == rank_b:
        return failed("Please choose a track and two different entrants.")
    if load_release(track) is None:
        return failed(NOT_RELEASED_MD)
    entry_a, entry_b = _entry(track, rank_a), _entry(track, rank_b)
    if entry_a is None or entry_b is None:
        return failed("Please choose both entrants from the list.")

    spec = TRACKS[track]
    try:
        progress(0.1, desc="Loading bootstrap scores…")
        summary = _summary(spec, entry_a, entry_b)
        progress(0.5, desc="Drawing the per-compound plots…")
        figures = per_compound_figures(spec, entry_a, entry_b)
        progress(1.0, desc="Done")
    except Exception as exc:
        logger.exception(f"Head-to-head comparison failed: {exc}")
        return failed(
            "Couldn't build this comparison. Please try again, or contact the "
            "organisers if it keeps happening."
        )

    columns = [gr.update(visible=t == track) for t in TRACKS]
    plots = []
    for t, s in TRACKS.items():
        if t == track:
            plots += [gr.update(value=fig) for fig in figures]
        else:
            plots += [gr.update(value=None)] * _plot_count(s)
    return (
        *done,
        gr.update(visible=False, value=""),
        gr.update(visible=True),
        *summary,
        *columns,
        *plots,
    )
