"""Leaderboard generation (ranking, significance testing) from S3 manifest."""

from datetime import datetime

import awswrangler as wr
import duckdb
import pandas as pd
from loguru import logger

from .aws_manifest import _configure_duckdb_s3_auth, create_manifest
from .config import (
    CLASSIFICATION_PATHS,
    FINAL_LEADERBOARD_DEADLINE,
    INTERIM_LEADERBOARD_DEADLINE,
    MACRO_ENDPOINT_LABEL,
    REGRESSION_PATHS,
    S3_BUCKET,
    STRUCTURE_PATHS,
    TrackPaths,
)
from .leaderboard import FinalLeaderboard, LeaderboardEntry
from .utils import validate_hf_username, validate_model_details

_TRACK_PATHS = {
    "regression": REGRESSION_PATHS,
    "classification": CLASSIFICATION_PATHS,
    "structure": STRUCTURE_PATHS,
}


def index_leaderboard_by_rank(leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    """Move the "rank" column into the DataFrame index, ready for CSV saving."""
    return leaderboard_df.set_index("rank")


def _leaderboard_file_path(
    track: str, stage: str, endpoint_slug: str, version: str
) -> str:
    """Return the S3 filepath (no bucket, no filename) for a leaderboard."""
    paths = _TRACK_PATHS[track]
    return {
        "live": paths.leaderboard_live,
        "interim": paths.leaderboard_interim,
        "final": paths.leaderboard_final,
    }[stage] + f"/{endpoint_slug}_leaderboard_{version}.csv"


def get_leaderboard(track: str, stage: str, endpoint_slug: str) -> pd.DataFrame | None:
    """Fetch the latest leaderboard of a given stage from S3.

    Parameters
    ----------
    track : str
        The track to fetch the leaderboard for ("regression", "classification", or
        "structure").
    stage : str
        "live", "interim", or "final".
    endpoint_slug : str
        Which leaderboard variant to fetch — one of ``track_paths.endpoints``
        (e.g. "ENDPOINT_1", or "Structure" for the structure track's one
        endpoint), or ``MACRO_ENDPOINT_LABEL`` ("MA") for a multi-endpoint track's
        macro-ranked master leaderboard.

    Returns
    -------
    pd.DataFrame | None
        The leaderboard DataFrame if it exists, otherwise None.

    """
    filepath = _leaderboard_file_path(track, stage, endpoint_slug, "latest")
    full_leaderboard_path = f"s3://{S3_BUCKET}/{filepath}"
    try:
        leaderboard_df = wr.s3.read_csv(full_leaderboard_path).set_index("rank")
    except wr.exceptions.NoFilesFound:
        return None
    leaderboard_df["submitted_at"] = pd.to_datetime(
        leaderboard_df["submitted_at"], utc=True
    ).astype("datetime64[ns, UTC]")
    leaderboard_df = leaderboard_df.fillna({"model_report_link": ""})
    logger.info("Fetched existing leaderboard with {} rows", len(leaderboard_df))
    return leaderboard_df


def save_leaderboard(
    leaderboard_df: pd.DataFrame,
    track: str,
    stage: str,
    endpoint_slug: str,
) -> str:
    """Save a leaderboard to S3 and overwrite the latest version for that stage.

    Parameters
    ----------
    leaderboard_df : pd.DataFrame
        The leaderboard DataFrame to save.
    track : str
        The track to save the leaderboard for ("regression", "classification", or
        "structure").
    stage : str
        "live", "interim", or "final".
    endpoint_slug : str
        Which leaderboard variant this is — see ``get_leaderboard``'s
        ``endpoint_slug`` parameter.

    Returns
    -------
    str
        The S3 path where the leaderboard was saved.

    """
    now = datetime.now().isoformat()
    timestamp_path = (
        f"s3://{S3_BUCKET}/{_leaderboard_file_path(track, stage, endpoint_slug, now)}"
    )
    latest_path = (
        f"s3://{S3_BUCKET}/"
        f"{_leaderboard_file_path(track, stage, endpoint_slug, 'latest')}"
    )
    wr.s3.to_csv(
        df=leaderboard_df,
        path=timestamp_path,
        index=True,
    )
    wr.s3.to_csv(df=leaderboard_df, path=latest_path, index=True)
    return latest_path


def _read_parquet_batch_by_uri(uris: list[str]) -> dict[str, pd.DataFrame]:
    """Read many parquet files from S3 in one DuckDB query and group by URI.

    Parameters
    ----------
    uris : list[str]
        List of S3 parquet URIs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from URI to DataFrame rows from that file.

    """
    logger.info("Reading {} parquet files in batch with DuckDB", len(uris))
    if not uris:
        return {}

    unique_uris = list(dict.fromkeys(uris))

    with duckdb.connect() as con:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        _configure_duckdb_s3_auth(con)
        combined_df = con.execute(
            "SELECT * FROM read_parquet(?, filename=true, union_by_name=true)",
            [unique_uris],
        ).df()
    logger.info("Created dataframe data with {} total rows", len(combined_df))

    grouped: dict[str, pd.DataFrame] = {}
    for uri, df_slice in combined_df.groupby("filename", sort=False):
        grouped[uri] = df_slice.drop(columns=["filename"]).reset_index(drop=True)
    return grouped


def get_averaged_scores_for_leaderboard(
    manifest: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Download averaged score files for a leaderboard manifest.

    Parameters
    ----------
    manifest : pd.DataFrame
        Manifest containing ``scores_uri_for_leaderboard``.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping of manifest row index to averaged-results DataFrame.

    """
    logger.info("Fetching averaged scores")
    uris = manifest["scores_uri_for_leaderboard"].tolist()
    averaged_by_uri = _read_parquet_batch_by_uri(uris)

    averaged_scores: dict[int, pd.DataFrame] = {}
    for idx, row in manifest.iterrows():
        averaged_scores[idx] = averaged_by_uri[row["scores_uri_for_leaderboard"]]
    return averaged_scores


def get_bootstrap_scores_for_leaderboard(
    manifest: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Download bootstrap score files for a leaderboard manifest.

    Parameters
    ----------
    manifest : pd.DataFrame
        Manifest containing ``scores_uri_for_leaderboard``.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping of manifest row index to bootstrap-results DataFrame.

    """
    logger.info("Fetching bootstrap scores")
    bootstrap_uris: list[str] = []
    for _, row in manifest.iterrows():
        averaged_uri = row["scores_uri_for_leaderboard"]
        bootstrap_uri = averaged_uri.replace(
            "averaged-results.parquet", "bootstrap-results.parquet"
        )
        bootstrap_uris.append(bootstrap_uri)

    bootstrap_by_uri = _read_parquet_batch_by_uri(bootstrap_uris)

    bootstrap_scores: dict[int, pd.DataFrame] = {}
    for idx, row in manifest.iterrows():
        averaged_uri = row["scores_uri_for_leaderboard"]
        bootstrap_uri = averaged_uri.replace(
            "averaged-results.parquet", "bootstrap-results.parquet"
        )
        bootstrap_scores[idx] = bootstrap_by_uri[bootstrap_uri]
    return bootstrap_scores


def _narrow_averaged_results_to_endpoint(
    wide_results: pd.DataFrame, endpoint: str
) -> pd.DataFrame:
    """Select one endpoint's columns from a wide averaged-results row.

    ``wide_results`` (as saved by ``evaluate_predictions.pivot_endpoint_results_wide``)
    has every endpoint's metrics prefixed, e.g. ``"ENDPOINT_1_MAE_mean"``.
    This selects just the columns for ``endpoint`` and strips the prefix, so the
    result has bare metric columns (e.g. ``"MAE_mean"``) regardless of which endpoint
    it came from — letting a single ``primary_metric`` string (e.g. ``"ST-RAE"``) work
    for any endpoint's leaderboard.

    Parameters
    ----------
    wide_results : pd.DataFrame
        Single-row wide averaged-results DataFrame.
    endpoint : str
        The endpoint to select, e.g. "MA" or "ENDPOINT_1".

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with that endpoint's bare metric columns.

    """
    prefix = f"{endpoint}_"
    columns = {
        c: c[len(prefix) :] for c in wide_results.columns if c.startswith(prefix)
    }
    return wide_results[list(columns.keys())].rename(columns=columns)


def _fetch_leaderboard_manifest_and_scores(
    track: str,
    stage: str,
    remove_invalid: bool,
    with_bootstrap: bool,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Fetch the manifest and un-narrowed score data shared by every endpoint variant.

    This is the expensive, endpoint-independent part of leaderboard generation (a
    DuckDB S3 glob query for the manifest, plus batched parquet reads for every
    submission's score files) — every endpoint variant of a track/stage's leaderboard
    (e.g. regression's master + 4 per-endpoint leaderboards) uses identical manifest
    and score data, differing only in which endpoint's columns get selected
    afterwards. Callers that need multiple variants should call this once and reuse
    the result; see ``create_track_leaderboards``.

    Parameters
    ----------
    track : str
        "regression", "classification", or "structure".
    stage : str
        "live", "interim", or "final" — see ``create_track_leaderboards``.
    remove_invalid : bool
        Whether to remove invalid submissions from the manifest.
    with_bootstrap : bool
        Whether to also fetch bootstrap score files (needed only for pairwise
        significance comparisons).

    Returns
    -------
    tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]
        The manifest (empty if no eligible submissions), and averaged/bootstrap
        score DataFrames keyed by manifest row index, not yet narrowed to any
        endpoint. The bootstrap dict is empty when ``with_bootstrap`` is False.

    """
    if stage == "live":
        cutoff = None
    elif stage == "interim":
        cutoff = INTERIM_LEADERBOARD_DEADLINE
    else:  # stage == "final"
        cutoff = FINAL_LEADERBOARD_DEADLINE
    uri_column = "phase_1_scores_uri" if stage == "live" else "all_scores_uri"

    manifest = create_manifest(track=track, only_latest=True, date_cutoff=cutoff)
    if manifest.empty:
        return manifest, {}, {}

    if remove_invalid:
        logger.info("Removing invalid submissions from leaderboard")
        manifest["valid_hf_username"] = manifest.username.apply(validate_hf_username)
        manifest["valid_model_details"] = manifest.model_report_link.apply(
            validate_model_details
        )
        start_len = len(manifest)
        model_details_ok = ~manifest["valid_model_details"].isin(
            ["Not submitted", "Invalid link"]
        )
        valid_mask = manifest["valid_hf_username"] & model_details_ok
        manifest = manifest[valid_mask]
        logger.info(
            "Removed {} invalid submissions, {} remaining",
            start_len - len(manifest),
            len(manifest),
        )

    manifest = manifest.copy()
    manifest["scores_uri_for_leaderboard"] = manifest[uri_column]
    manifest = manifest[manifest["scores_uri_for_leaderboard"].notna()]
    if manifest.empty:
        return manifest, {}, {}

    averaged_scores = get_averaged_scores_for_leaderboard(manifest)
    bootstrap_scores = (
        get_bootstrap_scores_for_leaderboard(manifest) if with_bootstrap else {}
    )
    return manifest, averaged_scores, bootstrap_scores


def _build_leaderboard(
    manifest: pd.DataFrame,
    averaged_scores: dict[int, pd.DataFrame],
    bootstrap_scores: dict[int, pd.DataFrame],
    primary_metric: str,
    metric_sort_ascending: bool,
    comparisons: str | None,
    endpoint: str,
    additional_columns: list[str],
) -> pd.DataFrame:
    """Build one endpoint's leaderboard from already-fetched manifest/score data.

    Parameters
    ----------
    manifest : pd.DataFrame
        Non-empty manifest, as returned by
        ``_fetch_leaderboard_manifest_and_scores``.
    averaged_scores : dict[int, pd.DataFrame]
        Un-narrowed averaged-results DataFrames keyed by manifest row index.
    bootstrap_scores : dict[int, pd.DataFrame]
        Un-narrowed bootstrap-results DataFrames keyed by manifest row index
        (empty if comparisons is None).
    primary_metric : str
        The primary metric to rank by (a bare metric name).
    metric_sort_ascending : bool
        Whether lower values of the primary metric are better.
    comparisons : str | None
        Pairwise significance method ("CLD", "tiers"), or None to skip
        significance testing.
    endpoint : str
        Which endpoint's leaderboard to build — one of ``track_paths.endpoints``
        (e.g. "Structure" for the structure track's one endpoint), or
        ``MACRO_ENDPOINT_LABEL`` ("MA") for the master leaderboard.
    additional_columns : list[str]
        Extra bare columns to copy from each entry's narrowed averaged results
        onto the leaderboard (e.g. ``["coverage_mean"]`` for structure).

    Returns
    -------
    pd.DataFrame
        The built leaderboard DataFrame.

    """
    # Every track's averaged-results/bootstrap files hold every endpoint's data, wide
    # and prefixed (see evaluate_predictions.pivot_endpoint_results_wide). Narrow both
    # down to just the endpoint this leaderboard targets, so `primary_metric` is a bare
    # metric name (e.g. "ST-RAE") regardless of which endpoint/track this leaderboard is
    # for. Bootstrap files hold one row per (Sample, Endpoint) pair — without
    # narrowing, EntryComparison's merge-on-"Sample" would fan out across every
    # endpoint.
    averaged_scores = {
        idx: _narrow_averaged_results_to_endpoint(df, endpoint)
        for idx, df in averaged_scores.items()
    }
    bootstrap_scores = {
        idx: df[df["Endpoint"] == endpoint].drop(columns=["Endpoint"])
        for idx, df in bootstrap_scores.items()
    }

    entries: list[LeaderboardEntry] = []
    for idx, row in manifest.iterrows():
        submitted_at = pd.to_datetime(row["submitted_at"], utc=True)
        entry = LeaderboardEntry(
            username=row["username"],
            anonymous=bool(row["anonymous"]),
            user_alias=row["user_alias"],
            submitted_at=submitted_at,
            model_report_link=row.get("model_report_link") or "",
            used_proprietary_data=bool(row.get("used_proprietary_data", False)),
            open_source_code=bool(row.get("open_source_code", False)),
            averaged_results=averaged_scores[idx],
            bootstrap_data=bootstrap_scores.get(idx),
        )
        entries.append(entry)

    leaderboard = FinalLeaderboard(
        entries=entries,
        primary_metric=primary_metric,
        metric_sort_ascending=metric_sort_ascending,
        significant_method=comparisons,
        additional_columns=additional_columns,
    )

    if leaderboard.leaderboard_df is not None:
        return leaderboard.leaderboard_df
    return pd.DataFrame()


def create_track_leaderboards(
    track_paths: TrackPaths,
    stage: str,
    primary_metric: str,
    metric_sort_ascending: bool,
    significant_method: str | None = "tiers",
    additional_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Build every leaderboard variant for a track: one per endpoint, plus a master.

    A track with a single endpoint (e.g. structure) gets exactly one leaderboard,
    keyed by that endpoint. A track with more than one endpoint (e.g. activity) gets
    one leaderboard per endpoint, plus one additional macro-ranked master leaderboard
    (slug ``MACRO_ENDPOINT_LABEL``, "MA") ranked on the macro-averaged metrics across
    all endpoints. Every variant shares the exact same manifest and score files —
    they differ only in which endpoint's columns get narrowed down to and ranked by —
    so that shared data is fetched once and reused across variants.

    Significance testing (pairwise bootstrap comparisons) is expensive and only ever
    applied to the master leaderboard (or the track's one leaderboard, for a
    single-endpoint track) — never to the individual per-endpoint leaderboards of a
    multi-endpoint track — and only for "interim"/"final" stages, never "live".

    Parameters
    ----------
    track_paths : TrackPaths
        The track to build for.
    stage : str
        Which leaderboard to generate:

        - "live": auto-generated throughout the entire competition, scored on the
          phase 1 compound subset, with no submission cutoff, no significance
          testing, and no invalid-submission filtering — always reflects every
          submission made so far.
        - "interim": generated manually, scored on all compounds, limited to
          submissions before ``INTERIM_LEADERBOARD_DEADLINE``. Includes
          significance testing on the master leaderboard; does not filter out
          invalid submissions.
        - "final": generated manually, scored on all compounds, limited to
          submissions before ``FINAL_LEADERBOARD_DEADLINE``. Includes
          significance testing on the master leaderboard and filters out invalid
          submissions.
    primary_metric : str
        The primary metric to rank by (a bare metric name).
    metric_sort_ascending : bool
        Whether lower values of the primary metric are better.
    significant_method : str | None
        Pairwise significance method ("CLD", "tiers") applied to the master
        leaderboard when ``stage`` is "interim" or "final". Ignored (no
        significance testing) when ``stage`` is "live". Defaults to "tiers".
        Multiple-testing correction is Benjamini-Hochberg (see
        ``FinalLeaderboard._perform_pairwise_comparisons``).
    additional_columns : list[str] | None
        Extra bare columns to copy onto every leaderboard variant. Defaults to
        none.

    Returns
    -------
    dict[str, pd.DataFrame | None]
        Maps each endpoint (plus ``MACRO_ENDPOINT_LABEL`` for a multi-endpoint
        track) to its leaderboard DataFrame (empty if there are no eligible
        submissions), or ``None`` if building that one variant raised. Each
        variant is built in isolation — one endpoint's bad data can't prevent the
        others from being returned.

    """
    if not track_paths.endpoints:
        # A track can legitimately have zero endpoints (e.g. CLASSIFICATION_ENDPOINTS
        # before that track launches, or if it's ever pulled) — without this guard,
        # track_paths.endpoints[0] below would raise IndexError.
        logger.info(
            "Track {} has no endpoints — skipping leaderboard generation.",
            track_paths.track,
        )
        return {}

    comparisons = significant_method if stage != "live" else None
    remove_invalid = stage == "final"
    logger.info(
        "Generating {} {} leaderboards with comparisons={} and remove_invalid={}",
        stage,
        track_paths.track,
        comparisons,
        remove_invalid,
    )

    manifest, averaged_scores, bootstrap_scores = (
        _fetch_leaderboard_manifest_and_scores(
            track=track_paths.track,
            stage=stage,
            remove_invalid=remove_invalid,
            with_bootstrap=bool(comparisons),
        )
    )

    is_multi_endpoint = len(track_paths.endpoints) > 1
    master_slug = (
        MACRO_ENDPOINT_LABEL if is_multi_endpoint else track_paths.endpoints[0]
    )
    slugs = (
        [*track_paths.endpoints, MACRO_ENDPOINT_LABEL]
        if is_multi_endpoint
        else list(track_paths.endpoints)
    )

    if manifest.empty:
        return dict.fromkeys(slugs, manifest)

    leaderboards: dict[str, pd.DataFrame | None] = {}
    for slug in slugs:
        try:
            leaderboards[slug] = _build_leaderboard(
                manifest=manifest,
                averaged_scores=averaged_scores,
                bootstrap_scores=bootstrap_scores,
                primary_metric=primary_metric,
                metric_sort_ascending=metric_sort_ascending,
                comparisons=comparisons if slug == master_slug else None,
                endpoint=slug,
                additional_columns=additional_columns or [],
            )
        except Exception:
            logger.exception(
                "Failed to build {} {} leaderboard", track_paths.track, slug
            )
            leaderboards[slug] = None
    return leaderboards


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def get_live_leaderboard(track: str, endpoint_slug: str) -> pd.DataFrame | None:
    """Fetch the latest live leaderboard from S3. Thin wrapper around ``get_leaderboard``."""
    return get_leaderboard(track, stage="live", endpoint_slug=endpoint_slug)


def save_live_leaderboard(
    leaderboard_df: pd.DataFrame, track: str, endpoint_slug: str
) -> str:
    """Save the live leaderboard to S3. Thin wrapper around ``save_leaderboard``."""
    return save_leaderboard(
        leaderboard_df, track, stage="live", endpoint_slug=endpoint_slug
    )
