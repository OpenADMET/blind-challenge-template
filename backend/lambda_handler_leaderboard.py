"""Lambda handler for generating leaderboard CSVs from scored submission outputs."""

import os

from loguru import logger

from .aws_leaderboards import (
    create_track_leaderboards,
    get_leaderboard,
    index_leaderboard_by_rank,
    save_leaderboard,
)
from .config import (
    CLASSIFICATION_PATHS,
    REGRESSION_PATHS,
    SORT_CLASSIFICATION_LEADERBOARD_BY,
    SORT_REGRESSION_LEADERBOARD_BY,
    SORT_STRUCTURE_LEADERBOARD_BY,
    STRUCTURE_PATHS,
    TrackPaths,
)

# Per-track ranking config. ``create_track_leaderboards`` picks the single- vs.
# multi-endpoint path from ``TrackPaths.endpoints``; everything else needed to build
# and save a track's leaderboard(s) lives here. The "classification" entry is built
# conditionally — a track with no endpoints (CLASSIFICATION_ENDPOINTS = [], e.g.
# before that track launches, or if it's ever pulled) shouldn't even attempt
# leaderboard generation; create_track_leaderboards also guards this itself, so this
# is belt-and-suspenders and avoids reporting a spurious "failed to generate" for a
# track that was never meant to exist yet.
_TRACKS: dict[str, tuple[TrackPaths, str, bool, list[str]]] = {
    "regression": (REGRESSION_PATHS, SORT_REGRESSION_LEADERBOARD_BY, True, []),
    "structure": (
        STRUCTURE_PATHS,
        SORT_STRUCTURE_LEADERBOARD_BY,
        False,
        ["coverage_mean"],
    ),
}
if CLASSIFICATION_PATHS.endpoints:
    _TRACKS["classification"] = (
        CLASSIFICATION_PATHS,
        SORT_CLASSIFICATION_LEADERBOARD_BY,
        False,
        [],
    )


def handler(event, context):
    """Generate and write leaderboard CSVs for all active tracks."""
    os.environ["HOME"] = "/tmp"
    os.environ["DUCKDB_HOME"] = "/tmp"

    results: dict = {"statusCode": 200}

    # Isolated per track: a failure generating one track's leaderboard(s) should not
    # prevent the other track from generating.
    for track_name, track_config in _TRACKS.items():
        try:
            results[track_name] = _generate_track_leaderboards(*track_config)
        except Exception:
            logger.exception("Failed to generate {} leaderboards", track_name)
            results[track_name] = {
                "message": f"Failed to generate {track_name} leaderboards."
            }

    return results


def _generate_track_leaderboards(
    track_paths: TrackPaths,
    primary_metric: str,
    metric_sort_ascending: bool,
    additional_columns: list[str],
) -> dict:
    """Generate and conditionally save every live leaderboard variant for a track.

    A single-endpoint track (e.g. structure) produces one variant; a multi-endpoint
    track (e.g. regression) produces one leaderboard per endpoint plus the
    macro-ranked master leaderboard — see ``aws_leaderboards.create_track_leaderboards``.

    Returns
    -------
    dict
        Status message per leaderboard variant.

    """
    leaderboards = create_track_leaderboards(
        track_paths=track_paths,
        stage="live",
        primary_metric=primary_metric,
        metric_sort_ascending=metric_sort_ascending,
        additional_columns=additional_columns,
    )

    results = {}
    for endpoint_slug, leaderboard in leaderboards.items():
        if leaderboard is None:
            # Already logged inside create_track_leaderboards — building this one
            # variant raised, isolated so it doesn't affect the others.
            results[endpoint_slug] = {
                "message": f"Failed to generate {track_paths.track} ({endpoint_slug}) "
                "leaderboard."
            }
            continue
        # Isolated per variant: a failure saving one leaderboard shouldn't prevent the
        # other variants from being saved.
        try:
            results[endpoint_slug] = _save_leaderboard_if_changed(
                leaderboard, track_paths.track, endpoint_slug
            )
        except Exception:
            logger.exception(
                "Failed to save {} ({}) leaderboard", track_paths.track, endpoint_slug
            )
            results[endpoint_slug] = {
                "message": f"Failed to generate {track_paths.track} ({endpoint_slug}) "
                "leaderboard."
            }
    return results


def _save_leaderboard_if_changed(leaderboard, track: str, endpoint_slug: str) -> dict:
    """Conditionally save one already-built leaderboard variant.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        The leaderboard DataFrame, as built by ``create_track_leaderboards``.
    track : str
        "regression", "classification", or "structure" — used for logging and the
        S3 path.
    endpoint_slug : str
        S3 filename slug for this variant.

    Returns
    -------
    dict
        Status message with row count.

    """
    if leaderboard.empty:
        return {"message": f"No scored {track} submissions found.", "rows": 0}

    leaderboard = index_leaderboard_by_rank(leaderboard)
    previous = get_leaderboard(track=track, stage="live", endpoint_slug=endpoint_slug)
    if previous is not None and leaderboard.round(2).equals(previous.round(2)):
        logger.info(
            "No changes to {} ({}) leaderboard since last generation.",
            track,
            endpoint_slug,
        )
        return {
            "message": f"No changes to {track} ({endpoint_slug}) leaderboard.",
            "rows": len(leaderboard),
        }

    save_path = save_leaderboard(
        leaderboard, track=track, stage="live", endpoint_slug=endpoint_slug
    )
    logger.info("Wrote {} ({}) leaderboard to {}", track, endpoint_slug, save_path)
    return {"message": f"Wrote {track} ({endpoint_slug}) leaderboard to {save_path}"}
