"""Build, check, save and upload the interim or final leaderboards.

The steps the ``interim_leaderboard.ipynb`` and ``final_leaderboard.ipynb``
notebooks run, one cell each. Needs AWS credentials for the challenge bucket, and
the repo root on ``sys.path`` (the notebooks add it).
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from backend.aws_leaderboards import (
    TrackLeaderboards,
    build_track_leaderboards,
    get_leaderboard,
    index_leaderboard_by_rank,
    save_leaderboard,
)
from backend.aws_submission_processing import (
    load_activity_ground_truth,
    load_track_identifiers,
)
from backend.config import (
    CLASSIFICATION_PATHS,
    FINAL_LEADERBOARD_DEADLINE,
    INTERIM_LEADERBOARD_DEADLINE,
    REGRESSION_PATHS,
    STRUCTURE_PATHS,
    TrackPaths,
)
from backend.head_to_head_release import (
    build_head_to_head_release,
    save_head_to_head_release,
)
from scripts.leaderboard_checks import CheckResult, check_track, master_slug

Boards = dict[str, TrackLeaderboards]  # track -> its built leaderboards

STAGE_DEADLINES = {
    "interim": INTERIM_LEADERBOARD_DEADLINE,
    "final": FINAL_LEADERBOARD_DEADLINE,
}
TRACK_PATHS: dict[str, TrackPaths] = {
    paths.spec.track: paths
    for paths in (REGRESSION_PATHS, CLASSIFICATION_PATHS, STRUCTURE_PATHS)
}
# Local copies for review (gitignored: they can include participants' details).
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _stage_deadline(stage: str) -> pd.Timestamp:
    if stage not in STAGE_DEADLINES:
        raise ValueError(f"stage must be one of {list(STAGE_DEADLINES)}, not {stage!r}")
    return pd.Timestamp(STAGE_DEADLINES[stage])


def build(stage: str) -> Boards:
    """Build every track's leaderboards for a stage.

    Parameters
    ----------
    stage : str
        "interim" or "final".

    Returns
    -------
    Boards
        Track to ``build_track_leaderboards``'s result. Tracks with no endpoints
        are left out.

    """
    _stage_deadline(stage)
    return {
        track: build_track_leaderboards(paths, stage=stage)
        for track, paths in TRACK_PATHS.items()
        if paths.spec.endpoints
    }


def check(
    stage: str,
    built: Boards,
    get_live: Callable[[str, str, str], pd.DataFrame | None] = get_leaderboard,
) -> CheckResult:
    """Check every track's master leaderboard and print what was found.

    Parameters
    ----------
    stage : str
        "interim" or "final".
    built : Boards
        ``build``'s result.
    get_live : Callable[[str, str, str], pd.DataFrame | None]
        Reads a published leaderboard (for tests); defaults to S3.

    Returns
    -------
    CheckResult
        Every track's findings.

    """
    deadline = _stage_deadline(stage)
    result = CheckResult()
    for track, track_boards in built.items():
        spec = TRACK_PATHS[track].spec
        live = get_live(track, "live", master_slug(spec))
        found, top = check_track(spec, track_boards.leaderboards, deadline, live)
        result.extend(found)
        print(f"\n{'=' * 72}\n{track} ({master_slug(spec)})\n{'=' * 72}")
        if not top.empty:
            with pd.option_context("display.width", 200, "display.max_rows", None):
                print(
                    f"Top {len(top)} compared with live:\n{top.to_string(index=False)}"
                )
    print(f"\n{'=' * 72}")
    for item in result.review:
        print("REVIEW:", item)
    for item in result.blockers:
        print("BLOCK: ", item)
    print(
        f"{len(result.blockers)} blocking, {len(result.review)} to review."
        + ("" if result.ok else " Fix the blocking issues before uploading.")
    )
    return result


def save_local(stage: str, built: Boards, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Save every built leaderboard as a CSV, for review and blog posts.

    Parameters
    ----------
    stage : str
        "interim" or "final".
    built : Boards
        ``build``'s result.
    output_dir : Path
        Where to save (a ``{stage}/`` folder is made inside it).

    Returns
    -------
    list[Path]
        The files written.

    """
    folder = output_dir / stage
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for track, track_boards in built.items():
        for slug, board in track_boards.leaderboards.items():
            if board is None or board.empty:
                continue
            path = folder / f"{track}_{slug}.csv"
            board.to_csv(path, index=False)
            written.append(path)
    print(f"Saved {len(written)} leaderboards to {folder}")
    return written


def upload(
    stage: str,
    built: Boards,
    result: CheckResult,
    save: Callable[..., str] = save_leaderboard,
) -> list[str]:
    """Upload every built leaderboard to where the Space reads it.

    Refuses to upload anything if the checks found a blocker or any board failed
    to build, so the Space never shows a partial set. Empty boards are skipped.

    Parameters
    ----------
    stage : str
        "interim" or "final".
    built : Boards
        ``build``'s result.
    result : CheckResult
        ``check``'s result for these boards.
    save : Callable[..., str]
        Writes one leaderboard (for tests); defaults to ``save_leaderboard``.

    Returns
    -------
    list[str]
        The S3 paths written.

    Raises
    ------
    RuntimeError
        If the checks found a blocker or a board failed to build.

    """
    _stage_deadline(stage)
    if not result.ok:
        raise RuntimeError(f"Not uploading: {len(result.blockers)} blocking issues")
    failed = sorted(
        (track, slug)
        for track, track_boards in built.items()
        for slug, board in track_boards.leaderboards.items()
        if board is None
    )
    if failed:
        raise RuntimeError(f"Not uploading: these failed to build: {failed}")
    written = []
    for track, track_boards in built.items():
        for slug, board in track_boards.leaderboards.items():
            assert board is not None  # checked above
            if board.empty:
                print(f"skip     {track:<15} {slug:<30} (no entries)")
                continue
            path = save(
                index_leaderboard_by_rank(board),
                track=track,
                stage=stage,
                endpoint_slug=slug,
            )
            print(f"uploaded {track:<15} {slug:<30} -> {path}")
            written.append(path)
    return written


def release_head_to_head(built: Boards, force: bool = False) -> list[str]:
    """Write the head-to-head tab's release files for every track with entries.

    Run after uploading the final leaderboards. Refused before
    ``FINAL_LEADERBOARD_DEADLINE`` (see ``backend.head_to_head_release``).

    Parameters
    ----------
    built : Boards
        ``build("final")``'s result.
    force : bool
        Write even before the final deadline. Only for testing against a
        non-production bucket.

    Returns
    -------
    list[str]
        The S3 paths written.

    """
    ground_truth = None
    written = []
    for track, track_boards in built.items():
        paths = TRACK_PATHS[track]
        master = track_boards.leaderboards.get(track_boards.master_slug or "")
        if master is None or master.empty:
            print(f"skip     {track:<15} (no final leaderboard)")
            continue
        if paths.spec.dataset == "activity" and ground_truth is None:
            ground_truth = load_activity_ground_truth()
        files = build_head_to_head_release(
            paths,
            track_boards,
            load_track_identifiers(paths),
            ground_truth if paths.spec.dataset == "activity" else None,
        )
        for path in save_head_to_head_release(paths, files, force=force):
            print(f"released {track:<15} -> {path}")
            written.append(path)
    return written
