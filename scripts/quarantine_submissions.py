"""Take submissions off the leaderboards by moving their manifests into ``quarantine/``.

The leaderboards are built from the manifests under ``manifests/{track}/``, one per
scored submission. Moving a submission's manifest to the same key under
``quarantine/`` drops it from the next leaderboard build and entries list::

    manifests/regression/alice_<sid>.parquet
    -> quarantine/manifests/regression/alice_<sid>.parquet

The upload and its scores stay where they are, and ``--unquarantine`` moves the
manifest back, so nothing is lost. See ``docs/removing-entries.md``.

The input is a CSV with ``submission_id``, ``safe_username`` and ``track`` columns
(extra columns are ignored), e.g. rows copied from the entries list. ``track`` can be
the S3 name (``regression``) or the Space's label (``Regression Prediction``);
usernames are normalised the same way the backend does.

It's a dry run unless ``--execute`` is given. Either way it writes a CSV log of
every move (default ``scripts/output/quarantine-<time>.csv``) for the record and
for reversing it.

Needs AWS credentials that can read and write the bucket (the developer group).

Usage::

    python scripts/quarantine_submissions.py TARGETS.csv --bucket BUCKET [--execute]
        [--unquarantine] [--overwrite] [--log PATH]

``--bucket`` defaults to the ``S3_BUCKET`` environment variable.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.config import (  # noqa: E402
    CLASSIFICATION_PATHS,
    REGRESSION_PATHS,
    STRUCTURE_PATHS,
    TrackPaths,
)
from backend.utils import _safeify_username  # noqa: E402

REQUIRED_COLUMNS = ["submission_id", "safe_username", "track"]
QUARANTINE_PREFIX = "quarantine"
TRACK_PATHS: dict[str, TrackPaths] = {
    paths.spec.track: paths
    for paths in (REGRESSION_PATHS, CLASSIFICATION_PATHS, STRUCTURE_PATHS)
}
LOG_COLUMNS = [
    "submission_id",
    "safe_username",
    "track",
    "source_key",
    "dest_key",
    "status",
]


def normalise_track(value: object) -> str:
    """Return the S3 track name for a ``track`` cell.

    Parameters
    ----------
    value : object
        A track's S3 name (``regression``) or the Space's label (``Regression
        Prediction``), in any case.

    Returns
    -------
    str
        The S3 track name.

    Raises
    ------
    ValueError
        If ``value`` isn't a known track.

    """
    key = str(value).strip().lower().removesuffix(" prediction")
    if key not in TRACK_PATHS:
        raise ValueError(
            f"unknown track {value!r}; expected one of {sorted(TRACK_PATHS)}"
        )
    return key


def load_targets(csv_path: Path) -> pd.DataFrame:
    """Read the submissions to move.

    Parameters
    ----------
    csv_path : Path
        A CSV with ``submission_id``, ``safe_username`` and ``track`` columns.

    Returns
    -------
    pd.DataFrame
        Those columns, with usernames normalised, tracks as S3 names, and blank and
        duplicate rows dropped.

    Raises
    ------
    ValueError
        If a column is missing, a track is unknown, or no rows are left.

    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    df = df[REQUIRED_COLUMNS].apply(lambda column: column.str.strip())
    df = df[~df.eq("").any(axis=1)]
    if df.empty:
        raise ValueError(f"no usable rows in {csv_path}")
    df["safe_username"] = df["safe_username"].map(_safeify_username)
    df["track"] = df["track"].map(normalise_track)
    return df.drop_duplicates().reset_index(drop=True)


def manifest_key(track: str, safe_username: str, submission_id: str) -> str:
    """Return a submission's manifest key.

    Parameters
    ----------
    track : str
        The track's S3 name.
    safe_username : str
        The normalised username.
    submission_id : str
        The submission ID.

    Returns
    -------
    str
        The key, without the bucket.

    """
    return f"{TRACK_PATHS[track].manifest}/{safe_username}_{submission_id}.parquet"


def _exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "404",
            "NoSuchKey",
            "NotFound",
        ):
            return False
        raise
    return True


def plan_moves(
    s3: Any,
    bucket: str,
    targets: pd.DataFrame,
    unquarantine: bool = False,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    """Work out every move, without changing anything.

    Parameters
    ----------
    s3 : Any
        A boto3 S3 client.
    bucket : str
        The challenge bucket.
    targets : pd.DataFrame
        ``load_targets``'s result.
    unquarantine : bool
        Move manifests back out of ``quarantine/``.
    overwrite : bool
        Move even if something is already at the destination.

    Returns
    -------
    list[dict[str, str]]
        One record per submission (``LOG_COLUMNS``), with ``status`` "pending" or
        why it's skipped: "not-found" or "destination-exists".

    """
    moves: list[dict[str, str]] = []
    for row in targets.to_dict("records"):
        submission_id = str(row["submission_id"])
        safe_username = str(row["safe_username"])
        track = str(row["track"])
        live_key = manifest_key(track, safe_username, submission_id)
        quarantined_key = f"{QUARANTINE_PREFIX}/{live_key}"
        source, dest = (
            (quarantined_key, live_key) if unquarantine else (live_key, quarantined_key)
        )
        status = "pending"
        if not _exists(s3, bucket, source):
            status = "not-found"
        elif not overwrite and _exists(s3, bucket, dest):
            status = "destination-exists"
        moves.append(
            {
                "submission_id": submission_id,
                "safe_username": safe_username,
                "track": track,
                "source_key": source,
                "dest_key": dest,
                "status": status,
            }
        )
    return moves


def execute_moves(
    s3: Any, bucket: str, moves: list[dict[str, str]], execute: bool
) -> None:
    """Make the pending moves (or mark them as a dry run), updating each status.

    S3 has no move: each is a copy, then a delete of the original. A failed move is
    recorded and the rest carry on.

    Parameters
    ----------
    s3 : Any
        A boto3 S3 client.
    bucket : str
        The challenge bucket.
    moves : list[dict[str, str]]
        ``plan_moves``'s result; each ``status`` becomes "moved", "dry-run" or
        "error: ...".
    execute : bool
        Actually move; otherwise mark pending moves "dry-run".

    Returns
    -------
    None

    """
    for move in moves:
        if move["status"] != "pending":
            continue
        if not execute:
            move["status"] = "dry-run"
            continue
        try:
            s3.copy_object(
                Bucket=bucket,
                Key=move["dest_key"],
                CopySource={"Bucket": bucket, "Key": move["source_key"]},
            )
            s3.delete_object(Bucket=bucket, Key=move["source_key"])
        except ClientError as exc:
            move["status"] = f"error: {exc.response.get('Error', {}).get('Code', exc)}"
        else:
            move["status"] = "moved"


def write_log(moves: list[dict[str, str]], path: Path, action: str) -> None:
    """Write the moves to a CSV log.

    Parameters
    ----------
    moves : list[dict[str, str]]
        The move records.
    path : Path
        Where to write the log.
    action : str
        "quarantine" or "unquarantine", recorded on every row.

    Returns
    -------
    None

    """
    df = pd.DataFrame(moves, columns=LOG_COLUMNS)
    df.insert(0, "action", action)
    df.insert(0, "timestamp", datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: list[str] | None = None, s3: Any = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments (default: the process's).
    s3 : Any
        A boto3 S3 client (for tests; default: a new one).

    Returns
    -------
    int
        The exit code: 0 if every move was made (or would be, in a dry run).

    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    parser.add_argument("--execute", action="store_true", help="Make the moves.")
    parser.add_argument(
        "--unquarantine", action="store_true", help="Move manifests back."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace what's at the destination."
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / f"quarantine-{stamp}.csv",
    )
    args = parser.parse_args(argv)
    if not args.bucket:
        print("Set --bucket or S3_BUCKET to the challenge bucket.", file=sys.stderr)
        return 2
    try:
        targets = load_targets(args.csv_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    s3 = s3 or boto3.client("s3")
    moves = plan_moves(s3, args.bucket, targets, args.unquarantine, args.overwrite)
    execute_moves(s3, args.bucket, moves, args.execute)
    action = "unquarantine" if args.unquarantine else "quarantine"
    write_log(moves, args.log, action)

    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(pd.DataFrame(moves, columns=LOG_COLUMNS).to_string(index=False))
    print(f"\nLog: {args.log}")
    if not args.execute:
        print("Dry run: nothing moved. Add --execute to move these manifests.")
    else:
        print(
            "The live leaderboard drops these entries on its next scheduled build; "
            "re-run the interim or final notebook if those are published."
        )
    problems = [m for m in moves if m["status"] not in ("moved", "dry-run")]
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
