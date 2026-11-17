"""Release files for the Space's head-to-head tab, published after the challenge.

The Space can't read ground truth (its IAM user is denied ``ground_truth/*``) or
anyone's predictions and scores (it only writes ``submissions/*``). After the final
deadline, ``scripts/final_leaderboard.ipynb`` writes everything the tab needs for
each track under ``TrackPaths.head_to_head``, which the Space can read:

- ``ground_truth``: one row per test compound: ``Molecule_Name``, ``live`` (scored
  on the live leaderboard) and, for tabular tracks, the endpoint values and any
  credible-interval columns.
- ``entries``: one row per final master-leaderboard entry: ``rank`` and
  ``display_name`` (the alias for anonymous entrants).
- ``pairwise``: the master leaderboard's pairwise comparisons, by rank.
- ``bootstrap``: each entrant's bootstrap scores, by rank.
- ``predictions`` (tabular tracks): each entrant's predictions, by rank.
- ``per_compound`` (structure): each entrant's per-compound scores, by rank.

Entrants are identified only by rank, so no username reaches the Space; anonymous
entrants appear only under their alias. Writing is refused before
``FINAL_LEADERBOARD_DEADLINE``, so nothing can be released mid-challenge.
"""

from collections.abc import Callable
from datetime import datetime, timezone

import awswrangler as wr
import pandas as pd
from loguru import logger

from .aws_leaderboards import TrackLeaderboards
from .config import (
    FINAL_LEADERBOARD_DEADLINE,
    REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX,
    REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX,
    S3_BUCKET,
    TrackPaths,
)

TABULAR_FILES = ("ground_truth", "entries", "pairwise", "bootstrap", "predictions")
STRUCTURE_FILES = ("ground_truth", "entries", "pairwise", "bootstrap", "per_compound")
PAIRWISE_COLUMNS = [
    "rank_a",
    "rank_b",
    "abs_mean_diff",
    "p_value",
    "adjustment_rank",
    "adjusted_threshold",
    "significant_difference",
]

Reader = Callable[[str], pd.DataFrame]


def release_files(track_paths: TrackPaths) -> tuple[str, ...]:
    """Return the names of a track's release files.

    Parameters
    ----------
    track_paths : TrackPaths
        The track.

    Returns
    -------
    tuple[str, ...]
        ``STRUCTURE_FILES`` for a structure dataset, else ``TABULAR_FILES``.

    """
    return STRUCTURE_FILES if track_paths.spec.dataset == "structure" else TABULAR_FILES


def read_s3_table(uri: str) -> pd.DataFrame:
    """Read a CSV or parquet file from S3.

    Parameters
    ----------
    uri : str
        Its ``s3://`` URI.

    Returns
    -------
    pd.DataFrame
        The table.

    """
    if uri.endswith(".csv"):
        return wr.s3.read_csv(uri)
    return wr.s3.read_parquet(uri)


def build_ground_truth(
    track_paths: TrackPaths,
    identifiers: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build ``ground_truth``: every test compound, flagged live or held out.

    Parameters
    ----------
    track_paths : TrackPaths
        The track.
    identifiers : pd.DataFrame
        The track's identifiers (``Molecule_Name``, ``set``, optionally ``SMILES``).
    ground_truth : pd.DataFrame | None
        The activity ground truth. Required for tabular tracks, ignored for
        structure.

    Returns
    -------
    pd.DataFrame
        One row per compound.

    Raises
    ------
    ValueError
        If a tabular track has no ground truth, or it lacks an endpoint or a
        compound.

    """
    columns = [c for c in ("Molecule_Name", "SMILES") if c in identifiers.columns]
    compounds = identifiers[columns].copy()
    compounds["live"] = (identifiers["set"] == "live").to_numpy()
    if track_paths.spec.dataset == "structure":
        return compounds.reset_index(drop=True)

    if ground_truth is None:
        raise ValueError(f"ground truth is required for the {track_paths.track} track")
    missing = set(track_paths.endpoints) - set(ground_truth.columns)
    if missing:
        raise ValueError(f"ground truth has no {sorted(missing)} columns")
    truth_columns = [
        column
        for endpoint in track_paths.endpoints
        for column in (
            endpoint,
            f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX}",
            f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX}",
        )
        if column in ground_truth.columns
    ]
    merged = compounds.merge(
        ground_truth[["Molecule_Name", *truth_columns]],
        on="Molecule_Name",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged.loc[merged["_merge"] != "both", "Molecule_Name"]
    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} compounds aren't in the ground truth, "
            f"e.g. {unmatched.iloc[0]}"
        )
    return merged.drop(columns="_merge").reset_index(drop=True)


def _entrants(master: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Join each master-board entry to its manifest row (rank, username, URIs)."""
    board = master.reset_index() if master.index.name == "rank" else master
    uris = manifest[["username", "original_data_uri", "scores_uri_for_leaderboard"]]
    entrants = board[["rank", "username", "user_alias", "anonymous"]].merge(
        uris, on="username", how="left", validate="one_to_one"
    )
    missing = entrants.loc[entrants["original_data_uri"].isna(), "username"]
    if not missing.empty:
        raise ValueError(f"no manifest row for {sorted(missing)}")
    return entrants.sort_values("rank").reset_index(drop=True)


def build_entries(entrants: pd.DataFrame) -> pd.DataFrame:
    """Build ``entries``: each entrant's rank and the name shown on the leaderboard.

    Parameters
    ----------
    entrants : pd.DataFrame
        The master board joined to the manifest.

    Returns
    -------
    pd.DataFrame
        ``rank`` and ``display_name`` (the alias for anonymous entrants).

    """
    anonymous = entrants["anonymous"].fillna(False).astype(bool)
    display = entrants["username"].where(~anonymous, entrants["user_alias"])
    return pd.DataFrame({"rank": entrants["rank"], "display_name": display})


def build_pairwise(comparisons: pd.DataFrame, entrants: pd.DataFrame) -> pd.DataFrame:
    """Build ``pairwise``: the master board's comparisons, by rank.

    Parameters
    ----------
    comparisons : pd.DataFrame
        ``FinalLeaderboard.comparison_df`` (``entry_a``/``entry_b`` usernames).
    entrants : pd.DataFrame
        The master board joined to the manifest.

    Returns
    -------
    pd.DataFrame
        ``PAIRWISE_COLUMNS``.

    """
    rank_of = dict(zip(entrants["username"], entrants["rank"]))
    pairwise = comparisons.assign(
        rank_a=comparisons["entry_a"].map(rank_of),
        rank_b=comparisons["entry_b"].map(rank_of),
    )
    return pairwise[PAIRWISE_COLUMNS].reset_index(drop=True)


def _per_entrant(
    entrants: pd.DataFrame, uri_of: Callable[[pd.Series], str], read: Reader
) -> pd.DataFrame:
    """Read one file per entrant and stack them, each row tagged with its rank."""
    frames = []
    for _, entrant in entrants.iterrows():
        frame = read(uri_of(entrant))
        frames.append(frame.assign(rank=entrant["rank"]))
    return pd.concat(frames, ignore_index=True)


def _scores_file(entrant: pd.Series, filename: str) -> str:
    """Swap the averaged-results filename in an entrant's scores URI."""
    return str(entrant["scores_uri_for_leaderboard"]).rsplit("/", 1)[0] + "/" + filename


def build_head_to_head_release(
    track_paths: TrackPaths,
    track_leaderboards: TrackLeaderboards,
    identifiers: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
    read: Reader = read_s3_table,
) -> dict[str, pd.DataFrame]:
    """Build every head-to-head release file for one track.

    Parameters
    ----------
    track_paths : TrackPaths
        The track.
    track_leaderboards : TrackLeaderboards
        The track's final leaderboards, from ``build_track_leaderboards``.
    identifiers : pd.DataFrame
        The track's identifiers.
    ground_truth : pd.DataFrame | None
        The activity ground truth (tabular tracks).
    read : Reader
        Reads an entrant's prediction or scores file (for tests); defaults to S3.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by ``release_files(track_paths)``.

    Raises
    ------
    ValueError
        If there's no master leaderboard or no pairwise comparisons.

    """
    master = track_leaderboards.leaderboards.get(track_leaderboards.master_slug or "")
    if master is None or master.empty:
        raise ValueError(f"no master leaderboard for the {track_paths.track} track")
    if track_leaderboards.comparisons is None:
        raise ValueError(
            f"no pairwise comparisons for the {track_paths.track} track; build the "
            "final leaderboards with significance testing"
        )
    entrants = _entrants(master, track_leaderboards.manifest)
    files = {
        "ground_truth": build_ground_truth(track_paths, identifiers, ground_truth),
        "entries": build_entries(entrants),
        "pairwise": build_pairwise(track_leaderboards.comparisons, entrants),
        "bootstrap": _per_entrant(
            entrants, lambda e: _scores_file(e, "bootstrap-results.parquet"), read
        ),
    }
    if track_paths.spec.dataset == "structure":
        files["per_compound"] = _per_entrant(
            entrants, lambda e: _scores_file(e, "per-compound-results.parquet"), read
        )
    else:
        predictions = _per_entrant(
            entrants, lambda e: str(e["original_data_uri"]), read
        )
        keep = ["rank", "Molecule_Name", *track_paths.endpoints]
        files["predictions"] = predictions[keep]
    return files


def save_head_to_head_release(
    track_paths: TrackPaths,
    files: dict[str, pd.DataFrame],
    force: bool = False,
    now: datetime | None = None,
) -> list[str]:
    """Upload a track's head-to-head release files.

    Parameters
    ----------
    track_paths : TrackPaths
        The track.
    files : dict[str, pd.DataFrame]
        From ``build_head_to_head_release``.
    force : bool
        Upload even before ``FINAL_LEADERBOARD_DEADLINE``. Only for testing against
        a non-production bucket.
    now : datetime | None
        The current time (for tests); defaults to now.

    Returns
    -------
    list[str]
        The S3 URIs written.

    Raises
    ------
    PermissionError
        If it's before the final deadline and ``force`` isn't set.
    ValueError
        If ``files`` isn't exactly the track's release files.

    """
    now = now or datetime.now(timezone.utc)
    if now <= pd.Timestamp(FINAL_LEADERBOARD_DEADLINE) and not force:
        raise PermissionError(
            "Refusing to release ground truth and predictions before "
            f"FINAL_LEADERBOARD_DEADLINE ({FINAL_LEADERBOARD_DEADLINE})"
        )
    expected = release_files(track_paths)
    if set(files) != set(expected):
        raise ValueError(f"expected files {expected}, got {sorted(files)}")
    paths = []
    for name in expected:
        path = f"s3://{S3_BUCKET}/{track_paths.head_to_head}/{name}.parquet"
        wr.s3.to_parquet(df=files[name], path=path, index=False)
        logger.info("Wrote {} rows to {}", len(files[name]), path)
        paths.append(path)
    return paths
