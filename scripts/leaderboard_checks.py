"""Checks run on freshly built interim and final leaderboards before they're uploaded.

Each finding is a *blocker* (stop the upload) or an item for *review* (a person
should look, but the boards may be fine). Only each track's master leaderboard (the
macro-averaged "MA" board, or a single-endpoint track's one board) is checked in
depth: the per-endpoint boards are built the same way. Everything is driven by the
track's ``TrackSpec``, so a new track needs no new code here.

Pure functions on DataFrames: no S3, so they're unit-tested directly.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from backend.config import MACRO_ENDPOINT_LABEL, TrackSpec

TIER_COLUMN = "Significance (tiers)"

# The values each metric can take; a value outside its range means a scoring bug.
# None means unbounded on that side. Metrics not listed aren't range-checked.
METRIC_RANGES: dict[str, tuple[float | None, float | None]] = {
    "MAE": (0, None),
    "ST-RAE": (0, None),
    "R2": (None, 1),
    "Spearman_R": (-1, 1),
    "Kendall_Tau": (-1, 1),
    "MCC": (-1, 1),
    "Accuracy": (0, 1),
    "Precision": (0, 1),
    "Recall": (0, 1),
    "F1": (0, 1),
    "LDDT-PLI": (0, 1),
    "LDDT-LP": (0, 1),
    "BiSyRMSD": (0, None),
}

# Movement from the live board worth a look.
RANK_MOVE = 5  # places
METRIC_MOVE_STDS = 2  # primary-metric change, in the board's own std
TOP_N = 20  # only the top of the board is checked for movement
MIN_RANK_CORRELATION = 0.7
# A newest entry this long before the deadline is suspicious (stale data?).
STALE_GAP = timedelta(days=3)


@dataclass
class CheckResult:
    """What the checks found.

    Attributes
    ----------
    blockers : list[str]
        Problems that must stop the upload.
    review : list[str]
        Things a person should look at before uploading.

    """

    blockers: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)

    def extend(self, other: "CheckResult") -> None:
        """Add another result's findings to this one.

        Parameters
        ----------
        other : CheckResult
            The findings to add.

        Returns
        -------
        None

        """
        self.blockers.extend(other.blockers)
        self.review.extend(other.review)

    @property
    def ok(self) -> bool:
        """Whether there's nothing blocking the upload."""
        return not self.blockers


def master_slug(spec: TrackSpec) -> str:
    """Return the slug of a track's master leaderboard.

    Parameters
    ----------
    spec : TrackSpec
        The track.

    Returns
    -------
    str
        ``MACRO_ENDPOINT_LABEL`` for a multi-endpoint track, else its one endpoint.

    """
    return MACRO_ENDPOINT_LABEL if len(spec.endpoints) > 1 else spec.endpoints[0]


def as_ranked(board: pd.DataFrame) -> pd.DataFrame:
    """Return a board with ``rank`` as a column, sorted by rank.

    Parameters
    ----------
    board : pd.DataFrame
        A leaderboard, with ``rank`` as a column or as the index.

    Returns
    -------
    pd.DataFrame
        The board, sorted by rank.

    """
    if board.index.name == "rank":
        board = board.reset_index()
    return board.sort_values("rank").reset_index(drop=True)


def check_build(
    spec: TrackSpec, boards: Mapping[str, pd.DataFrame | None]
) -> CheckResult:
    """Check every leaderboard variant built and the master board has entries.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    boards : Mapping[str, pd.DataFrame | None]
        ``create_track_leaderboards``'s result: slug to board, or None if that board
        failed to build.

    Returns
    -------
    CheckResult
        Blockers for failed builds, and a review item for an empty master board.

    """
    result = CheckResult()
    failed = sorted(slug for slug, board in boards.items() if board is None)
    if failed:
        result.blockers.append(f"{spec.track}: failed to build {failed}")
    master = boards.get(master_slug(spec))
    if master is not None and master.empty:
        result.review.append(f"{spec.track}: no eligible entries (empty leaderboard)")
    return result


def check_board(
    spec: TrackSpec, board: pd.DataFrame, deadline: pd.Timestamp, tiers: bool
) -> CheckResult:
    """Check a master leaderboard's structure, cutoff and values.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    board : pd.DataFrame
        The master leaderboard (not empty).
    deadline : pd.Timestamp
        The stage's submission deadline (UTC).
    tiers : bool
        Whether the board should have significance tiers.

    Returns
    -------
    CheckResult
        The findings.

    """
    result = CheckResult()
    board = as_ranked(board)
    name = spec.track
    primary = f"{spec.primary_metric}_mean"

    submitted = pd.to_datetime(board["submitted_at"], utc=True)
    late = int((submitted > deadline).sum())
    if late:
        result.blockers.append(f"{name}: {late} entries after the deadline {deadline}")
    elif deadline - submitted.max() > STALE_GAP:
        result.review.append(
            f"{name}: the newest entry is {deadline - submitted.max()} before the "
            "deadline (fine for a dry run; otherwise check the data is up to date)"
        )

    duplicates = board.loc[board["username"].duplicated(), "username"].tolist()
    if duplicates:
        result.blockers.append(f"{name}: users with more than one entry {duplicates}")
    if list(board["rank"]) != list(range(1, len(board) + 1)):
        result.blockers.append(f"{name}: ranks aren't 1 to {len(board)}")
    if primary not in board.columns:
        result.blockers.append(f"{name}: no {primary} column")
        return result
    if board[primary].isna().any():
        result.blockers.append(
            f"{name}: {int(board[primary].isna().sum())} entries have no {primary}"
        )
    ordered = (
        board[primary].is_monotonic_decreasing
        if spec.higher_is_better
        else board[primary].is_monotonic_increasing
    )
    if not ordered:
        result.blockers.append(f"{name}: rows aren't in {primary} order")

    if tiers:
        if TIER_COLUMN not in board.columns:
            result.blockers.append(f"{name}: no {TIER_COLUMN!r} column")
        elif not str(board[TIER_COLUMN].iloc[0]).strip():
            result.blockers.append(f"{name}: the top entry has no tier")

    for metric in spec.metrics:
        column = f"{metric}_mean"
        low, high = METRIC_RANGES.get(metric, (None, None))
        if column not in board.columns:
            continue
        values = board[column].dropna()
        if (low is not None and (values < low).any()) or (
            high is not None and (values > high).any()
        ):
            result.blockers.append(
                f"{name}: {column} outside its range [{low}, {high}]"
            )
    return result


def check_against_live(
    spec: TrackSpec,
    board: pd.DataFrame,
    live: pd.DataFrame | None,
    deadline: pd.Timestamp,
) -> tuple[CheckResult, pd.DataFrame]:
    """Compare a master leaderboard with the live one.

    The live board ranks only the live compound set and has no cutoff, so every
    entrant on this board should be on it, but ranks can move. Big moves near the
    top, and entrants who submitted before the deadline but aren't on this board
    (on the final board, usually a missing model report), are listed for review.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    board : pd.DataFrame
        The master leaderboard being checked (not empty).
    live : pd.DataFrame | None
        The live master leaderboard, or None if there isn't one.
    deadline : pd.Timestamp
        This stage's submission deadline (UTC).

    Returns
    -------
    tuple[CheckResult, pd.DataFrame]
        The findings, and the top of this board joined to the live ranks (empty
        without a live board).

    """
    result = CheckResult()
    name = spec.track
    if live is None or live.empty:
        result.review.append(f"{name}: no live leaderboard to compare with")
        return result, pd.DataFrame()
    board, live = as_ranked(board), as_ranked(live)
    primary = f"{spec.primary_metric}_mean"
    std = f"{spec.primary_metric}_std"

    not_on_live = sorted(set(board["username"]) - set(live["username"]))
    if not_on_live:
        result.blockers.append(
            f"{name}: {len(not_on_live)} entrants aren't on the live board: "
            f"{not_on_live[:10]}"
        )
    live_before = live[pd.to_datetime(live["submitted_at"], utc=True) <= deadline]
    missing = sorted(set(live_before["username"]) - set(board["username"]))
    if missing:
        result.review.append(
            f"{name}: {len(missing)} entrants submitted before the deadline but "
            f"aren't on this board (missing model report?): {missing[:20]}"
        )

    joined = board.merge(
        live[["username", "rank", primary]],
        on="username",
        suffixes=("", "_live"),
        how="inner",
    )
    if len(joined) >= 3:
        rho = joined["rank"].corr(joined["rank_live"], method="spearman")
        if rho < MIN_RANK_CORRELATION:
            result.review.append(
                f"{name}: ranks differ a lot from live (Spearman {rho:.2f} over "
                f"{len(joined)} entrants)"
            )
    joined["rank_change"] = joined["rank"] - joined["rank_live"]
    joined["metric_change"] = joined[primary] - joined[f"{primary}_live"]
    big_metric_move = (joined[std] > 0) & (
        joined["metric_change"].abs() > METRIC_MOVE_STDS * joined[std]
    )
    joined["big_move"] = (joined["rank_change"].abs() >= RANK_MOVE) | big_metric_move
    top = joined.sort_values("rank").head(TOP_N)
    movers = top.loc[top["big_move"], "username"].tolist()
    if movers:
        result.review.append(
            f"{name}: big moves from live in the top {TOP_N}: {movers}"
        )
    columns = [
        "rank",
        "rank_live",
        "rank_change",
        "username",
        primary,
        f"{primary}_live",
        "metric_change",
        "big_move",
    ]
    return result, top[columns]


def check_track(
    spec: TrackSpec,
    boards: Mapping[str, pd.DataFrame | None],
    deadline: pd.Timestamp,
    live: pd.DataFrame | None,
    tiers: bool = True,
) -> tuple[CheckResult, pd.DataFrame]:
    """Run every check on one track's freshly built leaderboards.

    Parameters
    ----------
    spec : TrackSpec
        The track.
    boards : Mapping[str, pd.DataFrame | None]
        ``create_track_leaderboards``'s result.
    deadline : pd.Timestamp
        The stage's submission deadline (UTC).
    live : pd.DataFrame | None
        The live master leaderboard, or None.
    tiers : bool
        Whether the master board should have significance tiers.

    Returns
    -------
    tuple[CheckResult, pd.DataFrame]
        The findings, and the top of the board compared with live.

    """
    result = check_build(spec, boards)
    master = boards.get(master_slug(spec))
    if master is None or master.empty:
        return result, pd.DataFrame()
    result.extend(check_board(spec, master, deadline, tiers))
    movement, top = check_against_live(spec, master, live, deadline)
    result.extend(movement)
    return result, top
