"""Configuration file for the blind challenge.

This module is the single source of truth for the challenge's endpoints, dataset
sizes, deadlines, and metrics. When standing up a new challenge from this
template, work through every ``TODO`` below and keep ``hf_space/config.py`` in
sync with the endpoint lists / dataset sizes set here.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

from .custom_scoring_functions import rae_soft_threshold_absolute_error

# Challenge information
# TODO: set these to this challenge's real cutoffs. Format must stay
# "%Y-%m-%dT%H:%M:%SZ" (UTC). Submissions after INTERIM_LEADERBOARD_DEADLINE are
# excluded from the interim leaderboard; after FINAL_LEADERBOARD_DEADLINE, from
# the final leaderboard. The 2099 values below are placeholders.
INTERIM_LEADERBOARD_DEADLINE = "2099-01-01T23:59:59Z"
FINAL_LEADERBOARD_DEADLINE = "2099-12-31T23:59:59Z"

# Multi-endpoint macro-averaging (see evaluate_predictions.compute_macro_bootstrap_results).
# A pseudo-endpoint, scored alongside the real endpoints in every bootstrap sample, whose
# per-metric values are macro-averages across endpoints rather than raw per-endpoint scores.
MACRO_ENDPOINT_LABEL = "MA"

# AWS S3 configuration
S3_BUCKET: str = os.environ.get("S3_BUCKET", "blind-challenge-template")

# Discord notifications — set DISCORD_NOTIFICATIONS=0 to disable
DISCORD_NOTIFICATIONS: bool = os.environ.get("DISCORD_NOTIFICATIONS", "1") != "0"

# Activity dataset: the compounds the regression and classification tracks share.
IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]
REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX = "_conf_high"
REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX = "_conf_low"
ENDPOINTS_TO_LOG_TRANSFORM: list[str] = []
# TODO: set to the number of compounds in this challenge's activity test set.
ACTIVITY_DATASET_SIZE = 750
# Rank correlations (Spearman_R, Kendall_Tau) are mathematically undefined — scipy
# returns NaN, not an exception — whenever a bootstrap sample has zero variance in
# y_pred (e.g. a submission that predicts the same value for every compound) or
# y_true. That's a legitimate, if uninformative, submission rather than a computation
# error, so bootstrap_metrics substitutes this fallback instead of raising. 0.0 is the
# "no correlation" value on both metrics' [-1, 1] scale, matching how an
# unconditionally-constant predictor should be scored: no better than chance, not a
# hard failure. Metrics not listed here still raise on a non-finite value (see
# bootstrap_metrics) since for e.g. MAE/RAE/R2 that would indicate a real bug rather
# than a valid degenerate submission.
METRIC_NAN_FALLBACK: dict[str, float] = {
    "Spearman_R": 0.0,
    "Kendall_Tau": 0.0,
}
# Scoring function for each activity metric name a TrackSpec can list. Structure
# metrics are computed by OpenStructure per compound, so they have no entry here.
# zero_division=0 matches sklearn's degenerate-case default, avoiding errors on
# bootstrap resamples with no positive predictions; matthews_corrcoef already
# returns 0.0 in that case.
METRIC_FUNCTIONS: dict[str, Callable] = {
    "ST-RAE": rae_soft_threshold_absolute_error,
    "MAE": mean_absolute_error,
    "R2": r2_score,
    "Spearman_R": spearmanr,
    "Kendall_Tau": kendalltau,
    "MCC": matthews_corrcoef,
    "Accuracy": accuracy_score,
    "Precision": partial(precision_score, zero_division=0),
    "Recall": partial(recall_score, zero_division=0),
    "F1": partial(f1_score, zero_division=0),
}
# Number of bootstrap resamples per submission. Samples are drawn once, at scoring
# time, with a fixed seed (utils.BOOTSTRAP_SEED), and pairwise comparisons join two
# submissions' samples on their "Sample" index. Fix B before the challenge opens and
# never change it while it runs: submissions scored with different B would be
# compared on different resample sets. The smallest attainable two-sided p-value
# is 2 / (B + 1).
#
# 10,000 (#11, 11-18): at 1,000, a p-value near the Benjamini-Hochberg cut-offs of a
# large leaderboard varied by up to 5x between seeds, so boundary tier decisions were
# partly chance; 10,000 cuts that noise about 3x and lowers the p-value floor to
# 0.0002. It costs ~70 s per classification scoring run (~35 s regression) on a
# laptop, well inside the evaluators' 900 s, files under 1 MB, and a few seconds
# more for the pairwise comparisons. Tests score with 1,000
# (tests/backend/backend_test_settings.py).
BOOTSTRAP_SAMPLES = 10000

# Structure dataset
# TODO: set to the number of compounds in this challenge's structure test set
# (0 / unused if this challenge has no structure track).
STRUCTURE_DATASET_SIZE = 184
# Penalty applied to BiSyRMSD when OST cannot match the ligand (lower is better,
# so a large value is used; 20 Å is well outside any reasonable binding-site RMSD)
BISYRMSD_NAN_PENALTY: float = 20.0
# Worst-case scores returned for a pose that fails the PoseBusters filter. LDDT values
# floor at 0; BiSyRMSD uses the same penalty as an unscored structure so its contribution
# to the bootstrap mean is identically bad.
POSEBUSTERS_MAX_FAILURES: int = 0
# TODO: filename of the reference protein structure in the ground-truth data.
PROTEIN_DATA_FILE = "protein_structure.pdb"


# Compound sets, from the "set" column of a dataset's identifiers parquet: "live"
# compounds, "hidden" compounds, and "all" of them.
COMPOUND_SETS = ("all", "live", "hidden")
LEADERBOARD_STAGES = ("live", "interim", "final")
# The live leaderboard ranks on the live compounds; interim and final on all of them.
DEFAULT_LEADERBOARD_SETS = {"live": "live", "interim": "all", "final": "all"}


def scores_uri_column(compound_set: str) -> str:
    """Name of the manifest column holding a compound set's averaged-results URI.

    Parameters
    ----------
    compound_set : str
        One of ``COMPOUND_SETS``.

    Returns
    -------
    str
        E.g. ``"live_scores_uri"``.

    """
    return f"{compound_set}_scores_uri"


@dataclass(frozen=True)
class TrackSpec:
    """What a competition track is: its data, endpoints, metrics and ranking.

    Mirrored in ``hf_space/config.py``; ``tests/backend/test_config_drift.py``
    checks the two agree.

    Attributes
    ----------
    track : str
        Track name, used in S3 prefixes ("regression", "classification",
        "structure").
    dataset : str
        The ground-truth dataset scored against: ``ground_truth/{dataset}-*`` in S3.
        Tracks scoring the same compounds share one ("activity").
    endpoints : list[str]
        Endpoint names, in display order. Activity endpoints must match columns in
        the ground-truth dataset. One leaderboard is built per endpoint, plus a
        macro-averaged master (``MACRO_ENDPOINT_LABEL``) when there are several.
    metrics : list[str]
        Metric names, in display order. Activity metrics are looked up in
        ``METRIC_FUNCTIONS``.
    primary_metric : str
        The metric leaderboards are ranked by.
    higher_is_better : bool
        Direction of ``primary_metric``.
    dataset_size : int
        Number of compounds a submission must cover.
    extra_columns : tuple[str, ...]
        Additional averaged-results columns to carry onto the leaderboard.
    leaderboard_sets : dict[str, str]
        The compound set each leaderboard stage ("live", "interim", "final") ranks
        on: "live", "hidden" or "all" (live + hidden). The sets come from the
        ``set`` column of the dataset's identifiers. A submission is scored on
        every set its stages use.

    """

    track: str
    dataset: str
    endpoints: list[str]
    metrics: list[str]
    primary_metric: str
    higher_is_better: bool
    dataset_size: int
    extra_columns: tuple[str, ...] = field(default=())
    leaderboard_sets: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LEADERBOARD_SETS)
    )

    def __post_init__(self) -> None:
        """Reject a spec that can't be ranked or scored consistently.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``primary_metric`` isn't one of ``metrics``, a metric or endpoint is
            listed twice, ``dataset_size`` is negative, or ``leaderboard_sets``
            doesn't map every stage to a known compound set.

        """
        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"{self.track}: primary_metric {self.primary_metric!r} is not one of "
                f"its metrics {self.metrics}"
            )
        for name, values in (("metrics", self.metrics), ("endpoints", self.endpoints)):
            duplicates = sorted({v for v in values if values.count(v) > 1})
            if duplicates:
                raise ValueError(f"{self.track}: duplicate {name} {duplicates}")
        if self.dataset_size < 0:
            raise ValueError(f"{self.track}: dataset_size must not be negative")
        if set(self.leaderboard_sets) != set(LEADERBOARD_STAGES) or not set(
            self.leaderboard_sets.values()
        ) <= set(COMPOUND_SETS):
            raise ValueError(
                f"{self.track}: leaderboard_sets must map each of {LEADERBOARD_STAGES} "
                f"to one of {COMPOUND_SETS}, got {self.leaderboard_sets}"
            )

    @property
    def scored_sets(self) -> list[str]:
        """The compound sets a submission is scored on: those any stage uses.

        Returns
        -------
        list[str]
            In ``COMPOUND_SETS`` order.

        """
        used = set(self.leaderboard_sets.values())
        return [s for s in COMPOUND_SETS if s in used]

    @property
    def metric_functions(self) -> list[tuple[str, Callable]]:
        """The track's ``(name, scoring function)`` pairs, from ``METRIC_FUNCTIONS``.

        Returns
        -------
        list[tuple[str, Callable]]
            One pair per metric, in ``metrics`` order.

        """
        return [(name, METRIC_FUNCTIONS[name]) for name in self.metrics]


# TODO: rename the endpoints to this challenge's. Regression endpoints must match
# columns in the activity ground truth (and their "_conf_low"/"_conf_high"
# credible-interval columns), e.g. "CYP3A4_pIC50_direct_inhibition". Set
# classification's endpoints to [] if this challenge has no classification track.
REGRESSION_SPEC = TrackSpec(
    track="regression",
    dataset="activity",
    endpoints=["ENDPOINT_1", "ENDPOINT_2", "ENDPOINT_3", "ENDPOINT_4"],
    metrics=["ST-RAE", "MAE", "R2", "Spearman_R", "Kendall_Tau"],
    primary_metric="ST-RAE",
    higher_is_better=False,
    dataset_size=ACTIVITY_DATASET_SIZE,
)
CLASSIFICATION_SPEC = TrackSpec(
    track="classification",
    dataset="activity",
    endpoints=["ENDPOINT_5", "ENDPOINT_6"],
    metrics=["MCC", "Accuracy", "Precision", "Recall", "F1"],
    primary_metric="MCC",
    higher_is_better=True,
    dataset_size=ACTIVITY_DATASET_SIZE,
)
# One pseudo-endpoint: structure has a single leaderboard.
STRUCTURE_SPEC = TrackSpec(
    track="structure",
    dataset="structure",
    endpoints=["structure"],
    metrics=["LDDT-PLI", "BiSyRMSD", "LDDT-LP"],
    primary_metric="LDDT-PLI",
    higher_is_better=True,
    dataset_size=STRUCTURE_DATASET_SIZE,
    extra_columns=("coverage_mean",),
)
TRACK_SPECS: dict[str, TrackSpec] = {
    spec.track: spec for spec in (REGRESSION_SPEC, CLASSIFICATION_SPEC, STRUCTURE_SPEC)
}

# Drop entries without a reachable model report link from the final leaderboard.
# Must mirror hf_space.config.REQUIRE_MODEL_REPORT.
REQUIRE_MODEL_REPORT = True

# Ground-truth and cross-track entries-list S3 prefixes, shared by every track.
GROUND_TRUTH_PREFIX = "ground_truth"
ENTRIES_PREFIX = "entries"


@dataclass(frozen=True)
class TrackPaths:
    """S3 path helpers for a single competition track.

    All properties return S3 key prefixes (no leading/trailing slashes).

    ``spec`` is the track's definition; ``track`` and ``endpoints`` come from it.
    """

    spec: TrackSpec
    valid_filenames: frozenset[str]

    @property
    def track(self) -> str:
        return self.spec.track

    @property
    def endpoints(self) -> list[str]:
        return self.spec.endpoints

    @property
    def submissions(self) -> str:
        return f"submissions/{self.track}"

    @property
    def manifest(self) -> str:
        return f"manifests/{self.track}"

    def scores(self, compound_set: str) -> str:
        """Scores prefix for one compound set, e.g. ``scores/live/regression``.

        Parameters
        ----------
        compound_set : str
            One of ``COMPOUND_SETS``.

        Returns
        -------
        str
            The S3 prefix.

        """
        return f"scores/{compound_set}/{self.track}"

    @property
    def leaderboard_live(self) -> str:
        return f"leaderboard/live/{self.track}"

    @property
    def leaderboard_interim(self) -> str:
        return f"leaderboard/interim/{self.track}"

    @property
    def leaderboard_final(self) -> str:
        return f"leaderboard/final/{self.track}"

    @property
    def head_to_head(self) -> str:
        """Release files for the Space's head-to-head tab, written after the challenge.

        The only place the Space can read ground truth, and other entrants'
        predictions and scores; see ``head_to_head_release``.
        """
        return f"{self.leaderboard_final}/head_to_head"

    @property
    def ground_truth(self) -> str:
        """Ground-truth data prefix. Same for every track; filenames disambiguate."""
        return GROUND_TRUTH_PREFIX

    @property
    def identifiers_file(self) -> str:
        """S3 key of the track's dataset identifiers parquet."""
        return f"{GROUND_TRUTH_PREFIX}/{self.spec.dataset}-identifiers.parquet"


REGRESSION_PATHS = TrackPaths(
    spec=REGRESSION_SPEC,
    valid_filenames=frozenset({"predictions.parquet", "predictions.csv"}),
)
CLASSIFICATION_PATHS = TrackPaths(
    spec=CLASSIFICATION_SPEC,
    valid_filenames=frozenset({"predictions.parquet", "predictions.csv"}),
)
STRUCTURE_PATHS = TrackPaths(
    spec=STRUCTURE_SPEC,
    valid_filenames=frozenset({"structures.zip"}),
)


@dataclass(frozen=True)
class SubmissionKey:
    """Parsed S3 key for a single submission file.

    Key format: submissions/{track}/{user_id}/{submission_id}/{filename}
    """

    track: str
    user_id: str
    submission_id: str
    filename: str

    @property
    def key(self) -> str:
        """Full S3 object key."""
        return f"submissions/{self.track}/{self.user_id}/{self.submission_id}/{self.filename}"

    @property
    def prefix(self) -> str:
        """S3 key prefix for the submission directory (without filename)."""
        return f"submissions/{self.track}/{self.user_id}/{self.submission_id}"

    @property
    def is_valid(self) -> bool:
        """True if the filename is a recognised submission file for any track."""
        return self.filename in {
            "predictions.parquet",
            "predictions.csv",
            "structures.zip",
        }

    def is_valid_for(self, track_paths: TrackPaths) -> bool:
        """Return if this key belongs to the given track and has a valid filename."""
        return (
            self.track == track_paths.track
            and self.filename in track_paths.valid_filenames
        )

    @classmethod
    def parse(cls, key: str) -> "SubmissionKey | None":
        """Parse an S3 object key into a SubmissionKey.

        Returns None if the key does not match the expected format
        (submissions/{track}/{user_id}/{submission_id}/{filename}).
        """
        parts = key.strip("/").split("/")
        if len(parts) != 5 or parts[0] != "submissions":
            return None
        _, track, user_id, submission_id, filename = parts
        return cls(
            track=track, user_id=user_id, submission_id=submission_id, filename=filename
        )
