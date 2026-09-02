"""Configuration file for the blind challenge.

This module is the single source of truth for the challenge's endpoints, dataset
sizes, deadlines, and metrics. When standing up a new challenge from this
template, work through every ``TODO`` below and keep ``hf_space/config.py`` in
sync with the endpoint lists / dataset sizes set here.
"""

import os
from dataclasses import dataclass
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

# Activity dataset
IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]
# TODO: rename to this challenge's regression endpoints. Each name must match a
# column in the ground-truth dataset (and its "{endpoint}_conf_low"/"_conf_high"
# credible-interval columns). Order is preserved through scoring and leaderboards.
# e.g. ["CYP3A4_pIC50_direct_inhibition", "CYP2D6_pIC50_direct_inhibition", ...]
REGRESSION_ENDPOINTS = [
    "ENDPOINT_1",
    "ENDPOINT_2",
    "ENDPOINT_3",
    "ENDPOINT_4",
]
REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX = "_conf_high"
REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX = "_conf_low"
# TODO: rename to this challenge's classification endpoints, or set to [] if this
# challenge has no classification track. e.g. ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]
CLASSIFICATION_ENDPOINTS = [
    "ENDPOINT_5",
    "ENDPOINT_6",
]
ACTIVITY_ENDPOINTS = REGRESSION_ENDPOINTS + CLASSIFICATION_ENDPOINTS
ENDPOINTS_TO_LOG_TRANSFORM: list[str] = []
# TODO: set to the number of compounds in this challenge's activity test set.
ACTIVITY_DATASET_SIZE = 750
ACTIVITY_METRICS = [
    ("ST-RAE", rae_soft_threshold_absolute_error),
    ("MAE", mean_absolute_error),
    ("R2", r2_score),
    ("Spearman_R", spearmanr),
    ("Kendall_Tau", kendalltau),
]
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
# zero_division=0 matches sklearn's documented degenerate-case default, avoiding
# warnings/errors on bootstrap resamples with no positive predictions (classification
# labels are often imbalanced). matthews_corrcoef already returns 0.0 (not NaN) in
# its own degenerate case, so it needs no wrapping.
CLASSIFICATION_METRICS = [
    ("MCC", matthews_corrcoef),
    ("Accuracy", accuracy_score),
    ("Precision", partial(precision_score, zero_division=0)),
    ("Recall", partial(recall_score, zero_division=0)),
    ("F1", partial(f1_score, zero_division=0)),
]
SORT_REGRESSION_LEADERBOARD_BY = "ST-RAE"
SORT_CLASSIFICATION_LEADERBOARD_BY = "MCC"
BOOTSTRAP_SAMPLES = 1000

# Structure dataset
# TODO: set to the number of compounds in this challenge's structure test set
# (0 / unused if this challenge has no structure track).
STRUCTURE_DATASET_SIZE = 184
SORT_STRUCTURE_LEADERBOARD_BY = "LDDT-PLI"
STRUCTURE_METRICS = ["LDDT-PLI", "BiSyRMSD", "LDDT-LP"]
# Penalty applied to BiSyRMSD when OST cannot match the ligand (lower is better,
# so a large value is used; 20 Å is well outside any reasonable binding-site RMSD)
BISYRMSD_NAN_PENALTY: float = 20.0
# TODO: filename of the reference protein structure in the ground-truth data.
PROTEIN_DATA_FILE = "protein_structure.pdb"
# Structure track currently scores a single pseudo-endpoint. A future structure track
# with multiple endpoints would also need its own STRUCTURE_LEADERBOARD_GROUPS.
STRUCTURE_ENDPOINTS = ["structure"]


@dataclass(frozen=True)
class TrackPaths:
    """S3 path helpers for a single competition track.

    All properties return S3 key prefixes (no leading/trailing slashes).

    Submissions are only ever scored against two compound sets:
    - Phase 0 ("all"): the full test set.
    - Phase 1 ("phase_1"): the half of the test set unblinded during phase 1.

    Those two score sets feed three leaderboards:
    - "live": auto-generated by the leaderboard Lambda from phase 1 scores.
    - "interim" / "final": generated manually from phase 0 (all-compound) scores,
      differing only in which submissions are included (by submission cutoff).

    ``endpoints`` drives leaderboard generation (see
    ``aws_leaderboards.create_track_leaderboards``): one leaderboard is built per
    endpoint, plus one additional macro-ranked master leaderboard (slug
    ``MACRO_ENDPOINT_LABEL``) whenever a track has more than one endpoint.
    """

    track: str
    valid_filenames: frozenset[str]
    endpoints: list[str]

    @property
    def submissions(self) -> str:
        return f"submissions/{self.track}"

    @property
    def manifest(self) -> str:
        return f"submissions/manifest/{self.track}"

    @property
    def phase_2_entries(self) -> str:
        return f"submissions/manifest/phase_2_entries/{self.track}"

    @property
    def scores_all(self) -> str:
        return f"scores/all/{self.track}"

    @property
    def scores_phase_1(self) -> str:
        return f"scores/phase_1/{self.track}"

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
    def scores_paths(self) -> dict[int, str]:
        """Map phase integer to scores S3 prefix (0=all, 1=phase_1)."""
        return {0: self.scores_all, 1: self.scores_phase_1}

    @property
    def ground_truth(self) -> str:
        """Ground-truth data prefix. Same for every track — filenames disambiguate."""
        return "ground_truth"

    @property
    def all_entries(self) -> str:
        """Cross-track entries list prefix."""
        return "entries"


# "activity" is no longer an upload track — nothing is ever uploaded to
# submissions/activity/* (hence the empty valid_filenames). ACTIVITY_PATHS is kept
# only as the shared ground-truth/identifiers source (activity-dataset.parquet,
# activity-identifiers.parquet) used by both the regression and classification
# tracks below, since both score against the same underlying compound set.
ACTIVITY_PATHS = TrackPaths(
    track="activity",
    valid_filenames=frozenset(),
    endpoints=ACTIVITY_ENDPOINTS,
)
REGRESSION_PATHS = TrackPaths(
    track="regression",
    valid_filenames=frozenset({"predictions.parquet", "predictions.csv"}),
    endpoints=REGRESSION_ENDPOINTS,
)
CLASSIFICATION_PATHS = TrackPaths(
    track="classification",
    valid_filenames=frozenset({"predictions.parquet", "predictions.csv"}),
    endpoints=CLASSIFICATION_ENDPOINTS,
)
STRUCTURE_PATHS = TrackPaths(
    track="structure",
    valid_filenames=frozenset({"structures.zip"}),
    endpoints=STRUCTURE_ENDPOINTS,
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
