"""Scoring of regression, classification, and structure submissions, saving to S3."""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

import awswrangler as wr
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from loguru import logger

from .config import (
    ACTIVITY_METRICS,
    ACTIVITY_PATHS,
    BOOTSTRAP_SAMPLES,
    CLASSIFICATION_ENDPOINTS,
    CLASSIFICATION_METRICS,
    CLASSIFICATION_PATHS,
    REGRESSION_ENDPOINTS,
    REGRESSION_PATHS,
    S3_BUCKET,
    STRUCTURE_DATASET_SIZE,
    STRUCTURE_PATHS,
    SubmissionKey,
    TrackPaths,
)
from .discord_bot import post_result_to_discord
from .evaluate_predictions import (
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    bootstrap_structure_metrics,
    pivot_endpoint_results_wide,
    score_activity_predictions,
    score_structure_predictions,
)
from .submission_validation import (
    ValidationResult,
    validate_classification_submission,
    validate_regression_submission,
    validate_structure_submission,
)

S3_CLIENT = boto3.client("s3")

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def fetch_submission_metadata(sk: SubmissionKey) -> dict:
    """Fetch the submission metadata for a given user and submission UUID.

    Parameters
    ----------
    sk : SubmissionKey
        The S3 key components for the submission to fetch, including the track,
        user ID, submission ID, and filename.

    Returns
    -------
    dict
        The metadata dictionary for the specified submission.

    Raises
    ------
    Exception
        If there is an error fetching the submission metadata from S3.

    """
    try:
        metadata_key = sk.prefix + "/metadata.json"
        response = S3_CLIENT.get_object(Bucket=S3_BUCKET, Key=metadata_key)
        metadata = json.loads(response["Body"].read())
        logger.info(
            "Successfully fetched submission metadata {} ({})",
            sk.user_id,
            sk.submission_id,
        )
        return metadata
    except Exception as e:
        logger.error("Error fetching submission metadata for {}: {}", metadata_key, e)
        raise e


def save_validated_submission_metadata(
    sk: SubmissionKey, metadata: pd.DataFrame, track_paths: TrackPaths
) -> None:
    """Save the metadata of a validated submission to S3.

    The metadata is saved in a manifest directory with the filename format:
        "{user_id}_{submission_uuid}.parquet"
    to allow for easy querying of the latest submission from each user when creating
    the leaderboards.

    Parameters
    ----------
    sk : SubmissionKey
        The S3 key components for the submission to fetch, including the track,
        user ID, submission ID, and filename.
    metadata : pd.DataFrame
        The metadata to save for the validated submission.
    track_paths : TrackPaths
        Which track's manifest to write to — e.g.
        ``REGRESSION_PATHS``/``CLASSIFICATION_PATHS`` for an activity submission
        (one submission writes a manifest row per track), or ``STRUCTURE_PATHS``.
        Not necessarily the same track as ``sk.track``: an activity submission's
        ``sk.track`` is always "activity" (the single upload prefix), but its
        scores/manifest rows are written per scoring track.

    """
    s3_metadata_path = (
        f"s3://{S3_BUCKET}/{track_paths.manifest}/"
        f"{sk.user_id}_{sk.submission_id}.parquet"
    )
    wr.s3.to_parquet(df=metadata, path=s3_metadata_path, index=False)
    logger.info(
        "Saved {} validated submission metadata to {}",
        track_paths.track,
        s3_metadata_path,
    )


def create_validation_metadata(
    valid_submission: bool,
    submission_metadata: dict,
    sk: SubmissionKey,
    track_paths: TrackPaths,
) -> pd.DataFrame:
    """Create a metadata DataFrame for a validated submission (either track).

    Parameters
    ----------
    valid_submission : bool
        Whether the submission passed validation.
    submission_metadata : dict
        Original metadata from the user's submission.
    sk : SubmissionKey
        Parsed S3 key for the submission — used to build ``original_data_uri`` and
        the score URIs.
    track_paths : TrackPaths
        Which track this manifest row is for — see
        ``save_validated_submission_metadata``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with validation outcome and score URIs.

    """
    scores_all_base = (
        f"s3://{S3_BUCKET}/{track_paths.scores_all}/{sk.user_id}/{sk.submission_id}"
    )
    scores_phase_1_base = (
        f"s3://{S3_BUCKET}/{track_paths.scores_phase_1}/{sk.user_id}/{sk.submission_id}"
    )
    metadata = {
        **submission_metadata,
        "valid_submission": valid_submission,
        "original_data_uri": f"s3://{S3_BUCKET}/{sk.key}",
        "all_scores_uri": f"{scores_all_base}/averaged-results.parquet",
        "phase_1_scores_uri": f"{scores_phase_1_base}/averaged-results.parquet",
    }
    df = pd.DataFrame([metadata])
    # Ensure correct types for faster loading of parquet files
    df["submitted_at"] = pd.to_datetime(df["submitted_at"], utc=True)
    df["valid_submission"] = df["valid_submission"].astype(bool)
    df["anonymous"] = df["anonymous"].astype(bool)
    df["used_proprietary_data"] = df["used_proprietary_data"].fillna(False).astype(bool)
    df["open_source_code"] = df["open_source_code"].fillna(False).astype(bool)
    return df


def load_track_identifiers(paths: TrackPaths) -> pd.DataFrame:
    """Load a track's identifiers parquet from S3.

    Parameters
    ----------
    paths : TrackPaths
        Track path helper (``ACTIVITY_PATHS`` or ``STRUCTURE_PATHS``) selecting
        where to load from.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``Molecule_Name`` column and a ``phase`` column
        (1 = interim, 2 = final).

    Raises
    ------
    FileNotFoundError
        If the identifiers file is not present in S3.

    """
    path = f"s3://{S3_BUCKET}/{paths.ground_truth}/{paths.track}-identifiers.parquet"
    try:
        return wr.s3.read_parquet(path)
    except wr.exceptions.NoFilesFound:
        raise FileNotFoundError(
            f"{paths.track.capitalize()} identifiers not found in S3: {path}"
        ) from None


def fetch_tabular_submission_data(sk: SubmissionKey) -> pd.DataFrame:
    """Fetch a tabular (parquet/csv) submission's data for a given submission.

    Shared by the regression and classification tracks — both submit a single
    parquet/csv file, differing only in which columns it's expected to contain.

    Parameters
    ----------
    sk : SubmissionKey
        The S3 key components for the submission to fetch, including the track,
        user ID, submission ID, and filename.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the submission predictions.

    Raises
    ------
    ValueError
        If the submission file format is unsupported.
    FileNotFoundError
        If no valid submission file is found in S3.
    Exception
        If there is an error fetching the submission data from S3.

    """
    try:
        suffix = PurePosixPath(sk.filename).suffix
        mapping: dict[str, Callable[[str], pd.DataFrame]] = {
            ".parquet": wr.s3.read_parquet,
            ".csv": wr.s3.read_csv,
        }
        reader = mapping.get(suffix)
        if reader is None:
            raise ValueError(f"Unsupported file format: {suffix}")
        s3_url = f"s3://{S3_BUCKET}/{sk.key}"
        try:
            submissions_df = reader(s3_url)
        except wr.exceptions.NoFilesFound:
            raise FileNotFoundError("No valid submission file found.") from None
        logger.info(
            "Successfully fetched submission data for user {}, submission ID {}",
            sk.user_id,
            sk.submission_id,
        )
        return submissions_df
    except Exception as e:
        logger.error("Error fetching submission data for {}: {}", sk.key, e)
        raise e


def load_activity_ground_truth() -> pd.DataFrame:
    """Load the full activity ground truth dataset from S3.

    Shared by the regression and classification tracks — both score against the same
    underlying compound set, just different columns. Returns the complete,
    unfiltered dataset — phase-based filtering (to the phase 1 subset) happens in
    ``score_regression_submission``/``score_classification_submission`` instead, so
    this only needs to be fetched once per submission regardless of how many phases
    get scored.

    Returns
    -------
    pd.DataFrame
        The full activity ground truth dataset.

    Raises
    ------
    FileNotFoundError
        If the ground truth file is not present in S3.

    """
    path = f"s3://{S3_BUCKET}/{ACTIVITY_PATHS.ground_truth}/activity-dataset.parquet"
    try:
        return wr.s3.read_parquet(path)
    except wr.exceptions.NoFilesFound:
        raise FileNotFoundError(
            f"Activity ground truth not found in S3: {path}"
        ) from None


def _score_tabular_submission(
    submissions_df: pd.DataFrame,
    phase: int,
    endpoints: list[str],
    metrics: list,
    ground_truth: pd.DataFrame | None,
    test_identifiers: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared scoring body for a single regression/classification submission.

    Parameters
    ----------
    submissions_df : pd.DataFrame
        The DataFrame containing the submission predictions.
    phase : int
        The phase number to determine which test data to use for scoring.
        0 = all compounds; 1 = interim subset only.
    endpoints : list[str]
        This track's endpoints — ``REGRESSION_ENDPOINTS`` or
        ``CLASSIFICATION_ENDPOINTS``.
    metrics : list
        This track's metrics — ``ACTIVITY_METRICS`` or ``CLASSIFICATION_METRICS``.
    ground_truth : pd.DataFrame | None
        The full activity ground truth. Required — must be provided by the caller
        (see ``load_activity_ground_truth``).
    test_identifiers : pd.DataFrame | None
        Optional preloaded activity identifiers (with "phase" and "Molecule_Name"
        columns), used to filter ground truth down to the requested phase's
        compounds. Loaded from S3 if not provided. Ignored when ``phase == 0``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        This track's scored bootstrap results (one row per bootstrap sample per
        endpoint, including its own macro pseudo-endpoint) and its wide,
        single-row averaged results (one column per endpoint/metric/statistic) —
        see ``evaluate_predictions.add_macro_endpoint``.

    Raises
    ------
    ValueError
        If ``ground_truth`` is not provided.

    """
    if ground_truth is None:
        raise ValueError("ground_truth is required")
    if phase != 0:
        if test_identifiers is None:
            test_identifiers = load_track_identifiers(ACTIVITY_PATHS)
        phase_ids = set(
            test_identifiers.loc[test_identifiers["phase"] == phase, "Molecule_Name"]
        )
        ground_truth = ground_truth[ground_truth["Molecule_Name"].isin(phase_ids)]

    bootstrap_results = score_activity_predictions(
        submissions_df, ground_truth, endpoints=endpoints
    )
    track_bootstrap_results = add_macro_endpoint(
        bootstrap_results, endpoints=endpoints, metrics=metrics
    )
    by_endpoint_results = average_bootstrap_results_by_endpoint(
        track_bootstrap_results
    )
    result_df = pivot_endpoint_results_wide(by_endpoint_results)
    return track_bootstrap_results, result_df


def score_regression_submission(
    submissions_df: pd.DataFrame,
    phase: int,
    ground_truth: pd.DataFrame | None = None,
    test_identifiers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a regression (pIC50) submission against the ground truth."""
    return _score_tabular_submission(
        submissions_df,
        phase,
        REGRESSION_ENDPOINTS,
        ACTIVITY_METRICS,
        ground_truth,
        test_identifiers,
    )


def score_classification_submission(
    submissions_df: pd.DataFrame,
    phase: int,
    ground_truth: pd.DataFrame | None = None,
    test_identifiers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a classification submission against the ground truth."""
    return _score_tabular_submission(
        submissions_df,
        phase,
        CLASSIFICATION_ENDPOINTS,
        CLASSIFICATION_METRICS,
        ground_truth,
        test_identifiers,
    )


def _process_new_tabular_submission(
    sk: SubmissionKey,
    track_paths: TrackPaths,
    validate: Callable[[pd.DataFrame, set[str] | None], ValidationResult],
    score: Callable[
        [pd.DataFrame, int, pd.DataFrame | None, pd.DataFrame | None],
        tuple[pd.DataFrame, pd.DataFrame],
    ],
) -> None:
    """Shared pipeline body for a single regression/classification submission.

    Parameters
    ----------
    sk : SubmissionKey
        The S3 key components for the submission to fetch, including the track,
        user ID, submission ID, and filename.
    track_paths : TrackPaths
        This track's ``TrackPaths`` (``REGRESSION_PATHS`` or
        ``CLASSIFICATION_PATHS``) — selects where scores/manifest are written.
    validate
        This track's validator (``validate_regression_submission`` or
        ``validate_classification_submission``).
    score
        This track's scorer (``score_regression_submission`` or
        ``score_classification_submission``).

    """
    logger.info(
        "Processing new {} submission for user {} with submission ID {}",
        track_paths.track,
        sk.user_id,
        sk.submission_id,
    )
    submissions_df = fetch_tabular_submission_data(sk)
    submission_metadata = fetch_submission_metadata(sk)

    try:
        identifiers_df = load_track_identifiers(ACTIVITY_PATHS)
        identifiers = set(identifiers_df["Molecule_Name"])
    except FileNotFoundError:
        logger.warning(
            "Activity identifiers not found — falling back to count-only validation."
        )
        identifiers_df = None
        identifiers = None

    validation_result = validate(submissions_df, identifiers)
    logger.info("Valid: {}", validation_result.is_valid)

    if validation_result.is_valid:
        try:
            ground_truth_df = load_activity_ground_truth()
            for phase in [0, 1]:
                scored_bootstrap_results, result_df = score(
                    submissions_df, phase, ground_truth_df, identifiers_df
                )
                scores_path = track_paths.scores_paths.get(phase)
                # Bootstrapped score saved for further analysis and plotting
                bootstrapped_score_path = (
                    f"s3://{S3_BUCKET}/{scores_path}/"
                    f"{sk.user_id}/{sk.submission_id}/bootstrap-results.parquet"
                )
                wr.s3.to_parquet(
                    df=scored_bootstrap_results, path=bootstrapped_score_path
                )
                # Averaged scores saved for the leaderboards
                score_path = (
                    f"s3://{S3_BUCKET}/{scores_path}/{sk.user_id}/"
                    f"{sk.submission_id}/averaged-results.parquet"
                )
                wr.s3.to_parquet(df=result_df, path=score_path)
                logger.info(
                    "Saved phase {} {} scores for user {} with submission ID {} to {}",
                    phase,
                    track_paths.track,
                    sk.user_id,
                    sk.submission_id,
                    score_path,
                )
            validation_result.scoring = "complete"
        except Exception as e:
            logger.error(
                "Error scoring submission for user {} with submission ID {}: {}",
                sk.user_id,
                sk.submission_id,
                e,
            )
            validation_result.scoring = "failed"
            validation_result.is_valid = False
    else:
        for error in validation_result.errors:
            logger.error(error.to_long())
        logger.info(
            "Not scoring invalid submission from user {} with submission ID {}",
            sk.user_id,
            sk.submission_id,
        )

    validation_metadata = create_validation_metadata(
        validation_result.is_valid, submission_metadata, sk, track_paths
    )
    save_validated_submission_metadata(sk, validation_metadata, track_paths)
    post_result_to_discord(validation_metadata, validation_result)


# ---------------------------------------------------------------------------
# Regression track
# ---------------------------------------------------------------------------


def process_new_regression_submission(sk: SubmissionKey) -> None:
    """Validate, score, and persist a new regression submission from S3.

    Parameters
    ----------
    sk : SubmissionKey
        The S3 key components for the submission to fetch, including the track,
        user ID, submission ID, and filename.

    """
    _process_new_tabular_submission(
        sk, REGRESSION_PATHS, validate_regression_submission, score_regression_submission
    )


# ---------------------------------------------------------------------------
# Classification track
# ---------------------------------------------------------------------------


def process_new_classification_submission(sk: SubmissionKey) -> None:
    """Validate, score, and persist a new classification submission from S3.

    Parameters
    ----------
    sk : SubmissionKey
        The S3 key components for the submission to fetch, including the track,
        user ID, submission ID, and filename.

    """
    _process_new_tabular_submission(
        sk,
        CLASSIFICATION_PATHS,
        validate_classification_submission,
        score_classification_submission,
    )


# ---------------------------------------------------------------------------
# Structure track
# ---------------------------------------------------------------------------


def _extract_pdb_files(zip_path: Path, extract_dir: Path) -> dict[str, str]:
    """Extract PDB files from a zip and return a mol_id → path mapping.

    Parameters
    ----------
    zip_path : Path
        Path to the zip file.
    extract_dir : Path
        Directory to extract into.

    Returns
    -------
    dict[str, str]
        Mapping from molecule ID (PDB stem) to extracted path.

    """
    extract_dir = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (extract_dir / member.filename).resolve()
            if not member_path.is_relative_to(extract_dir):
                raise ValueError(f"Unsafe path in zip: {member.filename!r}")
        zf.extractall(extract_dir)
        pdb_names = [name for name in zf.namelist() if name.endswith(".pdb")]
    return {Path(name).stem: str(extract_dir / name) for name in pdb_names}


def _is_missing_key_error(e: ClientError) -> bool:
    """Report if a boto3 ClientError means the requested S3 key does not exist."""
    return e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}


def _download_and_extract_structure_zip(
    key: str, tmp_prefix: str, zip_filename: str, missing_message: str
) -> tuple[Path, dict[str, str]]:
    """Download a structure zip from S3 to a fresh /tmp dir and extract its PDB files.

    Parameters
    ----------
    key : str
        S3 object key of the zip file (no bucket).
    tmp_prefix : str
        Prefix for the ``tempfile.mkdtemp`` directory.
    zip_filename : str
        Filename to give the downloaded zip on disk.
    missing_message : str
        Error message to raise if the key does not exist.

    Returns
    -------
    tuple[Path, dict[str, str]]
        The temporary directory (caller is responsible for cleaning it up) and a
        mapping from molecule ID (PDB stem) to extracted filesystem path.

    Raises
    ------
    FileNotFoundError
        If the zip object does not exist in S3.

    """
    tmp_dir = Path(tempfile.mkdtemp(prefix=tmp_prefix))
    try:
        zip_path = tmp_dir / zip_filename
        logger.info("Downloading {} to {}", key, zip_path)
        try:
            S3_CLIENT.download_file(S3_BUCKET, key, str(zip_path))
        except ClientError as e:
            if _is_missing_key_error(e):
                raise FileNotFoundError(missing_message) from None
            raise

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        extracted = _extract_pdb_files(zip_path, extract_dir)
        return tmp_dir, extracted
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def fetch_structure_submission_data(
    sk: SubmissionKey,
) -> tuple[Path, Path, dict[str, str]]:
    """Download a structure submission zip from S3 and extract PDB files to /tmp.

    Parameters
    ----------
    sk : SubmissionKey
        Parsed S3 key for the submission.

    Returns
    -------
    tuple[Path, Path, dict[str, str]]
        The temporary submission directory, the downloaded zip path, and a
        mapping from molecule ID (stem of each PDB filename) to its extracted
        filesystem path.

    Raises
    ------
    FileNotFoundError
        If the zip object does not exist in S3.

    """
    s3_url = f"s3://{S3_BUCKET}/{sk.key}"
    tmp_dir, predicted = _download_and_extract_structure_zip(
        key=sk.key,
        tmp_prefix=f"submission_{sk.submission_id}_",
        zip_filename=sk.filename,
        missing_message=f"Structure submission not found in S3: {s3_url}",
    )
    logger.info(
        "Extracted {} PDB files from submission {}", len(predicted), sk.submission_id
    )
    return tmp_dir, tmp_dir / sk.filename, predicted


def load_structure_ground_truth() -> tuple[Path, dict[str, str]]:
    """Download and extract the structure ground truth zip from S3 to /tmp.

    Returns
    -------
    tuple[Path, dict[str, str]]
        Temporary extraction directory and mapping from molecule ID (PDB stem) to
        extracted filesystem path.

    Raises
    ------
    FileNotFoundError
        If the ground truth zip is not present in S3.

    """
    key = f"{STRUCTURE_PATHS.ground_truth}/structure-dataset.zip"
    s3_url = f"s3://{S3_BUCKET}/{key}"
    tmp_dir, ground_truth = _download_and_extract_structure_zip(
        key=key,
        tmp_prefix="ground_truth_structure_",
        zip_filename="structure-dataset.zip",
        missing_message=f"Structure ground truth not found in S3: {s3_url}",
    )
    logger.info("Loaded {} ground truth structures", len(ground_truth))
    return tmp_dir, ground_truth


def score_structure_submission(
    predicted: dict[str, str],
    phase: int,
    ground_truth: dict[str, str] | None = None,
    identifiers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Score a structure submission against the ground truth.

    Parameters
    ----------
    predicted : dict[str, str]
        Mapping from molecule ID to predicted PDB path, as returned by
        ``fetch_structure_submission_data``.
    phase : int
        0 = all compounds; 1 = interim subset only (filtered via
        ``structure-identifiers.parquet``).
    ground_truth : dict[str, str] | None
        Optional preloaded ground-truth mapping to avoid repeated
        downloads/extraction.
    identifiers : pd.DataFrame | None
        Optional preloaded structure identifiers.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]
        A tuple of (per_compound_df, bootstrap_df, averaged_df, pb_failures)
        where per_compound_df contains per-compound raw scores, bootstrap_df the
        full bootstrap results, averaged_df the single-row, endpoint-prefixed
        mean/std per metric (see ``evaluate_predictions.pivot_endpoint_results_wide``),
        and pb_failures maps molecule ID to the PoseBusters check names that caused
        its scores to be zeroed (only compounds exceeding the failure threshold appear).

    """
    if ground_truth is None:
        _, ground_truth = load_structure_ground_truth()
    per_compound_df, pb_failures = score_structure_predictions(predicted, ground_truth)

    if phase != 0:
        phase_identifiers = (
            identifiers
            if identifiers is not None
            else load_track_identifiers(STRUCTURE_PATHS)
        )
        interim_ids = set(
            phase_identifiers.loc[phase_identifiers["phase"] == phase, "Molecule_Name"]
        )
        per_compound_df = per_compound_df[
            per_compound_df["Molecule_Name"].isin(interim_ids)
        ]

    bootstrap_df = bootstrap_structure_metrics(per_compound_df, BOOTSTRAP_SAMPLES)
    by_endpoint_results = average_bootstrap_results_by_endpoint(bootstrap_df)
    # Coverage is a scalar property of the submission, not a distributional quantity —
    # store the plain mean without bootstrapped uncertainty. Added alongside the other
    # metrics before pivoting so it flows through the same endpoint-prefixed wide format
    # (see pivot_endpoint_results_wide) as every other track.
    by_endpoint_results["coverage_mean"] = per_compound_df["coverage"].mean()
    averaged_df = pivot_endpoint_results_wide(by_endpoint_results)
    return per_compound_df, bootstrap_df, averaged_df, pb_failures


def process_new_structure_submission(sk: SubmissionKey) -> None:
    """Validate, score, and persist a new structure submission from S3.

    Parameters
    ----------
    sk : SubmissionKey
        Parsed S3 key for the submission file.

    """
    logger.info(
        "Processing new structure submission for user {} with submission ID {}",
        sk.user_id,
        sk.submission_id,
    )
    submission_metadata = fetch_submission_metadata(sk)

    submission_tmp_dir, zip_path, predicted = fetch_structure_submission_data(sk)
    ground_truth_tmp_dir: Path | None = None
    try:
        try:
            structure_identifiers = load_track_identifiers(STRUCTURE_PATHS)
            expected_ids = set(structure_identifiers["Molecule_Name"])
        except FileNotFoundError:
            logger.warning(
                "Structure identifiers not found in S3 — "
                "falling back to count-only validation."
            )
            expected_ids = None
            structure_identifiers = None
        validation_result = validate_structure_submission(
            zip_path, expected_ids=expected_ids
        )
        logger.info("Validation result: {}", validation_result.is_valid)

        if validation_result.is_valid:
            try:
                ground_truth_tmp_dir, ground_truth = load_structure_ground_truth()
                for phase in [0, 1]:
                    per_compound_df, bootstrap_df, averaged_df, phase_pb_failures = (
                        score_structure_submission(
                            predicted,
                            phase=phase,
                            ground_truth=ground_truth,
                            identifiers=structure_identifiers,
                        )
                    )
                    scores_path = STRUCTURE_PATHS.scores_paths.get(phase)
                    base = f"s3://{S3_BUCKET}/{scores_path}/{sk.user_id}/{sk.submission_id}"
                    wr.s3.to_parquet(
                        df=per_compound_df, path=f"{base}/per-compound-results.parquet"
                    )
                    wr.s3.to_parquet(
                        df=bootstrap_df, path=f"{base}/bootstrap-results.parquet"
                    )
                    wr.s3.to_parquet(
                        df=averaged_df, path=f"{base}/averaged-results.parquet"
                    )
                    logger.info(
                        "Saved phase {} structure scores for user {} "
                        "submission {} to {}",
                        phase,
                        sk.user_id,
                        sk.submission_id,
                        base,
                    )
                    # Check for failed scoring; collect pb failures from the full set
                    if phase == 0:
                        validation_result.pb_failures = phase_pb_failures
                        failed_scoring = per_compound_df[
                            per_compound_df["coverage"] == 0
                        ]
                        if not failed_scoring.empty:
                            if len(failed_scoring) == STRUCTURE_DATASET_SIZE:
                                validation_result.is_valid = False
                                validation_result.add_error(
                                    "Scoring failed for all structures, "
                                    "please check submission"
                                )
                                break
                            else:
                                validation_result.scoring = "partial"
                                validation_result.scoring_errors = failed_scoring[
                                    "Molecule_Name"
                                ].tolist()

                        else:
                            validation_result.scoring = "complete"
            except Exception as e:
                logger.error(
                    "Error scoring structure submission for user {} "
                    "with submission ID {}: {}",
                    sk.user_id,
                    sk.submission_id,
                    e,
                )
                validation_result.scoring = "failed"
                validation_result.is_valid = False
        else:
            for error in validation_result.errors:
                logger.error(error.to_long())
            logger.info(
                "Not scoring invalid structure submission from user {} submission {}",
                sk.user_id,
                sk.submission_id,
            )

        validation_metadata = create_validation_metadata(
            validation_result.is_valid, submission_metadata, sk, STRUCTURE_PATHS
        )
        save_validated_submission_metadata(sk, validation_metadata, STRUCTURE_PATHS)
        post_result_to_discord(validation_metadata, validation_result)
    finally:
        shutil.rmtree(submission_tmp_dir, ignore_errors=True)
        if ground_truth_tmp_dir is not None:
            shutil.rmtree(ground_truth_tmp_dir, ignore_errors=True)
