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
    BOOTSTRAP_SAMPLES,
    CLASSIFICATION_PATHS,
    COMPOUND_SETS,
    GROUND_TRUTH_PREFIX,
    REGRESSION_PATHS,
    S3_BUCKET,
    STRUCTURE_PATHS,
    SubmissionKey,
    TrackPaths,
    scores_uri_column,
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
    """Save a validated submission's manifest row to S3.

    Written to ``{track_paths.manifest}/{user_id}_{submission_id}.parquet``, one file
    per submission; the leaderboards read the latest per user.

    Parameters
    ----------
    sk : SubmissionKey
        The S3 key components for the submission to fetch, including the track,
        user ID, submission ID, and filename.
    metadata : pd.DataFrame
        The metadata to save for the validated submission.
    track_paths : TrackPaths
        The submission's track.

    Returns
    -------
    None

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
    metadata = {
        **submission_metadata,
        "valid_submission": valid_submission,
        "original_data_uri": f"s3://{S3_BUCKET}/{sk.key}",
        **{
            scores_uri_column(compound_set): (
                f"s3://{S3_BUCKET}/{track_paths.scores(compound_set)}/"
                f"{sk.user_id}/{sk.submission_id}/averaged-results.parquet"
            )
            for compound_set in track_paths.spec.scored_sets
        },
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
        Any track scoring the dataset to load: regression and classification both
        read ``activity-identifiers.parquet``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``Molecule_Name`` column and a ``set`` column ("live" or
        "hidden").

    Raises
    ------
    FileNotFoundError
        If the identifiers file is not present in S3.

    """
    path = f"s3://{S3_BUCKET}/{paths.identifiers_file}"
    try:
        return wr.s3.read_parquet(path)
    except wr.exceptions.NoFilesFound:
        raise FileNotFoundError(
            f"{paths.spec.dataset.capitalize()} identifiers not found in S3: {path}"
        ) from None


def compound_set_ids(identifiers: pd.DataFrame, compound_set: str) -> set[str] | None:
    """The molecule IDs in a compound set.

    Parameters
    ----------
    identifiers : pd.DataFrame
        The dataset's identifiers, with ``Molecule_Name`` and ``set`` columns.
    compound_set : str
        "live", "hidden" or "all".

    Returns
    -------
    set[str] | None
        The IDs whose ``set`` is ``compound_set``, or None for "all" (no filter).

    Raises
    ------
    ValueError
        If ``compound_set`` isn't known, or ``set`` holds a value other than "live"
        or "hidden".

    """
    if compound_set not in COMPOUND_SETS:
        raise ValueError(f"Unknown compound set {compound_set!r}")
    if compound_set == "all":
        return None
    unknown = set(identifiers["set"]) - {"live", "hidden"}
    if unknown:
        raise ValueError(
            f"Identifiers 'set' column has values other than live/hidden: {unknown}"
        )
    return set(identifiers.loc[identifiers["set"] == compound_set, "Molecule_Name"])


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
    unfiltered dataset; filtering to a compound set happens in
    ``score_regression_submission``/``score_classification_submission``, so this is
    fetched once per submission however many sets are scored.

    Returns
    -------
    pd.DataFrame
        The full activity ground truth dataset.

    Raises
    ------
    FileNotFoundError
        If the ground truth file is not present in S3.

    """
    path = f"s3://{S3_BUCKET}/{GROUND_TRUTH_PREFIX}/activity-dataset.parquet"
    try:
        return wr.s3.read_parquet(path)
    except wr.exceptions.NoFilesFound:
        raise FileNotFoundError(
            f"Activity ground truth not found in S3: {path}"
        ) from None


def _score_tabular_submission(
    submissions_df: pd.DataFrame,
    compound_set: str,
    track_paths: TrackPaths,
    ground_truth: pd.DataFrame | None,
    test_identifiers: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared scoring body for a single regression/classification submission.

    Parameters
    ----------
    submissions_df : pd.DataFrame
        The DataFrame containing the submission predictions.
    compound_set : str
        The compounds to score on: "live", "hidden" or "all".
    track_paths : TrackPaths
        ``REGRESSION_PATHS`` or ``CLASSIFICATION_PATHS``; its spec gives the
        endpoints and metrics scored.
    ground_truth : pd.DataFrame | None
        The full activity ground truth. Required — must be provided by the caller
        (see ``load_activity_ground_truth``).
    test_identifiers : pd.DataFrame | None
        Optional preloaded activity identifiers (with ``set`` and
        ``Molecule_Name`` columns), used to filter ground truth down to the
        compound set. Loaded from S3 if needed and not provided.

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
    if compound_set != "all":
        if test_identifiers is None:
            test_identifiers = load_track_identifiers(track_paths)
        set_ids = compound_set_ids(test_identifiers, compound_set) or set()
        ground_truth = ground_truth[ground_truth["Molecule_Name"].isin(set_ids)]

    spec = track_paths.spec
    bootstrap_results = score_activity_predictions(
        submissions_df, ground_truth, endpoints=spec.endpoints
    )
    track_bootstrap_results = add_macro_endpoint(
        bootstrap_results, endpoints=spec.endpoints, metrics=spec.metric_functions
    )
    by_endpoint_results = average_bootstrap_results_by_endpoint(track_bootstrap_results)
    result_df = pivot_endpoint_results_wide(by_endpoint_results)
    return track_bootstrap_results, result_df


def score_regression_submission(
    submissions_df: pd.DataFrame,
    compound_set: str,
    ground_truth: pd.DataFrame | None = None,
    test_identifiers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a regression (pIC50) submission against the ground truth."""
    return _score_tabular_submission(
        submissions_df,
        compound_set,
        REGRESSION_PATHS,
        ground_truth,
        test_identifiers,
    )


def score_classification_submission(
    submissions_df: pd.DataFrame,
    compound_set: str,
    ground_truth: pd.DataFrame | None = None,
    test_identifiers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a classification submission against the ground truth."""
    return _score_tabular_submission(
        submissions_df,
        compound_set,
        CLASSIFICATION_PATHS,
        ground_truth,
        test_identifiers,
    )


def _process_new_tabular_submission(
    sk: SubmissionKey,
    track_paths: TrackPaths,
    validate: Callable[[pd.DataFrame, set[str] | None], ValidationResult],
    score: Callable[
        [pd.DataFrame, str, pd.DataFrame | None, pd.DataFrame | None],
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
    validate : Callable[[pd.DataFrame, set[str] | None], ValidationResult]
        This track's validator (``validate_regression_submission`` or
        ``validate_classification_submission``).
    score : Callable[[pd.DataFrame, str, pd.DataFrame | None, pd.DataFrame | None], tuple[pd.DataFrame, pd.DataFrame]]
        This track's scorer (``score_regression_submission`` or
        ``score_classification_submission``).

    Returns
    -------
    None

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
        identifiers_df = load_track_identifiers(track_paths)
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
            for compound_set in track_paths.spec.scored_sets:
                scored_bootstrap_results, result_df = score(
                    submissions_df, compound_set, ground_truth_df, identifiers_df
                )
                scores_path = track_paths.scores(compound_set)
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
                    "Saved {} {} scores for user {} with submission ID {} to {}",
                    compound_set,
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

    Returns
    -------
    None

    """
    _process_new_tabular_submission(
        sk,
        REGRESSION_PATHS,
        validate_regression_submission,
        score_regression_submission,
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

    Returns
    -------
    None

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

    Raises
    ------
    ValueError
        If a member's path would extract outside ``extract_dir`` (zip-slip guard).

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
    ClientError
        If the S3 download fails for a reason other than a missing key.
    Exception
        Re-raised after cleaning up the temporary directory, for any other
        failure during download or extraction (e.g. an unsafe path caught by
        ``_extract_pdb_files``).

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

    Notes
    -----
    See ``_download_and_extract_structure_zip`` for exceptions that may
    propagate from the download/extraction it performs.

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

    Notes
    -----
    See ``_download_and_extract_structure_zip`` for exceptions that may
    propagate from the download/extraction it performs.

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


def structure_smiles_map(identifiers: pd.DataFrame | None) -> dict[str, str] | None:
    """Build the molecule ID -> ground-truth SMILES mapping used in scoring.

    The SMILES come from an optional ``SMILES`` column in
    ``structure-identifiers.parquet``. With them, each ligand's bonds are taken
    from its SMILES rather than guessed from atom distances (see
    ``evaluate_predictions.score_single_structure``).

    Parameters
    ----------
    identifiers : pd.DataFrame | None
        The structure identifiers, or None if they couldn't be loaded.

    Returns
    -------
    dict[str, str] | None
        The mapping, skipping blank SMILES, or None when there is no ``SMILES``
        column, in which case ligand bonds are guessed from distances.

    """
    if identifiers is None or "SMILES" not in identifiers.columns:
        logger.warning(
            "Structure identifiers have no SMILES column, so ligand bonds will be "
            "guessed from atom distances. Add SMILES to "
            "structure-identifiers.parquet."
        )
        return None
    with_smiles = identifiers.dropna(subset=["SMILES"])
    with_smiles = with_smiles[with_smiles["SMILES"].astype(str).str.strip() != ""]
    return dict(zip(with_smiles["Molecule_Name"], with_smiles["SMILES"].astype(str)))


def aggregate_structure_scores(
    per_compound_df: pd.DataFrame,
    compound_set: str,
    identifiers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate already-scored compounds into one compound set's scores.

    Cheap: filters ``per_compound_df`` to the set's compounds, bootstraps it and
    averages the bootstrap. No structure is re-scored, so a submission is scored
    once (``score_structure_predictions``) and aggregated once per set. This works
    because a compound's structure scores don't depend on which other compounds
    are in the set. The activity tracks can't do the same: R², Spearman and
    Kendall are whole-sample statistics and must be recomputed per set.

    Parameters
    ----------
    per_compound_df : pd.DataFrame
        Output of ``score_structure_predictions`` for the whole submission.
    compound_set : str
        "live", "hidden" or "all" (see ``compound_set_ids``).
    identifiers : pd.DataFrame | None
        Optional preloaded structure identifiers; loaded from S3 when needed and
        not given.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (per_compound_df, bootstrap_df, averaged_df) for the set: the set's rows
        of ``per_compound_df``, the full bootstrap results, and the
        single-row, endpoint-prefixed mean/std per metric (see
        ``evaluate_predictions.pivot_endpoint_results_wide``).

    """
    if compound_set != "all":
        if identifiers is None:
            identifiers = load_track_identifiers(STRUCTURE_PATHS)
        set_ids = compound_set_ids(identifiers, compound_set) or set()
        per_compound_df = per_compound_df[
            per_compound_df["Molecule_Name"].isin(set_ids)
        ]

    bootstrap_df = bootstrap_structure_metrics(per_compound_df, BOOTSTRAP_SAMPLES)
    by_endpoint_results = average_bootstrap_results_by_endpoint(bootstrap_df)
    # Coverage is a scalar property of the submission, not a distributional quantity —
    # store the plain mean without bootstrapped uncertainty. Added alongside the other
    # metrics before pivoting so it flows through the same endpoint-prefixed wide format
    # (see pivot_endpoint_results_wide) as every other track.
    by_endpoint_results["coverage_mean"] = per_compound_df["coverage"].mean()
    averaged_df = pivot_endpoint_results_wide(by_endpoint_results)
    return per_compound_df, bootstrap_df, averaged_df


def process_new_structure_submission(sk: SubmissionKey) -> None:
    """Validate, score, and persist a new structure submission from S3.

    Parameters
    ----------
    sk : SubmissionKey
        Parsed S3 key for the submission file.

    Returns
    -------
    None

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
                # The expensive step (OST + PoseBusters per compound) runs once;
                # each compound set is a cheap re-aggregation of the same scores.
                all_per_compound_df, pb_failures = score_structure_predictions(
                    predicted,
                    ground_truth,
                    smiles_map=structure_smiles_map(structure_identifiers),
                )
                validation_result.pb_failures = pb_failures
                failed_scoring = all_per_compound_df[
                    all_per_compound_df["coverage"] == 0
                ]
                if len(failed_scoring) == len(all_per_compound_df):
                    validation_result.is_valid = False
                    validation_result.add_error(
                        "Scoring failed for all structures, please check submission"
                    )
                else:
                    if failed_scoring.empty:
                        validation_result.scoring = "complete"
                    else:
                        validation_result.scoring = "partial"
                        validation_result.scoring_errors = failed_scoring[
                            "Molecule_Name"
                        ].tolist()
                    for compound_set in STRUCTURE_PATHS.spec.scored_sets:
                        per_compound_df, bootstrap_df, averaged_df = (
                            aggregate_structure_scores(
                                all_per_compound_df,
                                compound_set=compound_set,
                                identifiers=structure_identifiers,
                            )
                        )
                        base = (
                            f"s3://{S3_BUCKET}/{STRUCTURE_PATHS.scores(compound_set)}/"
                            f"{sk.user_id}/{sk.submission_id}"
                        )
                        wr.s3.to_parquet(
                            df=per_compound_df,
                            path=f"{base}/per-compound-results.parquet",
                        )
                        wr.s3.to_parquet(
                            df=bootstrap_df, path=f"{base}/bootstrap-results.parquet"
                        )
                        wr.s3.to_parquet(
                            df=averaged_df, path=f"{base}/averaged-results.parquet"
                        )
                        logger.info(
                            "Saved {} structure scores for user {} submission {} to {}",
                            compound_set,
                            sk.user_id,
                            sk.submission_id,
                            base,
                        )
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
