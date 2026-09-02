"""Tests for AWS submission scoring/processing functions.

Most of this module talks to S3 (via awswrangler/boto3) or delegates to heavy
scoring functions (openstructure-backed structure scoring, sklearn/scipy activity
metrics) that are exercised in their own test modules. These tests therefore mock
those collaborators at the ``backend.aws_submission_processing`` module boundary
and focus on this module's own control flow: path building, phase filtering,
error translation, and temp-file cleanup.
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import awswrangler as wr
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from backend import aws_submission_processing as asp
from backend.config import (
    ACTIVITY_PATHS,
    CLASSIFICATION_PATHS,
    REGRESSION_PATHS,
    S3_BUCKET,
    STRUCTURE_DATASET_SIZE,
    STRUCTURE_PATHS,
    SubmissionKey,
)
from backend.submission_validation import ValidationResult

REGRESSION_SK = SubmissionKey(
    track="regression",
    user_id="new-user",
    submission_id="submission-123",
    filename="predictions.parquet",
)
CLASSIFICATION_SK = SubmissionKey(
    track="classification",
    user_id="new-user",
    submission_id="submission-123",
    filename="predictions.parquet",
)
STRUCTURE_SK = SubmissionKey(
    track="structure",
    user_id="new-user",
    submission_id="submission-123",
    filename="structures.zip",
)


# ---------------------------------------------------------------------------
# Shared: fetch/save submission metadata
# ---------------------------------------------------------------------------


def test_fetch_submission_metadata_success(monkeypatch):
    """fetch_submission_metadata parses the metadata.json body from S3."""
    payload = {"username": "tester"}

    class FakeBody:
        def read(self):
            return json.dumps(payload).encode()

    def fake_get_object(Bucket, Key):
        assert Bucket == S3_BUCKET
        assert Key == REGRESSION_SK.prefix + "/metadata.json"
        return {"Body": FakeBody()}

    monkeypatch.setattr(asp.S3_CLIENT, "get_object", fake_get_object)
    assert asp.fetch_submission_metadata(REGRESSION_SK) == payload


def test_fetch_submission_metadata_error_propagates(monkeypatch):
    """Errors fetching submission metadata are logged and re-raised."""

    def fake_get_object(Bucket, Key):
        raise ClientError({"Error": {"Code": "404", "Message": "x"}}, "GetObject")

    monkeypatch.setattr(asp.S3_CLIENT, "get_object", fake_get_object)
    with pytest.raises(ClientError):
        asp.fetch_submission_metadata(REGRESSION_SK)


def test_save_validated_submission_metadata_success(monkeypatch):
    """save_validated_submission_metadata writes to the given track's manifest path."""
    captured = {}

    def fake_to_parquet(df, path, index=None):
        captured["df"] = df
        captured["path"] = path
        captured["index"] = index

    monkeypatch.setattr(asp.wr.s3, "to_parquet", fake_to_parquet)
    metadata_df = pd.DataFrame({"valid_submission": [True]})
    asp.save_validated_submission_metadata(
        REGRESSION_SK, metadata_df, REGRESSION_PATHS
    )

    assert captured["path"] == (
        f"s3://{S3_BUCKET}/{REGRESSION_PATHS.manifest}/new-user_submission-123.parquet"
    )
    assert captured["index"] is False


# ---------------------------------------------------------------------------
# Shared: create_validation_metadata
# ---------------------------------------------------------------------------


def test_create_validation_metadata_regression(example_activity_predictions_metadata):
    """Validation metadata for the regression track has the expected columns/dtypes."""
    metadata = asp.create_validation_metadata(
        valid_submission=True,
        submission_metadata=example_activity_predictions_metadata,
        sk=REGRESSION_SK,
        track_paths=REGRESSION_PATHS,
    )
    required_columns = {
        "valid_submission",
        "safe_username",
        "submitted_at",
        "original_data_uri",
        "all_scores_uri",
        "phase_1_scores_uri",
        "open_source_code",
    }
    assert required_columns.issubset(metadata.columns)
    assert metadata["valid_submission"].dtype == bool
    assert metadata["open_source_code"].dtype == bool
    assert metadata["submitted_at"].dtype == "datetime64[ns, UTC]"
    assert metadata["all_scores_uri"].iloc[0] == (
        f"s3://{S3_BUCKET}/{REGRESSION_PATHS.scores_all}/new-user/submission-123/"
        "averaged-results.parquet"
    )


def test_create_validation_metadata_classification(example_activity_predictions_metadata):
    """Validation metadata for the classification track has its own score URIs."""
    metadata = asp.create_validation_metadata(
        valid_submission=True,
        submission_metadata=example_activity_predictions_metadata,
        sk=CLASSIFICATION_SK,
        track_paths=CLASSIFICATION_PATHS,
    )
    assert metadata["original_data_uri"].iloc[0] == (
        f"s3://{S3_BUCKET}/{CLASSIFICATION_SK.key}"
    )
    assert metadata["all_scores_uri"].iloc[0] == (
        f"s3://{S3_BUCKET}/{CLASSIFICATION_PATHS.scores_all}/new-user/submission-123/"
        "averaged-results.parquet"
    )


def test_create_validation_metadata_structure(example_activity_predictions_metadata):
    """Validation metadata is track-aware: structure submissions use structure paths."""
    metadata = asp.create_validation_metadata(
        valid_submission=False,
        submission_metadata=example_activity_predictions_metadata,
        sk=STRUCTURE_SK,
        track_paths=STRUCTURE_PATHS,
    )
    assert not metadata["valid_submission"].iloc[0]
    assert metadata["all_scores_uri"].iloc[0] == (
        f"s3://{S3_BUCKET}/{STRUCTURE_PATHS.scores_all}/new-user/submission-123/"
        "averaged-results.parquet"
    )


# ---------------------------------------------------------------------------
# Shared: load_track_identifiers
# ---------------------------------------------------------------------------


def test_load_track_identifiers_success(monkeypatch):
    """load_track_identifiers returns whatever awswrangler reads back."""
    expected = pd.DataFrame({"Molecule_Name": ["a"], "phase": [1]})
    monkeypatch.setattr(asp.wr.s3, "read_parquet", lambda path: expected)
    assert asp.load_track_identifiers(ACTIVITY_PATHS) is expected


def test_load_track_identifiers_missing_raises_file_not_found(monkeypatch):
    """A missing identifiers file surfaces as FileNotFoundError, not NoFilesFound."""

    def raise_not_found(path):
        raise wr.exceptions.NoFilesFound(f"No files: {path}")

    monkeypatch.setattr(asp.wr.s3, "read_parquet", raise_not_found)
    with pytest.raises(FileNotFoundError, match="Activity identifiers not found"):
        asp.load_track_identifiers(ACTIVITY_PATHS)


def test_load_track_identifiers_reads_activity_path(monkeypatch):
    """load_track_identifiers(ACTIVITY_PATHS) reads from the activity ground-truth prefix."""
    captured = {}

    def fake_read_parquet(path):
        captured["path"] = path
        return pd.DataFrame()

    monkeypatch.setattr(asp.wr.s3, "read_parquet", fake_read_parquet)
    asp.load_track_identifiers(ACTIVITY_PATHS)
    assert captured["path"] == (
        f"s3://{S3_BUCKET}/{ACTIVITY_PATHS.ground_truth}/activity-identifiers.parquet"
    )


def test_load_track_identifiers_reads_structure_path(monkeypatch):
    """load_track_identifiers(STRUCTURE_PATHS) reads from the structure ground-truth prefix."""
    captured = {}

    def fake_read_parquet(path):
        captured["path"] = path
        return pd.DataFrame()

    monkeypatch.setattr(asp.wr.s3, "read_parquet", fake_read_parquet)
    asp.load_track_identifiers(STRUCTURE_PATHS)
    assert captured["path"] == (
        f"s3://{S3_BUCKET}/{STRUCTURE_PATHS.ground_truth}/structure-identifiers.parquet"
    )


# ---------------------------------------------------------------------------
# Shared: load_activity_ground_truth, fetch_tabular_submission_data
# ---------------------------------------------------------------------------


def test_load_activity_ground_truth_success(monkeypatch):
    """load_activity_ground_truth returns the full, unfiltered dataset."""
    expected = pd.DataFrame({"Molecule_Name": ["a"]})
    monkeypatch.setattr(asp.wr.s3, "read_parquet", lambda path: expected)
    assert asp.load_activity_ground_truth() is expected


def test_load_activity_ground_truth_missing_raises_file_not_found(monkeypatch):
    """A missing ground-truth file surfaces as FileNotFoundError, not NoFilesFound."""

    def raise_not_found(path):
        raise wr.exceptions.NoFilesFound("nope")

    monkeypatch.setattr(asp.wr.s3, "read_parquet", raise_not_found)
    with pytest.raises(FileNotFoundError, match="Activity ground truth not found"):
        asp.load_activity_ground_truth()


def test_fetch_tabular_submission_data_parquet(monkeypatch):
    """A .parquet submission is read with wr.s3.read_parquet."""
    expected = pd.DataFrame({"Molecule_Name": ["a"]})
    monkeypatch.setattr(asp.wr.s3, "read_parquet", lambda path: expected)
    assert asp.fetch_tabular_submission_data(REGRESSION_SK) is expected


def test_fetch_tabular_submission_data_csv(monkeypatch):
    """A .csv submission is read with wr.s3.read_csv."""
    csv_sk = SubmissionKey(
        track="classification",
        user_id="u1",
        submission_id="s1",
        filename="predictions.csv",
    )
    expected = pd.DataFrame({"Molecule_Name": ["a"]})
    monkeypatch.setattr(asp.wr.s3, "read_csv", lambda path: expected)
    assert asp.fetch_tabular_submission_data(csv_sk) is expected


def test_fetch_tabular_submission_data_missing_raises_file_not_found(monkeypatch):
    """A missing submission file surfaces as FileNotFoundError, not NoFilesFound."""

    def raise_not_found(path):
        raise wr.exceptions.NoFilesFound("nope")

    monkeypatch.setattr(asp.wr.s3, "read_parquet", raise_not_found)
    with pytest.raises(FileNotFoundError, match="No valid submission file found"):
        asp.fetch_tabular_submission_data(REGRESSION_SK)


def test_fetch_tabular_submission_data_unsupported_format():
    """An unrecognised file extension raises ValueError."""
    bad_sk = SubmissionKey(
        track="regression", user_id="u1", submission_id="s1", filename="predictions.txt"
    )
    with pytest.raises(ValueError, match="Unsupported file format"):
        asp.fetch_tabular_submission_data(bad_sk)


# ---------------------------------------------------------------------------
# Regression track: score_regression_submission
# ---------------------------------------------------------------------------


def test_score_regression_submission_requires_ground_truth():
    """ground_truth is contractually required; None fails fast with a clear error."""
    with pytest.raises(ValueError, match="ground_truth is required"):
        asp.score_regression_submission(pd.DataFrame(), phase=0, ground_truth=None)


def _patch_scoring_internals(monkeypatch, fake_score_activity_predictions):
    """Patch the shared scoring collaborators used by score_regression_submission/
    score_classification_submission, keeping only ground-truth phase-filtering (the
    thing under test) real."""
    monkeypatch.setattr(
        asp, "score_activity_predictions", fake_score_activity_predictions
    )
    # add_macro_endpoint's real narrowing/column-selection needs Endpoint/metric
    # columns that these fakes' dummy DataFrames don't have — bypass it here.
    monkeypatch.setattr(asp, "add_macro_endpoint", lambda df, endpoints, metrics: df)
    monkeypatch.setattr(
        asp,
        "average_bootstrap_results_by_endpoint",
        lambda df: pd.DataFrame({"x": [1]}),
    )
    monkeypatch.setattr(
        asp, "pivot_endpoint_results_wide", lambda df: pd.DataFrame({"y": [1]})
    )


def test_score_regression_submission_phase_1_filters_by_isin(monkeypatch):
    """Phase 1 narrows ground truth to that phase's Molecule_Name set via .isin()."""
    captured = {}

    def fake_score_activity_predictions(submissions_df, ground_truth, endpoints):
        captured["ground_truth_names"] = set(ground_truth["Molecule_Name"])
        captured["endpoints"] = endpoints
        return pd.DataFrame({"dummy": [1]})

    _patch_scoring_internals(monkeypatch, fake_score_activity_predictions)

    ground_truth = pd.DataFrame({"Molecule_Name": ["m1", "m2", "m3"]})
    identifiers = pd.DataFrame(
        {"Molecule_Name": ["m1", "m2", "m3"], "phase": [1, 1, 2]}
    )

    bootstrap_df, result_df = asp.score_regression_submission(
        pd.DataFrame(), phase=1, ground_truth=ground_truth, test_identifiers=identifiers
    )
    assert captured["ground_truth_names"] == {"m1", "m2"}
    assert captured["endpoints"] == asp.REGRESSION_ENDPOINTS
    assert isinstance(bootstrap_df, pd.DataFrame)
    assert isinstance(result_df, pd.DataFrame)


def test_score_regression_submission_phase_0_uses_full_ground_truth(monkeypatch):
    """Phase 0 scores against every compound, no filtering applied."""
    captured = {}

    def fake_score_activity_predictions(submissions_df, ground_truth, endpoints):
        captured["ground_truth_len"] = len(ground_truth)
        return pd.DataFrame({"dummy": [1]})

    _patch_scoring_internals(monkeypatch, fake_score_activity_predictions)

    ground_truth = pd.DataFrame({"Molecule_Name": ["m1", "m2", "m3"]})
    asp.score_regression_submission(pd.DataFrame(), phase=0, ground_truth=ground_truth)
    assert captured["ground_truth_len"] == 3


def test_score_regression_submission_does_not_mutate_caller_ground_truth(monkeypatch):
    """Filtering for one phase must not affect the ground truth used by later phases.

    ``score_regression_submission`` rebinds its local ``ground_truth`` name to a
    filtered copy — it must never mutate the caller's DataFrame, since the caller
    (``process_new_regression_submission``) reuses the same object across both phases
    in a loop.
    """
    _patch_scoring_internals(
        monkeypatch, lambda s, g, endpoints: pd.DataFrame({"dummy": [1]})
    )

    ground_truth = pd.DataFrame({"Molecule_Name": ["m1", "m2", "m3"]})
    identifiers = pd.DataFrame(
        {"Molecule_Name": ["m1", "m2", "m3"], "phase": [1, 1, 2]}
    )
    original_len = len(ground_truth)

    asp.score_regression_submission(
        pd.DataFrame(), phase=1, ground_truth=ground_truth, test_identifiers=identifiers
    )
    assert len(ground_truth) == original_len


# ---------------------------------------------------------------------------
# Classification track: score_classification_submission
# ---------------------------------------------------------------------------


def test_score_classification_submission_requires_ground_truth():
    """ground_truth is contractually required; None fails fast with a clear error."""
    with pytest.raises(ValueError, match="ground_truth is required"):
        asp.score_classification_submission(pd.DataFrame(), phase=0, ground_truth=None)


def test_score_classification_submission_routes_own_endpoints_and_metrics(monkeypatch):
    """score_classification_submission scores only CLASSIFICATION_ENDPOINTS."""
    captured = {}

    def fake_score_activity_predictions(submissions_df, ground_truth, endpoints):
        captured["endpoints"] = endpoints
        return pd.DataFrame({"dummy": [1]})

    _patch_scoring_internals(monkeypatch, fake_score_activity_predictions)

    ground_truth = pd.DataFrame({"Molecule_Name": ["m1", "m2"]})
    bootstrap_df, result_df = asp.score_classification_submission(
        pd.DataFrame(), phase=0, ground_truth=ground_truth
    )
    assert captured["endpoints"] == asp.CLASSIFICATION_ENDPOINTS
    assert isinstance(bootstrap_df, pd.DataFrame)
    assert isinstance(result_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Regression/Classification track: process_new_*_submission pipeline
# ---------------------------------------------------------------------------


def _patch_tabular_submission_pipeline(monkeypatch, validate_attr, score_attr, score_return):
    """Patch every collaborator of process_new_regression_submission /
    process_new_classification_submission except the pipeline control flow under
    test."""
    metadata = {
        "submitted_at": "2026-01-01T00:00:00",
        "username": "tester",
        "anonymous": False,
        "used_proprietary_data": False,
        "open_source_code": False,
    }
    monkeypatch.setattr(asp, "fetch_submission_metadata", lambda sk: metadata)
    monkeypatch.setattr(
        asp,
        "fetch_tabular_submission_data",
        lambda sk: pd.DataFrame({"Molecule_Name": ["mol1"]}),
    )
    monkeypatch.setattr(
        asp,
        "load_track_identifiers",
        lambda paths: pd.DataFrame({"Molecule_Name": ["mol1"], "phase": [1]}),
    )
    monkeypatch.setattr(
        asp,
        validate_attr,
        lambda df, expected_ids=None: ValidationResult(is_valid=True),
    )
    monkeypatch.setattr(
        asp,
        "load_activity_ground_truth",
        lambda: pd.DataFrame({"Molecule_Name": ["mol1"]}),
    )

    score_mock = MagicMock(return_value=score_return)
    monkeypatch.setattr(asp, score_attr, score_mock)

    to_parquet_mock = MagicMock()
    monkeypatch.setattr(asp.wr.s3, "to_parquet", to_parquet_mock)

    save_metadata_mock = MagicMock()
    monkeypatch.setattr(asp, "save_validated_submission_metadata", save_metadata_mock)
    monkeypatch.setattr(asp, "post_result_to_discord", lambda *a, **kw: None)

    return score_mock, to_parquet_mock, save_metadata_mock


_TABULAR_TRACK_PARAMS = [
    (
        "process_new_regression_submission",
        "validate_regression_submission",
        "score_regression_submission",
        REGRESSION_PATHS,
        REGRESSION_SK,
    ),
    (
        "process_new_classification_submission",
        "validate_classification_submission",
        "score_classification_submission",
        CLASSIFICATION_PATHS,
        CLASSIFICATION_SK,
    ),
]


@pytest.mark.parametrize(
    "process_fn_name,validate_attr,score_attr,track_paths,sk", _TABULAR_TRACK_PARAMS
)
def test_process_new_tabular_submission_success(
    monkeypatch, process_fn_name, validate_attr, score_attr, track_paths, sk
):
    """A valid submission scores both phases and writes one manifest row."""
    score_return = (pd.DataFrame({"dummy": [1]}), pd.DataFrame({"y": [1]}))
    score_mock, to_parquet_mock, save_metadata_mock = _patch_tabular_submission_pipeline(
        monkeypatch, validate_attr, score_attr, score_return
    )

    getattr(asp, process_fn_name)(sk)

    assert score_mock.call_count == 2  # phase 0 and phase 1
    assert to_parquet_mock.call_count == 4  # bootstrap + averaged, per phase
    saved_metadata = save_metadata_mock.call_args[0][1]
    saved_track_paths = save_metadata_mock.call_args[0][2]
    assert saved_metadata["valid_submission"].iloc[0] == True  # noqa: E712
    assert saved_track_paths is track_paths


@pytest.mark.parametrize(
    "process_fn_name,validate_attr,score_attr,track_paths,sk", _TABULAR_TRACK_PARAMS
)
def test_process_new_tabular_submission_invalid_skips_scoring(
    monkeypatch, process_fn_name, validate_attr, score_attr, track_paths, sk
):
    """An invalid submission is not scored, but a manifest row is still written."""
    score_return = (pd.DataFrame({"dummy": [1]}), pd.DataFrame({"y": [1]}))
    score_mock, to_parquet_mock, save_metadata_mock = _patch_tabular_submission_pipeline(
        monkeypatch, validate_attr, score_attr, score_return
    )
    monkeypatch.setattr(
        asp,
        validate_attr,
        lambda df, expected_ids=None: ValidationResult(is_valid=False),
    )

    getattr(asp, process_fn_name)(sk)

    assert score_mock.call_count == 0
    assert to_parquet_mock.call_count == 0
    saved_metadata = save_metadata_mock.call_args[0][1]
    assert saved_metadata["valid_submission"].iloc[0] == False  # noqa: E712


# ---------------------------------------------------------------------------
# Structure track: pure helpers
# ---------------------------------------------------------------------------


def test_extract_pdb_files_valid(tmp_path):
    """Only .pdb entries are returned, keyed by stem."""
    zip_path = tmp_path / "structures.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("mol1.pdb", "ATOM")
        zf.writestr("mol2.pdb", "ATOM")
        zf.writestr("readme.txt", "not a pdb")
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()

    result = asp._extract_pdb_files(zip_path, extract_dir)
    assert set(result.keys()) == {"mol1", "mol2"}
    assert Path(result["mol1"]).exists()


def test_extract_pdb_files_rejects_unsafe_paths(tmp_path):
    """A zip entry that would extract outside extract_dir is rejected."""
    zip_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.pdb", "ATOM")
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()

    with pytest.raises(ValueError, match="Unsafe path in zip"):
        asp._extract_pdb_files(zip_path, extract_dir)


def test_is_missing_key_error():
    """404 and NoSuchKey are treated as missing; other codes are not."""

    def make_error(code):
        return ClientError({"Error": {"Code": code, "Message": "x"}}, "GetObject")

    assert asp._is_missing_key_error(make_error("404"))
    assert asp._is_missing_key_error(make_error("NoSuchKey"))
    assert not asp._is_missing_key_error(make_error("403"))


# ---------------------------------------------------------------------------
# Structure track: download/extract + temp-dir cleanup
# ---------------------------------------------------------------------------


def test_download_and_extract_structure_zip_success(
    monkeypatch, example_structure_data_path
):
    """A successful download+extract returns the tmp dir and extracted PDB paths."""

    def fake_download_file(bucket, key, filename):
        shutil.copy(str(example_structure_data_path), filename)

    monkeypatch.setattr(asp.S3_CLIENT, "download_file", fake_download_file)

    tmp_dir, extracted = asp._download_and_extract_structure_zip(
        key="ground_truth/structure-dataset.zip",
        tmp_prefix="test_",
        zip_filename="structure-dataset.zip",
        missing_message="not found",
    )
    try:
        assert tmp_dir.exists()
        assert len(extracted) > 0
        for path in extracted.values():
            assert Path(path).exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_download_and_extract_structure_zip_missing_key_cleans_up_tmp_dir(
    monkeypatch,
):
    """A missing S3 key raises FileNotFoundError and removes the created tmp dir."""
    created_dirs = []
    real_mkdtemp = tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created_dirs.append(path)
        return path

    monkeypatch.setattr(asp.tempfile, "mkdtemp", spy_mkdtemp)

    def fake_download_file(bucket, key, filename):
        raise ClientError({"Error": {"Code": "404", "Message": "x"}}, "GetObject")

    monkeypatch.setattr(asp.S3_CLIENT, "download_file", fake_download_file)

    with pytest.raises(FileNotFoundError, match="not found"):
        asp._download_and_extract_structure_zip(
            key="k",
            tmp_prefix="test_",
            zip_filename="z.zip",
            missing_message="not found",
        )

    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists()


def test_download_and_extract_structure_zip_extraction_failure_cleans_up_tmp_dir(
    monkeypatch,
):
    """An extraction failure (e.g. unsafe zip) still removes the created tmp dir."""
    created_dirs = []
    real_mkdtemp = tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created_dirs.append(path)
        return path

    monkeypatch.setattr(asp.tempfile, "mkdtemp", spy_mkdtemp)

    def fake_download_file(bucket, key, filename):
        with zipfile.ZipFile(filename, "w") as zf:
            zf.writestr("../evil.pdb", "ATOM")

    monkeypatch.setattr(asp.S3_CLIENT, "download_file", fake_download_file)

    with pytest.raises(ValueError, match="Unsafe path in zip"):
        asp._download_and_extract_structure_zip(
            key="k",
            tmp_prefix="test_",
            zip_filename="z.zip",
            missing_message="not found",
        )

    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists()


def test_fetch_structure_submission_data_builds_correct_args(monkeypatch):
    """fetch_structure_submission_data passes the submission's own key/filename."""
    captured = {}

    def fake_download_and_extract(key, tmp_prefix, zip_filename, missing_message):
        captured.update(
            key=key,
            tmp_prefix=tmp_prefix,
            zip_filename=zip_filename,
            missing_message=missing_message,
        )
        return Path("/fake/tmp"), {"mol1": "/fake/tmp/extracted/mol1.pdb"}

    monkeypatch.setattr(
        asp, "_download_and_extract_structure_zip", fake_download_and_extract
    )

    tmp_dir, zip_path, predicted = asp.fetch_structure_submission_data(STRUCTURE_SK)

    assert captured["key"] == STRUCTURE_SK.key
    assert captured["tmp_prefix"] == f"submission_{STRUCTURE_SK.submission_id}_"
    assert captured["zip_filename"] == STRUCTURE_SK.filename
    assert STRUCTURE_SK.key in captured["missing_message"]
    assert tmp_dir == Path("/fake/tmp")
    assert zip_path == Path("/fake/tmp") / STRUCTURE_SK.filename
    assert predicted == {"mol1": "/fake/tmp/extracted/mol1.pdb"}


def test_load_structure_ground_truth_builds_correct_args(monkeypatch):
    """load_structure_ground_truth downloads the shared ground-truth key."""
    captured = {}

    def fake_download_and_extract(key, tmp_prefix, zip_filename, missing_message):
        captured.update(key=key, tmp_prefix=tmp_prefix, zip_filename=zip_filename)
        return Path("/fake/gt"), {"mol1": "/fake/gt/extracted/mol1.pdb"}

    monkeypatch.setattr(
        asp, "_download_and_extract_structure_zip", fake_download_and_extract
    )

    tmp_dir, ground_truth = asp.load_structure_ground_truth()

    assert captured["key"] == f"{STRUCTURE_PATHS.ground_truth}/structure-dataset.zip"
    assert captured["tmp_prefix"] == "ground_truth_structure_"
    assert captured["zip_filename"] == "structure-dataset.zip"
    assert tmp_dir == Path("/fake/gt")
    assert ground_truth == {"mol1": "/fake/gt/extracted/mol1.pdb"}


# ---------------------------------------------------------------------------
# Structure track: score_structure_submission
# ---------------------------------------------------------------------------


def test_score_structure_submission_phase_0_no_filtering(monkeypatch):
    """Phase 0 scores every compound and coverage_mean reflects the full set."""
    per_compound = pd.DataFrame(
        {"Molecule_Name": ["a", "b", "c"], "coverage": [1.0, 0.5, 0.0]}
    )
    monkeypatch.setattr(
        asp, "score_structure_predictions", lambda predicted, gt: per_compound
    )
    monkeypatch.setattr(
        asp, "bootstrap_structure_metrics", lambda df, n: pd.DataFrame({"dummy": [1]})
    )
    monkeypatch.setattr(
        asp,
        "average_bootstrap_results_by_endpoint",
        lambda df: pd.DataFrame({"metric_mean": [0.9]}),
    )
    monkeypatch.setattr(asp, "pivot_endpoint_results_wide", lambda df: df)

    result_per_compound, _bootstrap_df, averaged_df = asp.score_structure_submission(
        predicted={}, phase=0, ground_truth={}, identifiers=None
    )
    assert len(result_per_compound) == 3
    assert averaged_df["coverage_mean"].iloc[0] == pytest.approx(0.5)


def test_score_structure_submission_phase_1_filters_by_isin(monkeypatch):
    """Phase 1 narrows per-compound results to that phase's Molecule_Name set."""
    per_compound = pd.DataFrame(
        {"Molecule_Name": ["a", "b", "c"], "coverage": [1.0, 1.0, 0.0]}
    )
    identifiers = pd.DataFrame({"Molecule_Name": ["a", "b", "c"], "phase": [1, 2, 1]})
    monkeypatch.setattr(
        asp, "score_structure_predictions", lambda predicted, gt: per_compound
    )
    monkeypatch.setattr(
        asp, "bootstrap_structure_metrics", lambda df, n: pd.DataFrame({"dummy": [1]})
    )
    monkeypatch.setattr(
        asp,
        "average_bootstrap_results_by_endpoint",
        lambda df: pd.DataFrame({"metric_mean": [0.9]}),
    )
    monkeypatch.setattr(asp, "pivot_endpoint_results_wide", lambda df: df)

    result_per_compound, _bootstrap_df, averaged_df = asp.score_structure_submission(
        predicted={}, phase=1, ground_truth={}, identifiers=identifiers
    )
    assert set(result_per_compound["Molecule_Name"]) == {"a", "c"}
    assert averaged_df["coverage_mean"].iloc[0] == pytest.approx(0.5)


def test_score_structure_submission_loads_ground_truth_when_not_provided(monkeypatch):
    """A None ground_truth is loaded via load_structure_ground_truth."""
    captured = {}
    monkeypatch.setattr(
        asp, "load_structure_ground_truth", lambda: (Path("/gt"), {"loaded": "gt"})
    )

    def fake_score(predicted, gt):
        captured["gt"] = gt
        return pd.DataFrame({"Molecule_Name": ["a"], "coverage": [1.0]})

    monkeypatch.setattr(asp, "score_structure_predictions", fake_score)
    monkeypatch.setattr(
        asp, "bootstrap_structure_metrics", lambda df, n: pd.DataFrame({"dummy": [1]})
    )
    monkeypatch.setattr(
        asp,
        "average_bootstrap_results_by_endpoint",
        lambda df: pd.DataFrame({"metric_mean": [0.9]}),
    )
    monkeypatch.setattr(asp, "pivot_endpoint_results_wide", lambda df: df)

    asp.score_structure_submission(
        predicted={}, phase=0, ground_truth=None, identifiers=None
    )
    assert captured["gt"] == {"loaded": "gt"}


# ---------------------------------------------------------------------------
# Structure track: process_new_structure_submission phase-loop control
# ---------------------------------------------------------------------------


def _patch_structure_submission_pipeline(monkeypatch, tmp_path, per_compound_df):
    """Patch every collaborator of process_new_structure_submission except scoring."""
    metadata = {
        "submitted_at": "2026-01-01T00:00:00",
        "username": "tester",
        "anonymous": False,
        "used_proprietary_data": False,
        "open_source_code": False,
    }
    monkeypatch.setattr(asp, "fetch_submission_metadata", lambda sk: metadata)

    sub_dir = tmp_path / "submission"
    sub_dir.mkdir()
    monkeypatch.setattr(
        asp,
        "fetch_structure_submission_data",
        lambda sk: (sub_dir, sub_dir / "structures.zip", {"mol1": "path1"}),
    )
    monkeypatch.setattr(
        asp,
        "load_track_identifiers",
        lambda paths: pd.DataFrame({"Molecule_Name": ["mol1"], "phase": [1]}),
    )
    monkeypatch.setattr(
        asp,
        "validate_structure_submission",
        lambda zip_path, expected_ids=None: ValidationResult(is_valid=True),
    )

    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()
    monkeypatch.setattr(
        asp, "load_structure_ground_truth", lambda: (gt_dir, {"mol1": "gt"})
    )

    score_mock = MagicMock(
        return_value=(per_compound_df, pd.DataFrame(), pd.DataFrame({"x": [1]}))
    )
    monkeypatch.setattr(asp, "score_structure_submission", score_mock)

    to_parquet_mock = MagicMock()
    monkeypatch.setattr(asp.wr.s3, "to_parquet", to_parquet_mock)

    save_metadata_mock = MagicMock()
    monkeypatch.setattr(asp, "save_validated_submission_metadata", save_metadata_mock)
    monkeypatch.setattr(asp, "post_result_to_discord", lambda *a, **kw: None)

    return score_mock, to_parquet_mock, save_metadata_mock


def test_process_new_structure_submission_full_failure_stops_after_phase_0(
    monkeypatch, tmp_path
):
    """When every structure fails scoring at phase 0, phase 1 must not run.

    Regression test: previously the phase loop had no ``break`` here, so a
    submission that completely failed scoring still ran (and wrote S3 artifacts
    for) phase 1 before being marked invalid.
    """
    failed_df = pd.DataFrame(
        {
            "Molecule_Name": [f"mol{i}" for i in range(STRUCTURE_DATASET_SIZE)],
            "coverage": [0.0] * STRUCTURE_DATASET_SIZE,
        }
    )
    score_mock, to_parquet_mock, save_metadata_mock = (
        _patch_structure_submission_pipeline(monkeypatch, tmp_path, failed_df)
    )

    asp.process_new_structure_submission(STRUCTURE_SK)

    assert score_mock.call_count == 1
    assert to_parquet_mock.call_count == 3
    saved_metadata = save_metadata_mock.call_args[0][1]
    assert saved_metadata["valid_submission"].iloc[0] == False  # noqa: E712


def test_process_new_structure_submission_partial_failure_continues_to_phase_1(
    monkeypatch, tmp_path
):
    """A partial scoring failure at phase 0 still lets phase 1 run."""
    partial_df = pd.DataFrame(
        {
            "Molecule_Name": [f"mol{i}" for i in range(STRUCTURE_DATASET_SIZE)],
            "coverage": [0.0] * 5 + [1.0] * (STRUCTURE_DATASET_SIZE - 5),
        }
    )
    score_mock, to_parquet_mock, save_metadata_mock = (
        _patch_structure_submission_pipeline(monkeypatch, tmp_path, partial_df)
    )

    asp.process_new_structure_submission(STRUCTURE_SK)

    assert score_mock.call_count == 2
    assert to_parquet_mock.call_count == 6
    saved_metadata = save_metadata_mock.call_args[0][1]
    assert saved_metadata["valid_submission"].iloc[0] == True  # noqa: E712
