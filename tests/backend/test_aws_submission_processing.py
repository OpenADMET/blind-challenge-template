"""Tests for AWS submission scoring/processing functions.

Most of this module talks to S3 (via awswrangler/boto3) or delegates to heavy
scoring functions (openstructure-backed structure scoring, sklearn/scipy activity
metrics) that are exercised in their own test modules. These tests therefore mock
those collaborators at the ``backend.aws_submission_processing`` module boundary
and focus on this module's own control flow: path building, compound-set filtering,
error translation, and temp-file cleanup.
"""

import dataclasses
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import awswrangler as wr
import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from backend import aws_submission_processing as asp
from backend import evaluate_predictions
from backend.config import (
    CLASSIFICATION_PATHS,
    CLASSIFICATION_SPEC,
    GROUND_TRUTH_PREFIX,
    REGRESSION_PATHS,
    REGRESSION_SPEC,
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
    asp.save_validated_submission_metadata(REGRESSION_SK, metadata_df, REGRESSION_PATHS)

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
        "live_scores_uri",
        "open_source_code",
    }
    assert required_columns.issubset(metadata.columns)
    assert metadata["valid_submission"].dtype == bool
    assert metadata["open_source_code"].dtype == bool
    assert metadata["submitted_at"].dt.tz is not None
    assert metadata["all_scores_uri"].iloc[0] == (
        f"s3://{S3_BUCKET}/{REGRESSION_PATHS.scores('all')}/new-user/submission-123/"
        "averaged-results.parquet"
    )


def test_create_validation_metadata_classification(
    example_activity_predictions_metadata,
):
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
        f"s3://{S3_BUCKET}/{CLASSIFICATION_PATHS.scores('all')}/new-user/submission-123/"
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
        f"s3://{S3_BUCKET}/{STRUCTURE_PATHS.scores('all')}/new-user/submission-123/"
        "averaged-results.parquet"
    )


# ---------------------------------------------------------------------------
# Shared: load_track_identifiers
# ---------------------------------------------------------------------------


def test_load_track_identifiers_success(monkeypatch):
    """load_track_identifiers returns whatever awswrangler reads back."""
    expected = pd.DataFrame({"Molecule_Name": ["a"], "set": ["live"]})
    monkeypatch.setattr(asp.wr.s3, "read_parquet", lambda path: expected)
    assert asp.load_track_identifiers(REGRESSION_PATHS) is expected


def test_load_track_identifiers_missing_raises_file_not_found(monkeypatch):
    """A missing identifiers file surfaces as FileNotFoundError, not NoFilesFound."""

    def raise_not_found(path):
        raise wr.exceptions.NoFilesFound(f"No files: {path}")

    monkeypatch.setattr(asp.wr.s3, "read_parquet", raise_not_found)
    with pytest.raises(FileNotFoundError, match="Activity identifiers not found"):
        asp.load_track_identifiers(REGRESSION_PATHS)


def test_load_track_identifiers_reads_activity_path(monkeypatch):
    """load_track_identifiers(REGRESSION_PATHS) reads from the activity ground-truth prefix."""
    captured = {}

    def fake_read_parquet(path):
        captured["path"] = path
        return pd.DataFrame()

    monkeypatch.setattr(asp.wr.s3, "read_parquet", fake_read_parquet)
    asp.load_track_identifiers(REGRESSION_PATHS)
    assert captured["path"] == (
        f"s3://{S3_BUCKET}/{GROUND_TRUTH_PREFIX}/activity-identifiers.parquet"
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
        asp.score_regression_submission(
            pd.DataFrame(), compound_set="all", ground_truth=None
        )


def _patch_scoring_internals(monkeypatch, fake_score_activity_predictions):
    """Patch the shared scoring collaborators used by score_regression_submission/
    score_classification_submission, keeping only ground-truth compound-set filtering (the
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


def test_score_regression_submission_live_set_filters_by_isin(monkeypatch):
    """The live set narrows ground truth to the identifiers' live compounds."""
    captured = {}

    def fake_score_activity_predictions(submissions_df, ground_truth, endpoints):
        captured["ground_truth_names"] = set(ground_truth["Molecule_Name"])
        captured["endpoints"] = endpoints
        return pd.DataFrame({"dummy": [1]})

    _patch_scoring_internals(monkeypatch, fake_score_activity_predictions)

    ground_truth = pd.DataFrame({"Molecule_Name": ["m1", "m2", "m3"]})
    identifiers = pd.DataFrame(
        {"Molecule_Name": ["m1", "m2", "m3"], "set": ["live", "live", "hidden"]}
    )

    bootstrap_df, result_df = asp.score_regression_submission(
        pd.DataFrame(),
        compound_set="live",
        ground_truth=ground_truth,
        test_identifiers=identifiers,
    )
    assert captured["ground_truth_names"] == {"m1", "m2"}
    assert captured["endpoints"] == REGRESSION_SPEC.endpoints
    assert isinstance(bootstrap_df, pd.DataFrame)
    assert isinstance(result_df, pd.DataFrame)


def test_score_regression_submission_all_set_uses_full_ground_truth(monkeypatch):
    """Phase 0 scores against every compound, no filtering applied."""
    captured = {}

    def fake_score_activity_predictions(submissions_df, ground_truth, endpoints):
        captured["ground_truth_len"] = len(ground_truth)
        return pd.DataFrame({"dummy": [1]})

    _patch_scoring_internals(monkeypatch, fake_score_activity_predictions)

    ground_truth = pd.DataFrame({"Molecule_Name": ["m1", "m2", "m3"]})
    asp.score_regression_submission(
        pd.DataFrame(), compound_set="all", ground_truth=ground_truth
    )
    assert captured["ground_truth_len"] == 3


def test_score_regression_submission_does_not_mutate_caller_ground_truth(monkeypatch):
    """Filtering for one set must not affect the ground truth used for later sets.

    ``score_regression_submission`` rebinds its local ``ground_truth`` name to a
    filtered copy — it must never mutate the caller's DataFrame, since the caller
    (``process_new_regression_submission``) reuses the same object across every set
    in a loop.
    """
    _patch_scoring_internals(
        monkeypatch, lambda s, g, endpoints: pd.DataFrame({"dummy": [1]})
    )

    ground_truth = pd.DataFrame({"Molecule_Name": ["m1", "m2", "m3"]})
    identifiers = pd.DataFrame(
        {"Molecule_Name": ["m1", "m2", "m3"], "set": ["live", "live", "hidden"]}
    )
    original_len = len(ground_truth)

    asp.score_regression_submission(
        pd.DataFrame(),
        compound_set="live",
        ground_truth=ground_truth,
        test_identifiers=identifiers,
    )
    assert len(ground_truth) == original_len


# ---------------------------------------------------------------------------
# Classification track: score_classification_submission
# ---------------------------------------------------------------------------


def test_score_classification_submission_requires_ground_truth():
    """ground_truth is contractually required; None fails fast with a clear error."""
    with pytest.raises(ValueError, match="ground_truth is required"):
        asp.score_classification_submission(
            pd.DataFrame(), compound_set="all", ground_truth=None
        )


def test_score_classification_submission_routes_own_endpoints_and_metrics(monkeypatch):
    """score_classification_submission scores only CLASSIFICATION_SPEC.endpoints."""
    captured = {}

    def fake_score_activity_predictions(submissions_df, ground_truth, endpoints):
        captured["endpoints"] = endpoints
        return pd.DataFrame({"dummy": [1]})

    _patch_scoring_internals(monkeypatch, fake_score_activity_predictions)

    ground_truth = pd.DataFrame({"Molecule_Name": ["m1", "m2"]})
    bootstrap_df, result_df = asp.score_classification_submission(
        pd.DataFrame(), compound_set="all", ground_truth=ground_truth
    )
    assert captured["endpoints"] == CLASSIFICATION_SPEC.endpoints
    assert isinstance(bootstrap_df, pd.DataFrame)
    assert isinstance(result_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Regression/Classification track: process_new_*_submission pipeline
# ---------------------------------------------------------------------------


def _patch_tabular_submission_pipeline(
    monkeypatch, validate_attr, score_attr, score_return
):
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
        lambda paths: pd.DataFrame({"Molecule_Name": ["mol1"], "set": ["live"]}),
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

    # Autospecced, so a call with arguments the real function doesn't accept fails.
    to_parquet_mock = create_autospec(asp.wr.s3.to_parquet)
    monkeypatch.setattr(asp.wr.s3, "to_parquet", to_parquet_mock)

    save_metadata_mock = create_autospec(asp.save_validated_submission_metadata)
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
    """A valid submission is scored on each of its track's sets (all and live by
    default) and writes one manifest row."""
    score_return = (pd.DataFrame({"dummy": [1]}), pd.DataFrame({"y": [1]}))
    score_mock, to_parquet_mock, save_metadata_mock = (
        _patch_tabular_submission_pipeline(
            monkeypatch, validate_attr, score_attr, score_return
        )
    )

    getattr(asp, process_fn_name)(sk)

    assert score_mock.call_count == 2  # the all and live sets
    assert to_parquet_mock.call_count == 4  # bootstrap + averaged, per set
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
    score_mock, to_parquet_mock, save_metadata_mock = (
        _patch_tabular_submission_pipeline(
            monkeypatch, validate_attr, score_attr, score_return
        )
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
# Structure track: aggregate_structure_scores
# ---------------------------------------------------------------------------


def _fake_aggregation(monkeypatch):
    """Make bootstrapping and averaging trivial, to test filtering and coverage."""
    monkeypatch.setattr(
        asp, "bootstrap_structure_metrics", lambda df, n: pd.DataFrame({"dummy": [1]})
    )
    monkeypatch.setattr(
        asp,
        "average_bootstrap_results_by_endpoint",
        lambda df: pd.DataFrame({"metric_mean": [0.9]}),
    )
    monkeypatch.setattr(asp, "pivot_endpoint_results_wide", lambda df: df)


def test_aggregate_structure_scores_all_set_no_filtering(monkeypatch):
    """The "all" set keeps every compound and coverage_mean reflects them all."""
    _fake_aggregation(monkeypatch)
    per_compound = pd.DataFrame(
        {"Molecule_Name": ["a", "b", "c"], "coverage": [1.0, 0.5, 0.0]}
    )
    result, _bootstrap_df, averaged_df = asp.aggregate_structure_scores(
        per_compound, "all"
    )
    assert len(result) == 3
    assert averaged_df["coverage_mean"].iloc[0] == pytest.approx(0.5)


def test_aggregate_structure_scores_live_set_filters(monkeypatch):
    """The live set narrows per-compound results to the live compounds."""
    _fake_aggregation(monkeypatch)
    per_compound = pd.DataFrame(
        {"Molecule_Name": ["a", "b", "c"], "coverage": [1.0, 1.0, 0.0]}
    )
    identifiers = pd.DataFrame(
        {"Molecule_Name": ["a", "b", "c"], "set": ["live", "hidden", "live"]}
    )
    result, _bootstrap_df, averaged_df = asp.aggregate_structure_scores(
        per_compound, "live", identifiers
    )
    assert set(result["Molecule_Name"]) == {"a", "c"}
    assert averaged_df["coverage_mean"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Structure track: process_new_structure_submission compound-set loop
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
        lambda paths: pd.DataFrame({"Molecule_Name": ["mol1"], "set": ["live"]}),
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

    # The expensive per-compound scoring; must run once per submission.
    score_mock = MagicMock(return_value=(per_compound_df, {}))
    monkeypatch.setattr(asp, "score_structure_predictions", score_mock)
    # The cheap per-set aggregation.
    aggregate_mock = MagicMock(
        side_effect=lambda df, compound_set, identifiers: (
            df,
            pd.DataFrame(),
            pd.DataFrame({"x": [1]}),
        )
    )
    monkeypatch.setattr(asp, "aggregate_structure_scores", aggregate_mock)

    # Autospecced, so a call with arguments the real function doesn't accept fails.
    to_parquet_mock = create_autospec(asp.wr.s3.to_parquet)
    monkeypatch.setattr(asp.wr.s3, "to_parquet", to_parquet_mock)

    save_metadata_mock = create_autospec(asp.save_validated_submission_metadata)
    monkeypatch.setattr(asp, "save_validated_submission_metadata", save_metadata_mock)
    monkeypatch.setattr(asp, "post_result_to_discord", lambda *a, **kw: None)

    return score_mock, aggregate_mock, to_parquet_mock, save_metadata_mock


def test_process_new_structure_submission_full_failure_saves_no_scores(
    monkeypatch, tmp_path
):
    """When every structure fails scoring, the submission is invalid and no
    compound set is aggregated or saved."""
    failed_df = pd.DataFrame(
        {
            "Molecule_Name": [f"mol{i}" for i in range(STRUCTURE_DATASET_SIZE)],
            "coverage": [0.0] * STRUCTURE_DATASET_SIZE,
        }
    )
    score_mock, aggregate_mock, to_parquet_mock, save_metadata_mock = (
        _patch_structure_submission_pipeline(monkeypatch, tmp_path, failed_df)
    )

    asp.process_new_structure_submission(STRUCTURE_SK)

    assert score_mock.call_count == 1
    assert aggregate_mock.call_count == 0
    assert to_parquet_mock.call_count == 0
    saved_metadata = save_metadata_mock.call_args[0][1]
    assert saved_metadata["valid_submission"].iloc[0] == False  # noqa: E712


def test_process_new_structure_submission_partial_failure_still_saves_scores(
    monkeypatch, tmp_path
):
    """A partial scoring failure still aggregates and saves every set."""
    partial_df = pd.DataFrame(
        {
            "Molecule_Name": [f"mol{i}" for i in range(STRUCTURE_DATASET_SIZE)],
            "coverage": [0.0] * 5 + [1.0] * (STRUCTURE_DATASET_SIZE - 5),
        }
    )
    score_mock, aggregate_mock, to_parquet_mock, save_metadata_mock = (
        _patch_structure_submission_pipeline(monkeypatch, tmp_path, partial_df)
    )

    asp.process_new_structure_submission(STRUCTURE_SK)

    # #15: the structures are scored once; only the aggregation runs per set.
    assert score_mock.call_count == 1
    assert [c.kwargs["compound_set"] for c in aggregate_mock.call_args_list] == [
        "all",
        "live",
    ]
    assert to_parquet_mock.call_count == 6
    saved_metadata = save_metadata_mock.call_args[0][1]
    assert saved_metadata["valid_submission"].iloc[0] == True  # noqa: E712


# ---------------------------------------------------------------------------
# Structure track: score once, aggregate per compound set (#15)
# ---------------------------------------------------------------------------


def _fake_single_structure_scores(monkeypatch, calls: list[str]):
    """Deterministic per-compound scores, with "mol2" unscoreable; records calls."""

    def _score(model_path, ref_path, max_pb_failures, smiles):
        calls.append(model_path)
        if model_path.endswith("mol2.pdb"):
            return {"LDDT-PLI": np.nan, "BiSyRMSD": np.nan, "LDDT-LP": np.nan}, []
        x = float(model_path[-5])
        return {"LDDT-PLI": x / 10, "BiSyRMSD": x, "LDDT-LP": x / 20}, []

    monkeypatch.setattr(evaluate_predictions, "score_single_structure", _score)


def test_aggregating_once_scored_compounds_matches_scoring_per_set(monkeypatch):
    """Scoring every compound once and filtering per set gives the same scores as
    scoring only that set's compounds."""
    calls: list[str] = []
    _fake_single_structure_scores(monkeypatch, calls)
    predicted = {f"mol{i}": f"/pred/mol{i}.pdb" for i in range(1, 7)}
    ground_truth = {f"mol{i}": f"/gt/mol{i}.pdb" for i in range(1, 7)}
    sets = ["live", "hidden", "live", "hidden", "live", "hidden"]
    identifiers = pd.DataFrame({"Molecule_Name": list(predicted), "set": sets})

    all_scores, _ = asp.score_structure_predictions(predicted, ground_truth)
    assert len(calls) == 6  # every compound scored once
    for compound_set in ["live", "hidden"]:
        in_set = {m for m, s in zip(predicted, sets) if s == compound_set}
        set_scores, _ = asp.score_structure_predictions(
            {m: p for m, p in predicted.items() if m in in_set}, ground_truth
        )
        expected = asp.aggregate_structure_scores(set_scores, "all")
        actual = asp.aggregate_structure_scores(all_scores, compound_set, identifiers)
        for got, want in zip(actual, expected):
            pd.testing.assert_frame_equal(
                got.reset_index(drop=True), want.reset_index(drop=True)
            )

    live = asp.aggregate_structure_scores(all_scores, "live", identifiers)[0]
    assert list(live["Molecule_Name"]) == ["mol1", "mol3", "mol5"]


def test_aggregate_structure_scores_coverage_is_per_set():
    per_compound = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c", "d"],
            "LDDT-PLI": [0.5, 0.0, 0.7, 0.9],
            "BiSyRMSD": [1.0, 10.0, 1.0, 1.0],
            "LDDT-LP": [0.5, 0.0, 0.5, 0.5],
            "coverage": [1.0, 0.0, 1.0, 1.0],
        }
    )
    identifiers = pd.DataFrame(
        {"Molecule_Name": list("abcd"), "set": ["live", "live", "hidden", "hidden"]}
    )
    _, _, all_avg = asp.aggregate_structure_scores(per_compound, "all", identifiers)
    _, _, live_avg = asp.aggregate_structure_scores(per_compound, "live", identifiers)
    coverage_col = [c for c in all_avg.columns if "coverage" in c][0]
    assert all_avg[coverage_col].iloc[0] == pytest.approx(0.75)
    assert live_avg[coverage_col].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Structure track: SMILES for ligand bonds (#27)
# ---------------------------------------------------------------------------


def test_structure_smiles_map_from_identifiers():
    identifiers = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c", "d"],
            "SMILES": ["CCO", None, "  ", "c1ccccc1"],
            "set": ["live", "live", "hidden", "hidden"],
        }
    )
    assert asp.structure_smiles_map(identifiers) == {"a": "CCO", "d": "c1ccccc1"}


@pytest.mark.parametrize(
    "identifiers", [None, pd.DataFrame({"Molecule_Name": ["a"], "set": ["live"]})]
)
def test_structure_smiles_map_without_smiles_is_none(identifiers):
    assert asp.structure_smiles_map(identifiers) is None


def test_process_new_structure_submission_passes_smiles_to_scoring(
    monkeypatch, tmp_path
):
    per_compound = pd.DataFrame({"Molecule_Name": ["mol1"], "coverage": [1.0]})
    score_mock, *_ = _patch_structure_submission_pipeline(
        monkeypatch, tmp_path, per_compound
    )
    monkeypatch.setattr(
        asp,
        "load_track_identifiers",
        lambda paths: pd.DataFrame(
            {"Molecule_Name": ["mol1"], "SMILES": ["C1CNC1"], "set": ["live"]}
        ),
    )

    asp.process_new_structure_submission(STRUCTURE_SK)

    assert score_mock.call_args.kwargs["smiles_map"] == {"mol1": "C1CNC1"}


@pytest.mark.parametrize(
    "paths", [REGRESSION_PATHS, CLASSIFICATION_PATHS, STRUCTURE_PATHS]
)
def test_manifests_are_outside_the_delete_protected_prefix(paths):
    """#12: removing a leaderboard entry means moving its manifest, which must not
    need the submissions/* delete protection lifted."""
    assert paths.manifest == f"manifests/{paths.track}"
    assert not paths.manifest.startswith("submissions/")


# ---------------------------------------------------------------------------
# Compound sets (#16)
# ---------------------------------------------------------------------------

IDENTIFIERS = pd.DataFrame(
    {"Molecule_Name": ["a", "b", "c"], "set": ["live", "hidden", "live"]}
)


@pytest.mark.parametrize(
    ("compound_set", "expected"),
    [("all", None), ("live", {"a", "c"}), ("hidden", {"b"})],
)
def test_compound_set_ids(compound_set, expected):
    assert asp.compound_set_ids(IDENTIFIERS, compound_set) == expected


def test_compound_set_ids_rejects_unknown_values():
    with pytest.raises(ValueError, match="other than live/hidden"):
        asp.compound_set_ids(IDENTIFIERS.assign(set=["live", "2", "live"]), "live")
    with pytest.raises(ValueError, match="Unknown compound set"):
        asp.compound_set_ids(IDENTIFIERS, "phase_1")


def _hidden_final_paths():
    """Classification ranked on live compounds live and hidden ones at the end."""
    spec = dataclasses.replace(
        CLASSIFICATION_SPEC,
        leaderboard_sets={"live": "live", "interim": "hidden", "final": "hidden"},
    )
    return dataclasses.replace(CLASSIFICATION_PATHS, spec=spec)


def test_validation_metadata_has_a_uri_per_scored_set(
    example_activity_predictions_metadata,
):
    paths = _hidden_final_paths()
    metadata = asp.create_validation_metadata(
        True, example_activity_predictions_metadata, CLASSIFICATION_SK, paths
    )
    assert "all_scores_uri" not in metadata.columns
    assert metadata["hidden_scores_uri"].iloc[0] == (
        f"s3://{S3_BUCKET}/scores/hidden/classification/new-user/submission-123/"
        "averaged-results.parquet"
    )
    assert "live_scores_uri" in metadata.columns
