"""Test the submission validation logic."""

import zipfile

import pytest

from backend.config import CLASSIFICATION_ENDPOINTS, REGRESSION_ENDPOINTS
from backend.submission_validation import (
    SubmissionError,
    ValidationResult,
    validate_classification_submission,
    validate_regression_submission,
    validate_structure_submission,
)


def test_submission_error_to_short():
    """Test the to_short method of SubmissionError."""
    error = SubmissionError("Test error", details=["item1", "item2"])
    assert error.to_short() == "Test error (2 items)"

    error_no_details = SubmissionError("Test error")
    assert error_no_details.to_short() == "Test error"


def test_submission_error_to_long():
    """Test the to_long method of SubmissionError."""
    error = SubmissionError("Test error", details=["item1", "item2"])
    assert error.to_long() == "Test error: item1, item2"

    error_no_details = SubmissionError("Test error")
    assert error_no_details.to_long() == "Test error"


def test_validation_result_add_error():
    """Test the add_error method of ValidationResult."""
    result = ValidationResult()
    result.add_error("Test error", details=["item1", "item2"])
    assert not result.is_valid
    assert len(result.errors) == 1
    assert result.errors[0].message == "Test error"
    assert result.errors[0].details == ["item1", "item2"]
    assert result.get_error_summary() == "Test error (2 items)"
    assert result.get_full_log() == "Test error: item1, item2"


def test_validation_result_add_error_no_details():
    """Test the add_error method of ValidationResult when no details are provided."""
    result = ValidationResult()
    result.add_error("Test error")
    assert not result.is_valid
    assert len(result.errors) == 1
    assert result.errors[0].message == "Test error"
    assert result.errors[0].details == []
    assert result.get_error_summary() == "Test error"
    assert result.get_full_log() is None


def test_validation_result_add_system_error():
    """Test the add_system_error method of ValidationResult."""
    result = ValidationResult()
    result.add_system_error("System error")
    assert not result.is_valid
    assert len(result.system_errors) == 1
    assert result.system_errors[0] == "System error"
    assert (
        result.get_error_summary()
        == "An unknown error has occurred, we are looking into it."
    )
    assert result.get_full_log() is None


def test_validation_result_get_error_summary_valid():
    """Test that get_error_summary raises an error when the submission is valid."""
    result = ValidationResult()
    with pytest.raises(
        ValueError, match="Submission is valid, no errors to summarize."
    ):
        result.get_error_summary()
    with pytest.raises(ValueError, match="Submission is valid, no errors to log."):
        result.get_full_log()


def test_validate_regression_submission_valid(example_activity_predictions_data):
    """Test that a valid regression submission passes validation."""
    validation_result = validate_regression_submission(
        example_activity_predictions_data
    )
    assert validation_result.is_valid
    assert len(validation_result.errors) == 0


def test_validate_regression_missing_predictions(example_activity_predictions_data):
    """Test that a submission with missing predictions fails validation."""
    short_df = example_activity_predictions_data.sample(100)
    validation_result = validate_regression_submission(short_df)
    log_text = validation_result.get_error_summary()
    assert not validation_result.is_valid
    assert "Submission contains 100 molecules" in log_text


def test_validate_regression_submission_missing_ids(example_activity_predictions_data):
    """Test that a submission missing expected molecule IDs fails validation."""
    expected_ids = set(example_activity_predictions_data["Molecule_Name"])
    expected_ids.add("E-9999999")
    validation_result = validate_regression_submission(
        example_activity_predictions_data, expected_ids=expected_ids
    )
    log_text = validation_result.get_full_log()
    assert not validation_result.is_valid
    assert "Missing expected molecule(s): E-9999999" in log_text


def test_validate_regression_submission_unexpected_ids(example_activity_predictions_data):
    """Test that a submission with unexpected molecule IDs fails validation."""
    expected_ids = set(example_activity_predictions_data["Molecule_Name"])
    unexpected_id = expected_ids.pop()
    validation_result = validate_regression_submission(
        example_activity_predictions_data, expected_ids=expected_ids
    )
    log_text = validation_result.get_full_log()
    assert not validation_result.is_valid
    assert f"Unexpected molecule(s) present: {unexpected_id}" in log_text


def test_validate_regression_missing_column(example_activity_predictions_data):
    """Test that a submission missing a required regression column fails validation."""
    missing_col_name = REGRESSION_ENDPOINTS[-1]
    missing_col = example_activity_predictions_data.drop(columns=[missing_col_name])
    validation_result = validate_regression_submission(missing_col)
    log_text = validation_result.get_error_summary()
    assert "Validation failed" in log_text
    assert not validation_result.is_valid
    assert f"Missing required column: '{missing_col_name}'" in log_text


def test_validate_regression_nan_values(example_activity_predictions_data):
    """Test that a submission with NaN values fails validation."""
    example_activity_predictions_data.loc[0, REGRESSION_ENDPOINTS[0]] = float("nan")
    validation_result = validate_regression_submission(
        example_activity_predictions_data
    )
    log_text = validation_result.get_error_summary()
    assert not validation_result.is_valid
    assert "Validation failed" in log_text
    assert (
        f"- Column '{REGRESSION_ENDPOINTS[0]}' Failed check 'not_nullable'" in log_text
    )


def test_validate_regression_inf_values(example_activity_predictions_data):
    """Test that a submission with infinite values fails validation."""
    example_activity_predictions_data.loc[0, REGRESSION_ENDPOINTS[0]] = float("inf")
    validation_result = validate_regression_submission(
        example_activity_predictions_data
    )
    log_text = validation_result.get_error_summary()
    assert not validation_result.is_valid
    assert "Validation failed" in log_text
    assert (
        f"- Column '{REGRESSION_ENDPOINTS[0]}' Failed check 'Column contains infinite values "
        "(inf).'" in log_text
    )


def test_validate_classification_submission_valid(example_activity_predictions_data):
    """Test that a valid classification submission passes validation."""
    validation_result = validate_classification_submission(
        example_activity_predictions_data
    )
    assert validation_result.is_valid
    assert len(validation_result.errors) == 0


def test_validate_classification_missing_predictions(example_activity_predictions_data):
    """Test that a submission with missing predictions fails validation."""
    short_df = example_activity_predictions_data.sample(100)
    validation_result = validate_classification_submission(short_df)
    log_text = validation_result.get_error_summary()
    assert not validation_result.is_valid
    assert "Submission contains 100 molecules" in log_text


def test_validate_classification_submission_missing_ids(
    example_activity_predictions_data,
):
    """Test that a submission missing expected molecule IDs fails validation."""
    expected_ids = set(example_activity_predictions_data["Molecule_Name"])
    expected_ids.add("E-9999999")
    validation_result = validate_classification_submission(
        example_activity_predictions_data, expected_ids=expected_ids
    )
    log_text = validation_result.get_full_log()
    assert not validation_result.is_valid
    assert "Missing expected molecule(s): E-9999999" in log_text


def test_validate_classification_submission_unexpected_ids(
    example_activity_predictions_data,
):
    """Test that a submission with unexpected molecule IDs fails validation."""
    expected_ids = set(example_activity_predictions_data["Molecule_Name"])
    unexpected_id = expected_ids.pop()
    validation_result = validate_classification_submission(
        example_activity_predictions_data, expected_ids=expected_ids
    )
    log_text = validation_result.get_full_log()
    assert not validation_result.is_valid
    assert f"Unexpected molecule(s) present: {unexpected_id}" in log_text


def test_validate_classification_missing_column(example_activity_predictions_data):
    """Test that a submission missing a required classification column fails."""
    missing_col_name = CLASSIFICATION_ENDPOINTS[-1]
    missing_col = example_activity_predictions_data.drop(columns=[missing_col_name])
    validation_result = validate_classification_submission(missing_col)
    log_text = validation_result.get_error_summary()
    assert "Validation failed" in log_text
    assert not validation_result.is_valid
    assert f"Missing required column: '{missing_col_name}'" in log_text


def test_validate_classification_non_binary_value(
    example_activity_predictions_data,
):
    """Test that a non-binary value (e.g. a float) in a classification column fails validation."""
    endpoint = CLASSIFICATION_ENDPOINTS[0]
    # Cast to object first — the fixture's column is int64, and assigning a float
    # into it would trigger a pandas FutureWarning about implicit whole-column
    # upcasting. object dtype also mirrors a real mixed-type submission column.
    example_activity_predictions_data[endpoint] = example_activity_predictions_data[
        endpoint
    ].astype(object)
    example_activity_predictions_data.loc[0, endpoint] = 0.37
    validation_result = validate_classification_submission(
        example_activity_predictions_data
    )
    log_text = validation_result.get_error_summary()
    assert not validation_result.is_valid
    assert "Validation failed" in log_text
    assert (
        f"- Column '{endpoint}' Failed check 'Column must be boolean "
        "(True/False or 1/0).'" in log_text
    )


@pytest.mark.parametrize("value", [0, 1, True, False])
def test_validate_classification_binary_values_pass(
    example_activity_predictions_data, value
):
    """Test that 0, 1, True, and False all pass the classification column's binary check."""
    endpoint = CLASSIFICATION_ENDPOINTS[0]
    # Cast to object first — see comment in test_validate_classification_non_binary_value.
    example_activity_predictions_data[endpoint] = example_activity_predictions_data[
        endpoint
    ].astype(object)
    example_activity_predictions_data.loc[0, endpoint] = value
    validation_result = validate_classification_submission(
        example_activity_predictions_data
    )
    assert validation_result.is_valid


def test_validate_structure_submission_valid(
    example_structure_data_path, example_structure_expected_ids
):
    """Test that a valid structure submission passes validation."""
    validation_result = validate_structure_submission(
        example_structure_data_path, expected_ids=example_structure_expected_ids
    )
    assert validation_result.is_valid
    assert len(validation_result.errors) == 0


def test_validate_structure_submission_not_a_zip(tmp_path):
    """Test that a non-zip file fails structure validation."""
    not_a_zip = tmp_path / "structures.csv"
    not_a_zip.write_text("not a zip file")
    validation_result = validate_structure_submission(not_a_zip)
    log_text = validation_result.get_error_summary()
    assert not validation_result.is_valid
    assert "Structure predictions file must be a zip file" in log_text


def test_validate_structure_submission_empty_zip(tmp_path):
    """Test that an empty zip file fails structure validation."""
    empty_zip = tmp_path / "structures.zip"
    with zipfile.ZipFile(empty_zip, "w"):
        pass
    validation_result = validate_structure_submission(empty_zip)
    log_text = validation_result.get_error_summary()
    assert not validation_result.is_valid
    assert "Zip file contains no PDB files" in log_text


def test_validate_structure_submission_missing_ids(
    example_structure_data_path, example_structure_expected_ids
):
    """Test that a zip missing expected molecule IDs fails validation."""
    extra_id = "E-9999999"
    expected_with_extra = example_structure_expected_ids | {extra_id}
    validation_result = validate_structure_submission(
        example_structure_data_path, expected_ids=expected_with_extra
    )
    log_text = validation_result.get_error_summary()
    assert not validation_result.is_valid
    assert "Zip file is missing 1 expected structure(s)" in log_text


def test_validate_structure_submission_unexpected_ids(
    tmp_path, example_structure_expected_ids
):
    """Test that a zip with unexpected molecule IDs fails validation."""
    bad_zip = tmp_path / "structures.zip"
    unexpected_id = "E-9999999"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        for stem in list(example_structure_expected_ids)[:-1]:
            zf.writestr(f"{stem}.pdb", "")
        zf.writestr(f"{unexpected_id}.pdb", "")
    validation_result = validate_structure_submission(
        bad_zip, expected_ids=example_structure_expected_ids
    )
    log_text = validation_result.get_full_log()
    assert not validation_result.is_valid
    assert "Zip file contains 1 unexpected structure(s)" in log_text
    assert unexpected_id in log_text
