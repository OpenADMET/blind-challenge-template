"""Test the CLI commands."""

from click.testing import CliRunner

from backend.cli import validate


def test_validate_cli_three_files(
    example_activity_predictions_path, example_structure_data_path
):
    """Test the validate CLI command with all three prediction files."""
    runner = CliRunner()
    result = runner.invoke(
        validate,
        [
            "--regression-predictions",
            str(example_activity_predictions_path),
            "--classification-predictions",
            str(example_activity_predictions_path),
            "--structure-predictions",
            str(example_structure_data_path),
        ],
    )
    assert result.exit_code == 0, f"CLI command failed with error: {result.output}"


def test_validate_cli_regression_file_only(example_activity_predictions_csv_path):
    """Test the validate CLI command with only the regression predictions file."""
    runner = CliRunner()
    result = runner.invoke(
        validate,
        ["--regression-predictions", str(example_activity_predictions_csv_path)],
    )
    assert result.exit_code == 0, f"CLI command failed with error: {result.output}"


def test_validate_cli_classification_file_only(example_activity_predictions_csv_path):
    """Test the validate CLI command with only the classification predictions file."""
    runner = CliRunner()
    result = runner.invoke(
        validate,
        ["--classification-predictions", str(example_activity_predictions_csv_path)],
    )
    assert result.exit_code == 0, f"CLI command failed with error: {result.output}"


def test_validate_cli_structure_file_only(example_structure_data_path):
    """Test the validate CLI command with only the structure predictions file."""
    runner = CliRunner()
    result = runner.invoke(
        validate,
        ["--structure-predictions", str(example_structure_data_path)],
    )
    assert result.exit_code == 0, f"CLI command failed with error: {result.output}"


def test_validate_cli_no_files():
    """Test the validate CLI command with no files provided."""
    runner = CliRunner()
    result = runner.invoke(validate)
    assert result.exit_code != 0, "CLI command should fail when no files are provided"
    assert (
        "At least one file is required" in result.output
    ), "Error message not found in output"


def test_validate_cli_missing_file():
    """Test the validate CLI command with a missing file path."""
    runner = CliRunner()
    result = runner.invoke(
        validate,
        ["--regression-predictions", "invalid/path/to/predictions.parquet"],
    )
    assert result.exit_code != 0, "CLI command should fail with an invalid file path"
    assert (
        "'invalid/path/to/predictions.parquet' does not exist." in result.output
    ), "Error message not found in output for invalid file path"


def test_validate_cli_invalid_file_format(example_structure_data_path):
    """Test the validate CLI command with an invalid file format."""
    runner = CliRunner()
    result = runner.invoke(
        validate,
        ["--regression-predictions", str(example_structure_data_path)],
    )
    assert result.exit_code == 0, "CLI command should not fail"
    assert (
        "Regression predictions file must be a parquet or csv file." in result.output
    ), "Error message not found in CLI streams for invalid file format"
