"""Configuration file for pytest."""

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger


@pytest.fixture()
def loguru_caplog():
    """Capture Loguru messages for assertions in tests."""

    @contextmanager
    def _capture(level: str = "INFO"):
        messages = []
        sink_id = logger.add(messages.append, level=level, format="{message}")
        try:
            yield messages
        finally:
            logger.remove(sink_id)

    return _capture


@pytest.fixture()
def example_activity_gound_truth_path():
    """Example activity ground truth data path for testing."""
    return (
        Path(__file__).parent
        / "test_data"
        / "activity-dataset"
        / "activity-dataset.parquet"
    )


@pytest.fixture()
def example_activity_predictions_path():
    """Example activity predictions data path for testing."""
    return (
        Path(__file__).parent
        / "test_data"
        / "activity-submissions"
        / "example-submission.parquet"
    )


@pytest.fixture()
def example_activity_predictions_csv_path():
    """Example activity predictions CSV data path for testing."""
    return (
        Path(__file__).parent
        / "test_data"
        / "activity-submissions"
        / "example-submission.csv"
    )


@pytest.fixture()
def example_activity_predictions_metadata():
    """Example activity predictions metadata for testing."""
    metadata_path = (
        Path(__file__).parent / "test_data" / "activity-submissions" / "metadata.json"
    )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


@pytest.fixture()
def example_structure_data_path():
    """Example structure data path for testing."""
    return (
        Path(__file__).parent / "test_data" / "structure-submissions" / "structures.zip"
    )


@pytest.fixture()
def example_structure_expected_ids():
    """Expected molecule IDs matching the test structure zip."""
    df = pd.read_parquet(
        Path(__file__).parent
        / "test_data"
        / "structure-submissions"
        / "structure-identifiers.parquet"
    )
    return set(df["Molecule_Name"].to_list())


@pytest.fixture()
def example_activity_gound_truth_data(example_activity_gound_truth_path):
    """Example activity ground truth data for testing."""
    return pd.read_parquet(example_activity_gound_truth_path)


@pytest.fixture()
def example_activity_predictions_data(example_activity_predictions_path):
    """Example activity predictions data for testing."""
    return pd.read_parquet(example_activity_predictions_path).copy(deep=True)


@pytest.fixture()
def example_regression_bootstrap_scores():
    """Example regression-track bootstrapped scores for testing (endpoints + MA)."""
    return pd.read_parquet(
        (
            Path(__file__).parent
            / "test_data"
            / "activity-scores"
            / "example-bootstrap-scores-regression.parquet"
        )
    )


@pytest.fixture()
def example_classification_bootstrap_scores():
    """Example classification-track bootstrapped scores for testing (endpoints + MA)."""
    return pd.read_parquet(
        (
            Path(__file__).parent
            / "test_data"
            / "activity-scores"
            / "example-bootstrap-scores-classification.parquet"
        )
    )


@pytest.fixture()
def example_validation_metadata():
    """Example validation metadata for testing."""
    return pd.DataFrame(
        {
            "valid_submission": [True],
            "user_alias": ["test_user_anonymous"],
            "anonymous": [False],
            "username": ["test_user_full"],
            "submitted_at": [datetime(2024, 1, 1, 12, 0, 0)],
            "track": ["Structure Prediction"],
        }
    )


@pytest.fixture
def mock_averaged_results():
    """Mock averaged results for testing leaderboard entry parsing."""
    return pd.DataFrame(
        {
            "accuracy_mean": [0.90],
            "accuracy_std": [0.02],
            "f1_mean": [0.85],
            "f1_std": [0.03],
            "coverage_mean": [1.0],  # Should be ignored by remotesuffix logic
        }
    )


@pytest.fixture
def mock_bootstrap_data():
    """Mock bootstrapped data for testing bootstrap score parsing."""
    return pd.DataFrame(
        {
            "Sample": list(range(100)),
            "accuracy": [0.9 + (i * 0.001) for i in range(100)],
        }
    )
