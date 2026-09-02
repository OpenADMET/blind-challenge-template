"""Tests for the Discord bot functionality."""

import pandas as pd

from backend.discord_bot import (
    VALID_SUBMISSION_RESPONSES,
    _unpack_metadata,
    prepare_discord_message,
)
from backend.submission_validation import ValidationResult


def test_unpack_metadata(example_validation_metadata):
    """Test unpacking of validation metadata."""
    valid, name, time, track = _unpack_metadata(example_validation_metadata)
    assert isinstance(valid, bool)
    assert isinstance(name, str)
    assert isinstance(time, str)
    assert pd.to_datetime(time)
    assert isinstance(track, str)
    assert track in ["regression", "classification", "structure"]


def test_unpack_metadata_anonymous(example_validation_metadata):
    """Test unpacking of validation metadata for anonymous submission."""
    example_validation_metadata["anonymous"] = True
    _, name, _, _ = _unpack_metadata(example_validation_metadata)
    assert name == "test_user_anonymous"


def test_prepare_discord_message_system_errors(example_validation_metadata):
    """System errors should produce a red generic error notification."""
    validation_result = ValidationResult()
    validation_result.add_system_error("unexpected failure")

    color, content, long_errors = prepare_discord_message(
        example_validation_metadata, validation_result
    )

    assert color == 15158332
    assert (
        "❌ Sorry **test_user_full**, something went wrong with your structure submission"
        in content
    )
    assert long_errors is None


def test_prepare_discord_message_partial_scoring(example_validation_metadata):
    """Partial scoring should produce warning content and attach full details."""
    validation_result = ValidationResult(is_valid=False, scoring="partial")
    validation_result.scoring_errors = ["OADMET-00001", "OADMET-00002"]

    color, content, long_errors = prepare_discord_message(
        example_validation_metadata, validation_result
    )

    assert color == 15105570
    assert "Partially valid structure submission from" in content
    assert long_errors is not None
    assert "OADMET-00001" in long_errors


def test_prepare_discord_message_scoring_failed(example_validation_metadata):
    """Scoring failures should produce a red generic error notification."""
    validation_result = ValidationResult(is_valid=True, scoring="failed")

    color, content, long_errors = prepare_discord_message(
        example_validation_metadata, validation_result
    )

    assert color == 15158332
    assert (
        "❌ Sorry **test_user_full**, something went wrong with your structure submission"
        in content
    )
    assert long_errors is None


def test_prepare_discord_message_system_error_takes_precedence(
    example_validation_metadata,
):
    """System errors should override detailed validation messages."""
    example_validation_metadata["valid_submission"] = False
    validation_result = ValidationResult(is_valid=False)
    validation_result.add_error(
        "Missing expected molecule(s)", ["OADMET-00001", "OADMET-00002"]
    )
    validation_result.add_system_error("unexpected failure")

    color, content, long_errors = prepare_discord_message(
        example_validation_metadata, validation_result
    )

    assert color == 15158332
    assert (
        "❌ Sorry **test_user_full**, something went wrong with your structure submission"
        in content
    )
    assert long_errors is None


def test_prepare_discord_message_partial_scoring_empty_ids(example_validation_metadata):
    """Partial scoring with no IDs still returns an orange warning and attachment."""
    validation_result = ValidationResult(is_valid=True, scoring="partial")
    validation_result.scoring_errors = []

    color, content, long_errors = prepare_discord_message(
        example_validation_metadata, validation_result
    )

    assert color == 15105570
    assert "Partially valid" in content
    assert long_errors is not None
    assert long_errors.startswith("Scoring failed for the following compounds:")


def test_prepare_discord_message_valid_submission(example_validation_metadata):
    """Fully valid submissions should produce a green success notification."""
    validation_result = ValidationResult(is_valid=True, scoring="complete")

    color, content, long_errors = prepare_discord_message(
        example_validation_metadata, validation_result
    )
    _, name, time, track = _unpack_metadata(example_validation_metadata)
    expected_messages = [
        template.format(track=track, name=name, time=time)
        for template in VALID_SUBMISSION_RESPONSES
    ]

    assert color == 3066993
    assert content in expected_messages
    assert "{" not in content
    assert long_errors is None


def test_prepare_discord_message_invalid_with_long_error_details(
    example_validation_metadata,
):
    """Invalid submissions with detailed errors should mention attached details."""
    example_validation_metadata["valid_submission"] = False
    validation_result = ValidationResult(is_valid=False)
    validation_result.add_error(
        "Missing expected molecule(s)", ["OADMET-00001", "OADMET-00002"]
    )

    color, content, long_errors = prepare_discord_message(
        example_validation_metadata, validation_result
    )

    assert color == 15158332
    assert "Invalid" in content
    assert "See file above for more details" in content
    assert long_errors is not None
    assert "OADMET-00001" in long_errors


def test_prepare_discord_message_invalid_without_long_error_details(
    example_validation_metadata,
):
    """Invalid submissions with short errors should not mention file attachment."""
    example_validation_metadata["valid_submission"] = False
    validation_result = ValidationResult(is_valid=False)
    validation_result.add_error("Structure predictions file must be a zip file.")

    color, content, long_errors = prepare_discord_message(
        example_validation_metadata, validation_result
    )

    assert color == 15158332
    assert "Invalid" in content
    assert "See file above for more details" not in content
    assert long_errors is None
