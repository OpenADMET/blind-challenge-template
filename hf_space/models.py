"""Pydantic models for blind challenge submission data."""

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from utils import _safeify_username


class Submission(BaseModel):
    """A single competition submission.

    Serialisable to JSON for storage in S3 alongside the uploaded prediction
    file. All fields are stored; only non-private fields are ever surfaced on
    the leaderboard.

    Attributes:
        submission_id: Auto-generated UUID, used as the S3 key component.
        submitted_at: UTC timestamp of submission.
        username: HuggingFace username (required, used for deduplication).
        safe_username: Sanitised username for use in S3 keys and file paths.
        user_alias: Optional display alias for anonymous submissions.
        anonymous: If True, display user_alias on leaderboard instead of username.
        participant_name: Real name — stored privately, never displayed.
        discord_username: Discord handle — stored privately.
        email: Contact email — stored privately.
        affiliation: Institutional affiliation — stored privately.
        model_report_link: URL to method report (required before deadline).
        include_in_publication: Opt-in for Challenge publication authorship.
        used_proprietary_data: Whether proprietary data was used in training.
        open_source_code: Whether the participant's code is open-source and
            publicly available.
        track: Competition track.
        filename: Original uploaded filename.
        s3_key: Full S3 object key for the uploaded prediction file.
            Populated by submission_store.upload_submission() after upload.

    """

    model_config = ConfigDict(protected_namespaces=())

    # --- generated ---
    submission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- public identity ---
    username: str
    safe_username: str = ""
    user_alias: str = ""
    anonymous: bool = True

    # --- private contact ---
    participant_name: str = ""
    discord_username: str = ""
    email: str = ""
    affiliation: str = ""
    model_report_link: str = ""
    include_in_publication: bool = False
    used_proprietary_data: bool = False
    open_source_code: bool = False

    # --- submission ---
    track: Literal[
        "Regression Prediction", "Classification Prediction", "Structure Prediction"
    ]
    filename: str
    s3_key: str = ""

    @model_validator(mode="after")
    def _populate_safe_username(self) -> "Submission":
        self.safe_username = _safeify_username(self.username)
        return self

    @property
    def display_name(self) -> str:
        """Name to show on the leaderboard."""
        if self.anonymous and self.user_alias:
            return self.user_alias
        return self.username

    @property
    def s3_prefix(self) -> str:
        """S3 prefix for all objects belonging to this submission.

        Layout::

            submissions/
              regression/{username}/{submission_id}/
                metadata.json
                predictions.parquet   # or .csv
              classification/{username}/{submission_id}/
                metadata.json
                predictions.parquet   # or .csv
              structure/{username}/{submission_id}/
                metadata.json
                structures.zip
        """
        track_slug = {
            "Regression Prediction": "regression",
            "Classification Prediction": "classification",
            "Structure Prediction": "structure",
        }[self.track]
        return f"submissions/{track_slug}/{self.safe_username}/{self.submission_id}"
