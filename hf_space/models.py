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

    Attributes
    ----------
    model_config
        Pydantic model configuration. ``protected_namespaces=()`` disables
        Pydantic's "model_" protected-namespace warning, which would otherwise
        fire on fields like ``model_report_link``.
    submission_id : str
        Auto-generated UUID, used as the S3 key component.
    submitted_at : datetime
        UTC timestamp of submission.
    username : str
        HuggingFace username (required, used for deduplication).
    safe_username : str
        Sanitised username for use in S3 keys and file paths.
    user_alias : str
        Optional display alias for anonymous submissions.
    anonymous : bool
        If True, display user_alias on leaderboard instead of username.
    participant_name : str
        Real name — stored privately, never displayed.
    discord_username : str
        Discord handle — stored privately.
    email : str
        Contact email — stored privately.
    affiliation : str
        Institutional affiliation — stored privately.
    model_report_link : str
        URL to method report (required before deadline).
    used_proprietary_data : bool
        Whether proprietary data was used in training.
    open_source_code : bool
        Whether the participant's code is open-source and publicly available.
    track : Literal["Regression Prediction", "Classification Prediction", "Structure Prediction"]
        Competition track.
    filename : str
        Original uploaded filename.
    s3_key : str
        Full S3 object key for the uploaded prediction file. Populated by
        submission_store.upload_submission() after upload.

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
