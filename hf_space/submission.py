"""Submission persistence — uploads prediction files and metadata to S3.

Architecture
------------
Each submission is stored in S3 under a deterministic prefix::

    s3://{S3_BUCKET}/submissions/{track}/{username}/{submission_id}/
        metadata.json       <- Submission model serialised as JSON
        {original_filename} <- The uploaded prediction file

A scheduled AWS Lambda function (``backend/lambda_handler.py``) polls this
prefix for unscored submissions, runs evaluation, and writes scores back.

Environment variables
---------------------
S3_BUCKET
    Name of the S3 bucket (required at runtime, not at import time).
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
    Standard boto3 credentials — set via HuggingFace Space secrets.
"""

import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import boto3
import gradio as gr
import numpy as np
import pandas as pd
from config import (
    ACTIVITY_DATASET_SIZE,
    AWS_DEFAULT_REGION,
    CLASSIFICATION_ENDPOINTS,
    DATASET_DOWNLOAD_LINK,
    HOURS_BETWEEN_SUBMISSIONS,
    MIN_PREDICTION_STD,
    REGRESSION_ENDPOINTS,
    REQUIRED_CLASSIFICATION_COLUMNS,
    REQUIRED_REGRESSION_COLUMNS,
    S3_BUCKET,
    STRUCTURE_DATASET_SIZE,
    SUBMISSION_CSV_EXAMPLE_CLASSIFICATION_MD,
    SUBMISSION_CSV_EXAMPLE_REGRESSION_MD,
    SUBMISSION_SCHEMA_TABLE_CLASSIFICATION_MD,
    SUBMISSION_SCHEMA_TABLE_REGRESSION_MD,
    TUTORIAL_LINK,
)
from loguru import logger
from models import Submission
from utils import (
    BANNED_USERNAMES,
    _safeify_username,
    validate_hf_username,
    validate_model_details,
)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def render_submission_tab(
    phase: int,
    regression_active: bool,
    classification_active: bool,
    structure_track_live: bool,
) -> None:
    """Render the submission tab content based on the current challenge phase.

    ``regression_active`` / ``classification_active`` / ``structure_track_live``
    each gate whether that track's schema block and Track radio option appear —
    submissions only, not leaderboards.
    """
    if phase == 0:
        gr.Markdown("""
# Submission Portal (Coming Soon)
The submission portal will open when the challenge begins. Please check back once the challenge opens.
""")
    elif phase in [3, 4]:
        gr.Markdown("""
# Submission Portal (Closed)
The submission portal is now closed. Thank you for your participation!

If you were unable to get your submission in before the deadline, please contact us through Discord and we *may* be able to help.
""")
    else:
        gr.Markdown("""# Submit Your Predictions""")
        intro_markdown = f"""New to the challenge? See the [challenge tutorial]({TUTORIAL_LINK}) for a step-by-step walkthrough of data loading, model building, and preparing your submission. The tutorial repository also includes a validation script to check your files before uploading. Training and test sets are available on the [HuggingFace dataset page]({DATASET_DOWNLOAD_LINK})."""

        regression_markdown = f"""
#### Regression Track
Submit a **`.parquet` or `.csv`** file with exactly **{ACTIVITY_DATASET_SIZE} rows** and {len(REQUIRED_REGRESSION_COLUMNS)} columns — a numeric prediction for all {len(REGRESSION_ENDPOINTS)} regression endpoints, for every compound, no `NaN` or `inf`:

{SUBMISSION_SCHEMA_TABLE_REGRESSION_MD}

**Example (CSV):**
{SUBMISSION_CSV_EXAMPLE_REGRESSION_MD}
"""
        classification_markdown = f"""
#### Classification Track
Submit a **`.parquet` or `.csv`** file with exactly **{ACTIVITY_DATASET_SIZE} rows** and {len(REQUIRED_CLASSIFICATION_COLUMNS)} columns — a boolean prediction for each of the {len(CLASSIFICATION_ENDPOINTS)} classification endpoints, for every compound:

{SUBMISSION_SCHEMA_TABLE_CLASSIFICATION_MD}

**Example (CSV):**
{SUBMISSION_CSV_EXAMPLE_CLASSIFICATION_MD}
"""
        structure_markdown = f"""
#### Structure Track
Submit a **`.zip`** archive containing exactly **{STRUCTURE_DATASET_SIZE} `.pdb` files**, one per compound, named after the compound identifier (e.g. `x00011-1.pdb`). Each file must be a full protein–ligand complex with the ligand residue named **`LIG`**.
"""
        submission_markdown = intro_markdown
        if regression_active:
            submission_markdown += regression_markdown
        if classification_active:
            submission_markdown += classification_markdown
        if structure_track_live:
            submission_markdown += structure_markdown
        gr.Markdown(submission_markdown)

        with gr.Row():
            # --- Column 1: Leaderboard identity ---
            with gr.Column(scale=1):
                gr.Markdown("### Leaderboard Identity")
                username_input = gr.Textbox(
                    label="HuggingFace Username *",
                    placeholder="your-hf-username",
                    info="Required. Used to track submissions.",
                )
                user_alias = gr.Textbox(
                    label="Alias (Optional display name for the leaderboard. "
                    "Anonymous by default.)",
                    placeholder="Chemprop Wizard",
                )
                anon_checkbox = gr.Checkbox(
                    label="Submit anonymously (show alias instead of username on "
                    "leaderboard and Discord submission validation bot)",
                    value=True,
                )
                proprietary_data_checkbox = gr.Checkbox(
                    label="I used proprietary data (not publicly available) in training my model",
                    value=False,
                    info="Displayed publicly on the leaderboard.",
                )
                open_code_checkbox = gr.Checkbox(
                    label="My code is open-source and publicly available",
                    value=False,
                    info="Include a link to your code as your model report.",
                )

            # --- Column 2: Contact & publication details ---
            with gr.Column(scale=1):
                gr.Markdown("### Contact & Publication *(optional, private)*")
                participant_name = gr.Textbox(
                    label="Full Name",
                    placeholder="Jane Smith",
                    info="Not displayed publicly; used for publication tracking.",
                )
                discord_username = gr.Textbox(
                    label="Discord Username",
                    placeholder="janesmith#1234",
                )
                email = gr.Textbox(
                    label="Email",
                    placeholder="jane@example.com",
                )
                affiliation = gr.Textbox(
                    label="Affiliation",
                    placeholder="University / Company",
                )
                model_tag = gr.Textbox(
                    label="Method Report Link",
                    placeholder="https://...",
                    info="Required before the deadline to appear on the final "
                    "leaderboard. Required if you have checked the open-source code "
                    "box. Must be a reachable link that starts with 'https://'.",
                )

            # --- Column 3: Track & file ---
            with gr.Column(scale=1):
                gr.Markdown("### Submission")
                tracks = []
                if regression_active:
                    tracks.append("Regression Prediction")
                if classification_active:
                    tracks.append("Classification Prediction")
                if structure_track_live:
                    tracks.append("Structure Prediction")
                track_select = gr.Radio(
                    tracks,
                    label="Track *",
                )
                file_input = gr.File(label="Upload File *")

            # --- Submit row ---
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    submit_btn = gr.Button(
                        "Submit predictions", variant="primary", size="lg"
                    )
                    submit_msg = gr.Textbox(
                        label="Submission Status",
                        lines=2,
                        visible=False,
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    pass

            submit_btn.click(
                fn=submit_predictions,
                inputs=[
                    username_input,
                    user_alias,
                    anon_checkbox,
                    participant_name,
                    discord_username,
                    email,
                    affiliation,
                    model_tag,
                    paper_checkbox,
                    proprietary_data_checkbox,
                    open_code_checkbox,
                    track_select,
                    file_input,
                ],
                outputs=[submit_msg],
            )


def upload_submission(submission: Submission, file_path: Path) -> Submission:
    """Upload the prediction file and serialised metadata to S3.

    On success, returns the submission with ``s3_key`` populated.
    On failure, logs the error and returns the submission unchanged so the
    caller can still surface a user-facing message.

    Args:
        submission (Submission): Validated Submission instance (s3_key will be set
            here).
        file_path (Path): Local path to the uploaded prediction file.

    Returns:
        Submission: The submission with s3_key set to the uploaded object key.

    Todo:
        - Consider server-side encryption (SSE-S3 or SSE-KMS).

    """
    bucket = S3_BUCKET
    if not bucket:
        logger.warning(
            "S3_BUCKET not set — submission will not be persisted. "
            "Set S3_BUCKET and AWS credentials as Space secrets to enable storage."
        )
        return submission

    if submission.track == "Structure Prediction":
        canonical_filename = "structures.zip"
    else:
        canonical_filename = f"predictions{Path(file_path).suffix}"
    file_key = f"{submission.s3_prefix}/{canonical_filename}"
    metadata_key = f"{submission.s3_prefix}/metadata.json"
    submission = submission.model_copy(update={"s3_key": file_key})

    try:
        s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)

        # Upload prediction file
        logger.info(f"Uploading prediction file to s3://{bucket}/{file_key}")
        s3.upload_file(str(file_path), bucket, file_key)

        # Upload metadata JSON
        logger.info(f"Uploading metadata to s3://{bucket}/{metadata_key}")
        s3.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=submission.model_dump_json(indent=2).encode(),
            ContentType="application/json",
        )

        logger.info(f"Submission {submission.submission_id!r} stored successfully.")

    except Exception as exc:
        logger.error(
            f"S3 upload failed for submission {submission.submission_id!r}: {exc}"
        )

    return submission


def _fetch_last_submission_date(track: str, user_id: str) -> datetime | None:
    """Fetch the submission date of the most recent submission for a track and user.

    Args:
        track (str): The track name (e.g., "regression", "classification", or
            "structure").
        user_id (str): The user ID to check for previous submissions.

    Returns:
        datetime | None: The submission date of the most recent submission, or None if
                         no previous submissions are found.

    """
    bucket = S3_BUCKET
    if not bucket:
        logger.warning(
            "S3_BUCKET not set — cannot fetch last submission date. "
            "Set S3_BUCKET and AWS credentials as Space secrets to enable this feature."
        )
        return None

    s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    prefix = f"submissions/{track}/{user_id}/"
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if response["IsTruncated"]:  # Unlikely to be > 1000 submissions per user
            logger.warning(
                f"ListObjectsV2 response truncated for prefix {prefix!r}. "
                "Only the first 1000 objects will be considered."
            )
        if "Contents" not in response:
            return None  # No submissions found

        logger.info(
            f"Found {len(response['Contents'])} objects under prefix {prefix!r}."
        )
        submission_dates = []
        for obj in response["Contents"]:
            if obj["Key"].endswith("metadata.json"):
                submission_dates.append(obj["LastModified"])

        if not submission_dates:  # Shouldn't be possible
            logger.warning(f"No metadata.json files found under prefix {prefix!r}.")
            return None

        return max(submission_dates).astimezone(timezone.utc)

    except Exception as exc:
        logger.error(f"Failed to fetch last submission date for {user_id!r}: {exc}")
        return None


def _format_submission_time_message(last_submission: datetime, track: str) -> str:
    """Format a message indicating when the user can next submit next."""
    next_submission_time = last_submission + pd.Timedelta(
        hours=HOURS_BETWEEN_SUBMISSIONS
    )
    time_remaining = next_submission_time - datetime.now(timezone.utc)
    seconds_left = max(0, int(time_remaining.total_seconds()))
    hours, rem = divmod(seconds_left, 3600)
    minutes, seconds = divmod(rem, 60)
    wait_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return (
        f"Error: You submitted a {track} prediction on "
        f"{last_submission.strftime('%Y-%m-%d %H:%M:%S (UTC)')}.\n"
        f"Please wait {wait_str} before submitting again."
    )


def _read_tabular_submission(
    file_path: Path, required_columns: list[str], track_label: str
) -> tuple[pd.DataFrame | None, str | None]:
    """Read and apply the shared row/column checks for a tabular submission.

    Args:
        file_path (Path): Path to the uploaded file.
        required_columns (list[str]): Columns the file must contain — this track's
            identifiers plus its own endpoint columns.
        track_label (str): Human-readable track name for error messages (e.g.
            "Regression", "Classification").

    Returns:
        tuple[pd.DataFrame | None, str | None]: The parsed DataFrame and ``None`` on
            success, or ``None`` and a user-facing error message on failure.

    """
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        try:
            df = pd.read_parquet(file_path)
        except Exception as exc:
            return None, f"Error: Could not read parquet file — {exc}"
    elif suffix == ".csv":
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            return None, f"Error: Could not read CSV file — {exc}"
    else:
        return (
            None,
            f"Error: {track_label} submissions must be a .parquet or .csv file.",
        )
    if len(df) != ACTIVITY_DATASET_SIZE:
        return None, f"Error: Expected {ACTIVITY_DATASET_SIZE} rows, got {len(df)}."
    missing = set(required_columns) - set(df.columns)
    if missing:
        return None, f"Error: Missing required columns: {missing}"
    return df, None


def submit_predictions(
    username: str,
    user_alias: str,
    anon_checkbox: bool,
    participant_name: str,
    discord_username: str,
    email: str,
    affiliation: str,
    model_tag: str,
    paper_checkbox: bool,
    proprietary_data_checkbox: bool,
    open_code_checkbox: bool,
    track_select: Literal[
        "Regression Prediction", "Classification Prediction", "Structure Prediction"
    ],
    file_input: str | None,
) -> dict:
    """Handle a competition submission from the web UI.

    Validates required fields and the uploaded file, then records the submission.

    Args:
        username (str): HuggingFace username (required). Checked against Hugging
            Face to confirm the account exists — this is not an identity proof, just
            a sanity check that the username is real.
        user_alias (str): Optional alias for anonymous display on the leaderboard.
        anon_checkbox (bool): If True, display alias instead of username.
        participant_name (str): Real name (private, not displayed publicly).
        discord_username (str): Discord handle (optional).
        email (str): Contact email (optional).
        affiliation (str): Institutional affiliation (optional).
        model_tag (str): Link to method report (optional). Only checked for
            reachability when ``open_code_checkbox`` is True.
        paper_checkbox (bool): Opt-in for future publication inclusion.
        proprietary_data_checkbox (bool): Whether proprietary data was used in
            training.
        open_code_checkbox (bool): Whether the participant's code is open-source
            and publicly available. When True, ``model_tag`` is validated as a
            reachable link.
        track_select (Literal["Regression Prediction", "Classification Prediction",
            "Structure Prediction"]): The selected competition track.
        file_input (str | None): Path to the uploaded submission file.

    Returns:
        dict: gr.update with a status message and visible=True.

    """
    # --- required field validation ---
    if not username or not username.strip():
        return gr.update(
            value="Error: Hugging Face username is required.", visible=True
        )
    if not validate_hf_username(username.strip()):
        logger.warning(f"Invalid Hugging Face username format: {username.strip()!r}")
        return gr.update(
            value=f"Error: Hugging Face username {username.strip()!r} could not be "
            "found. Please check for typos.",
            visible=True,
        )
    banned_reason = BANNED_USERNAMES.get(username.strip().lower())
    if banned_reason:
        logger.warning(
            f"Blocked submission attempt from banned username: {username.strip()!r}"
        )
        return gr.update(
            value=f"Error: Your account ({username.strip()!r}) has been blocked. Reason: {banned_reason}. "
            "If you believe this is a mistake, please contact the organisers.",
            visible=True,
        )
    if anon_checkbox and (not user_alias or not user_alias.strip()):
        return gr.update(
            value="Error: An alias is required when submitting anonymously. "
            "Uncheck the 'Submit anonymously' box or provide an alias.",
            visible=True,
        )
    if email and email.strip() and not EMAIL_RE.match(email.strip()):
        logger.warning(f"Invalid email format: {email.strip()!r}")
        return gr.update(
            value="Error: Please enter a valid email address.", visible=True
        )
    if open_code_checkbox:
        model_status = validate_model_details(model_tag)
        if model_status in ["Invalid link", "Not submitted"]:
            logger.warning(f"Invalid model report link: {model_tag.strip()!r}")
            return gr.update(
                value="Error: Could not open the Method Report Link. Please check "
                "the URL and try again, or uncheck the 'open code' box.",
                visible=True,
            )
    else:
        model_status = model_tag.strip() if model_tag else "Not submitted"
    if not track_select:
        return gr.update(value="Error: Please select a track.", visible=True)
    if file_input is None:
        return gr.update(value="Error: Please upload a submission file.", visible=True)

    file_path = Path(file_input)

    # --- file format validation ---
    if track_select == "Regression Prediction":
        df, error = _read_tabular_submission(
            file_path, REQUIRED_REGRESSION_COLUMNS, "Regression"
        )
        if error:
            return gr.update(value=error, visible=True)
        for col in REGRESSION_ENDPOINTS:
            if df[col].isnull().any():
                return gr.update(
                    value=f"Error: {col} column contains NaN values.", visible=True
                )
            if not np.isfinite(df[col]).all():
                return gr.update(
                    value=f"Error: {col} column contains infinite values.", visible=True
                )
            if df[col].nunique() <= 1 or df[col].std() < MIN_PREDICTION_STD:
                return gr.update(
                    value=f"Error: {col} predictions are constant or near-constant — "
                    "please submit real model outputs.",
                    visible=True,
                )
        last_submission = _fetch_last_submission_date(
            "regression", _safeify_username(username.strip())
        )
        logger.info(
            f"Last submission date for user {username.strip()!r}: {last_submission}"
        )
        if (
            last_submission
            and (datetime.now(timezone.utc) - last_submission).total_seconds()
            < HOURS_BETWEEN_SUBMISSIONS * 3600
        ):
            return gr.update(
                value=_format_submission_time_message(
                    last_submission, track="regression"
                ),
                visible=True,
            )

    elif track_select == "Classification Prediction":
        df, error = _read_tabular_submission(
            file_path, REQUIRED_CLASSIFICATION_COLUMNS, "Classification"
        )
        if error:
            return gr.update(value=error, visible=True)
        for col in CLASSIFICATION_ENDPOINTS:
            if df[col].isnull().any():
                return gr.update(
                    value=f"Error: {col} column contains NaN values.", visible=True
                )
            if not df[col].isin([0, 1, True, False]).all():
                return gr.update(
                    value=f"Error: {col} column contains non-binary values.",
                    visible=True,
                )
            if df[col].nunique() <= 1:
                return gr.update(
                    value=f"Error: {col} predictions are all one class — "
                    "a constant prediction cannot be scored.",
                    visible=True,
                )
        last_submission = _fetch_last_submission_date(
            "classification", _safeify_username(username.strip())
        )
        logger.info(
            f"Last submission date for user {username.strip()!r}: {last_submission}"
        )
        if (
            last_submission
            and (datetime.now(timezone.utc) - last_submission).total_seconds()
            < HOURS_BETWEEN_SUBMISSIONS * 3600
        ):
            return gr.update(
                value=_format_submission_time_message(
                    last_submission, track="classification"
                ),
                visible=True,
            )

    elif track_select == "Structure Prediction":
        if file_path.suffix.lower() != ".zip":
            return gr.update(
                value="Error: Structure submissions must be a .zip file.", visible=True
            )
        try:
            with zipfile.ZipFile(file_path) as zf:
                n_files = len(zf.namelist())
        except Exception as exc:
            return gr.update(
                value=f"Error: Could not read zip file — {exc}", visible=True
            )
        if n_files != STRUCTURE_DATASET_SIZE:
            return gr.update(
                value=f"Error: Expected {STRUCTURE_DATASET_SIZE} files in zip, got {n_files}.",
                visible=True,
            )
        last_submission = _fetch_last_submission_date(
            "structure", _safeify_username(username.strip())
        )
        logger.info(
            f"Last submission date for user {username.strip()!r}: {last_submission}"
        )
        if (
            last_submission
            and (datetime.now(timezone.utc) - last_submission).total_seconds()
            < HOURS_BETWEEN_SUBMISSIONS * 3600
        ):
            return gr.update(
                value=_format_submission_time_message(
                    last_submission, track="structure"
                ),
                visible=True,
            )

    # --- build submission model and persist to S3 ---
    submission = Submission(
        username=username.strip(),
        user_alias=user_alias.strip(),
        anonymous=anon_checkbox,
        participant_name=participant_name.strip(),
        discord_username=discord_username.strip(),
        email=email.strip(),
        affiliation=affiliation.strip(),
        model_report_link=model_status,
        include_in_publication=paper_checkbox,
        used_proprietary_data=proprietary_data_checkbox,
        open_source_code=open_code_checkbox,
        track=track_select,
        filename=file_path.name,
    )
    logger.info(
        f"Submission received: id={submission.submission_id!r} "
        f"user={submission.username!r} track={submission.track!r}"
    )
    upload_submission(submission, file_path)

    display_name = submission.display_name
    return gr.update(
        value=(
            f"Submission received from {display_name!r} for the {track_select} track. "
            "Your predictions are being processed and should appear on the leaderboard within 2 hours."
        ),
        visible=True,
    )
