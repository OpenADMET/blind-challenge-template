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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import boto3
import gradio as gr
import numpy as np
import pandas as pd
from config import (
    ACTIVITY_DATASET_SIZE,
    AWS_DEFAULT_REGION,
    CLASSIFICATION_NAME,
    CLASSIFICATION_SPEC,
    CLASSIFICATION_TRACK,
    DATASET_DOWNLOAD_LINK,
    HOURS_BETWEEN_SUBMISSIONS,
    MIN_ACCOUNT_AGE_DAYS,
    MIN_PREDICTION_STD,
    REGRESSION_NAME,
    REGRESSION_SPEC,
    REGRESSION_TRACK,
    REQUIRED_CLASSIFICATION_COLUMNS,
    REQUIRE_MODEL_REPORT,
    REQUIRED_REGRESSION_COLUMNS,
    S3_BUCKET,
    STRUCTURE_DATASET_SIZE,
    STRUCTURE_NAME,
    STRUCTURE_TRACK,
    SUBMISSION_CSV_EXAMPLE_CLASSIFICATION_MD,
    SUBMISSION_CSV_EXAMPLE_REGRESSION_MD,
    SUBMISSION_SCHEMA_TABLE_CLASSIFICATION_MD,
    SUBMISSION_SCHEMA_TABLE_REGRESSION_MD,
    TRACK_VISIBILITY,
    TUTORIAL_LINK,
    TrackVisibility,
)
from gradio.utils import get_space
from loguru import logger
from models import Submission
from utils import (
    BANNED_USERNAMES,
    HFAccount,
    _safeify_username,
    fetch_hf_account,
    validate_model_details,
)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# The track names on the form, and the S3 track prefix each submission goes under.
TRACK_SLUGS = {
    REGRESSION_NAME: REGRESSION_TRACK,
    CLASSIFICATION_NAME: CLASSIFICATION_TRACK,
    STRUCTURE_NAME: STRUCTURE_TRACK,
}

# Aliases are shown on the leaderboards, which render cells as HTML. Rendering
# escapes them as well, but markup and control characters are refused up front.
MAX_ALIAS_LENGTH = 40
ALIAS_FORBIDDEN_CHARS = "<>&\"'`"
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
# Characters that can't appear unencoded in a URL and could break out of the
# leaderboard's href attribute.
_URL_FORBIDDEN_RE = re.compile(r"[\s<>\"'`\x00-\x1f\x7f]")


def alias_error(alias: str) -> str | None:
    """Explain why an alias can't be used, or return None if it's fine.

    Parameters
    ----------
    alias : str
        The alias as entered, already stripped of surrounding whitespace.

    Returns
    -------
    str | None
        A message for the participant naming the rule the alias breaks, or None.

    """
    if len(alias) > MAX_ALIAS_LENGTH:
        return (
            f"Error: Your alias is {len(alias)} characters long. The maximum is "
            f"{MAX_ALIAS_LENGTH} characters, so please shorten it."
        )
    bad_chars = sorted({c for c in alias if c in ALIAS_FORBIDDEN_CHARS})
    if bad_chars:
        listed = " ".join(bad_chars)
        return (
            f"Error: Your alias contains {listed}, which can't be used in an alias. "
            f"Aliases can't contain any of {' '.join(ALIAS_FORBIDDEN_CHARS)} "
            "because they are shown on the leaderboard web page."
        )
    if _CONTROL_CHAR_RE.search(alias):
        return (
            "Error: Your alias contains a control character (such as a tab or a "
            "line break). Please use a single line of plain text."
        )
    return None


def model_report_link_error(link: str) -> str | None:
    """Explain why a model report link can't be used, or return None if it's fine.

    Only checks for characters that can't appear unencoded in a URL; whether the
    link starts with ``https://`` and is reachable is checked separately
    (``utils.validate_model_details``).

    Parameters
    ----------
    link : str
        The link as entered, already stripped of surrounding whitespace.

    Returns
    -------
    str | None
        A message for the participant, or None.

    """
    if _URL_FORBIDDEN_RE.search(link):
        return (
            "Error: The Model Report Link contains a space, quote, < or > or "
            "another character that can't appear in a web address. Please paste "
            "the link exactly as it appears in your browser's address bar."
        )
    return None


def render_submission_tab(
    phase: int, visibility: dict[str, TrackVisibility] | None = None
) -> None:
    """Render the submission tab content based on the current challenge phase.

    A track's schema block and Track option appear only if its visibility has
    ``submissions_open``. ``visibility`` defaults to ``config.TRACK_VISIBILITY``.
    """
    visibility = TRACK_VISIBILITY if visibility is None else visibility
    open_tracks = {track for track, v in visibility.items() if v.submissions_open}
    if phase == 0:
        gr.Markdown("""
## Submission Portal (Coming Soon)
The submission portal will open when the challenge begins. Please check back once the challenge opens.
""")
    elif phase in [3, 4]:
        gr.Markdown("""
## Submission Portal (Closed)
The submission portal is now closed. Thank you for your participation!

If you were unable to get your submission in before the deadline, please contact us through Discord and we *may* be able to help.
""")
    else:
        gr.Markdown("## Submit Your Predictions")
        gr.Markdown(
            f"New to the challenge? See the [challenge tutorial]({TUTORIAL_LINK}) for a "
            "step-by-step walkthrough of data loading, model building, and preparing "
            "your submission. The tutorial repository also includes a validation "
            "script to check your files before uploading. Training and test sets are "
            f"available on the [Hugging Face dataset page]({DATASET_DOWNLOAD_LINK})."
        )
        formats = [
            (REGRESSION_TRACK, REGRESSION_NAME, _regression_format_md()),
            (CLASSIFICATION_TRACK, CLASSIFICATION_NAME, _classification_format_md()),
            (STRUCTURE_TRACK, STRUCTURE_NAME, _structure_format_md()),
        ]
        for track, name, format_md in formats:
            if track in open_tracks:
                with gr.Accordion(f"{name}: file format", open=False):
                    gr.Markdown(format_md)

        gr.Markdown("### 1. Log in with Hugging Face")
        with gr.Column(elem_classes="login-card"):
            gr.Markdown(_login_note(), elem_classes="login-note")
            with gr.Row(elem_classes="login-button-row"):
                gr.LoginButton(size="md", scale=0, min_width=260)

        gr.Markdown("### 2. Choose a track and upload your file")
        with gr.Row(equal_height=True, elem_classes="track-file-row"):
            track_select = gr.Radio(
                [name for track, name, _ in formats if track in open_tracks],
                label="Track *",
                scale=1,
                elem_classes="track-select",
            )
            file_input = gr.File(label="Prediction file *", scale=1, height=190)

        with gr.Row(equal_height=True, elem_classes="entry-row"):
            with gr.Column(scale=1):
                gr.Markdown("### 3. How your entry appears")
                user_alias = gr.Textbox(
                    label="Alias (display name for the leaderboard)",
                    placeholder="Chemprop Wizard",
                    info=f"Up to {MAX_ALIAS_LENGTH} characters. Can't contain "
                    f"{' '.join(ALIAS_FORBIDDEN_CHARS)}.",
                    max_lines=1,
                )
                anon_checkbox = gr.Checkbox(
                    label="Submit anonymously (show my alias instead of my username "
                    "on the leaderboard and the Discord submission bot)",
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
                model_tag = gr.Textbox(
                    label="Model Report Link",
                    placeholder="https://...",
                    info=(
                        "Required before the deadline to appear on the final "
                        "leaderboard. "
                        if REQUIRE_MODEL_REPORT
                        else ""
                    )
                    + "Required if you have checked the open-source code box. Must "
                    "be a reachable link that starts with 'https://'.",
                )
            with gr.Column(scale=1):
                gr.Markdown("### 4. Contact details (optional, private)")
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

        submit_btn = gr.Button("Submit predictions", variant="primary", size="lg")
        submit_msg = gr.Textbox(
            label="Submission Status",
            lines=2,
            visible=False,
            interactive=False,
        )
        # The logged-in profile isn't an input: Gradio passes it to
        # submit_predictions's ``profile`` parameter from the session.
        submit_btn.click(
            fn=submit_predictions,
            inputs=[
                user_alias,
                anon_checkbox,
                participant_name,
                discord_username,
                email,
                affiliation,
                model_tag,
                proprietary_data_checkbox,
                open_code_checkbox,
                track_select,
                file_input,
            ],
            outputs=[submit_msg],
        )


def _login_note() -> str:
    """Explain what logging in is for, next to the login button."""
    age = (
        f" Accounts must be at least {MIN_ACCOUNT_AGE_DAYS} "
        f"day{'s' if MIN_ACCOUNT_AGE_DAYS != 1 else ''} old."
        if MIN_ACCOUNT_AGE_DAYS
        else ""
    )
    return (
        "Your Hugging Face account identifies your submissions; no other account "
        f"details are shared.{age} You can submit to each track once every "
        f"{HOURS_BETWEEN_SUBMISSIONS} hours. Submissions are made through this page "
        "only, not through the API."
    )


def _regression_format_md() -> str:
    """Describe the regression track's file format."""
    return f"""Submit a **`.parquet` or `.csv`** file with exactly **{ACTIVITY_DATASET_SIZE} rows** and {len(REQUIRED_REGRESSION_COLUMNS)} columns — a numeric prediction for all {len(REGRESSION_SPEC.endpoints)} regression endpoints, for every compound, no `NaN` or `inf`:

{SUBMISSION_SCHEMA_TABLE_REGRESSION_MD}

**Example (CSV):**
{SUBMISSION_CSV_EXAMPLE_REGRESSION_MD}
"""


def _classification_format_md() -> str:
    """Describe the classification track's file format."""
    return f"""Submit a **`.parquet` or `.csv`** file with exactly **{ACTIVITY_DATASET_SIZE} rows** and {len(REQUIRED_CLASSIFICATION_COLUMNS)} columns — a boolean prediction for each of the {len(CLASSIFICATION_SPEC.endpoints)} classification endpoints, for every compound:

{SUBMISSION_SCHEMA_TABLE_CLASSIFICATION_MD}

**Example (CSV):**
{SUBMISSION_CSV_EXAMPLE_CLASSIFICATION_MD}
"""


def _structure_format_md() -> str:
    """Describe the structure track's file format."""
    return f"""Submit a **`.zip`** archive containing exactly **{STRUCTURE_DATASET_SIZE} `.pdb` files**, one per compound, named after the compound identifier (e.g. `x00011-1.pdb`). Each file must be a full protein–ligand complex containing **exactly one** predicted ligand pose, as a single residue named **`LIG`**. A submission where any file has no `LIG` residue, or more than one, is rejected as a whole, and the rejection still counts towards the submission cooldown, so please check your files with the validation code first.
"""


class UploadError(RuntimeError):
    """A submission couldn't be saved to S3."""


def upload_submission(submission: Submission, file_path: Path) -> Submission:
    """Upload the prediction file and serialised metadata to S3.

    The prediction file goes first and the metadata second. The backend scores, and
    the cooldown counts, only submissions with metadata, so a failure part-way
    leaves nothing that counts.

    Parameters
    ----------
    submission : Submission
        Validated Submission instance (s3_key will be set here).
    file_path : Path
        Local path to the uploaded prediction file.

    Returns
    -------
    Submission
        The submission with s3_key set to the uploaded object key.

    Raises
    ------
    UploadError
        If ``S3_BUCKET`` isn't set or either upload fails.

    """
    bucket = S3_BUCKET
    if not bucket:
        raise UploadError(
            "S3_BUCKET not set; set it and AWS credentials as Space secrets."
        )

    if submission.track == "Structure Prediction":
        canonical_filename = "structures.zip"
    else:
        canonical_filename = f"predictions{Path(file_path).suffix}"
    file_key = f"{submission.s3_prefix}/{canonical_filename}"
    metadata_key = f"{submission.s3_prefix}/metadata.json"
    submission = submission.model_copy(update={"s3_key": file_key})

    try:
        s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
        logger.info(f"Uploading prediction file to s3://{bucket}/{file_key}")
        s3.upload_file(str(file_path), bucket, file_key)
        logger.info(f"Uploading metadata to s3://{bucket}/{metadata_key}")
        s3.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=submission.model_dump_json(indent=2).encode(),
            ContentType="application/json",
        )
    except Exception as exc:
        raise UploadError(
            f"S3 upload failed for submission {submission.submission_id!r}: {exc}"
        ) from exc

    logger.info(f"Submission {submission.submission_id!r} stored successfully.")
    return submission


def _fetch_last_submission_date(track: str, user_id: str) -> datetime | None:
    """Fetch the submission date of the most recent submission for a track and user.

    Parameters
    ----------
    track : str
        The track's S3 prefix ("regression", "classification" or "structure").
    user_id : str
        The user's safe username.

    Returns
    -------
    datetime | None
        The submission date of the most recent submission, or None if there are no
        previous submissions.

    Raises
    ------
    RuntimeError
        If ``S3_BUCKET`` isn't set.

    """
    bucket = S3_BUCKET
    if not bucket:
        raise RuntimeError("S3_BUCKET not set; can't check the submission cooldown.")

    s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    prefix = f"submissions/{track}/{user_id}/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if response["IsTruncated"]:  # Unlikely to be > 1000 submissions per user
        logger.warning(
            f"ListObjectsV2 response truncated for prefix {prefix!r}. "
            "Only the first 1000 objects will be considered."
        )
    submission_dates = [
        obj["LastModified"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith("metadata.json")
    ]
    if not submission_dates:
        return None
    return max(submission_dates).astimezone(timezone.utc)


def _cooldown_error(track: str, user_id: str) -> str | None:
    """Check the submission cooldown, failing closed.

    Parameters
    ----------
    track : str
        The track's S3 prefix.
    user_id : str
        The user's safe username.

    Returns
    -------
    str | None
        A message if the user must wait, or if their last submission can't be
        checked (so the cooldown can't be bypassed when S3 has a problem); None if
        they can submit.

    """
    try:
        last_submission = _fetch_last_submission_date(track, user_id)
    except Exception as exc:
        logger.error(f"Cooldown check failed for {user_id!r} on {track}: {exc}")
        return (
            "Error: We couldn't check when you last submitted, so we can't accept "
            "this submission yet. Please try again in a few minutes."
        )
    logger.info(f"Last {track} submission for {user_id!r}: {last_submission}")
    if (
        last_submission
        and (datetime.now(timezone.utc) - last_submission).total_seconds()
        < HOURS_BETWEEN_SUBMISSIONS * 3600
    ):
        return _format_submission_time_message(last_submission, track=track)
    return None


def _on_space() -> bool:
    """Whether the app is running in a Hugging Face Space (not locally)."""
    return get_space() is not None


def account_error(
    profile: gr.OAuthProfile, now: datetime | None = None
) -> tuple[str | None, HFAccount | None]:
    """Check the logged-in Hugging Face account can submit, failing closed.

    Parameters
    ----------
    profile : gr.OAuthProfile
        The logged-in user's profile.
    now : datetime | None
        The current time (for tests); defaults to now.

    Returns
    -------
    tuple[str | None, HFAccount | None]
        A message and None if the account can't submit (too new, not confirmed,
        or the check failed), or None and the account's details.

    """
    now = now or datetime.now(timezone.utc)
    try:
        account = fetch_hf_account(profile.username)
    except Exception as exc:
        logger.error(f"Account lookup failed for {profile.username!r}: {exc}")
        return (
            "Error: We couldn't check your Hugging Face account just now. Please "
            "try again in a few minutes.",
            None,
        )
    # Locally, Gradio fakes the login with a placeholder ID, so only a Space
    # can compare the account's ID with the login's.
    if _on_space() and account.id != profile.get("sub"):
        logger.warning(
            f"Account ID mismatch for {profile.username!r}: "
            f"{account.id!r} vs {profile.get('sub')!r}"
        )
        return (
            "Error: We couldn't confirm your Hugging Face account. Please log out, "
            "log in again and retry.",
            None,
        )
    if MIN_ACCOUNT_AGE_DAYS:
        allowed_from = account.created_at + timedelta(days=MIN_ACCOUNT_AGE_DAYS)
        if now < allowed_from:
            return (
                "Error: New Hugging Face accounts can submit "
                f"{MIN_ACCOUNT_AGE_DAYS} day{'s' if MIN_ACCOUNT_AGE_DAYS != 1 else ''} "
                "after they're created. Yours can submit from "
                f"{allowed_from:%Y-%m-%d %H:%M} UTC.",
                None,
            )
    return None, account


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

    Parameters
    ----------
    file_path : Path
        Path to the uploaded file.
    required_columns : list[str]
        Columns the file must contain — this track's identifiers plus its own
        endpoint columns.
    track_label : str
        Human-readable track name for error messages (e.g. "Regression",
        "Classification").

    Returns
    -------
    tuple[pd.DataFrame | None, str | None]
        The parsed DataFrame and ``None`` on success, or ``None`` and a
        user-facing error message on failure.

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


def _status(message: str) -> dict:
    """Show a message in the submission status box."""
    return gr.update(value=message, visible=True)


def _validate_regression(file_path: Path) -> str | None:
    """Check a regression file; return a message for the participant, or None."""
    df, error = _read_tabular_submission(
        file_path, REQUIRED_REGRESSION_COLUMNS, "Regression"
    )
    if error:
        return error
    assert df is not None  # guaranteed by _read_tabular_submission when error is None
    for col in REGRESSION_SPEC.endpoints:
        if df[col].isnull().any():
            return f"Error: {col} column contains NaN values."
        if not np.isfinite(df[col]).all():
            return f"Error: {col} column contains infinite values."
        if df[col].nunique() <= 1 or df[col].std() < MIN_PREDICTION_STD:
            return (
                f"Error: {col} predictions are constant or near-constant — "
                "please submit real model outputs."
            )
    return None


def _validate_classification(file_path: Path) -> str | None:
    """Check a classification file; return a message for the participant, or None."""
    df, error = _read_tabular_submission(
        file_path, REQUIRED_CLASSIFICATION_COLUMNS, "Classification"
    )
    if error:
        return error
    assert df is not None  # guaranteed by _read_tabular_submission when error is None
    for col in CLASSIFICATION_SPEC.endpoints:
        if df[col].isnull().any():
            return f"Error: {col} column contains NaN values."
        if not df[col].isin([0, 1, True, False]).all():
            return f"Error: {col} column contains non-binary values."
        if df[col].nunique() <= 1:
            return (
                f"Error: {col} predictions are all one class — "
                "a constant prediction cannot be scored."
            )
    return None


def _validate_structure(file_path: Path) -> str | None:
    """Check a structure zip; return a message for the participant, or None."""
    if file_path.suffix.lower() != ".zip":
        return "Error: Structure submissions must be a .zip file."
    try:
        with zipfile.ZipFile(file_path) as zf:
            n_files = len(zf.namelist())
    except Exception as exc:
        return f"Error: Could not read zip file — {exc}"
    if n_files != STRUCTURE_DATASET_SIZE:
        return f"Error: Expected {STRUCTURE_DATASET_SIZE} files in zip, got {n_files}."
    return None


_FILE_CHECKS = {
    REGRESSION_NAME: _validate_regression,
    CLASSIFICATION_NAME: _validate_classification,
    STRUCTURE_NAME: _validate_structure,
}


def submit_predictions(
    user_alias: str,
    anon_checkbox: bool,
    participant_name: str,
    discord_username: str,
    email: str,
    affiliation: str,
    model_tag: str,
    proprietary_data_checkbox: bool,
    open_code_checkbox: bool,
    track_select: Literal[
        "Regression Prediction", "Classification Prediction", "Structure Prediction"
    ],
    file_input: str | None,
    profile: gr.OAuthProfile | None = None,
) -> dict:
    """Handle a competition submission from the web UI.

    The submitter is whoever is logged in with Hugging Face: Gradio passes their
    profile from the session, and it can't come from the form or the API. Checks
    run in order and the first failure is reported; nothing is uploaded, so a
    refused submission doesn't use up the cooldown.

    Parameters
    ----------
    user_alias : str
        Optional alias for anonymous display on the leaderboard.
    anon_checkbox : bool
        If True, display alias instead of username.
    participant_name : str
        Real name (private, not displayed publicly).
    discord_username : str
        Discord handle (optional).
    email : str
        Contact email (optional).
    affiliation : str
        Institutional affiliation (optional).
    model_tag : str
        Link to model report (optional). Only checked for reachability when
        ``open_code_checkbox`` is True.
    proprietary_data_checkbox : bool
        Whether proprietary data was used in training.
    open_code_checkbox : bool
        Whether the participant's code is open-source and publicly available.
        When True, ``model_tag`` is validated as a reachable link.
    track_select : Literal["Regression Prediction", "Classification Prediction", "Structure Prediction"]
        The selected competition track.
    file_input : str | None
        Path to the uploaded submission file.
    profile : gr.OAuthProfile | None
        The logged-in user, from the session; None if nobody is logged in.

    Returns
    -------
    dict
        gr.update with a status message and visible=True.

    """
    # --- who is submitting ---
    if profile is None:
        return _status("Error: Please log in with Hugging Face (step 1) to submit.")
    username = profile.username
    error, account = account_error(profile)
    if error:
        return _status(error)
    assert account is not None  # guaranteed by account_error when error is None
    banned_reason = BANNED_USERNAMES.get(username.lower())
    if banned_reason:
        logger.warning(f"Blocked submission attempt from banned username: {username!r}")
        return _status(
            f"Error: Your account ({username!r}) has been blocked. Reason: "
            f"{banned_reason}. If you believe this is a mistake, please contact the "
            "organisers."
        )

    # --- the form ---
    if anon_checkbox and (not user_alias or not user_alias.strip()):
        return _status(
            "Error: An alias is required when submitting anonymously. "
            "Uncheck the 'Submit anonymously' box or provide an alias."
        )
    if user_alias and user_alias.strip():
        error = alias_error(user_alias.strip())
        if error:
            logger.warning(f"Rejected alias: {user_alias.strip()!r}")
            return _status(error)
    if model_tag and model_tag.strip():
        error = model_report_link_error(model_tag.strip())
        if error:
            logger.warning(f"Rejected model report link: {model_tag.strip()!r}")
            return _status(error)
    if email and email.strip() and not EMAIL_RE.match(email.strip()):
        logger.warning(f"Invalid email format: {email.strip()!r}")
        return _status("Error: Please enter a valid email address.")
    if open_code_checkbox:
        model_status = validate_model_details(model_tag)
        if model_status in ["Invalid link", "Not submitted"]:
            logger.warning(f"Invalid model report link: {model_tag.strip()!r}")
            return _status(
                "Error: Could not open the Model Report Link. Please check "
                "the URL and try again, or uncheck the 'open code' box."
            )
    else:
        model_status = model_tag.strip() if model_tag else "Not submitted"
    if not track_select:
        return _status("Error: Please select a track.")
    if file_input is None:
        return _status("Error: Please upload a submission file.")

    # --- the file, then the cooldown ---
    file_path = Path(file_input)
    error = _FILE_CHECKS[track_select](file_path)
    if error:
        return _status(error)
    error = _cooldown_error(TRACK_SLUGS[track_select], _safeify_username(username))
    if error:
        return _status(error)

    # --- build submission model and persist to S3 ---
    submission = Submission(
        username=username,
        hf_user_id=str(profile.get("sub", "")),
        hf_account_created_at=account.created_at,
        user_alias=user_alias.strip(),
        anonymous=anon_checkbox,
        participant_name=participant_name.strip(),
        discord_username=discord_username.strip(),
        email=email.strip(),
        affiliation=affiliation.strip(),
        model_report_link=model_status,
        used_proprietary_data=proprietary_data_checkbox,
        open_source_code=open_code_checkbox,
        track=track_select,
        filename=file_path.name,
    )
    logger.info(
        f"Submission received: id={submission.submission_id!r} "
        f"user={submission.username!r} track={submission.track!r}"
    )
    try:
        upload_submission(submission, file_path)
    except UploadError as exc:
        logger.error(str(exc))
        return _status(
            "Error: Your submission couldn't be saved, so it hasn't been entered and "
            "doesn't count towards your cooldown. Please try again in a few minutes."
        )

    return _status(
        f"Submission received from {submission.display_name!r} for the "
        f"{track_select} track. Your predictions are being processed and should "
        "appear on the leaderboard within 2 hours."
    )
