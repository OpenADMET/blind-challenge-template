import io
import json
import os
import random
from functools import lru_cache

import boto3
import pandas as pd
import requests
from loguru import logger

from .config import DISCORD_NOTIFICATIONS, STRUCTURE_DATASET_SIZE
from .submission_validation import ValidationResult

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
SECRETS_MANAGER_CLIENT = boto3.client("secretsmanager", region_name=AWS_REGION)
_TRACK_LABELS = {
    "Regression Prediction": "regression",
    "Classification Prediction": "classification",
    "Structure Prediction": "structure",
}
VALID_SUBMISSION_RESPONSES = [
    "✅ Thanks for your {track} submission at {time}, **{name}**! It looks great!",
    "✅ We've received your {track} submission at {time}, **{name}**. Nice work!",
    "✅ Awesome {track} submission from **{name}** at {time}! 🚀",
    "✅ New valid {track} submission from **{name}** at {time}! Looks good!",
    "✅ **{name}**, your {track} submission at {time} is all set! Keep it up!",
    "✅ Excellent {track} submission from **{name}** at {time}! 👏",
    "✅ **{name}** just submitted a {track} prediction at {time}. Looking sharp!",
    "✅ Your {track} submission at {time} is valid, **{name}**! Great job!",
    "✅ **{name}** just dropped a {track} submission at {time}. Fantastic work! 🔥",
    "✅ Nice one, **{name}**! Your {track} submission at {time} is locked in. 🎯",
    "✅ Boom! **{name}**'s {track} submission at {time} landed safely. 💪",
    "✅ Your {track} submission at {time} checks out, **{name}**! On a roll! 🎉",
    "✅ **{name}** brings another {track} submission at {time}. Consistency wins! ⭐",
    "✅ Solid {track} submission from **{name}** at {time}! Keep the momentum going!",
    "✅ **{name}**'s {track} submission at {time} is in the books. Well done! 📈",
    "✅ Validated! **{name}**'s {track} submission at {time}. Crushing it! 🙌",
]


@lru_cache(maxsize=1)
def get_discord_webhook() -> str | None:
    """Get the Discord webhook URL from AWS Secrets Manager.

    Returns
    -------
    str | None
        Discord webhook URL if available, otherwise None.

    """
    secret_name = os.environ.get("DISCORD_WEBHOOK_SECRET_NAME")
    if not secret_name:
        logger.warning(
            "DISCORD_WEBHOOK_SECRET_NAME is not set; Discord notifications disabled. "
            "The evaluator Lambdas set it from Tofu (see opentofu/lambda.tf)."
        )
        return None
    try:
        return SECRETS_MANAGER_CLIENT.get_secret_value(SecretId=secret_name)[
            "SecretString"
        ]
    except Exception as e:
        logger.warning(
            "Could not load Discord webhook secret '{}' from Secrets Manager: {}",
            secret_name,
            e,
        )
        return None


def _unpack_metadata(validation_metadata_df: pd.DataFrame) -> tuple[str, str, str, str]:
    """Unpack relevant metadata from the validation metadata DataFrame."""
    submission_metadata = validation_metadata_df.iloc[0].to_dict()
    valid = submission_metadata["valid_submission"]
    name = (
        submission_metadata["user_alias"]
        if submission_metadata["anonymous"]
        else submission_metadata["username"]
    )
    time = submission_metadata["submitted_at"].strftime("%Y-%m-%d %H:%M:%S UTC")
    track = _TRACK_LABELS[submission_metadata["track"]]
    return valid, name, time, track


def prepare_discord_message(
    validation_metadata_df: pd.DataFrame, validation_result: ValidationResult
) -> tuple[int, str, str | None]:
    """Prepare a Discord post with feedback on a submission.

    Parameters
    ----------
    validation_metadata_df : pd.DataFrame
        Validation metadata for the specified submission.
    validation_result : ValidationResult
        The result of the submission validation, including validity and any errors.

    Returns
    -------
    tuple[int, str, str | None]
        A tuple containing the color code for the Discord embed, the content of the
        message, and an optional string with long error details.

    """
    valid, name, time, track = _unpack_metadata(validation_metadata_df)
    long_errors: str | None = None
    if (len(validation_result.system_errors) > 0) or (
        validation_result.scoring == "failed"
    ):
        content = (
            f"❌ Sorry **{name}**, something went wrong with your {track} submission at"
            f" {time}. we are looking into it."
        )
        color = 15158332  # Red

    elif not valid:
        long_errors = validation_result.get_full_log()
        content = (
            f"❌ Invalid {track} submission from user **{name}** at {time} with errors:"
            f" \n```{validation_result.get_error_summary()}```"
        )
        if long_errors is not None:
            content += "\nSee file above for more details."
        color = 15158332  # Red
    else:
        if validation_result.scoring == "partial":
            content = (
                f"⚠️ Partially valid {track} submission from user **{name}** at {time}!"
                f"\n{len(validation_result.scoring_errors)}/{STRUCTURE_DATASET_SIZE} "
                "compounds failed scoring, see file above for details."
            )
            color = 15105570  # Orange
            long_errors = (
                f"Scoring failed for the following compounds: "
                f"{', '.join(validation_result.scoring_errors)}"
            )
            if validation_result.pb_failures:
                long_errors += "\n\n" + _format_pb_failures(validation_result.pb_failures)
        elif validation_result.pb_failures:
            n_pb = len(validation_result.pb_failures)
            content = (
                f"⚠️ Valid {track} submission from **{name}** at {time}, but"
                f" {n_pb} compound(s) failed PoseBusters checks and had their scores"
                " zeroed. See file above for details."
            )
            color = 15105570  # Orange
            long_errors = _format_pb_failures(validation_result.pb_failures)
        else:
            content = random.choice(VALID_SUBMISSION_RESPONSES).format(
                track=track, name=name, time=time
            )
            color = 3066993  # Green
    return color, content, long_errors


def _format_pb_failures(pb_failures: dict[str, list[str]]) -> str:
    """Format a molecule-to-failed-checks mapping as a human-readable string."""
    lines = ["PoseBusters checks failed (scores zeroed for these compounds):"]
    for mol_id, checks in sorted(pb_failures.items()):
        lines.append(f"  {mol_id}: {', '.join(checks)}")
    return "\n".join(lines)


def post_result_to_discord(
    validation_metadata_df: pd.DataFrame, validation_result: ValidationResult
) -> None:
    """Post the result of a submission validation to a Discord channel using a webhook.

    Parameters
    ----------
    validation_metadata_df : pd.DataFrame
        Validation metadata for the specified submission.
    validation_result : ValidationResult
        The result of the submission validation, including validity and any errors.

    """
    if not DISCORD_NOTIFICATIONS:
        logger.info("Discord notifications disabled — skipping.")
        return

    name = validation_metadata_df.iloc[0]["username"]
    color, content, long_errors = prepare_discord_message(
        validation_metadata_df, validation_result
    )

    payload = {
        "embeds": [
            {"title": "Submission Status", "description": content, "color": color}
        ],
        # user_alias is free-text and user-controlled — disable mention parsing so a
        # crafted alias (e.g. "@everyone") can't ping the channel.
        "allowed_mentions": {"parse": []},
    }
    discord_webhook = get_discord_webhook()
    if discord_webhook is None:
        logger.info("Skipping Discord notification: webhook secret is unavailable")
        return

    try:
        if long_errors is not None:
            file = {
                "file": (
                    "errors.txt",
                    io.BytesIO(long_errors.encode("utf-8")),
                    "text/plain",
                )
            }
            response = requests.post(
                discord_webhook, data={"payload_json": json.dumps(payload)}, files=file
            )
        else:
            response = requests.post(discord_webhook, json=payload)
        response.raise_for_status()
        logger.info(
            "Successfully posted submission result to Discord for user {}", name
        )
    except requests.exceptions.RequestException as e:
        logger.error(
            "Failed to post submission result to Discord for user {}: {}", name, e
        )
