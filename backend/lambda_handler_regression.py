"""Lambda handler for regression prediction evaluation.

Triggered by EventBridge on a schedule. Scans S3 for unscored regression
submissions, scores them against the ground truth, and writes results back
to S3.
"""

from loguru import logger

from .aws_submission_processing import process_new_regression_submission
from .config import REGRESSION_PATHS, SubmissionKey


def handler(event, context):
    """Lambda handler for processing new regression submissions."""
    detail = event.get("detail", {})
    file_path = detail.get("file_path") or detail.get("object", {}).get("key")
    if file_path:
        sk = SubmissionKey.parse(file_path)
        # Expected: submissions/regression/{user_id}/{submission_id}/{predictions file}
        if sk and sk.is_valid_for(REGRESSION_PATHS):
            logger.info(f"Received new regression submission: {file_path}")
            process_new_regression_submission(sk)
        else:
            logger.warning(f"Invalid file path structure: {file_path}")
    else:
        logger.info("No new regression submission detected in event.")

    return {"statusCode": 200}
