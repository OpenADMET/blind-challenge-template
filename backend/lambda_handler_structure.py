"""Lambda handler for structure prediction evaluation.

Triggered by EventBridge on a schedule. Scans S3 for unscored structure
submissions, scores them against the ground truth, and writes results back
to S3.
"""

from loguru import logger

from .aws_submission_processing import process_new_structure_submission
from .config import STRUCTURE_PATHS, SubmissionKey


def handler(event, context):
    detail = event.get("detail", {})
    file_path = detail.get("file_path") or detail.get("object", {}).get("key")
    if file_path:
        sk = SubmissionKey.parse(file_path)
        # Expected: submissions/structure/{user_id}/{submission_id}/{predictions file}
        if sk and sk.is_valid_for(STRUCTURE_PATHS):
            logger.info(f"Received new structure submission: {file_path}")
            process_new_structure_submission(sk)
        else:
            logger.warning(f"Invalid file path structure: {file_path}")
    else:
        logger.info("No new structure submission detected in event.")
