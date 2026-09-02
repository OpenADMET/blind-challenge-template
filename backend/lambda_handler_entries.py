"""Lambda handler for generating the full cross-track entries list.

Runs on a daily schedule (see ``opentofu/eventbridge.tf``) and writes an unfiltered
list of every valid, scored submission from every user, across both the activity and
structure tracks — intended for entrant/publication tracking, not for ranking.
"""

import os

from loguru import logger

from .aws_manifest import create_all_entries_list, save_all_entries_list


def handler(event, context):
    """Generate and save the full cross-track entries list.

    Returns:
        dict: Status message with row count.

    """
    os.environ["HOME"] = "/tmp"
    os.environ["DUCKDB_HOME"] = "/tmp"

    logger.info("Generating full entries list for both tracks")
    entries = create_all_entries_list()
    if entries.empty:
        return {
            "statusCode": 200,
            "message": "No valid, scored submissions found for either track.",
            "rows": 0,
        }

    save_path = save_all_entries_list(entries)
    logger.info("Wrote {} entries to {}", len(entries), save_path)
    return {
        "statusCode": 200,
        "message": f"Wrote entries list to {save_path}",
        "rows": len(entries),
    }
