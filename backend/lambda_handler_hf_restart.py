"""Lambda handler to restart the HuggingFace Space.

Packaging note: include `huggingface_hub` in the function zip (or the shared
layer) — it is not part of the Lambda standard library.
"""

import os

import boto3
from huggingface_hub import HfApi
from loguru import logger

SECRETS_MANAGER_CLIENT = boto3.client("secretsmanager")


def handler(event, context):
    owner = os.environ["HF_OWNER"]
    space = os.environ["HF_SPACE_NAME"]
    secret_name = os.environ["HF_TOKEN_SECRET_NAME"]
    token = SECRETS_MANAGER_CLIENT.get_secret_value(SecretId=secret_name)[
        "SecretString"
    ]

    api = HfApi(token=token)
    api.restart_space(repo_id=f"{owner}/{space}")
    logger.info("Restarted HF Space {}/{}", owner, space)

    return {"status": "restarted", "space": f"{owner}/{space}"}
