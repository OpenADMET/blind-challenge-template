"""Utility functions for the blind challenge Gradio app."""

import io
import ipaddress
import re
import socket
import time
from urllib.parse import urljoin, urlparse

import boto3
import pandas as pd
import requests
from config import AWS_DEFAULT_REGION, S3_BUCKET
from loguru import logger

_MAX_REDIRECTS = 5

# S3FS was causing 403 errors in testing on some machines, unclear why.
# Switching to boto3 client which works reliably
s3_client = boto3.client("s3", region_name=AWS_DEFAULT_REGION)


def _load_csv_from_s3(key: str, parquet: bool = False) -> pd.DataFrame:
    """Load a CSV file from S3 into a DataFrame."""
    logger.info(f"Downloading from S3: {key}")
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    if parquet:
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _safeify_username(username: str) -> str:
    """Sanitise a HuggingFace username for use in S3 keys and file paths.

    HF usernames for organisations use the format ``org/user``, which would create
    unintended S3 path nesting. Spaces are also replaced for safety. Lowercased so
    that HF-username casing variants (which all resolve to the same account) map to
    the same S3 prefix — otherwise the per-user submission cooldown
    (``_fetch_last_submission_date``) could be bypassed by alternating case.
    """
    return str(username.strip()).lower().replace("/", "_").replace(" ", "_")


def _is_safe_public_url(url: str) -> bool:
    """Reject URLs whose host resolves to a private/loopback/link-local address.

    Guards ``check_page_exists`` against SSRF: without this, a user-supplied URL
    (e.g. the "Method Report Link" field) could point the server at internal
    infrastructure and have it echo back reachability.
    """
    hostname = urlparse(url).hostname
    if not hostname:
        return False
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


# TODO: Maybe use Tenacity for retrying?
def check_page_exists(
    url: str,
    delay: float = 0.2,
    max_retries: int = 3,
    current_retries: int = 0,
    restrict_to_public: bool = True,
):
    """Check if a web page exists at the given URL with a retry limit for 429 errors.

    Redirects are followed manually (rather than via ``requests``' built-in
    ``allow_redirects``) so every hop can be checked against
    ``_is_safe_public_url`` before it is requested, closing off SSRF via a
    redirect to an internal address.

    Params:
        url (str): The URL of the page to check.
        delay (float, optional): Seconds to wait until submitting another request.
            Defaults to 0.
        max_retries (int, optional): Maximum number of times to retry on a 429 error.
            Defaults to 3.
        current_retries (int, optional): Current number of retries performed (internal
            counter). Defaults to 0.
        restrict_to_public (bool, optional): Reject hosts that resolve to a
            private/loopback/internal address. Only meaningful protection when
            ``url`` (or its host) is attacker-controlled — e.g. a user-supplied
            link. Should be disabled for calls against a hardcoded, trusted
            domain (e.g. huggingface.co), since some platforms resolve their own
            domain to an internal address for intra-network callers (split-horizon
            DNS), which this check would otherwise incorrectly reject. Defaults to
            True.

    Returns:
        bool: True if the page exists (status code 200), False otherwise.

    """
    safe_url = str(url).strip()

    # Attempt to fix url
    if not safe_url.startswith(("http://", "https://")):
        safe_url = f"https://{safe_url}"

    try:
        response = None
        for _ in range(_MAX_REDIRECTS + 1):
            if restrict_to_public and not _is_safe_public_url(safe_url):
                logger.warning(f"Refusing to fetch non-public URL: {safe_url}")
                return False
            response = requests.get(safe_url, timeout=5, allow_redirects=False)
            if (
                response.status_code in (301, 302, 303, 307, 308)
                and "Location" in response.headers
            ):
                safe_url = urljoin(safe_url, response.headers["Location"])
                continue
            break
        else:
            logger.warning(f"Too many redirects for {url}")
            return False

        # Check for Rate Limit Error and retry if under the limit
        if response.status_code == 429:
            if current_retries < max_retries:
                # Make wait time exponential
                wait_time = 5 * (2**current_retries)
                logger.warning(
                    f"Warning: Rate limit hit on {safe_url}. Attempt "
                    f"{current_retries + 1}/{max_retries}. Waiting for {wait_time} "
                    "seconds..."
                )
                time.sleep(wait_time)
                # Recurse with an incremented retry counter
                return check_page_exists(
                    safe_url,
                    delay=delay,
                    max_retries=max_retries,
                    current_retries=current_retries + 1,
                    restrict_to_public=restrict_to_public,
                )
            else:
                logger.error(
                    f"Error: Max retries ({max_retries}) reached for rate limit on "
                    f"{safe_url}."
                )
                return False  # Give up after max retries

        # Return True only for a successful status code (200)
        return response.status_code == 200

    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking URL {safe_url}: {e}")
        return False

    finally:
        # Sleep after every request to avoid HTTPS error
        time.sleep(delay)


def validate_hf_username(username: str) -> bool:
    """Validate that the Hugging Face username exists by checking the profile page."""
    # restrict_to_public=False: the target here is always the literal huggingface.co
    # domain (hardcoded below, or enforced by the regex above), never an
    # attacker-supplied host, so the SSRF guard in check_page_exists is not needed —
    # and huggingface.co can legitimately resolve to an internal address when called
    # from within HF's own infrastructure (split-horizon DNS), which that guard
    # would otherwise reject.
    if re.match(r"^https?://huggingface\.co/([^/]+)/?$", str(username).strip()):
        return check_page_exists(
            str(username).strip(), delay=1, max_retries=10, restrict_to_public=False
        )
    username = str(username).strip()
    hf_url = f"https://huggingface.co/{username}"
    return check_page_exists(hf_url, delay=1, max_retries=10, restrict_to_public=False)


def validate_model_details(tag: str) -> str:
    """Validate that the model details link is a valid URL and exists."""
    if tag is None or str(tag).strip() == "":
        return "Not submitted"
    safe_tag = str(tag).strip()
    if not safe_tag.startswith("https://"):
        return "Invalid link"
    is_real_url = check_page_exists(safe_tag, delay=2)
    if not is_real_url:
        return "Invalid link"
    else:
        return safe_tag
