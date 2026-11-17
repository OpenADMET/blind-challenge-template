"""Utility functions for the blind challenge."""

import ipaddress
import socket
import time
from functools import lru_cache
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from loguru import logger

from .config import (
    BOOTSTRAP_SAMPLES,
)

BOOTSTRAP_SEED = 0
_MAX_REDIRECTS = 5


def clip_and_log_transform(y: np.ndarray) -> np.ndarray:
    """Clip the input array to zero then apply a log10(y + 1) transformation.

    Parameters
    ----------
    y : np.ndarray
        The input array to be transformed.

    Returns
    -------
    np.ndarray
        The transformed array.

    """
    y = np.clip(y, a_min=0, a_max=None)
    return np.log10(y + 1)


@lru_cache(maxsize=3)
def bootstrap_sampling(
    original_dataset_size: int, n_bootstrap_repeats: int = BOOTSTRAP_SAMPLES
) -> np.ndarray:
    """Generate bootstrap sample indices for a dataset of a given size.

    The random seed is fixed so all submissions are evaluated on the same bootstrap
    samples. Best practices for bootstrap sampling involve sampling the same number of
    samples as the original dataset, with replacement, at least 1000 times.

    Parameters
    ----------
    original_dataset_size : int
        The size of the original dataset.
    n_bootstrap_repeats : int
        The number of bootstrap samples to generate. Defaults to
        ``config.BOOTSTRAP_SAMPLES``.

    Returns
    -------
    np.ndarray
        An array of bootstrap sample indices.

    """
    rng = np.random.default_rng(seed=BOOTSTRAP_SEED)
    return rng.choice(
        original_dataset_size,
        size=(n_bootstrap_repeats, original_dataset_size),
        replace=True,
    )


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


def check_page_exists(
    url: str,
    delay: float = 0.2,
    max_retries: int = 3,
    current_retries: int = 0,
    restrict_to_public: bool = True,
) -> bool:
    """Check if a web page exists at the given URL with a retry limit for 429 errors.

    Redirects are followed manually (rather than via ``requests``' built-in
    ``allow_redirects``) so every hop can be checked against
    ``_is_safe_public_url`` before it is requested, closing off SSRF via a
    redirect to an internal address.

    Parameters
    ----------
    url : str
        The URL of the page to check.
    delay : float
        Seconds to wait until submitting another request. Defaults to 0.
    max_retries : int
        Maximum number of times to retry on a 429 error. Defaults to 3.
    current_retries : int
        Current number of retries performed (internal counter). Defaults to 0.
    restrict_to_public : bool
        Reject hosts that resolve to a private/loopback/internal address. Only
        meaningful protection when ``url`` (or its host) is attacker-controlled —
        e.g. a user-supplied link. Should be disabled for calls against a
        hardcoded, trusted domain (e.g. huggingface.co), since some platforms
        resolve their own domain to an internal address for intra-network callers
        (split-horizon DNS), which this check would otherwise incorrectly reject.
        Defaults to True.

    Returns
    -------
    bool
        True if the page exists (status code 200), False otherwise.

    """
    logger.debug("Checking if page exists at URL: {}", url)
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
                    f"Warning: Rate limit hit on {safe_url}. Attempt {current_retries + 1}/{max_retries}. Waiting for {wait_time} seconds..."
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
                    f"Error: Max retries ({max_retries}) reached for rate limit on {safe_url}."
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


def validate_model_details(tag: str | float | None) -> str:
    """Validate that the model details link is a valid URL and exists."""
    logger.debug("Validating model details link: {}", tag)
    if tag is None or pd.isna(tag) or str(tag).strip() == "":
        return "Not submitted"
    safe_tag = str(tag).strip()
    if not safe_tag.startswith("https://"):
        return "Invalid link"
    is_real_url = check_page_exists(safe_tag, delay=2)
    if not is_real_url:
        return "Invalid link"
    else:
        return safe_tag


def _safeify_username(username: str) -> str:
    """Sanitise a HuggingFace username for use in S3 keys and file paths.

    HF usernames for organisations use the format ``org/user``, which would create
    unintended S3 path nesting. Spaces are also replaced for safety. Lowercased so
    that HF-username casing variants (which all resolve to the same account) map to
    the same S3 prefix — otherwise the per-user submission cooldown
    (``_fetch_last_submission_date``) could be bypassed by alternating case.
    """
    return str(username.strip()).lower().replace("/", "_").replace(" ", "_")
