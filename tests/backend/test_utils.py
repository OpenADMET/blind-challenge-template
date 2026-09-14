"""Tests for the utils module."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from backend.config import FINAL_LEADERBOARD_DEADLINE, INTERIM_LEADERBOARD_DEADLINE
from backend.utils import (
    _safeify_username,
    bootstrap_sampling,
    clip_and_log_transform,
    current_phase,
)


def test_current_phase_before_phase_1():
    """Test that current_phase returns 1 before the interim leaderboard deadline."""
    interim_deadline = datetime.strptime(
        INTERIM_LEADERBOARD_DEADLINE, "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=UTC)
    test_time = interim_deadline - timedelta(seconds=1)
    assert current_phase(test_time) == 1


def test_current_phase_during_phase_2():
    """Test that current_phase returns 2 after the interim leaderboard deadline."""
    interim_deadline = datetime.strptime(
        INTERIM_LEADERBOARD_DEADLINE, "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=UTC)
    test_time = interim_deadline + timedelta(seconds=1)
    assert current_phase(test_time) == 2


def test_current_phase_after_phase_2():
    """Test that current_phase returns 0 after the final leaderboard deadline."""
    final_deadline = datetime.strptime(
        FINAL_LEADERBOARD_DEADLINE, "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=UTC)
    test_time = final_deadline + timedelta(seconds=1)
    assert current_phase(test_time) == 0


def test_clip_and_log_transform():
    """Test that clip_and_log_transform correctly processes the input array."""
    y = np.array([-1, 0, 1, 10])
    transformed = clip_and_log_transform(y)
    expected = np.log10(np.array([1, 1, 2, 11]))
    assert np.allclose(transformed, expected)


def test_bootstrap_sampling():
    """Test that bootstrap_sampling generates the correct shape of indices."""
    original_dataset_size = 100
    n_bootstrap_repeats = 1000
    indices = bootstrap_sampling(original_dataset_size, n_bootstrap_repeats)
    assert indices.shape == (n_bootstrap_repeats, original_dataset_size)
    assert indices.max() < original_dataset_size
    assert indices.min() >= 0


def test_bootstrap_sampling_reproducibility():
    """Test that bootstrap_sampling generates the same indices on multiple calls."""
    original_dataset_size = 100
    n_bootstrap_repeats = 1000
    indices1 = bootstrap_sampling(original_dataset_size, n_bootstrap_repeats)
    indices2 = bootstrap_sampling(original_dataset_size, n_bootstrap_repeats)
    assert np.array_equal(indices1, indices2)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("alice", "alice"),  # already safe — unchanged
        ("Alice", "alice"),  # lower-cased
        ("ALICE", "alice"),
        ("AlIcE", "alice"),
        ("org/User", "org_user"),  # '/' (HF org accounts) -> '_'
        ("a/b/c", "a_b_c"),  # every '/' replaced
        ("a b", "a_b"),  # internal space -> '_'
        ("a  b", "a__b"),  # each space replaced individually
        ("  Alice  ", "alice"),  # surrounding whitespace stripped
        ("\tBob\n", "bob"),  # strip() handles tabs/newlines too
        ("Org/Sub User", "org_sub_user"),  # combined: case + '/' + space
    ],
)
def test_safeify_username(raw: str, expected: str) -> None:
    """Usernames are lower-cased with '/' and spaces mapped to '_'."""
    assert _safeify_username(raw) == expected


@pytest.mark.parametrize(
    "variants",
    [
        ("Alice", "alice", "ALICE", "  aLiCe "),
        ("Org/User", "org/user", "ORG/USER"),
    ],
)
def test_safeify_username_collapses_casing_variants(variants: tuple[str, ...]) -> None:
    """Casing variants of one HF account map to a single key.

    This is the property the leaderboard de-duplication and the per-user
    submission cooldown rely on — alternating case must not create a distinct
    identity (see ``create_manifest`` and ``_fetch_last_submission_date``).
    """
    safe = {_safeify_username(v) for v in variants}
    assert len(safe) == 1


@pytest.mark.parametrize("raw", ["Alice", "Org/User", "  A B  ", "already_safe"])
def test_safeify_username_is_idempotent(raw: str) -> None:
    """Re-safeifying an already-safe username is a no-op."""
    once = _safeify_username(raw)
    assert _safeify_username(once) == once
