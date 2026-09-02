"""Tests for the utils module."""

from datetime import UTC, datetime, timedelta

import numpy as np

from backend.config import FINAL_LEADERBOARD_DEADLINE, INTERIM_LEADERBOARD_DEADLINE
from backend.utils import bootstrap_sampling, clip_and_log_transform, current_phase


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
