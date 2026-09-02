"""Test custom scoring functions used by the activity tracks."""

import numpy as np
import pytest

from backend.custom_scoring_functions import rae, rae_soft_threshold_absolute_error

Y_TRUE = np.array([1.0, 5.0, 10.0])
Y_PRED = np.array([1.5, 5.0, 12.0])


def test_soft_threshold_rae_matches_hand_computed_value():
    """Manually-derived value for a 3-point example with a symmetric band of width 1.

    y_true_upper/lower = y_true +/- 0.5. Only the third point falls outside its band
    (pred 12 vs upper bound 10.5), contributing 1.5 to the numerator. The mean-true
    baseline (16/3) falls outside the first and third points' bands, contributing
    3.8333... + 4.1667 = 8.0 to the denominator.
    """
    result = rae_soft_threshold_absolute_error(Y_TRUE, Y_PRED, confidence_interval=1.0)
    assert result == pytest.approx(1.5 / 8.0)


def test_soft_threshold_rae_explicit_bounds_match_symmetric_confidence_interval():
    """A symmetric confidence_interval is equivalent to explicit +/- half-width bounds."""
    via_interval = rae_soft_threshold_absolute_error(
        Y_TRUE, Y_PRED, confidence_interval=1.0
    )
    via_bounds = rae_soft_threshold_absolute_error(
        Y_TRUE, Y_PRED, y_true_upper=Y_TRUE + 0.5, y_true_lower=Y_TRUE - 0.5
    )
    assert via_interval == pytest.approx(via_bounds)


def test_soft_threshold_rae_array_confidence_interval_matches_scalar():
    """A per-point confidence_interval array of a constant value matches the scalar."""
    scalar_result = rae_soft_threshold_absolute_error(
        Y_TRUE, Y_PRED, confidence_interval=1.0
    )
    array_result = rae_soft_threshold_absolute_error(
        Y_TRUE, Y_PRED, confidence_interval=np.full_like(Y_TRUE, 1.0)
    )
    assert scalar_result == pytest.approx(array_result)


def test_soft_threshold_rae_converges_to_plain_rae_as_band_shrinks():
    """A near-zero-width band recovers plain RAE (soft-thresholding disappears)."""
    result = rae_soft_threshold_absolute_error(
        Y_TRUE, Y_PRED, confidence_interval=1e-9
    )
    assert result == pytest.approx(rae(Y_TRUE, Y_PRED), abs=1e-6)


def test_soft_threshold_rae_zero_when_all_predictions_within_band():
    """Predictions falling entirely inside their credible interval score zero error."""
    result = rae_soft_threshold_absolute_error(
        Y_TRUE, Y_TRUE, confidence_interval=1.0
    )
    assert result == 0.0


def test_soft_threshold_rae_no_bounds_reduces_to_plain_rae():
    """Omitting bounds and confidence_interval entirely is a valid, not an error case.

    Both sides of the band then default to y_true itself (see _resolve_bounds),
    collapsing the band to a point — soft-thresholding becomes a no-op and the result
    is identical to plain rae().
    """
    result = rae_soft_threshold_absolute_error(Y_TRUE, Y_PRED)
    assert result == pytest.approx(rae(Y_TRUE, Y_PRED))


def test_soft_threshold_rae_rejects_both_bounds_and_confidence_interval():
    with pytest.raises(ValueError, match="Cannot provide both"):
        rae_soft_threshold_absolute_error(
            Y_TRUE, Y_PRED, confidence_interval=1.0, y_true_upper=Y_TRUE + 1
        )


def test_soft_threshold_rae_missing_upper_bound_defaults_to_y_true():
    """Omitting just one side defaults it to y_true (no tolerance on that side).

    y_true_upper defaults to Y_TRUE = [1, 5, 10], so above_upper is the positive part
    of (y_pred - y_true) at every point: [0.5, 0, 2]. y_true_lower = Y_TRUE - 0.5 gives
    below_lower = [0, 0, 0] (every prediction is >= its lower bound). Numerator = 2.5.
    The mean-true baseline (16/3) contributes analogously to the denominator.
    """
    result = rae_soft_threshold_absolute_error(
        Y_TRUE, Y_PRED, y_true_lower=Y_TRUE - 0.5
    )
    assert result == pytest.approx(0.28301886792452835)
