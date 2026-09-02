"""Test functions for evaluating predictions."""

import numpy as np
import pandas as pd
import pytest

from backend.config import (
    ACTIVITY_ENDPOINTS,
    ACTIVITY_METRICS,
    BOOTSTRAP_SAMPLES,
    CLASSIFICATION_ENDPOINTS,
    CLASSIFICATION_METRICS,
    MACRO_ENDPOINT_LABEL,
    REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX,
    REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX,
    REGRESSION_ENDPOINTS,
)
from backend.evaluate_predictions import (
    _metric_needs_credible_interval_bounds,
    _metrics_for_endpoint,
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    bootstrap_metrics,
    compute_macro_bootstrap_results,
    pivot_endpoint_results_wide,
    score_activity_predictions,
)


def test_metrics_for_endpoint_dispatch():
    """Classification endpoints get CLASSIFICATION_METRICS, others get ACTIVITY_METRICS."""
    for endpoint in REGRESSION_ENDPOINTS:
        assert _metrics_for_endpoint(endpoint) is ACTIVITY_METRICS
    for endpoint in CLASSIFICATION_ENDPOINTS:
        assert _metrics_for_endpoint(endpoint) is CLASSIFICATION_METRICS


def test_score_activity_predictions(
    example_activity_predictions_data, example_activity_gound_truth_data
):
    """Test scoring of activity predictions — no macro pseudo-endpoint included.

    score_activity_predictions scores whichever endpoints it's given, using each
    endpoint's own metric list — the macro "MA" row is no longer computed here (see
    add_macro_endpoint). Passing the full combined ACTIVITY_ENDPOINTS list here
    exercises both metric-dispatch branches in one call; a real regression or
    classification submission would instead pass only its own track's endpoints.
    """
    scored_df = score_activity_predictions(
        example_activity_predictions_data,
        example_activity_gound_truth_data,
        endpoints=ACTIVITY_ENDPOINTS,
    )
    assert not scored_df.empty
    assert scored_df.shape[0] == BOOTSTRAP_SAMPLES * len(ACTIVITY_ENDPOINTS)
    assert set(scored_df["Endpoint"]) == set(ACTIVITY_ENDPOINTS)
    assert MACRO_ENDPOINT_LABEL not in set(scored_df["Endpoint"])

    # Columns are the union of both metric lists — concatenating per-endpoint frames
    # with different metric columns leaves NaN wherever a metric doesn't apply to
    # that row's endpoint (add_macro_endpoint is what narrows this away per-track).
    expected_columns = (
        {"Sample", "Endpoint"}
        | {name for name, _ in ACTIVITY_METRICS}
        | {name for name, _ in CLASSIFICATION_METRICS}
    )
    assert set(scored_df.columns) == expected_columns

    regression_rows = scored_df[scored_df["Endpoint"].isin(REGRESSION_ENDPOINTS)]
    classification_rows = scored_df[
        scored_df["Endpoint"].isin(CLASSIFICATION_ENDPOINTS)
    ]
    for name, _ in ACTIVITY_METRICS:
        assert regression_rows[name].notna().all()
        assert classification_rows[name].isna().all()
    for name, _ in CLASSIFICATION_METRICS:
        assert classification_rows[name].notna().all()
        assert regression_rows[name].isna().all()


def test_score_activity_predictions_excludes_missing_classification_ground_truth(
    example_activity_predictions_data, example_activity_gound_truth_data
):
    """Compounds with a missing classification label are excluded, not crashed on.

    Confirms the pd.isna (not np.isnan) fix: a classification endpoint's ground
    truth has a couple of genuinely missing labels and lands as object dtype after
    merge, which np.isnan can't handle.
    """
    endpoint = CLASSIFICATION_ENDPOINTS[0]
    assert example_activity_gound_truth_data[endpoint].isna().any(), (
        "fixture must have >=1 missing classification label to exercise this path"
    )
    scored_df = score_activity_predictions(
        example_activity_predictions_data,
        example_activity_gound_truth_data,
        endpoints=ACTIVITY_ENDPOINTS,
    )
    endpoint_samples = scored_df[scored_df["Endpoint"] == endpoint]
    # bootstrap_metrics always produces one row per bootstrap iteration regardless of
    # pool size — the missing-label compound(s) are dropped from the resampling pool
    # before bootstrapping, not from the row count.
    assert len(endpoint_samples) == BOOTSTRAP_SAMPLES


def test_bootstrap_metrics_regression(
    example_activity_predictions_data, example_activity_gound_truth_data
):
    """Test bootstrapping of metrics for a regression endpoint.

    ACTIVITY_METRICS includes ST-RAE, which needs the per-compound credible-interval
    bounds (y_true_upper/y_true_lower) — omitting them would make bootstrap_metrics
    raise (see test_bootstrap_metrics_regression_requires_credible_interval_bounds).
    """
    endpoint = REGRESSION_ENDPOINTS[0]
    # A couple of compounds genuinely lack ground truth for this endpoint —
    # bootstrap_metrics itself doesn't filter (score_activity_predictions does that
    # before calling it), so mirror that exclusion here.
    has_ground_truth = example_activity_gound_truth_data[endpoint].notna()
    upper_col = f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX}"
    lower_col = f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX}"
    bootstrap_df = bootstrap_metrics(
        example_activity_predictions_data.loc[has_ground_truth, endpoint].to_numpy(),
        example_activity_gound_truth_data.loc[has_ground_truth, endpoint].to_numpy(),
        endpoint,
        n_bootstrap_samples=BOOTSTRAP_SAMPLES,
        metrics=ACTIVITY_METRICS,
        y_true_upper=example_activity_gound_truth_data.loc[
            has_ground_truth, upper_col
        ].to_numpy(),
        y_true_lower=example_activity_gound_truth_data.loc[
            has_ground_truth, lower_col
        ].to_numpy(),
    )
    assert not bootstrap_df.empty
    assert bootstrap_df.shape == (
        BOOTSTRAP_SAMPLES,
        len(ACTIVITY_METRICS) + 2,  # +2 for Endpoint and Sample columns
    )
    assert set(bootstrap_df.columns) == set(
        [met[0] for met in ACTIVITY_METRICS] + ["Endpoint", "Sample"]
    )


def test_bootstrap_metrics_regression_requires_credible_interval_bounds(
    example_activity_predictions_data, example_activity_gound_truth_data
):
    """ST-RAE needs credible-interval bounds — omitting them raises, not silently 0s."""
    endpoint = REGRESSION_ENDPOINTS[0]
    has_ground_truth = example_activity_gound_truth_data[endpoint].notna()
    with pytest.raises(RuntimeError, match="ST-RAE"):
        bootstrap_metrics(
            example_activity_predictions_data.loc[
                has_ground_truth, endpoint
            ].to_numpy(),
            example_activity_gound_truth_data.loc[
                has_ground_truth, endpoint
            ].to_numpy(),
            endpoint,
            n_bootstrap_samples=1,
            metrics=ACTIVITY_METRICS,
        )


def test_metric_needs_credible_interval_bounds_dispatch():
    """Only metrics with y_true_upper/y_true_lower params are flagged as needing bounds."""
    metrics_by_name = dict(ACTIVITY_METRICS)
    assert _metric_needs_credible_interval_bounds(metrics_by_name["ST-RAE"])
    assert not _metric_needs_credible_interval_bounds(metrics_by_name["MAE"])
    assert not _metric_needs_credible_interval_bounds(metrics_by_name["R2"])


def test_bootstrap_metrics_constant_prediction_falls_back_instead_of_raising():
    """A submission predicting the same value for every compound makes Spearman_R
    and Kendall_Tau mathematically undefined (NaN) rather than erroring — these
    should fall back to 0.0 ("no correlation") rather than raising.

    y_true_upper/y_true_lower are supplied as a wide dummy band purely so ST-RAE
    (also in ACTIVITY_METRICS) can run without hitting the separate
    credible-interval-bounds-required error this test isn't about.
    """
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = np.full_like(y_true, 5.0)
    bootstrap_df = bootstrap_metrics(
        y_pred,
        y_true,
        "some_endpoint",
        n_bootstrap_samples=10,
        metrics=ACTIVITY_METRICS,
        y_true_upper=y_true + 1.0,
        y_true_lower=y_true - 1.0,
    )
    assert (bootstrap_df["Spearman_R"] == 0.0).all()
    assert (bootstrap_df["Kendall_Tau"] == 0.0).all()


def test_bootstrap_metrics_classification(
    example_activity_predictions_data, example_activity_gound_truth_data
):
    """Test bootstrapping of metrics for a classification endpoint."""
    endpoint = CLASSIFICATION_ENDPOINTS[0]
    has_ground_truth = example_activity_gound_truth_data[endpoint].notna()
    y_true = (
        example_activity_gound_truth_data.loc[has_ground_truth, endpoint]
        .astype(bool)
        .to_numpy()
    )
    y_pred = (
        example_activity_predictions_data.loc[has_ground_truth, endpoint]
        .astype(bool)
        .to_numpy()
    )
    bootstrap_df = bootstrap_metrics(
        y_pred,
        y_true,
        endpoint,
        n_bootstrap_samples=BOOTSTRAP_SAMPLES,
        metrics=CLASSIFICATION_METRICS,
    )
    assert not bootstrap_df.empty
    assert bootstrap_df.shape == (
        BOOTSTRAP_SAMPLES,
        len(CLASSIFICATION_METRICS) + 2,  # +2 for Endpoint and Sample columns
    )
    assert set(bootstrap_df.columns) == set(
        [met[0] for met in CLASSIFICATION_METRICS] + ["Endpoint", "Sample"]
    )


def test_add_macro_endpoint_regression(example_regression_bootstrap_scores):
    """add_macro_endpoint narrows to the regression track and appends its MA row."""
    raw_only = example_regression_bootstrap_scores[
        example_regression_bootstrap_scores["Endpoint"] != MACRO_ENDPOINT_LABEL
    ]
    result = add_macro_endpoint(
        raw_only, endpoints=REGRESSION_ENDPOINTS, metrics=ACTIVITY_METRICS
    )
    assert set(result.columns) == {"Sample", "Endpoint"} | {
        name for name, _ in ACTIVITY_METRICS
    }
    assert set(result["Endpoint"]) == set(REGRESSION_ENDPOINTS) | {
        MACRO_ENDPOINT_LABEL
    }
    assert (result["Endpoint"] == MACRO_ENDPOINT_LABEL).sum() == BOOTSTRAP_SAMPLES


def test_add_macro_endpoint_classification(example_classification_bootstrap_scores):
    """add_macro_endpoint narrows to the classification track and appends its MA row."""
    raw_only = example_classification_bootstrap_scores[
        example_classification_bootstrap_scores["Endpoint"] != MACRO_ENDPOINT_LABEL
    ]
    result = add_macro_endpoint(
        raw_only, endpoints=CLASSIFICATION_ENDPOINTS, metrics=CLASSIFICATION_METRICS
    )
    assert set(result.columns) == {"Sample", "Endpoint"} | {
        name for name, _ in CLASSIFICATION_METRICS
    }
    assert set(result["Endpoint"]) == set(CLASSIFICATION_ENDPOINTS) | {
        MACRO_ENDPOINT_LABEL
    }
    assert (result["Endpoint"] == MACRO_ENDPOINT_LABEL).sum() == BOOTSTRAP_SAMPLES


def test_add_macro_endpoint_drops_other_track_columns():
    """No cross-track NaN metric columns leak into a track's narrowed output."""
    mixed = pd.DataFrame(
        {
            "Sample": [0, 0],
            "Endpoint": ["A", "X"],
            "MAE": [1.0, np.nan],
            "MCC": [np.nan, 0.5],
        }
    )
    result = add_macro_endpoint(mixed, endpoints=["A"], metrics=[("MAE", None)])
    assert list(result.columns) == ["Sample", "Endpoint", "MAE"]
    assert "MCC" not in result.columns
    # A single endpoint — no macro row to add.
    assert set(result["Endpoint"]) == {"A"}


def test_average_bootstrap_results_by_endpoint(example_regression_bootstrap_scores):
    """Test averaging of bootstrap results by endpoint, including the macro row."""
    averaged_df = average_bootstrap_results_by_endpoint(
        example_regression_bootstrap_scores
    )
    assert not averaged_df.empty
    # +1 for the "MA" pseudo-endpoint row alongside the real endpoints.
    assert averaged_df.shape == (
        len(REGRESSION_ENDPOINTS) + 1,
        len(ACTIVITY_METRICS) * 2,
    )  # *2 for mean and std columns
    assert MACRO_ENDPOINT_LABEL in averaged_df.index


def test_compute_macro_bootstrap_results_matches_fixture(
    example_regression_bootstrap_scores,
):
    """Recomputing the macro block from the raw endpoints should match the fixture.

    The fixture was generated by running the real scoring pipeline, so re-deriving
    the "MA" rows from just the real-endpoint rows should reproduce them exactly.
    """
    raw_only = example_regression_bootstrap_scores[
        example_regression_bootstrap_scores["Endpoint"] != MACRO_ENDPOINT_LABEL
    ]
    fixture_macro = example_regression_bootstrap_scores[
        example_regression_bootstrap_scores["Endpoint"] == MACRO_ENDPOINT_LABEL
    ].sort_values("Sample")

    recomputed_macro = compute_macro_bootstrap_results(
        raw_only, metrics=ACTIVITY_METRICS
    ).sort_values("Sample")

    assert recomputed_macro.shape == (BOOTSTRAP_SAMPLES, len(ACTIVITY_METRICS) + 2)
    assert (recomputed_macro["Endpoint"] == MACRO_ENDPOINT_LABEL).all()
    for metric_name, _ in ACTIVITY_METRICS:
        np.testing.assert_allclose(
            recomputed_macro[metric_name].to_numpy(),
            fixture_macro[metric_name].to_numpy(),
            atol=1e-8,
        )


def test_compute_macro_bootstrap_results_metric_semantics():
    """Check every metric — including Spearman_R — is a plain arithmetic mean.

    Spearman_R previously used a Fisher z-transform (arctanh/tanh), the standard
    treatment for combining several noisy estimates of the *same* correlation. That
    doesn't apply to macro-averaging across different endpoints (unrelated true
    correlations): arctanh blows up near +/-1, so a submission with a
    near-perfect Spearman on one endpoint and ~0 on the rest could macro-average to
    ~0.99 instead of the expected ~0.33 — a couple of easy/lucky endpoints dominating
    the score. So Spearman_R now gets a plain mean like every other metric.
    """
    raw = pd.DataFrame(
        {
            "Sample": [0, 0, 0, 1, 1, 1],
            "Endpoint": ["A", "B", "C", "A", "B", "C"],
            "MAE": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0],
            "ST-RAE": [1.0, 2.0, 30.0, 4.0, 5.0, 9.0],
            "R2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
            "Spearman_R": [0.9, 0.9, -0.9, 0.5, 0.5, 0.5],
            "Kendall_Tau": [0.9, 0.9, -0.9, 0.5, 0.5, 0.5],
        }
    )
    macro = compute_macro_bootstrap_results(raw, metrics=ACTIVITY_METRICS).set_index(
        "Sample"
    )

    # ST-RAE, MAE, R2, Kendall_Tau, Spearman_R: all plain arithmetic mean.
    assert macro.loc[0, "MAE"] == pytest.approx((1.0 + 2.0 + 3.0) / 3)
    assert macro.loc[0, "R2"] == pytest.approx((0.1 + 0.2 + 0.3) / 3)
    assert macro.loc[1, "MAE"] == pytest.approx((4.0 + 5.0 + 9.0) / 3)
    assert macro.loc[0, "Kendall_Tau"] == pytest.approx((0.9 + 0.9 - 0.9) / 3)
    assert macro.loc[1, "Kendall_Tau"] == pytest.approx(0.5)
    assert macro.loc[0, "Spearman_R"] == pytest.approx((0.9 + 0.9 - 0.9) / 3)
    assert macro.loc[1, "Spearman_R"] == pytest.approx(0.5)


def test_pivot_endpoint_results_wide(example_regression_bootstrap_scores):
    """Check every endpoint's columns, including the macro pseudo-endpoint, are prefixed."""
    by_endpoint = average_bootstrap_results_by_endpoint(
        example_regression_bootstrap_scores
    )
    wide = pivot_endpoint_results_wide(by_endpoint)

    assert wide.shape[0] == 1
    # Macro pseudo-endpoint: prefixed with MACRO_ENDPOINT_LABEL, same as real endpoints.
    assert f"{MACRO_ENDPOINT_LABEL}_ST-RAE_mean" in wide.columns
    assert f"{MACRO_ENDPOINT_LABEL}_ST-RAE_std" in wide.columns
    assert "ST-RAE_mean" not in wide.columns

    # Real endpoints: prefixed with the endpoint name.
    for endpoint in REGRESSION_ENDPOINTS:
        assert f"{endpoint}_ST-RAE_mean" in wide.columns
        assert f"{endpoint}_ST-RAE_std" in wide.columns

    assert wide.shape[1] == (len(REGRESSION_ENDPOINTS) + 1) * len(ACTIVITY_METRICS) * 2
