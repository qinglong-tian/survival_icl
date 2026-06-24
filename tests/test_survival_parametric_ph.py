"""Unit tests for parametric proportional-hazards utilities."""

from __future__ import annotations

import numpy as np
import pytest

from tabicl.survival._parametric_ph import (
    EPS,
    FAMILIES,
    FAMILY_KEYS,
    ParametricPHEstimate,
    baseline_survival,
    cumulative_hazard0,
    fit_parametric_ph_mle,
    initial_ph_theta,
    inverse_cumulative_hazard0,
    log_hazard0,
    ph_negative_log_likelihood,
    unpack_params,
)


def _synthetic_ph_task(family_key: str, n: int = 200, n_features: int = 3):
    rng = np.random.default_rng(42)
    beta = np.array([0.8, -0.6, 0.45], dtype=np.float64)[:n_features]
    params = FAMILIES[family_key].true_params
    X = rng.normal(size=(n, n_features)).astype(np.float64)
    eta = X @ beta
    u = np.clip(rng.uniform(size=n), 1e-10, 1.0 - 1e-10)
    event_hazard = -np.log(u) / np.exp(eta)
    t_event = inverse_cumulative_hazard0(event_hazard, family_key, params)
    censor_time = rng.exponential(scale=18.0, size=n)
    t_obs = np.minimum(t_event, censor_time)
    delta = (t_event <= censor_time).astype(np.float64)
    return X, t_obs, delta, beta, params


@pytest.mark.parametrize("family_key", FAMILY_KEYS)
def test_unpack_params_roundtrip(family_key):
    params = FAMILIES[family_key].true_params
    param_names = FAMILIES[family_key].param_names
    if family_key == "weibull":
        theta = np.array([np.log(params["shape"]), np.log(params["scale"])])
    elif family_key == "gompertz":
        theta = np.array([np.log(params["rate"]), np.log(params["gamma"])])
    elif family_key == "loglogistic":
        theta = np.array([np.log(params["shape"]), np.log(params["scale"])])
    elif family_key == "lognormal":
        theta = np.array([params["mu"], np.log(params["sigma"])])
    else:
        pytest.skip(f"unknown family: {family_key}")

    unpacked = unpack_params(family_key, theta=theta)
    for key in params:
        assert np.isclose(unpacked[key], params[key], rtol=1e-10)


def test_unpack_params_direct_passthrough():
    params = {"shape": 2.0, "scale": 5.0}
    result = unpack_params("weibull", params=params)
    assert result is not params
    assert result == params


def test_unpack_params_raises_on_missing_args():
    with pytest.raises(ValueError, match="Either theta or params must be provided"):
        unpack_params("weibull")


def test_unpack_params_raises_on_unknown_family():
    with pytest.raises(ValueError):
        unpack_params("unknown_family", theta=np.array([0.0, 0.0]))


def test_cumulative_hazard0_is_positive_and_increasing():
    t = np.linspace(EPS, 20.0, 100)
    for family_key in FAMILY_KEYS:
        params = FAMILIES[family_key].true_params
        h0 = cumulative_hazard0(t, family_key, params)
        assert np.all(h0 >= 0.0)
        assert np.all(np.diff(h0) >= -EPS)
        assert np.all(np.isfinite(h0))


def test_baseline_survival_bounds():
    t = np.linspace(EPS, 20.0, 100)
    for family_key in FAMILY_KEYS:
        params = FAMILIES[family_key].true_params
        s0 = baseline_survival(t, family_key, params)
        assert np.all(s0 >= 0.0)
        assert np.all(s0 <= 1.0)
        assert np.all(np.diff(s0) <= EPS)


def test_log_hazard0_is_finite():
    t = np.linspace(EPS * 10, 20.0, 100)
    for family_key in FAMILY_KEYS:
        params = FAMILIES[family_key].true_params
        log_h = log_hazard0(t, family_key, params)
        assert np.all(np.isfinite(log_h))


def test_inverse_cumulative_hazard0_roundtrip():
    t_orig = np.linspace(0.1, 10.0, 30)
    for family_key in FAMILY_KEYS:
        params = FAMILIES[family_key].true_params
        h = cumulative_hazard0(t_orig, family_key, params)
        t_hat = inverse_cumulative_hazard0(h, family_key, params)
        assert np.allclose(t_hat, t_orig, rtol=1e-4)


@pytest.mark.parametrize("family_key", FAMILY_KEYS)
def test_ph_negative_log_likelihood_is_finite(family_key):
    X, t, delta, beta, params = _synthetic_ph_task(family_key)
    n_features = X.shape[1]
    if family_key == "weibull":
        theta = np.concatenate([np.zeros(n_features), [np.log(1.2), np.log(np.median(t))]])
    elif family_key == "gompertz":
        theta = np.concatenate([np.zeros(n_features), [np.log(1.0), np.log(0.05)]])
    elif family_key == "loglogistic":
        theta = np.concatenate([np.zeros(n_features), [np.log(1.2), np.log(np.median(t))]])
    elif family_key == "lognormal":
        theta = np.concatenate([np.zeros(n_features), [np.log(np.median(t)), np.log(1.0)]])
    else:
        pytest.skip(f"unknown family: {family_key}")

    nll = ph_negative_log_likelihood(theta, family_key, X, t, delta)
    assert np.isfinite(nll)
    assert nll > 0.0


def test_ph_negative_log_likelihood_with_l2_penalty():
    X, t, delta, _, _ = _synthetic_ph_task("weibull")
    theta = np.array([0.5, -0.3, 0.2, np.log(1.2), np.log(10.0)], dtype=np.float64)
    nll_no_penalty = ph_negative_log_likelihood(theta, "weibull", X, t, delta, l2_penalty=0.0)
    nll_with_penalty = ph_negative_log_likelihood(theta, "weibull", X, t, delta, l2_penalty=0.1)
    assert nll_with_penalty > nll_no_penalty


def test_ph_negative_log_likelihood_returns_penalty_for_nonfinite():
    X = np.array([[1.0, 2.0]], dtype=np.float64)
    t = np.array([np.nan, 1.0], dtype=np.float64)
    delta = np.array([1.0, 1.0], dtype=np.float64)
    theta = np.array([0.5, -0.3, np.log(1.2), np.log(10.0)], dtype=np.float64)
    nll = ph_negative_log_likelihood(theta, "weibull", X, t, delta)
    assert nll == 1e100


def test_ph_negative_log_likelihood_handles_nan_inf():
    family_key = "weibull"
    X, t, delta, _, _ = _synthetic_ph_task(family_key, n=50)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    theta = np.array([0.0, 0.0, 0.0, np.log(1.2), np.log(10.0)], dtype=np.float64)
    nll = ph_negative_log_likelihood(theta, family_key, X_nan, t, delta)
    assert nll == 1e100


def test_initial_ph_theta_shape():
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    for family_key in FAMILY_KEYS:
        theta = initial_ph_theta(family_key, 5, t)
        assert theta.shape == (7,)
        assert theta.dtype == np.float64


@pytest.mark.parametrize("family_key", FAMILY_KEYS)
def test_fit_parametric_ph_mle_recovery(family_key):
    X, t, delta, true_beta, true_params = _synthetic_ph_task(family_key, n=500)
    estimate = fit_parametric_ph_mle(family_key, X, t, delta, maxiter=700)
    assert isinstance(estimate, ParametricPHEstimate)
    assert estimate.beta.shape == true_beta.shape
    assert set(estimate.baseline_params.keys()) == set(true_params.keys())
    assert np.all(np.isfinite(estimate.beta))
    for key, value in estimate.baseline_params.items():
        assert np.isfinite(value)
        if key != "mu":
            assert value > 0.0


def test_fit_parametric_ph_mle_raises_on_failure():
    X = np.zeros((10, 2), dtype=np.float64)
    t = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)
    delta = np.zeros(10, dtype=np.float64)
    with pytest.raises(RuntimeError, match="PH MLE failed"):
        fit_parametric_ph_mle("weibull", X, t, delta, maxiter=5)


def test_all_family_keys_are_valid():
    for key in FAMILY_KEYS:
        assert key in FAMILIES
        assert isinstance(FAMILIES[key].key, str)
        assert isinstance(FAMILIES[key].label, str)
        assert isinstance(FAMILIES[key].param_names, tuple)
        assert isinstance(FAMILIES[key].true_params, dict)


def test_gompertz_preserves_rate_positive():
    t = np.linspace(EPS, 10.0, 100)
    params = {"rate": 0.5, "gamma": 0.2}
    h0 = cumulative_hazard0(t, "gompertz", params)
    assert np.all(np.isfinite(h0))
    assert np.all(h0 >= 0.0)
