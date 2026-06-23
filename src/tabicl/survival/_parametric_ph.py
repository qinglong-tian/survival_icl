"""Parametric proportional-hazards baseline utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr as norm_cdf
from scipy.special import ndtri as norm_ppf


EPS = 1e-12


@dataclass(frozen=True)
class PHFamily:
    """Baseline family for proportional-hazards generation and fitting."""

    key: str
    label: str
    param_names: tuple[str, ...]
    true_params: dict[str, float]


@dataclass
class ParametricPHEstimate:
    """Fitted parametric PH model state."""

    beta: np.ndarray
    baseline_params: dict[str, float]


FAMILIES: dict[str, PHFamily] = {
    "weibull": PHFamily("weibull", "Weibull", ("log_shape", "log_scale"), {"shape": 1.6, "scale": 10.0}),
    "gompertz": PHFamily("gompertz", "Gompertz", ("log_rate", "log_gamma"), {"rate": 1.0, "gamma": 0.15}),
    "loglogistic": PHFamily(
        "loglogistic",
        "LogLogistic",
        ("log_shape", "log_scale"),
        {"shape": 1.8, "scale": 1.0},
    ),
    "lognormal": PHFamily("lognormal", "LogNormal", ("mu", "log_sigma"), {"mu": 0.0, "sigma": 0.8}),
}
FAMILY_KEYS = tuple(FAMILIES)


def _clip_time(t: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(t, dtype=np.float64), EPS)


def unpack_params(
    family_key: str,
    theta: np.ndarray | None = None,
    params: dict[str, float] | None = None,
) -> dict[str, float]:
    """Unpack unconstrained baseline parameters for one PH family."""
    if params is not None:
        return dict(params)
    if theta is None:
        raise ValueError("Either theta or params must be provided.")
    if family_key == "weibull":
        return {"shape": float(np.exp(theta[0])), "scale": float(np.exp(theta[1]))}
    if family_key == "gompertz":
        return {"rate": float(np.exp(theta[0])), "gamma": float(np.exp(theta[1]))}
    if family_key == "loglogistic":
        return {"shape": float(np.exp(theta[0])), "scale": float(np.exp(theta[1]))}
    if family_key == "lognormal":
        return {"mu": float(theta[0]), "sigma": float(np.exp(theta[1]))}
    raise ValueError(family_key)


def cumulative_hazard0(t: np.ndarray, family_key: str, params: dict[str, float]) -> np.ndarray:
    """Return baseline cumulative hazard ``H0(t)``."""
    t = _clip_time(t)
    if family_key == "weibull":
        return (t / params["scale"]) ** params["shape"]
    if family_key == "gompertz":
        gamma = params["gamma"]
        return params["rate"] * np.expm1(np.minimum(gamma * t, 50.0)) / gamma
    if family_key == "loglogistic":
        return np.log1p((t / params["scale"]) ** params["shape"])
    if family_key == "lognormal":
        z = (params["mu"] - np.log(t)) / params["sigma"]
        return -np.log(np.clip(norm_cdf(z), EPS, 1.0))
    raise ValueError(family_key)


def log_hazard0(t: np.ndarray, family_key: str, params: dict[str, float]) -> np.ndarray:
    """Return baseline log-hazard ``log h0(t)``."""
    t = _clip_time(t)
    if family_key == "weibull":
        shape = params["shape"]
        scale = params["scale"]
        return np.log(shape) - np.log(scale) + (shape - 1.0) * (np.log(t) - np.log(scale))
    if family_key == "gompertz":
        return np.log(params["rate"]) + np.minimum(params["gamma"] * t, 50.0)
    if family_key == "loglogistic":
        shape = params["shape"]
        scale = params["scale"]
        z = (t / scale) ** shape
        return np.log(shape) - np.log(scale) + (shape - 1.0) * (np.log(t) - np.log(scale)) - np.log1p(z)
    if family_key == "lognormal":
        mu = params["mu"]
        sigma = params["sigma"]
        z = (np.log(t) - mu) / sigma
        log_pdf = -np.log(t) - np.log(sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * z**2
        log_surv = -cumulative_hazard0(t, family_key, params)
        return log_pdf - log_surv
    raise ValueError(family_key)


def inverse_cumulative_hazard0(h: np.ndarray, family_key: str, params: dict[str, float]) -> np.ndarray:
    """Return ``H0^{-1}(h)`` for synthetic PH sampling."""
    h = np.maximum(np.asarray(h, dtype=np.float64), EPS)
    if family_key == "weibull":
        return params["scale"] * h ** (1.0 / params["shape"])
    if family_key == "gompertz":
        inner = 1.0 + params["gamma"] * h / params["rate"]
        return np.log(np.maximum(inner, 1.0 + EPS)) / params["gamma"]
    if family_key == "loglogistic":
        return params["scale"] * np.expm1(np.minimum(h, 50.0)) ** (1.0 / params["shape"])
    if family_key == "lognormal":
        s0 = np.exp(-np.minimum(h, 50.0))
        z = norm_ppf(np.clip(s0, 1e-10, 1.0 - 1e-10))
        return np.exp(params["mu"] - params["sigma"] * z)
    raise ValueError(family_key)


def baseline_survival(t: np.ndarray, family_key: str, params: dict[str, float]) -> np.ndarray:
    """Return baseline survival ``S0(t)``."""
    return np.exp(-cumulative_hazard0(t, family_key, params))


def ph_negative_log_likelihood(
    theta: np.ndarray,
    family_key: str,
    X: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
    *,
    l2_penalty: float = 0.0,
) -> float:
    """Right-censored parametric PH negative log-likelihood."""
    beta = theta[: X.shape[1]]
    params = unpack_params(family_key, theta[X.shape[1] :])
    eta = np.clip(np.asarray(X, dtype=np.float64) @ beta, -20.0, 20.0)
    h0 = np.clip(cumulative_hazard0(t, family_key, params), EPS, 1e100)
    log_h0 = log_hazard0(t, family_key, params)
    log_likelihood = delta * (log_h0 + eta) - h0 * np.exp(eta)
    value = -float(np.sum(log_likelihood))
    if not np.isfinite(value):
        return 1e100
    return value + l2_penalty * float(np.sum(beta**2))


def initial_ph_theta(family_key: str, n_features: int, t: np.ndarray) -> np.ndarray:
    """Return a deterministic PH optimizer initialization."""
    beta0 = np.zeros(n_features, dtype=np.float64)
    median_time = float(np.median(t))
    if family_key == "weibull":
        baseline0 = np.array([np.log(1.2), np.log(median_time)], dtype=np.float64)
    elif family_key == "gompertz":
        baseline0 = np.array([np.log(1.0), np.log(0.05)], dtype=np.float64)
    elif family_key == "loglogistic":
        baseline0 = np.array([np.log(1.2), np.log(median_time)], dtype=np.float64)
    elif family_key == "lognormal":
        baseline0 = np.array([np.log(median_time), np.log(1.0)], dtype=np.float64)
    else:
        raise ValueError(family_key)
    return np.concatenate([beta0, baseline0])


def fit_parametric_ph_mle(
    family_key: str,
    X: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
    *,
    maxiter: int = 700,
    l2_penalty: float = 0.0,
) -> ParametricPHEstimate:
    """Fit a parametric PH family by censored maximum likelihood."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    result = minimize(
        lambda theta, fam, X_arg, t_arg, delta_arg: ph_negative_log_likelihood(
            theta,
            fam,
            X_arg,
            t_arg,
            delta_arg,
            l2_penalty=l2_penalty,
        ),
        initial_ph_theta(family_key, X.shape[1], t),
        args=(family_key, X, t, delta),
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    if not result.success:
        raise RuntimeError(f"{FAMILIES[family_key].label} PH MLE failed: {result.message}")
    theta = np.asarray(result.x, dtype=np.float64)
    return ParametricPHEstimate(
        beta=theta[: X.shape[1]],
        baseline_params=unpack_params(family_key, theta[X.shape[1] :]),
    )
