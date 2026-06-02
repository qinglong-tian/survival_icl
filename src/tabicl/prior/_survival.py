from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union

import math
import numpy as np
import torch
from torch import Tensor
from torch.nested import nested_tensor

import joblib

from tabicl.prior._dataset import SCMPrior, Prior, DisablePrinting
from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from tabicl.prior._mlp_scm import MLPSCM
from tabicl.prior._tree_scm import TreeSCM
from tabicl.prior._reg2cls import Reg2Cls


DEFAULT_RAW_TIME_MAX = 1e30
MIN_RAW_TIME = 1e-8


def calibrate_censor_scale_by_quantile(
    t_event: Tensor,
    c_base: Tensor,
    target_event_rate: float,
    eps: float = 1e-12,
) -> tuple[float, dict]:
    """Find the censoring scale that achieves a target event rate under strict ``<``.

    Computes per-subject ratios ``r = t_event / c_base``, sorts them, then
    finds the unique-regime threshold whose strict-``<`` empirical event rate
    is closest to ``target_event_rate``.

    Complexity: O(n log n) from the sort; the candidate search is O(n)
    because there are at most n+1 distinct regimes.

    Parameters
    ----------
    t_event : Tensor, shape ``(n,)``
        Event times (already sanitized via ``_finite_positive_time``).
    c_base : Tensor, shape ``(n,)``
        Base censoring times (same).
    target_event_rate : float, in (0, 1)
        Desired fraction of events.
    eps : float, default=1e-12
        Clamp the returned scale to at least this value.

    Returns
    -------
    censor_scale : float
        Threshold that (approximately) achieves the target rate.
    diagnostics : dict
        Target, achieved rate, scale, absolute error, and method info.
    """
    r = (t_event / c_base.clamp_min(eps)).reshape(-1)
    r_sorted = torch.sort(r, stable=True).values
    n = r_sorted.numel()

    # Build candidate thresholds from unique tie-groups.
    # We use the upper unique value as the threshold for each interior
    # regime: s = unique_vals[i].  With strict <, this guarantees that
    # all ratios in groups 0..i-1 are < s and all ratios in group i are
    # = s (therefore ≥ s), so the achieved event count is cumcount[i-1].
    # Midpoints between adjacent unique values can round back to one
    # endpoint in float32, making diagnostics['achieved'] unreliable.
    unique_vals, counts = torch.unique_consecutive(r_sorted, return_counts=True)
    cumcount = torch.cumsum(counts, dim=0)
    n_uniq = len(unique_vals)

    # Build (threshold, achieved_rate) for each distinct regime.
    # Regime 0: threshold well below smallest ratio → 0 events
    # Regime i (1..n_uniq-1): s = unique_vals[i] → cumcount[i-1] events
    # Regime n_uniq: threshold well above largest ratio → n events
    targets = []  # list of (threshold, achieved_rate)
    for i in range(n_uniq + 1):
        if i == 0:
            s = float((unique_vals[0] * 0.5).item())
            k = 0
        elif i == n_uniq:
            s = float((unique_vals[-1] * 2.0).item())
            k = n
        else:
            # s = unique_vals[i] — strict < excludes this tie group
            s = float(unique_vals[i].item())
            k = int(cumcount[i - 1].item())
        rate = float(k) / float(n)
        targets.append((s, rate))

    # Find the regime whose achieved rate is closest to target.
    best_scale = targets[0][0]
    best_err = float("inf")
    for s, rate in targets:
        err = abs(rate - target_event_rate)
        if err < best_err:
            best_err = err
            best_scale = s

    best_scale = max(float(best_scale), eps)
    # Recompute diagnostics from the clamped scale so that achieved/error
    # are always consistent with the returned value.  The eps clamp can
    # move the threshold into a different regime when ratios are tiny.
    achieved = (r < best_scale).float().mean().item()
    diagnostics = {
        "target": target_event_rate,
        "achieved": achieved,
        "scale": best_scale,
        "absolute_error": abs(achieved - target_event_rate),
        "n_subjects": n,
    }
    return best_scale, diagnostics


def _finite_positive_time(t: Tensor, max_time: float) -> Tensor:
    """Sanitize raw time tensor: clamp to [MIN_RAW_TIME, max_time].

    ``-inf`` is replaced by ``MIN_RAW_TIME`` (a tiny positive value) rather
    than ``max_time`` because negative event times are physically meaningless
    and should map near zero, not to the far horizon.  ``nan`` and ``+inf``
    both map to ``max_time`` (conservative: "we don't know when it happens").
    """
    return torch.nan_to_num(
        t, nan=max_time, posinf=max_time, neginf=MIN_RAW_TIME,
    ).clamp(min=MIN_RAW_TIME, max=max_time)


class BaselineHazard(ABC):
    """Abstract base class for baseline hazard distributions.

    Subclasses implement a parametric baseline hazard with scale fixed at 1.
    """

    @abstractmethod
    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sample shape parameters for one GP group.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        dict
            Sampled parameters (e.g., ``{"k": 1.5}`` for Weibull shape).
        """
        ...

    @abstractmethod
    def inverse_cdf(self, u: Tensor, log_risk: Tensor, params: Dict[str, float]) -> Tensor:
        """Compute event times given Uniform(0,1) samples and log relative risk.

        Parameters
        ----------
        u : Tensor
            Uniform(0,1) samples, shape ``(n,)``.
        log_risk : Tensor
            Log relative risk for each observation, shape ``(n,)``.
        params : dict
            Sampled baseline parameters.

        Returns
        -------
        Tensor
            Event times, shape ``(n,)``.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this baseline hazard."""
        ...


class WeibullHazard(BaselineHazard):
    """Weibull baseline hazard with scale fixed at 1.

    Baseline hazard: ``h_0(t) = k * t^(k-1)``
    Individual hazard: ``h(t|X) = k * t^(k-1) * exp(log_risk)``
    Inverse CDF: ``T = [-log(U) / exp(log_risk)]^(1/k)``

    Parameters
    ----------
    k_min : float, default=0.5
        Minimum shape parameter.
    k_max : float, default=3.0
        Maximum shape parameter.
    """

    def __init__(self, k_min: float = 0.5, k_max: float = 3.0):
        self.k_min = k_min
        self.k_max = k_max

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        return {"k": float(rng.uniform(self.k_min, self.k_max))}

    def inverse_cdf(self, u: Tensor, log_risk: Tensor, params: Dict[str, float]) -> Tensor:
        k = params["k"]
        arg = (-torch.log(u) / torch.exp(log_risk)).clamp(max=36.0)
        return arg.pow(1.0 / k)

    @property
    def name(self) -> str:
        return "weibull"


class GompertzHazard(BaselineHazard):
    """Gompertz baseline hazard with scale fixed at 1.

    Baseline hazard: ``h_0(t) = exp(gamma * t)``
    Individual hazard: ``h(t|X) = exp(gamma * t) * exp(log_risk)``
    Inverse CDF: ``T = (1/gamma) * log(1 - gamma * log(U) / exp(log_risk))``

    Parameters
    ----------
    gamma_log_min : float, default=-4.605
        log(0.01), lower bound for log-uniform sampling.
    gamma_log_max : float, default=-0.693
        log(0.5), upper bound for log-uniform sampling.
    """

    def __init__(self, gamma_log_min: float = -4.605, gamma_log_max: float = -0.693):
        self.gamma_log_min = gamma_log_min
        self.gamma_log_max = gamma_log_max

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        log_gamma = rng.uniform(self.gamma_log_min, self.gamma_log_max)
        return {"gamma": float(np.exp(log_gamma))}

    def inverse_cdf(self, u: Tensor, log_risk: Tensor, params: Dict[str, float]) -> Tensor:
        gamma = params["gamma"]
        inner = 1.0 - gamma * torch.log(u) / torch.exp(log_risk)
        inner = inner.clamp(min=1.0 + 1e-7, max=1e10)
        return (1.0 / gamma) * torch.log(inner)

    @property
    def name(self) -> str:
        return "gompertz"


class LogLogisticHazard(BaselineHazard):
    """Log-logistic baseline hazard with scale fixed at 1.

    Baseline survival: ``S_0(t) = 1 / (1 + t^beta)``
    Baseline hazard: ``h_0(t) = beta * t^(beta-1) / (1 + t^beta)``
    Inverse CDF: ``T = (exp(-log(U) / exp(log_risk)) - 1)^(1/beta)``

    Parameters
    ----------
    beta_min : float, default=0.5
        Minimum shape parameter.
    beta_max : float, default=3.0
        Maximum shape parameter.
    """

    def __init__(self, beta_min: float = 0.5, beta_max: float = 3.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        return {"beta": float(rng.uniform(self.beta_min, self.beta_max))}

    def inverse_cdf(self, u: Tensor, log_risk: Tensor, params: Dict[str, float]) -> Tensor:
        beta = params["beta"]
        arg = (-torch.log(u) / torch.exp(log_risk)).clamp(max=36.0)
        return (torch.exp(arg) - 1.0).pow(1.0 / beta)

    @property
    def name(self) -> str:
        return "loglogistic"


class LogNormalHazard(BaselineHazard):
    """Log-normal baseline hazard with scale fixed at 1.

    Baseline distribution: ``log(T) ~ N(mu, 1)``
    Baseline survival: ``S_0(t) = Phi(-(log(t) - mu))`` where Phi is standard normal CDF
    Inverse CDF: ``T = exp(mu - Phi^(-1)(U^(1/exp(log_risk))))``

    Uses ``torch.special.ndtri`` for the standard normal quantile function.

    Parameters
    ----------
    mu_min : float, default=-2.0
        Minimum location parameter.
    mu_max : float, default=2.0
        Maximum location parameter.
    """

    def __init__(self, mu_min: float = -2.0, mu_max: float = 2.0):
        self.mu_min = mu_min
        self.mu_max = mu_max

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        return {"mu": float(rng.uniform(self.mu_min, self.mu_max))}

    def inverse_cdf(self, u: Tensor, log_risk: Tensor, params: Dict[str, float]) -> Tensor:
        mu = params["mu"]
        p = u.pow(1.0 / torch.exp(log_risk))
        p = p.clamp(min=1e-7, max=1.0 - 1e-7)
        return torch.exp(mu - torch.special.ndtri(p))

    @property
    def name(self) -> str:
        return "lognormal"


class ProportionalHazardSampler:
    """Convert log relative risk into event times using a pool of baseline hazards.

    Parameters
    ----------
    baseline_pool : dict
        Mapping from name (str) to :class:`BaselineHazard` instance.
    beta : float, default=1.0
        Multiplier for the log relative risk: ``log_risk = beta * y``.
    baseline_mode : str, default="mix"
        ``"mix"`` randomly selects a baseline per dataset, or a fixed name
        like ``"weibull"``.
    max_time : float, default=1e30
        Numerical safety maximum for raw times.  The model-facing horizon is
        set later by per-task standardized log-time scaling.
    u_eps : float, default=1e-6
        Epsilon for clipping uniform samples away from 0 and 1.
    """

    def __init__(
        self,
        baseline_pool: Dict[str, BaselineHazard],
        beta: float = 1.0,
        baseline_mode: str = "mix",
        max_time: float = DEFAULT_RAW_TIME_MAX,
        u_eps: float = 1e-6,
    ):
        self.baseline_pool = baseline_pool
        self.beta = beta
        self.baseline_mode = baseline_mode
        self.max_time = max_time
        self.u_eps = u_eps

    def sample(
        self,
        y: Tensor,
        baseline_name: str,
        baseline_params: Dict[str, float],
        rng: np.random.Generator,
        device: str = "cpu",
        censor_scale: float = 1.0,
        censoring_strategy: str = "uniform_scale",
        target_event_rate: float | None = None,
        calibration_eps: float = 1e-12,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert continuous regression target into event times.

        Parameters
        ----------
        y : Tensor
            Continuous target, shape ``(seq_len,)``.
        baseline_name : str
            Which baseline hazard to use (or ``"mix"``).
        baseline_params : dict
            Sampled parameters for the baseline hazard.
        rng : numpy.random.Generator
            Random number generator for selecting baseline in mix mode.
        device : str, default="cpu"
            Device for tensor operations.
        censor_scale : float, default=1.0
            Multiplier for censoring times under ``uniform_scale`` strategy.
        censoring_strategy : str, default="uniform_scale"
            ``"uniform_scale"`` uses the provided ``censor_scale`` directly.
            ``"target_event_rate"`` calibrates the scale from ``t_event / c_base``
            to hit ``target_event_rate``.
        target_event_rate : float or None, default=None
            Required when ``censoring_strategy="target_event_rate"``.
        calibration_eps : float, default=1e-12
            Epsilon for the calibration helper.

        Returns
        -------
        t_obs : Tensor
            Observed times (event or censoring), shape ``(seq_len,)``.
        delta : Tensor
            Event indicators (1=event, 0=censored), shape ``(seq_len,)``.
        t_event : Tensor
            Underlying event time (before censoring), shape ``(seq_len,)``.
        """
        if baseline_name == "mix":
            names = list(self.baseline_pool.keys())
            baseline_name = names[int(rng.integers(0, len(names)))]

        baseline = self.baseline_pool[baseline_name]
        log_risk = self.beta * y

        u = torch.rand(y.shape, device=device)
        u = u.clamp(min=self.u_eps, max=1.0 - self.u_eps)
        t_event = baseline.inverse_cdf(u, log_risk, baseline_params)
        t_event = _finite_positive_time(t_event, self.max_time)

        # Generate independent base censoring times (no covariate effect)
        u_c = torch.rand(y.shape, device=device)
        u_c = u_c.clamp(min=self.u_eps, max=1.0 - self.u_eps)
        c_base = baseline.inverse_cdf(u_c, torch.zeros_like(log_risk), baseline_params)
        c_base = _finite_positive_time(c_base, self.max_time)

        if censoring_strategy == "target_event_rate":
            assert target_event_rate is not None, (
                "target_event_rate is required when censoring_strategy='target_event_rate'"
            )
            censor_scale, _diag = calibrate_censor_scale_by_quantile(
                t_event, c_base, target_event_rate, eps=calibration_eps,
            )
        elif censoring_strategy != "uniform_scale":
            raise ValueError(
                f"Unknown censoring_strategy '{censoring_strategy}'. "
                "Options: 'target_event_rate', 'uniform_scale'."
            )

        c = c_base * censor_scale
        c = _finite_positive_time(c, self.max_time)

        t_obs = torch.minimum(t_event, c)
        delta = (t_event < c).float()

        return t_obs, delta, t_event


class AFTBaselineHazard(ABC):
    """Abstract base class for AFT baseline time distributions.

    Simpler than :class:`BaselineHazard` — the acceleration factor
    ``exp(-beta * y)`` is applied externally by the AFT sampler.
    """

    @abstractmethod
    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        ...

    @abstractmethod
    def baseline_time(self, u: Tensor, params: Dict[str, float]) -> Tensor:
        """Compute baseline event time T₀ from Uniform(0,1) samples.

        Parameters
        ----------
        u : Tensor
            Uniform(0,1) samples, shape ``(n,)``.
        params : dict
            Sampled parameters.

        Returns
        -------
        Tensor
            Baseline event times, shape ``(n,)``.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class WeibullAFT(AFTBaselineHazard):
    """Weibull AFT baseline with scale fixed at 1.

    Baseline time: ``T_0 = (-log(U))^(1/k)``
    Final time: ``T = T_0 * exp(-beta * y)``

    Parameters
    ----------
    k_min : float, default=0.5
    k_max : float, default=3.0
    """

    def __init__(self, k_min: float = 0.5, k_max: float = 3.0):
        self.k_min = k_min
        self.k_max = k_max

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        return {"k": float(rng.uniform(self.k_min, self.k_max))}

    def baseline_time(self, u: Tensor, params: Dict[str, float]) -> Tensor:
        k = params["k"]
        arg = (-torch.log(u)).clamp(max=36.0)
        return arg.pow(1.0 / k)

    @property
    def name(self) -> str:
        return "weibull"


class LogNormalAFT(AFTBaselineHazard):
    """Log-normal AFT baseline with scale sigma=1.

    Baseline time: ``T_0 = exp(mu - Phi^(-1)(U))``
    Final time: ``T = T_0 * exp(-beta * y)``

    Parameters
    ----------
    mu_min : float, default=-2.0
    mu_max : float, default=2.0
    """

    def __init__(self, mu_min: float = -2.0, mu_max: float = 2.0):
        self.mu_min = mu_min
        self.mu_max = mu_max

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        return {"mu": float(rng.uniform(self.mu_min, self.mu_max))}

    def baseline_time(self, u: Tensor, params: Dict[str, float]) -> Tensor:
        mu = params["mu"]
        u = u.clamp(min=1e-7, max=1.0 - 1e-7)
        return torch.exp(mu - torch.special.ndtri(u))

    @property
    def name(self) -> str:
        return "lognormal"


class LogLogisticAFT(AFTBaselineHazard):
    """Log-logistic AFT baseline with scale fixed at 1.

    Baseline time: ``T_0 = (1/U - 1)^(1/beta)``
    Final time: ``T = T_0 * exp(-beta * y)``

    Parameters
    ----------
    beta_min : float, default=0.5
    beta_max : float, default=3.0
    """

    def __init__(self, beta_min: float = 0.5, beta_max: float = 3.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        return {"beta": float(rng.uniform(self.beta_min, self.beta_max))}

    def baseline_time(self, u: Tensor, params: Dict[str, float]) -> Tensor:
        beta = params["beta"]
        u = u.clamp(min=1e-7, max=1.0 - 1e-7)
        return (1.0 / u - 1.0).pow(1.0 / beta)

    @property
    def name(self) -> str:
        return "loglogistic"


class AcceleratedFailureTimeSampler:
    """Convert continuous target into event times via AFT model.

    For each observation: ``T = T_0 * exp(-beta * y)`` where ``T_0`` is
    sampled from the baseline time distribution.

    Parameters
    ----------
    baseline_pool : dict
        Mapping from name (str) to :class:`AFTBaselineHazard` instance.
    beta : float, default=1.0
        Multiplier for acceleration: ``T = T_0 * exp(-beta * y)``.
    baseline_mode : str, default="mix"
        ``"mix"`` randomly selects a baseline per dataset, or a fixed name.
    max_time : float, default=1e30
        Numerical safety maximum for raw times.  The model-facing horizon is
        set later by per-task standardized log-time scaling.
    u_eps : float, default=1e-6
        Epsilon for clipping uniform samples away from 0 and 1.
    """

    def __init__(
        self,
        baseline_pool: Dict[str, AFTBaselineHazard],
        beta: float = 1.0,
        baseline_mode: str = "mix",
        max_time: float = DEFAULT_RAW_TIME_MAX,
        u_eps: float = 1e-6,
    ):
        self.baseline_pool = baseline_pool
        self.beta = beta
        self.baseline_mode = baseline_mode
        self.max_time = max_time
        self.u_eps = u_eps

    def sample(
        self,
        y: Tensor,
        baseline_name: str,
        baseline_params: Dict[str, float],
        rng: np.random.Generator,
        device: str = "cpu",
        censor_scale: float = 1.0,
        censoring_strategy: str = "uniform_scale",
        target_event_rate: float | None = None,
        calibration_eps: float = 1e-12,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if baseline_name == "mix":
            names = list(self.baseline_pool.keys())
            baseline_name = names[int(rng.integers(0, len(names)))]

        baseline = self.baseline_pool[baseline_name]

        u = torch.rand(y.shape, device=device)
        u = u.clamp(min=self.u_eps, max=1.0 - self.u_eps)
        t0 = baseline.baseline_time(u, baseline_params)
        t_event = t0 * torch.exp(-self.beta * y)
        t_event = _finite_positive_time(t_event, self.max_time)

        # Generate independent base censoring times (no covariate effect)
        u_c = torch.rand(y.shape, device=device)
        u_c = u_c.clamp(min=self.u_eps, max=1.0 - self.u_eps)
        c_base = baseline.baseline_time(u_c, baseline_params)
        c_base = _finite_positive_time(c_base, self.max_time)

        if censoring_strategy == "target_event_rate":
            assert target_event_rate is not None, (
                "target_event_rate is required when censoring_strategy='target_event_rate'"
            )
            censor_scale, _diag = calibrate_censor_scale_by_quantile(
                t_event, c_base, target_event_rate, eps=calibration_eps,
            )
        elif censoring_strategy != "uniform_scale":
            raise ValueError(
                f"Unknown censoring_strategy '{censoring_strategy}'. "
                "Options: 'target_event_rate', 'uniform_scale'."
            )

        c = c_base * censor_scale
        c = _finite_positive_time(c, self.max_time)

        t_obs = torch.minimum(t_event, c)
        delta = (t_event < c).float()

        return t_obs, delta, t_event


class SurvivalSCMPrior(SCMPrior):
    """SCM-based prior that generates survival (time-to-event) datasets.

    Identical to :class:`RegressionSCMPrior` in SCM logic, but converts the
    continuous target ``y`` into event times ``(t, delta)`` via a proportional
    hazard model or an accelerated failure time model.

    Parameters
    ----------
    See :class:`tabicl.prior._dataset.SCMPrior` for all base parameters.

    model_type : str, default="ph"
        ``"ph"`` for proportional hazard, ``"aft"`` for accelerated failure time.
    beta : float, default=1.0
        PH: ``log_risk = beta * y``.  AFT: ``T = T_0 * exp(-beta * y)``.
    baseline_types : list of str, default=["weibull", "gompertz", "loglogistic", "lognormal"]
        Which baseline hazards to include in the pool. Gompertz is ignored in AFT mode.
    baseline_mode : str, default="mix"
        ``"mix"`` randomly selects a baseline per dataset, or a fixed name
        like ``"weibull"``.
    max_time : float, default=1e30
        Numerical safety maximum for raw event/censoring times.  The
        model-facing horizon is set later by standardized log-time scaling.
    u_eps : float, default=1e-6
        Epsilon for clipping uniform samples away from 0 and 1.
    min_censor_scale : float, default=1.0
        Minimum censoring time scale factor, sampled per GP group.
    max_censor_scale : float, default=5.0
        Maximum censoring time scale factor, sampled per GP group.
    min_event_rate: float = 0.40
        Minimum acceptable fraction of observed events per dataset.
    max_event_rate: float = 1.0
        Maximum acceptable fraction of observed events per dataset.
    """

    def __init__(self, *args, model_type: str = "ph", beta: float = 1.0,
                 baseline_types=None, baseline_mode: str = "mix",
                 max_time: float = DEFAULT_RAW_TIME_MAX, u_eps: float = 1e-6,
                 min_censor_scale: float = 1.0, max_censor_scale: float = 5.0,
                 min_event_rate: float = 0.40, max_event_rate: float = 0.90,
                 censoring_strategy: str = "target_event_rate",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.beta = beta
        self.baseline_types = baseline_types or ["weibull", "gompertz", "loglogistic", "lognormal"]
        self.baseline_mode = baseline_mode
        self.max_time = max_time
        self.u_eps = u_eps
        self.min_censor_scale = min_censor_scale
        self.max_censor_scale = max_censor_scale
        self.min_event_rate = min_event_rate
        self.max_event_rate = max_event_rate
        if censoring_strategy not in ("target_event_rate", "uniform_scale"):
            raise ValueError(
                f"Unknown censoring_strategy '{censoring_strategy}'. "
                "Options: 'target_event_rate', 'uniform_scale'."
            )
        self.censoring_strategy = censoring_strategy
        if model_type == "ph":
            self._setup_ph_baselines()
        elif model_type == "aft":
            self._setup_aft_baselines()
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Options: 'ph', 'aft'.")

    def _setup_ph_baselines(self):
        self.baseline_pool: Dict[str, BaselineHazard] = {}
        if "weibull" in self.baseline_types:
            self.baseline_pool["weibull"] = WeibullHazard()
        if "gompertz" in self.baseline_types:
            self.baseline_pool["gompertz"] = GompertzHazard()
        if "loglogistic" in self.baseline_types:
            self.baseline_pool["loglogistic"] = LogLogisticHazard()
        if "lognormal" in self.baseline_types:
            self.baseline_pool["lognormal"] = LogNormalHazard()
        if not self.baseline_pool:
            raise ValueError(f"No valid baseline types in {self.baseline_types}")
        self.sampler = ProportionalHazardSampler(
            baseline_pool=self.baseline_pool,
            beta=self.beta,
            baseline_mode=self.baseline_mode,
            max_time=self.max_time,
            u_eps=self.u_eps,
        )

    def _setup_aft_baselines(self):
        self.baseline_pool: Dict[str, AFTBaselineHazard] = {}
        if "weibull" in self.baseline_types:
            self.baseline_pool["weibull"] = WeibullAFT()
        if "loglogistic" in self.baseline_types:
            self.baseline_pool["loglogistic"] = LogLogisticAFT()
        if "lognormal" in self.baseline_types:
            self.baseline_pool["lognormal"] = LogNormalAFT()
        if not self.baseline_pool:
            raise ValueError(f"No valid AFT baseline types in {self.baseline_types}")
        self.sampler = AcceleratedFailureTimeSampler(
            baseline_pool=self.baseline_pool,
            beta=self.beta,
            baseline_mode=self.baseline_mode,
            max_time=self.max_time,
            u_eps=self.u_eps,
        )

    @staticmethod
    def _regression_sanity_check(y: Tensor, train_size: int, min_std: float = 1e-6) -> bool:
        if not torch.isfinite(y).all():
            return False
        y_train = y[:, :train_size]
        y_test = y[:, train_size:]
        if (y_train.numel() > 1 and y_train.std() < min_std) or (y_test.numel() > 1 and y_test.std() < min_std):
            return False
        return True

    @staticmethod
    def _survival_sanity_check(t: Tensor, delta: Tensor, min_time: float = 1e-6,
                               min_event_rate: float = 0.05, max_event_rate: float = 0.98,
                               calibrating: bool = False) -> bool:
        if not torch.isfinite(t).all():
            return False
        if (t <= min_time).any():
            return False
        if t.std() < min_time:
            return False
        event_rate = delta.float().mean().item()
        if calibrating:
            # Under target_event_rate strategy, the calibration chooses the
            # scale to hit the requested rate — use broad hard bounds only.
            if event_rate < 0.05 or event_rate > 0.98:
                return False
        else:
            if event_rate < min_event_rate or event_rate > max_event_rate:
                return False
        return True

    @torch.no_grad()
    def generate_dataset(self, params: Dict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        params = {**params, "num_classes": 0}

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        max_attempts = 5000
        for _ in range(max_attempts):
            X, y = prior_cls(**params)()
            X, y = Reg2Cls(params)(X, y)

            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            X, d = self.delete_unique_features(X, d)
            if (d > 0).all() and self._regression_sanity_check(y, params["train_size"]):
                y_flat = y.squeeze(0)
                t, delta, t_event = self.sampler.sample(
                    y=y_flat,
                    baseline_name=params["baseline_type"],
                    baseline_params=params["baseline_params"],
                    rng=params["_rng"],
                    device=self.device,
                    censor_scale=params["censor_scale"],
                    censoring_strategy=self.censoring_strategy,
                    target_event_rate=params.get("target_event_rate"),
                )

                X_out = X.squeeze(0)
                d_out = d.squeeze(0)

                if self._survival_sanity_check(
                    t, delta,
                    min_event_rate=params["min_event_rate"],
                    max_event_rate=params["max_event_rate"],
                    calibrating=(self.censoring_strategy == "target_event_rate"),
                ):
                    return X_out, t, delta, t_event, d_out

        raise RuntimeError(
            f"SurvivalSCMPrior failed to generate valid dataset after {max_attempts} total attempts. "\
            f"params: prior_type={params.get('prior_type')}, "\
            f"seq_len={params.get('seq_len')}, "\
            f"baseline_type={params.get('baseline_type')}, "\
            f"min_event_rate={params.get('min_event_rate')}, "\
            f"max_event_rate={params.get('max_event_rate')}"\
        )

    @torch.no_grad()
    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = batch_size or self.batch_size

        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        param_list = []
        global_seq_len = None
        global_train_size = None

        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
            )
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

        rng = np.random.default_rng()

        for gp_idx in range(num_gps):
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()
            group_baseline_type = self.baseline_mode
            if group_baseline_type == "mix":
                names = list(self.baseline_pool.keys())
                group_baseline_type = names[int(rng.integers(0, len(names)))]
            group_baseline_params = self.baseline_pool[group_baseline_type].sample_params(rng)
            group_censor_scale = float(np.random.uniform(self.min_censor_scale, self.max_censor_scale))

            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
                )
                gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            for subgp_idx in range(num_subgps_in_gp):
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                subgp_prior_type = self.get_prior()
                subgp_num_features = round(float(np.random.uniform(self.min_features, gp_max_features)))
                subgp_sampled_hp = {k: v() if callable(v) else v for k, v in group_sampled_hp.items()}

                for ds_idx in range(actual_subgp_size):
                    if np.random.random() > 0.5:
                        ds_num_classes = np.random.randint(2, self.max_classes + 1)
                    else:
                        ds_num_classes = 2

                    target_event_rate = None
                    if self.censoring_strategy == "target_event_rate":
                        target_event_rate = float(
                            np.random.uniform(self.min_event_rate, self.max_event_rate)
                        )

                    params = {
                        **self.fixed_hp,
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
                        **subgp_sampled_hp,
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "device": self.device,
                        "baseline_type": group_baseline_type,
                        "baseline_params": group_baseline_params,
                        "censor_scale": group_censor_scale,
                        "target_event_rate": target_event_rate,
                        "min_event_rate": self.min_event_rate,
                        "max_event_rate": self.max_event_rate,
                        "_rng": rng,
                    }
                    param_list.append(params)

        if self.n_jobs > 1 and self.device == "cpu":
            with joblib.parallel_config(
                n_jobs=self.n_jobs, backend="threading", prefer="threads"
            ):
                results = joblib.Parallel()(joblib.delayed(self.generate_dataset)(params) for params in param_list)
        else:
            results = [self.generate_dataset(params) for params in param_list]

        X_list, t_list, delta_list, t_event_list, d_list = zip(*results)

        if self.seq_len_per_gp:
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            t = nested_tensor([ti.to(self.device) for ti in t_list], device=self.device)
            delta = nested_tensor([de.to(self.device) for de in delta_list], device=self.device)
            t_event = nested_tensor([te.to(self.device) for te in t_event_list], device=self.device)
        else:
            X = torch.stack(X_list).to(self.device)
            t = torch.stack(t_list).to(self.device)
            delta = torch.stack(delta_list).to(self.device)
            t_event = torch.stack(t_event_list).to(self.device)

        d = torch.stack(d_list).to(self.device)
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
        )

        return X, t, delta, t_event, d, seq_lens, train_sizes
