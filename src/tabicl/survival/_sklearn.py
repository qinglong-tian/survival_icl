"""Scikit-learn style survival inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from tabicl._sklearn.preprocessing import TransformToNumerical
from tabicl._sklearn.sklearn_utils import validate_data
from tabicl.survival._inference import TabICLSurvivalPredictor


def _as_1d_float(name: str, value, n_samples: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if n_samples is not None and array.shape[0] != n_samples:
        raise ValueError(
            f"{name} has length {array.shape[0]}, expected {n_samples}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _unpack_survival_target(
    y,
    *,
    t=None,
    delta=None,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if t is not None or delta is not None:
        if t is None or delta is None:
            raise ValueError("Pass both t and delta, or pass neither.")
        t_arr = _as_1d_float("t", t, n_samples)
        delta_arr = _as_1d_float("delta", delta, n_samples)
    elif isinstance(y, dict):
        time_key = "time" if "time" in y else "t" if "t" in y else "t_obs"
        event_key = "event" if "event" in y else "delta"
        if time_key not in y or event_key not in y:
            raise ValueError("Survival target dict must contain time/t/t_obs and event/delta.")
        t_arr = _as_1d_float(time_key, y[time_key], n_samples)
        delta_arr = _as_1d_float(event_key, y[event_key], n_samples)
    else:
        array = np.asarray(y)
        if array.dtype.names is not None:
            names = set(array.dtype.names)
            time_key = "time" if "time" in names else "t" if "t" in names else "t_obs"
            event_key = "event" if "event" in names else "delta"
            if time_key not in names or event_key not in names:
                raise ValueError(
                    "Structured survival target must contain time/t/t_obs "
                    "and event/delta fields."
                )
            t_arr = _as_1d_float(time_key, array[time_key], n_samples)
            delta_arr = _as_1d_float(event_key, array[event_key], n_samples)
        else:
            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError(
                    "Pass survival targets as y with shape (n_samples, 2), "
                    "as a structured array/dict, or via t= and delta=."
                )
            t_arr = _as_1d_float("y[:, 0]", array[:, 0], n_samples)
            delta_arr = _as_1d_float("y[:, 1]", array[:, 1], n_samples)

    if not (t_arr > 0).all():
        raise ValueError("Observed survival times must be strictly positive.")
    if not np.isin(delta_arr, [0.0, 1.0]).all():
        raise ValueError("delta/event indicators must contain only 0 or 1.")
    return t_arr, delta_arr


class TabICLSurvivalEstimator(BaseEstimator):
    """Sklearn-style wrapper for checkpoint-based TabICL survival inference.

    The estimator stores a fitted in-context support set in :meth:`fit` and
    uses it as the prompt for later query predictions. Features are converted
    to numeric values and standardized using support rows only. Survival times
    are scaled internally by the checkpoint predictor using context-only
    ``(t_obs, delta)`` information, and predictions are returned on raw time
    units.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to a modern ``km_hybrid_log`` TabICL survival checkpoint.
    device : str, default="cpu"
        Torch device for model inference.
    max_context_size : int or None, default=None
        Maximum number of support rows retained for the in-context prompt.
        When ``None`` (the default), all rows from ``fit`` are used without
        subsampling. Pass an explicit integer to cap the context set size;
        larger support sets are then deterministically subsampled.

        This limit exists because TabICL was trained with a maximum sequence
        length (context + query rows combined). For a Stage-1 checkpoint the
        training limit is ~1,024 total positions; for a Stage-3 checkpoint it
        is ~60,000. If your context set exceeds the training sequence length,
        set ``max_context_size`` to stay within it.
    query_batch_size : int, default=512
        Number of query rows evaluated per model forward pass. Lower values
        reduce peak memory; raise this if you have spare RAM/VRAM and want
        faster inference on large query sets. Most users can leave this at
        the default.
    standardize_features : bool, default=True
        Whether to z-score features using context rows only.
    random_state : int, default=42
        Random seed used when subsampling support rows.
    verbose : bool, default=False
        Whether to print feature conversion details.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        max_context_size: int | None = None,
        query_batch_size: int = 512,
        standardize_features: bool = True,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_context_size = max_context_size
        self.query_batch_size = query_batch_size
        self.standardize_features = standardize_features
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None, *, t=None, delta=None) -> "TabICLSurvivalEstimator":
        """Fit the in-context support set.

        ``y`` can be an array with columns ``[time, delta]``, a structured
        array with ``time``/``event`` or ``t``/``delta`` fields, or a dict with
        those keys. Alternatively pass ``t=`` and ``delta=`` explicitly.
        """
        if self.max_context_size is not None and self.max_context_size < 1:
            raise ValueError("max_context_size must be positive or None.")
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be positive.")

        X_valid = validate_data(self, X, y="no_validation", dtype=None, skip_check_array=True)
        n_samples = X_valid.shape[0]
        t_arr, delta_arr = _unpack_survival_target(
            y, t=t, delta=delta, n_samples=n_samples,
        )

        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X_num = np.asarray(self.X_encoder_.fit_transform(X_valid), dtype=np.float32)
        if X_num.ndim != 2:
            raise ValueError("X must be two-dimensional after numeric encoding.")
        if not np.isfinite(X_num).all():
            raise ValueError("Encoded features must contain only finite values.")

        self.n_samples_in_ = n_samples
        self.n_features_in_ = X_num.shape[1]
        self.predictor_ = TabICLSurvivalPredictor.from_checkpoint(
            self.checkpoint_path, device=self.device,
        )
        self.checkpoint_path_ = Path(self.checkpoint_path)

        if self.n_features_in_ > 100:
            raise ValueError(
                f"Stage 1 survival checkpoints were trained with at most 100 features; "
                f"got {self.n_features_in_}."
            )

        if self.standardize_features:
            self.feature_scaler_ = StandardScaler()
            X_num = self.feature_scaler_.fit_transform(X_num).astype(np.float32)
        else:
            self.feature_scaler_ = None

        if self.max_context_size is not None and n_samples > self.max_context_size:
            rng = np.random.default_rng(self.random_state)
            indices = np.sort(rng.choice(n_samples, self.max_context_size, replace=False))
            warnings.warn(
                f"Support set has {n_samples} rows; using a deterministic "
                f"subsample of {self.max_context_size} rows for the TabICL prompt.",
                UserWarning,
                stacklevel=2,
            )
        else:
            indices = np.arange(n_samples)

        self.context_indices_ = indices
        self.X_context_ = X_num[indices]
        self.t_context_ = t_arr[indices]
        self.delta_context_ = delta_arr[indices]
        return self

    def _transform_query_features(self, X) -> np.ndarray:
        check_is_fitted(self, ["X_encoder_", "X_context_", "predictor_"])
        X_valid = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)
        X_num = np.asarray(self.X_encoder_.transform(X_valid), dtype=np.float32)
        if X_num.ndim != 2:
            raise ValueError("X must be two-dimensional after numeric encoding.")
        if X_num.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_num.shape[1]} features, expected {self.n_features_in_}."
            )
        if not np.isfinite(X_num).all():
            raise ValueError("Encoded features must contain only finite values.")
        if self.feature_scaler_ is not None:
            X_num = self.feature_scaler_.transform(X_num).astype(np.float32)
        return X_num

    @staticmethod
    def _interpolate_survival(
        grid: np.ndarray,
        survival: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        curves = np.empty((survival.shape[0], times.shape[0]), dtype=np.float32)
        for row_idx, row in enumerate(survival):
            values = np.interp(times, grid, row, left=1.0, right=row[-1])
            curves[row_idx] = np.minimum.accumulate(np.clip(values, 0.0, 1.0))
        return curves

    @staticmethod
    def _condition_on_censoring(
        curves: np.ndarray,
        grid: np.ndarray,
        survival_grid: np.ndarray,
        times: np.ndarray,
        conditional_time,
    ) -> np.ndarray:
        c = _as_1d_float("conditional_time", conditional_time, curves.shape[0])
        if not (c > 0).all():
            raise ValueError("conditional_time values must be strictly positive.")
        conditioned = curves.copy()
        for row_idx, c_i in enumerate(c):
            s_c = float(
                np.interp(
                    c_i,
                    grid,
                    survival_grid[row_idx],
                    left=1.0,
                    right=survival_grid[row_idx, -1],
                )
            )
            denom = max(s_c, 1e-8)
            after = times > c_i
            conditioned[row_idx, ~after] = 1.0
            conditioned[row_idx, after] = np.clip(conditioned[row_idx, after] / denom, 0.0, 1.0)
            conditioned[row_idx] = np.minimum.accumulate(conditioned[row_idx])
        return conditioned

    def _predict_batches(self, X_query: np.ndarray, quantile_levels: Sequence[float]):
        X_context = torch.from_numpy(self.X_context_).unsqueeze(0)
        t_context = torch.from_numpy(self.t_context_).unsqueeze(0)
        delta_context = torch.from_numpy(self.delta_context_).unsqueeze(0)
        predictions = []
        for start in range(0, X_query.shape[0], self.query_batch_size):
            stop = min(start + self.query_batch_size, X_query.shape[0])
            prediction = self.predictor_.predict(
                X_context,
                t_context,
                delta_context,
                torch.from_numpy(X_query[start:stop]).unsqueeze(0),
                quantile_levels=quantile_levels,
            )
            predictions.append(prediction)
        return predictions

    def predict_survival_function(
        self,
        X,
        *,
        times: Sequence[float] | None = None,
        conditional_time: Sequence[float] | None = None,
        return_times: bool = False,
    ):
        """Estimate survival curves for query rows.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query features.
        times : sequence of float, optional
            Raw time points at which to evaluate survival. If omitted, the
            checkpoint's raw time grid fitted from the support set is used.
        conditional_time : sequence of float, optional
            Censored query times. If provided, returns
            ``S(t | T > conditional_time, X)`` without passing those times to
            the model prompt.
        return_times : bool, default=False
            If True, return ``(times, survival)``.

        Returns
        -------
        survival : ndarray of shape (n_samples, n_times)
            Survival probabilities in raw time units, or ``(times, survival)``
            when ``return_times=True``.
        """
        X_query = self._transform_query_features(X)
        predictions = self._predict_batches(X_query, quantile_levels=(0.5,))
        grid = predictions[0].raw_time_grid[0].numpy().astype(np.float64)
        if times is None:
            eval_times = grid
        else:
            eval_times = _as_1d_float("times", times).astype(np.float64)
            if not (eval_times > 0).all():
                raise ValueError("times must be strictly positive.")
            if np.any(np.diff(eval_times) < 0):
                raise ValueError("times must be sorted in nondecreasing order.")

        curves = []
        survival_grids = []
        for prediction in predictions:
            survival_grid = prediction.survival_probabilities[0].numpy().astype(np.float32)
            survival_grids.append(survival_grid)
            curves.append(self._interpolate_survival(grid, survival_grid, eval_times))
        survival = np.concatenate(curves, axis=0)

        if conditional_time is not None:
            survival_grid = np.concatenate(survival_grids, axis=0)
            survival = self._condition_on_censoring(
                survival, grid, survival_grid, eval_times, conditional_time,
            )

        if return_times:
            return eval_times, survival
        return survival

    def predict_quantiles(
        self,
        X,
        *,
        quantile_levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    ) -> np.ndarray:
        """Return raw-time event quantiles for query rows."""
        X_query = self._transform_query_features(X)
        predictions = self._predict_batches(X_query, quantile_levels=quantile_levels)
        return np.concatenate([
            prediction.raw_quantiles[0].numpy() for prediction in predictions
        ], axis=0)

    def predict(self, X) -> np.ndarray:
        """Return predicted median event time for sklearn-style ``predict``."""
        return self.predict_quantiles(X, quantile_levels=(0.5,))[:, 0]
