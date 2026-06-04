from __future__ import annotations

import time
import json
import warnings
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from tabicl.prior._dataset import DummyPrior, Prior, DisablePrinting
from tabicl.prior._genload import (
    dense2sparse,
    sparse2dense,
    SliceNestedTensor,
    cat_slice_nested_tensors,
)
from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from tabicl.prior._survival import DEFAULT_RAW_TIME_MAX, SurvivalSCMPrior

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


class SurvivalPriorDataset(IterableDataset):
    """Infinite iterator over synthetic tabular **survival** datasets.

    Uses the same hierarchical meta-distribution as the TabICL prior, but
    converts the continuous regression target into event times ``(t, delta)``
    via a proportional hazard model.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets per batch.

    batch_size_per_gp : int, default=4
        Datasets per group (share similar characteristics).

    batch_size_per_subgp : int, optional
        Datasets per subgroup (share causal structure).  Defaults to
        ``batch_size_per_gp``.

    min_features : int, default=2
        Minimum number of features per dataset.

    max_features : int, default=100
        Maximum number of features per dataset.

    max_classes : int, default=10
        Ignored for survival (targets are continuous event times).

    min_seq_len : int, optional
        Minimum samples per dataset.  If None, uses ``max_seq_len``.

    max_seq_len : int, default=4096
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        Sample sequence length from a log-uniform distribution.

    seq_len_per_gp : bool, default=False
        Sample sequence length per group (variable-length datasets).

    min_train_size : int or float, default=1.0
        Train/test split lower bound.

    max_train_size : int or float, default=1.0
        Train/test split upper bound.

    replay_small : bool, default=False
        Occasionally sample smaller sequence lengths for robustness.

    prior_type : str, default="mlp_scm"
        ``"mlp_scm"``, ``"tree_scm"``, ``"mix_scm"``, or ``"dummy"``.

    model_type : str, default="ph"
        ``"ph"`` for proportional hazard, ``"aft"`` for accelerated failure time,
        ``"mix"`` samples PH/aft with equal probability per GP group.

    beta : float, default=1.0
        PH: log relative risk ``beta * y``. AFT: time scaling ``exp(-beta * y)``.
        Only used when ``beta_sampling="fixed"``.

    beta_sampling : str, default="fixed"
        ``"fixed"`` uses ``beta``. ``"log_uniform"`` samples per GP group.

    min_beta : float, default=0.25
    max_beta : float, default=2.0

    baseline_param_prior : str, default="current"
        ``"current"`` or ``"broad"`` — broader parameter ranges.

    time_scale_sampling : str, default="fixed"
        ``"fixed"`` or ``"log_uniform"`` — per-GP time scale multiplier.

    min_time_scale : float, default=0.2
    max_time_scale : float, default=5.0

    baseline_types : list of str, default=["weibull", "gompertz", "loglogistic", "lognormal"]
        Baseline hazard distributions in the pool. Gompertz is ignored in AFT mode.

    baseline_mode : str, default="mix"
        ``"mix"`` randomly selects a baseline per dataset with equal probability,
        or a fixed name like ``"weibull"``.

    max_time : float, default=1e30
        Numerical safety maximum for raw event/censoring times.  Model-facing
        horizons are handled by per-task standardized log-time scaling.

    u_eps : float, default=1e-6
        Epsilon for clipping uniform samples away from 0 and 1. Prevents
        ``log(0)`` and other boundary singularities.

    min_censor_scale : float, default=1.0
        Minimum censoring time scale factor, sampled per GP group from
        ``U[min_censor_scale, max_censor_scale]``.

    max_censor_scale : float, default=5.0
        Maximum censoring time scale factor.

    min_event_rate: float = 0.40
        Minimum fraction of observed events required for a valid dataset.
        Datasets with fewer events are rejected and regenerated.

    max_event_rate: float = 0.90
        Maximum fraction of observed events allowed.  Under
        ``target_event_rate``, defines the upper bound of the sampled
        target range rather than a rejection threshold.

    censoring_strategy : str, default="target_event_rate"
        ``"target_event_rate"`` calibrates a per-dataset censoring scale
        from the ratio ``t_event / c_base`` to achieve a target event rate
        sampled from ``U[min_event_rate, max_event_rate]``.
        ``"uniform_scale"`` uses ``censor_scale ~ U[min_censor_scale, max_censor_scale]``
        as before.

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed hyperparameters for SCM priors.

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Sampled hyperparameters for SCM priors.

    n_jobs : int, default=1
        Parallel jobs (-1 uses all processors).

    num_threads_per_generate : int, default=1
        Threads per generation job.

    device : str, default="cpu"
        Computation device.
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 4096,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 1.0,
        max_train_size: Union[int, float] = 1.0,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        model_type: str = "ph",
        beta: float = 1.0,
        beta_sampling: str = "fixed",
        min_beta: float = 0.25,
        max_beta: float = 2.0,
        baseline_param_prior: str = "current",
        time_scale_sampling: str = "fixed",
        min_time_scale: float = 0.2,
        max_time_scale: float = 5.0,
        baseline_types: Optional[List[str]] = None,
        baseline_mode: str = "mix",
        max_time: float = DEFAULT_RAW_TIME_MAX,
        u_eps: float = 1e-6,
        min_censor_scale: float = 1.0,
        max_censor_scale: float = 5.0,
        min_event_rate: float = 0.40,
        max_event_rate: float = 0.90,
        censoring_strategy: str = "target_event_rate",
        scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = 1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        default_kwargs = dict(
            batch_size=batch_size,
            batch_size_per_gp=batch_size_per_gp,
            batch_size_per_subgp=batch_size_per_subgp,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            seq_len_per_gp=seq_len_per_gp,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
            prior_type=prior_type,
            n_jobs=n_jobs,
            num_threads_per_generate=num_threads_per_generate,
            device=device,
        )

        self.baseline_types = baseline_types or ["weibull", "gompertz", "loglogistic", "lognormal"]
        self.baseline_mode = baseline_mode
        self.model_type = model_type
        self.beta = beta
        self.beta_sampling = beta_sampling
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.baseline_param_prior = baseline_param_prior
        self.time_scale_sampling = time_scale_sampling
        self.min_time_scale = min_time_scale
        self.max_time_scale = max_time_scale
        self.max_time = max_time
        self.u_eps = u_eps
        self.min_censor_scale = min_censor_scale
        self.max_censor_scale = max_censor_scale
        self.min_event_rate = min_event_rate
        self.max_event_rate = max_event_rate
        self.censoring_strategy = censoring_strategy

        if prior_type == "dummy":
            self.prior = DummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
            )
        elif prior_type in ("mlp_scm", "tree_scm", "mix_scm"):
            self.prior = SurvivalSCMPrior(
                **default_kwargs,
                model_type=model_type,
                beta=beta,
                beta_sampling=beta_sampling,
                min_beta=min_beta,
                max_beta=max_beta,
                baseline_param_prior=baseline_param_prior,
                time_scale_sampling=time_scale_sampling,
                min_time_scale=min_time_scale,
                max_time_scale=max_time_scale,
                baseline_types=self.baseline_types,
                baseline_mode=baseline_mode,
                max_time=max_time,
                u_eps=u_eps,
                min_censor_scale=min_censor_scale,
                max_censor_scale=max_censor_scale,
                min_event_rate=min_event_rate,
                max_event_rate=max_event_rate,
                censoring_strategy=censoring_strategy,
                fixed_hp=scm_fixed_hp,
                sampled_hp=scm_sampled_hp,
            )
        else:
            raise ValueError(
                f"Unknown prior_type '{prior_type}'. "
                "Options: 'mlp_scm', 'tree_scm', 'mix_scm', 'dummy'."
            )

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type

    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.prior.get_batch(batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        return (
            f"SurvivalPriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  model_type: {self.model_type}\n"
            f"  beta: {self.beta} (sampling: {self.beta_sampling})\n"
            f"  baseline_param_prior: {self.baseline_param_prior}\n"
            f"  time_scale_sampling: {self.time_scale_sampling}\n"
            f"  baseline_types: {self.baseline_types}\n"
            f"  baseline_mode: {self.baseline_mode}\n"
            f"  max_time: {self.max_time}\n"
            f"  u_eps: {self.u_eps}\n"
            f"  censor_scale: {self.min_censor_scale} - {self.max_censor_scale}\n"
            f"  censoring_strategy: {self.censoring_strategy}\n"
            f"  event_rate: {self.min_event_rate} - {self.max_event_rate}\n"
            f"  device: {self.device}\n"
            f")"
        )


class SaveSurvivalPriorDataset:
    """Generate and save batches of survival prior datasets to disk.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments (see CLI for details).
    """

    def __init__(self, args):
        self.args = args
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_metadata()

        self.prior = SurvivalPriorDataset(
            batch_size=self.args.batch_size,
            batch_size_per_gp=self.args.batch_size_per_gp,
            min_features=self.args.min_features,
            max_features=self.args.max_features,
            max_classes=self.args.max_classes,
            min_seq_len=self.args.min_seq_len,
            max_seq_len=self.args.max_seq_len,
            log_seq_len=self.args.log_seq_len,
            seq_len_per_gp=self.args.seq_len_per_gp,
            min_train_size=self.args.min_train_size,
            max_train_size=self.args.max_train_size,
            replay_small=self.args.replay_small,
            prior_type=self.args.prior_type,
            model_type=self.args.model_type,
            beta=self.args.beta,
            beta_sampling=self.args.beta_sampling,
            min_beta=self.args.min_beta,
            max_beta=self.args.max_beta,
            baseline_param_prior=self.args.baseline_param_prior,
            time_scale_sampling=self.args.time_scale_sampling,
            min_time_scale=self.args.min_time_scale,
            max_time_scale=self.args.max_time_scale,
            baseline_types=self.args.baseline_types,
            baseline_mode=self.args.baseline_mode,
            max_time=self.args.max_time,
            u_eps=self.args.u_eps,
            min_censor_scale=self.args.min_censor_scale,
            max_censor_scale=self.args.max_censor_scale,
            min_event_rate=self.args.min_event_rate,
            max_event_rate=self.args.max_event_rate,
            censoring_strategy=self.args.censoring_strategy,
            scm_fixed_hp=DEFAULT_FIXED_HP,
            scm_sampled_hp=DEFAULT_SAMPLED_HP,
            n_jobs=self.args.n_jobs,
            num_threads_per_generate=self.args.num_threads_per_generate,
            device=self.args.device,
        )
        print(self.prior)

    def _save_metadata(self):
        metadata = {
            "model_type": self.args.model_type,
            "prior_type": self.args.prior_type,
            "batch_size": self.args.batch_size,
            "batch_size_per_gp": self.args.batch_size_per_gp,
            "min_seq_len": self.args.min_seq_len,
            "max_seq_len": self.args.max_seq_len,
            "log_seq_len": self.args.log_seq_len,
            "seq_len_per_gp": self.args.seq_len_per_gp,
            "min_features": self.args.min_features,
            "max_features": self.args.max_features,
            "min_train_size": self.args.min_train_size,
            "max_train_size": self.args.max_train_size,
            "replay_small": self.args.replay_small,
            "beta": self.args.beta,
            "beta_sampling": self.args.beta_sampling,
            "min_beta": self.args.min_beta,
            "max_beta": self.args.max_beta,
            "baseline_param_prior": self.args.baseline_param_prior,
            "time_scale_sampling": self.args.time_scale_sampling,
            "min_time_scale": self.args.min_time_scale,
            "max_time_scale": self.args.max_time_scale,
            "baseline_types": self.args.baseline_types,
            "baseline_mode": self.args.baseline_mode,
            "max_time": self.args.max_time,
            "u_eps": self.args.u_eps,
            "min_censor_scale": self.args.min_censor_scale,
            "max_censor_scale": self.args.max_censor_scale,
            "min_event_rate": self.args.min_event_rate,
            "max_event_rate": self.args.max_event_rate,
            "censoring_strategy": self.args.censoring_strategy,
            "calibration_scope": getattr(self.args, "censor_calibration_scope", "dataset"),
        }
        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_batch_sparse(self, batch_idx, X, t, delta, t_event, d, seq_lens, train_sizes):
        if self.args.seq_len_per_gp:
            B = len(d)
        else:
            B, T, H = X.shape
            X = dense2sparse(X.view(-1, H), d.repeat_interleave(T), dtype=torch.float32)

        batch_file = self.save_dir / f"batch_{batch_idx:06d}.pt"
        temp_file = self.save_dir / f"batch_{batch_idx:06d}.pt.tmp"
        torch.save(
            {
                "X": X,
                "t": t,
                "delta": delta,
                "t_event": t_event,
                "d": d,
                "seq_lens": seq_lens,
                "train_sizes": train_sizes,
                "batch_size": B,
            },
            temp_file,
        )
        temp_file.replace(batch_file)

    def run(self):
        print(f"Save directory: {self.save_dir}")
        print(f"Generating {self.args.num_batches} batches starting from index {self.args.resume_from}")

        for batch_idx in tqdm(
            range(self.args.resume_from, self.args.resume_from + self.args.num_batches),
            desc="Generating survival batches",
        ):
            X, t, delta, t_event, d, seq_lens, train_sizes = self.prior.get_batch()
            X = X.cpu()
            t = t.cpu()
            delta = delta.cpu()
            t_event = t_event.cpu()
            d = d.cpu()
            seq_lens = seq_lens.cpu()
            train_sizes = train_sizes.cpu()
            self._save_batch_sparse(batch_idx, X, t, delta, t_event, d, seq_lens, train_sizes)


class LoadSurvivalPriorDataset(IterableDataset):
    """Load pre-generated survival prior datasets for training.

    Compatible with data saved by :class:`SaveSurvivalPriorDataset`.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing batch files.

    batch_size : int, default=512
        Number of datasets per iteration.

    ddp_world_size : int, default=1
        Total number of distributed processes.

    ddp_rank : int, default=0
        Rank of the current process.

    start_from : int, default=0
        Batch index to start from.

    max_batches : int, optional
        Maximum batches to load.  If None, load indefinitely.

    timeout : int, default=60
        Maximum time (seconds) to wait for a batch file.

    delete_after_load : bool, default=False
        Delete batch files after loading.

    device : str, default="cpu"
        Device to load tensors to.

    censor_calibration_scope : str, default="dataset"
        Expected calibration scope.  Rejects on-disk data whose metadata
        is missing or incompatible when ``"context"`` is requested.
    """

    def __init__(
        self,
        data_dir,
        batch_size=512,
        ddp_world_size=1,
        ddp_rank=0,
        start_from=0,
        max_batches=None,
        timeout=60,
        delete_after_load=False,
        censor_calibration_scope="dataset",
        device="cpu",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.current_idx = ddp_rank + start_from
        self.start_from = start_from
        self.max_batches = max_batches
        self.timeout = timeout
        self.delete_after_load = delete_after_load
        self.device = device
        self.censor_calibration_scope = censor_calibration_scope

        self.metadata = None
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load or parse metadata.json: {e}")

        # Reject disk-based priors when context calibration is required
        # but the on-disk metadata is missing or incompatible.
        if censor_calibration_scope == "context":
            if self.metadata is None:
                raise RuntimeError(
                    "censor_calibration_scope='context' requires metadata.json "
                    f"in the prior directory ({self.data_dir}), but none was found."
                )
            saved_scope = self.metadata.get("calibration_scope", "dataset")
            if saved_scope != "context":
                raise RuntimeError(
                    f"Disk prior was generated with calibration_scope='{saved_scope}', "
                    f"but censor_calibration_scope='context' was requested. "
                    f"Regenerate the data with --censor_calibration_scope context."
                )

        self.buffer_X = None
        self.buffer_t = None
        self.buffer_delta = None
        self.buffer_t_event = None
        self.buffer_d = None
        self.buffer_seq_lens = None
        self.buffer_train_sizes = None
        self.buffer_size = 0
        self._batches_loaded = 0

    def __iter__(self):
        return self

    def _load_batch_file(self):
        batch_file = self.data_dir / f"batch_{self.current_idx:06d}.pt"

        wait_time = 0
        while not batch_file.exists():
            if wait_time >= self.timeout:
                raise RuntimeError(f"Timeout waiting for batch file {batch_file}")
            time.sleep(5)
            wait_time += 5

        batch = torch.load(batch_file, map_location=self.device, weights_only=True)
        X = batch["X"]
        t = batch["t"]
        delta = batch["delta"]
        t_event = batch.get("t_event")
        d = batch["d"]
        seq_lens = batch["seq_lens"]
        train_sizes = batch["train_sizes"]
        file_batch_size = batch["batch_size"]

        if X.is_nested:
            X = SliceNestedTensor(X)
            t = SliceNestedTensor(t)
            delta = SliceNestedTensor(delta)
            if t_event is not None and t_event.is_nested:
                t_event = SliceNestedTensor(t_event)
        else:
            X = sparse2dense(X, d.repeat_interleave(seq_lens[0]), dtype=torch.float32).view(
                file_batch_size, seq_lens[0], -1
            )

        if self.delete_after_load and batch_file.exists():
            batch_file.unlink()

        self.current_idx += self.ddp_world_size

        return X, t, delta, t_event, d, seq_lens, train_sizes, file_batch_size

    def __next__(self):
        while self.buffer_size < self.batch_size:
            if self.max_batches is not None and self._batches_loaded >= self.max_batches:
                break
            try:
                X, t, delta, t_event, d, seq_lens, train_sizes, file_batch_size = self._load_batch_file()
                self._batches_loaded += 1
                if self.buffer_X is None:
                    self.buffer_X = X
                    self.buffer_t = t
                    self.buffer_delta = delta
                    self.buffer_t_event = t_event
                    self.buffer_d = d
                    self.buffer_seq_lens = seq_lens
                    self.buffer_train_sizes = train_sizes
                    self.buffer_size = file_batch_size
                else:
                    if isinstance(X, SliceNestedTensor):
                        self.buffer_X = cat_slice_nested_tensors([self.buffer_X, X], dim=0)
                        self.buffer_t = cat_slice_nested_tensors([self.buffer_t, t], dim=0)
                        self.buffer_delta = cat_slice_nested_tensors([self.buffer_delta, delta], dim=0)
                        if t_event is not None:
                            self.buffer_t_event = cat_slice_nested_tensors([self.buffer_t_event, t_event], dim=0)
                    else:
                        self.buffer_X = torch.cat([self.buffer_X, X], dim=0)
                        self.buffer_t = torch.cat([self.buffer_t, t], dim=0)
                        self.buffer_delta = torch.cat([self.buffer_delta, delta], dim=0)
                        if t_event is not None:
                            self.buffer_t_event = torch.cat([self.buffer_t_event, t_event], dim=0)
                    self.buffer_d = torch.cat([self.buffer_d, d], dim=0)
                    self.buffer_seq_lens = torch.cat([self.buffer_seq_lens, seq_lens], dim=0)
                    self.buffer_train_sizes = torch.cat([self.buffer_train_sizes, train_sizes], dim=0)
                    self.buffer_size += file_batch_size
            except Exception as e:
                if self.max_batches is not None:
                    # Finite mode — give up after exhausting all files
                    print(f"Warning: Could not load more files: {str(e)}")
                    break
                else:
                    # Infinite mode — cycle back to start
                    if self.current_idx >= self.ddp_rank + 1000000:
                        print("Warning: cycled past 1M batches; if this is intentional, increase start_from")
                    self.current_idx = self.ddp_rank + self.start_from
                    continue

        if self.buffer_size == 0:
            raise StopIteration

        output_size = min(self.batch_size, self.buffer_size)
        X_out = self.buffer_X[:output_size]
        t_out = self.buffer_t[:output_size]
        delta_out = self.buffer_delta[:output_size]
        t_event_out = self.buffer_t_event[:output_size] if self.buffer_t_event is not None else None
        d_out = self.buffer_d[:output_size]
        seq_lens_out = self.buffer_seq_lens[:output_size]
        train_sizes_out = self.buffer_train_sizes[:output_size]

        if output_size < self.buffer_size:
            self.buffer_X = self.buffer_X[output_size:]
            self.buffer_t = self.buffer_t[output_size:]
            self.buffer_delta = self.buffer_delta[output_size:]
            if self.buffer_t_event is not None:
                self.buffer_t_event = self.buffer_t_event[output_size:]
            self.buffer_d = self.buffer_d[output_size:]
            self.buffer_seq_lens = self.buffer_seq_lens[output_size:]
            self.buffer_train_sizes = self.buffer_train_sizes[output_size:]
            self.buffer_size -= output_size
        else:
            self.buffer_X = None
            self.buffer_t = None
            self.buffer_delta = None
            self.buffer_t_event = None
            self.buffer_d = None
            self.buffer_seq_lens = None
            self.buffer_train_sizes = None
            self.buffer_size = 0

        if isinstance(X_out, SliceNestedTensor):
            X_out = X_out.nested_tensor
            t_out = t_out.nested_tensor
            delta_out = delta_out.nested_tensor
            if t_event_out is not None:
                t_event_out = t_event_out.nested_tensor

        return X_out, t_out, delta_out, t_event_out, d_out, seq_lens_out, train_sizes_out

    def __repr__(self) -> str:
        repr_str = (
            f"LoadSurvivalPriorDataset(\n"
            f"  data_dir: {self.data_dir}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  ddp_world_size: {self.ddp_world_size}\n"
            f"  ddp_rank: {self.ddp_rank}\n"
            f"  start_from: {self.current_idx - self.ddp_rank}\n"
            f"  max_batches: {self.max_batches or 'Infinite'}\n"
            f"  timeout: {self.timeout}\n"
            f"  delete_after_load: {self.delete_after_load}\n"
            f"  device: {self.device}\n"
        )
        if self.metadata:
            repr_str += "  Loaded Metadata:\n"
            repr_str += f"    prior_type: {self.metadata.get('prior_type', 'N/A')}\n"
            repr_str += f"    model_type: {self.metadata.get('model_type', 'N/A')}\n"
            repr_str += f"    batch_size (generated): {self.metadata.get('batch_size', 'N/A')}\n"
            repr_str += f"    batch_size_per_gp: {self.metadata.get('batch_size_per_gp', 'N/A')}\n"
            repr_str += f"    min features: {self.metadata.get('min_features', 'N/A')}\n"
            repr_str += f"    max features: {self.metadata.get('max_features', 'N/A')}\n"
            repr_str += f"    seq_len: {self.metadata.get('min_seq_len', 'N/A') or 'None'} - {self.metadata.get('max_seq_len', 'N/A')}\n"
            repr_str += f"    log_seq_len: {self.metadata.get('log_seq_len', 'N/A')}\n"
            repr_str += f"    sequence length varies across groups: {self.metadata.get('seq_len_per_gp', 'N/A')}\n"
            repr_str += f"    train_size: {self.metadata.get('min_train_size', 'N/A')} - {self.metadata.get('max_train_size', 'N/A')}\n"
            repr_str += f"    replay_small: {self.metadata.get('replay_small', 'N/A')}\n"
            repr_str += f"    beta: {self.metadata.get('beta', 'N/A')}\n"
            repr_str += f"    baseline_types: {self.metadata.get('baseline_types', 'N/A')}\n"
            repr_str += f"    baseline_mode: {self.metadata.get('baseline_mode', 'N/A')}\n"
            repr_str += f"    max_time: {self.metadata.get('max_time', 'N/A')}\n"
            repr_str += f"    u_eps: {self.metadata.get('u_eps', 'N/A')}\n"
        repr_str += ")"
        return repr_str


if __name__ == "__main__":

    def _str2bool(value):
        return value.lower() == "true"

    def _train_size_type(value):
        value = float(value)
        if 0 < value < 1:
            return value
        elif value.is_integer():
            return int(value)
        else:
            raise argparse.ArgumentTypeError(
                "Train size must be either an integer (absolute position) "
                "or a float between 0 and 1 (ratio of sequence length)."
            )

    def _baseline_types_type(value):
        return value.split(",")

    parser = argparse.ArgumentParser(description="Generate survival prior datasets")
    parser.add_argument("--save_dir", type=str, default="survival_data", help="Directory to save generated data")
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of batches to generate")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this batch index")
    parser.add_argument("--batch_size", type=int, default=512, help="Total batch size")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=2, help="Minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="(Legacy, ignored — targets are event times)")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=_str2bool,
        help="Sample sequence length from log-uniform distribution",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=_str2bool,
        help="Sample sequence length independently per group",
    )
    parser.add_argument(
        "--min_train_size", type=_train_size_type, default=1.0, help="Training size (full dataset position/ratio"
    )
    parser.add_argument(
        "--max_train_size", type=_train_size_type, default=1.0, help="Training size (full dataset position/ratio"
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=_str2bool,
        help="Occasionally sample smaller sequence lengths",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="mlp_scm",
        choices=["mlp_scm", "tree_scm", "mix_scm"],
        help="Type of prior to use",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ph",
        choices=["ph", "aft", "mix"],
        help="Survival model: ph, aft, or mix (per-GP group random)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Effect size multiplier (used when beta_sampling=fixed)",
    )
    parser.add_argument(
        "--beta_sampling",
        type=str,
        default="fixed",
        choices=["fixed", "log_uniform"],
        help="Beta sampling: fixed (single value) or log_uniform (per-GP)",
    )
    parser.add_argument(
        "--min_beta", type=float, default=0.25, help="Min beta under log_uniform"
    )
    parser.add_argument(
        "--max_beta", type=float, default=2.0, help="Max beta under log_uniform"
    )
    parser.add_argument(
        "--baseline_param_prior",
        type=str,
        default="current",
        choices=["current", "broad"],
        help="Baseline parameter prior: current or broad (wider ranges)",
    )
    parser.add_argument(
        "--time_scale_sampling",
        type=str,
        default="fixed",
        choices=["fixed", "log_uniform"],
        help="Time scale sampling: fixed (1.0) or log_uniform (per-GP)",
    )
    parser.add_argument(
        "--min_time_scale", type=float, default=0.2, help="Min time scale under log_uniform"
    )
    parser.add_argument(
        "--max_time_scale", type=float, default=5.0, help="Max time scale under log_uniform"
    )
    parser.add_argument(
        "--baseline_types",
        type=_baseline_types_type,
        default="weibull,gompertz,loglogistic,lognormal",
        help="Comma-separated baseline hazard types (weibull,gompertz,loglogistic,lognormal)",
    )
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default="mix",
        choices=["mix", "weibull", "gompertz", "loglogistic", "lognormal"],
        help="Baseline selection mode",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--num_threads_per_generate", type=int, default=1, help="Threads per generation")
    parser.add_argument(
        "--max_time", type=float, default=DEFAULT_RAW_TIME_MAX,
        help="Numerical safety maximum for raw times; model-facing time is standardized later"
    )
    parser.add_argument(
        "--u_eps", type=float, default=1e-6, help="Epsilon for U clipping away from 0 and 1"
    )
    parser.add_argument(
        "--min_censor_scale", type=float, default=1.0,
        help="Minimum censoring time scale factor"
    )
    parser.add_argument(
        "--max_censor_scale", type=float, default=5.0,
        help="Maximum censoring time scale factor"
    )
    parser.add_argument(
        "--min_event_rate", type=float, default=0.40,
        help="Minimum fraction of observed events per dataset"
    )
    parser.add_argument(
        "--max_event_rate", type=float, default=0.90,
        help="Maximum fraction of observed events per dataset (target range upper bound under target_event_rate)"
    )
    parser.add_argument(
        "--censoring_strategy",
        type=str,
        default="target_event_rate",
        choices=["target_event_rate", "uniform_scale"],
        help="Censoring strategy: target_event_rate calibrates scale per dataset, uniform_scale samples censor_scale from U[min,max]",
    )
    parser.add_argument(
        "--censor_calibration_scope", type=str, default="dataset",
        choices=["dataset", "context"],
        help="'dataset' calibrates on full dataset, 'context' calibrates on context rows only"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for generation"
    )

    main_args = parser.parse_args()
    np.random.seed(main_args.np_seed)
    torch.manual_seed(main_args.torch_seed)
    saver = SaveSurvivalPriorDataset(main_args)
    saver.run()
