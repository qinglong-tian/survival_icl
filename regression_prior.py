from __future__ import annotations

import time
import json
import warnings
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from tabicl.prior._dataset import SCMPrior, Prior, DisablePrinting
from tabicl.prior._genload import (
    dense2sparse,
    sparse2dense,
    SliceNestedTensor,
    cat_slice_nested_tensors,
)
from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from tabicl.prior._reg2cls import Reg2Cls
from tabicl.prior._mlp_scm import MLPSCM
from tabicl.prior._tree_scm import TreeSCM

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


class RegressionSCMPrior(SCMPrior):
    """SCM-based prior that generates regression datasets instead of classification.

    Identical to ``SCMPrior`` except that it forces ``num_classes=0`` so that
    ``Reg2Cls`` skips discretization and returns continuous targets.  The
    class-balance sanity check is replaced with a variance-based check suited
    for regression.

    Parameters
    ----------
    See :class:`tabicl.prior._dataset.SCMPrior` for all parameters.
    """

    @staticmethod
    def _regression_sanity_check(y: Tensor, train_size: int, min_std: float = 1e-6) -> bool:
        """Verify regression targets are finite and have non-trivial variance.

        Parameters
        ----------
        y : Tensor of shape ``(1, T)``
            Targets with batch dim added by ``generate_dataset``.

        train_size : int
            Train/test split position.

        min_std : float, default=1e-6
            Minimum acceptable standard deviation.

        Returns
        -------
        bool
            True if the dataset passes all checks.
        """
        if not torch.isfinite(y).all():
            return False
        y_train = y[:, :train_size]
        y_test = y[:, train_size:]
        if y_train.std() < min_std or y_test.std() < min_std:
            return False
        return True

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        params = {**params, "num_classes": 0}

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        while True:
            X, y = prior_cls(**params)()
            X, y = Reg2Cls(params)(X, y)

            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            X, d = self.delete_unique_features(X, d)
            if (d > 0).all() and self._regression_sanity_check(y, params["train_size"]):
                return X.squeeze(0), y.squeeze(0), d.squeeze(0)


class RegressionPriorDataset(IterableDataset):
    """Infinite iterator over synthetic tabular **regression** datasets.

    Uses the same hierarchical meta-distribution as the TabICL classification
    prior, but keeps targets continuous instead of discretizing into class
    labels.  Supports ``mlp_scm``, ``tree_scm``, ``mix_scm``, and ``dummy``
    prior types.

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
        Ignored for regression (target is always continuous).

    min_seq_len : int, optional
        Minimum samples per dataset.  If None, uses ``max_seq_len``.

    max_seq_len : int, default=4096
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        Sample sequence length from a log-uniform distribution.

    seq_len_per_gp : bool, default=False
        Sample sequence length per group (variable-length datasets).

    min_train_size : int or float, default=0.1
        Train/test split lower bound.

    max_train_size : int or float, default=0.9
        Train/test split upper bound.

    replay_small : bool, default=False
        Occasionally sample smaller sequence lengths for robustness.

    prior_type : str, default="mix_scm"
        ``"mlp_scm"``, ``"tree_scm"``, ``"mix_scm"``, or ``"dummy"``.

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
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mix_scm",
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

        if prior_type == "dummy":
            from tabicl.prior._dataset import DummyPrior

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
            self.prior = RegressionSCMPrior(
                **default_kwargs,
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

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.prior.get_batch(batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        return (
            f"RegressionPriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  device: {self.device}\n"
            f")"
        )


class SaveRegressionPriorDataset:
    """Generate and save batches of regression prior datasets to disk.

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

        self.prior = RegressionPriorDataset(
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
            scm_fixed_hp=DEFAULT_FIXED_HP,
            scm_sampled_hp=DEFAULT_SAMPLED_HP,
            n_jobs=self.args.n_jobs,
            num_threads_per_generate=self.args.num_threads_per_generate,
            device=self.args.device,
        )
        print(self.prior)

    def _save_metadata(self):
        metadata = {
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
        }
        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_batch_sparse(self, batch_idx, X, y, d, seq_lens, train_sizes):
        if self.args.seq_len_per_gp:
            B = len(d)
        else:
            B, T, H = X.shape
            X = dense2sparse(X.view(-1, H), d.repeat_interleave(T), dtype=torch.float32)

        batch_file = self.save_dir / f"batch_{batch_idx:06d}.pt"
        temp_file = self.save_dir / f"batch_{batch_idx:06d}.pt.tmp"
        torch.save(
            {"X": X, "y": y, "d": d, "seq_lens": seq_lens, "train_sizes": train_sizes, "batch_size": B},
            temp_file,
        )
        temp_file.replace(batch_file)

    def run(self):
        print(f"Save directory: {self.save_dir}")
        print(f"Generating {self.args.num_batches} batches starting from index {self.args.resume_from}")

        for batch_idx in tqdm(
            range(self.args.resume_from, self.args.resume_from + self.args.num_batches),
            desc="Generating regression batches",
        ):
            X, y, d, seq_lens, train_sizes = self.prior.get_batch()
            X = X.cpu()
            y = y.cpu()
            d = d.cpu()
            seq_lens = seq_lens.cpu()
            train_sizes = train_sizes.cpu()
            self._save_batch_sparse(batch_idx, X, y, d, seq_lens, train_sizes)


class LoadRegressionPriorDataset(IterableDataset):
    """Load pre-generated regression prior datasets for training.

    Same format and logic as ``LoadPriorDataset``, but semantically named
    for regression data.  Compatible with data saved by
    ``SaveRegressionPriorDataset``.

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
        device="cpu",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.current_idx = ddp_rank + start_from
        self.max_batches = max_batches
        self.timeout = timeout
        self.delete_after_load = delete_after_load
        self.device = device

        self.metadata = None
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load or parse metadata.json: {e}")

        self.buffer_X = None
        self.buffer_y = None
        self.buffer_d = None
        self.buffer_seq_lens = None
        self.buffer_train_sizes = None
        self.buffer_size = 0

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
        y = batch["y"]
        d = batch["d"]
        seq_lens = batch["seq_lens"]
        train_sizes = batch["train_sizes"]
        batch_size = batch["batch_size"]

        if X.is_nested:
            X = SliceNestedTensor(X)
            y = SliceNestedTensor(y)
        else:
            X = sparse2dense(X, d.repeat_interleave(seq_lens[0]), dtype=torch.float32).view(
                batch_size, seq_lens[0], -1
            )

        if self.delete_after_load and batch_file.exists():
            batch_file.unlink()

        self.current_idx += self.ddp_world_size

        return X, y, d, seq_lens, train_sizes, batch_size

    def __next__(self):
        while self.buffer_size < self.batch_size:
            if self.max_batches is not None and self.current_idx >= self.max_batches:
                break
            try:
                X, y, d, seq_lens, train_sizes, file_batch_size = self._load_batch_file()
                if self.buffer_X is None:
                    self.buffer_X = X
                    self.buffer_y = y
                    self.buffer_d = d
                    self.buffer_seq_lens = seq_lens
                    self.buffer_train_sizes = train_sizes
                    self.buffer_size = file_batch_size
                else:
                    if isinstance(X, SliceNestedTensor):
                        self.buffer_X = cat_slice_nested_tensors([self.buffer_X, X], dim=0)
                        self.buffer_y = cat_slice_nested_tensors([self.buffer_y, y], dim=0)
                    else:
                        self.buffer_X = torch.cat([self.buffer_X, X], dim=0)
                        self.buffer_y = torch.cat([self.buffer_y, y], dim=0)
                    self.buffer_d = torch.cat([self.buffer_d, d], dim=0)
                    self.buffer_seq_lens = torch.cat([self.buffer_seq_lens, seq_lens], dim=0)
                    self.buffer_train_sizes = torch.cat([self.buffer_train_sizes, train_sizes], dim=0)
                    self.buffer_size += file_batch_size
            except Exception as e:
                print(f"Warning: Could not load more files: {str(e)}")
                break

        if self.buffer_size == 0:
            raise StopIteration

        output_size = min(self.batch_size, self.buffer_size)
        X_out = self.buffer_X[:output_size]
        y_out = self.buffer_y[:output_size]
        d_out = self.buffer_d[:output_size]
        seq_lens_out = self.buffer_seq_lens[:output_size]
        train_sizes_out = self.buffer_train_sizes[:output_size]

        if output_size < self.buffer_size:
            self.buffer_X = self.buffer_X[output_size:]
            self.buffer_y = self.buffer_y[output_size:]
            self.buffer_d = self.buffer_d[output_size:]
            self.buffer_seq_lens = self.buffer_seq_lens[output_size:]
            self.buffer_train_sizes = self.buffer_train_sizes[output_size:]
            self.buffer_size -= output_size
        else:
            self.buffer_X = None
            self.buffer_y = None
            self.buffer_d = None
            self.buffer_seq_lens = None
            self.buffer_train_sizes = None
            self.buffer_size = 0

        if isinstance(X_out, SliceNestedTensor):
            X_out = X_out.nested_tensor
            y_out = y_out.nested_tensor

        return X_out, y_out, d_out, seq_lens_out, train_sizes_out

    def __repr__(self) -> str:
        repr_str = (
            f"LoadRegressionPriorDataset(\n"
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
            repr_str += f"    batch_size (generated): {self.metadata.get('batch_size', 'N/A')}\n"
            repr_str += f"    batch_size_per_gp: {self.metadata.get('batch_size_per_gp', 'N/A')}\n"
            repr_str += f"    min features: {self.metadata.get('min_features', 'N/A')}\n"
            repr_str += f"    max features: {self.metadata.get('max_features', 'N/A')}\n"
            repr_str += f"    seq_len: {self.metadata.get('min_seq_len', 'N/A') or 'None'} - {self.metadata.get('max_seq_len', 'N/A')}\n"
            repr_str += f"    log_seq_len: {self.metadata.get('log_seq_len', 'N/A')}\n"
            repr_str += f"    sequence length varies across groups: {self.metadata.get('seq_len_per_gp', 'N/A')}\n"
            repr_str += f"    train_size: {self.metadata.get('min_train_size', 'N/A')} - {self.metadata.get('max_train_size', 'N/A')}\n"
            repr_str += f"    replay_small: {self.metadata.get('replay_small', 'N/A')}\n"
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

    parser = argparse.ArgumentParser(description="Generate regression prior datasets")
    parser.add_argument("--save_dir", type=str, default="regression_data", help="Directory to save generated data")
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of batches to generate")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this batch index")
    parser.add_argument("--batch_size", type=int, default=512, help="Total batch size")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=2, help="Minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="(Legacy, ignored — targets are continuous)")
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
        "--min_train_size", type=_train_size_type, default=0.1, help="Minimum training size position/ratio"
    )
    parser.add_argument(
        "--max_train_size", type=_train_size_type, default=0.9, help="Maximum training size position/ratio"
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
        default="mix_scm",
        choices=["mlp_scm", "tree_scm", "mix_scm"],
        help="Type of prior to use",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--num_threads_per_generate", type=int, default=1, help="Threads per generation")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for generation"
    )

    main_args = parser.parse_args()
    np.random.seed(main_args.np_seed)
    torch.manual_seed(main_args.torch_seed)
    saver = SaveRegressionPriorDataset(main_args)
    saver.run()
