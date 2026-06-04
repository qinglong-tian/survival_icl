from __future__ import annotations

import functools
import inspect
import math
import os
import timeit
import warnings
from contextlib import nullcontext

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce, init_process_group, destroy_process_group, ReduceOp

from tqdm import tqdm

try:
    from torch.cuda import OutOfMemoryError as _OOMError
except ImportError:
    _OOMError = RuntimeError

try:
    import wandb
except ImportError:
    wandb = None

from tabicl._model.tabicl import TabICL
from tabicl.prior._dataset import PriorDataset
from tabicl.prior._genload import LoadPriorDataset
from tabicl.train._optim import get_scheduler
from tabicl.train._train_config import build_parser

# --- survival imports (guarded at use sites) ---

# Ensure project root is on path so survival_prior.py is importable
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def _parse_baseline_types(config):
    """Parse baseline_types from config, returning a list of strings."""
    raw = getattr(config, "baseline_types", "weibull")
    if isinstance(raw, str):
        return [b.strip() for b in raw.split(",") if b.strip()]
    return raw


def _masked_discrete_survival_nll(h_raw, bin_idx, delta, valid_mask):
    """Compute survival NLL over valid query observations."""
    from tabicl.survival import discrete_survival_nll

    if not valid_mask.any():
        raise ValueError("Survival micro-batch contains no valid query observations.")
    if not valid_mask.all():
        h_raw = h_raw[valid_mask]
        bin_idx = bin_idx[valid_mask]
        delta = delta[valid_mask]
    return discrete_survival_nll(h_raw.float(), bin_idx, delta.float())


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper


class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config):
        self.config = config
        self.survival = getattr(config, "task", "classification") == "survival"

        # --- validate survival supervision settings ---
        if self.survival:
            self.query_supervision = getattr(config, "survival_query_supervision", "observed")
            if self.query_supervision not in ("observed", "event"):
                raise ValueError(
                    f"survival_query_supervision must be 'observed' or 'event', "
                    f"got '{self.query_supervision}'."
                )
            self.censor_calibration_scope = getattr(config, "censor_calibration_scope", "dataset")
            if self.censor_calibration_scope not in ("dataset", "context"):
                raise ValueError(
                    f"censor_calibration_scope must be 'dataset' or 'context', "
                    f"got '{self.censor_calibration_scope}'."
                )
            self.query_pinball_weight = getattr(config, "survival_query_pinball_weight", 0.0)
            if not isinstance(self.query_pinball_weight, (int, float)) or self.query_pinball_weight < 0.0:
                raise ValueError(
                    f"survival_query_pinball_weight must be ≥ 0, got {self.query_pinball_weight}."
                )
            if self.query_supervision == "observed" and self.query_pinball_weight > 0.0:
                raise ValueError(
                    "survival_query_pinball_weight must be 0.0 when "
                    "query_supervision='observed' (oracle pinball mixes objective semantics)."
                )

            raw_q = getattr(config, "survival_query_pinball_quantiles", "0.1,0.25,0.5,0.75,0.9")
            try:
                quantiles = [float(x.strip()) for x in raw_q.split(",") if x.strip()]
            except ValueError:
                raise ValueError(
                    f"survival_query_pinball_quantiles must be comma-separated floats, "
                    f"got '{raw_q}'."
                )
            if len(quantiles) < 1:
                raise ValueError(
                    "survival_query_pinball_quantiles must contain at least one value."
                )
            if any(q <= 0.0 or q >= 1.0 for q in quantiles):
                raise ValueError(
                    f"All pinball quantiles must be in (0, 1), got {quantiles}."
                )
            if quantiles != sorted(set(quantiles)):
                raise ValueError(
                    f"Pinball quantiles must be unique and strictly increasing, "
                    f"got {quantiles}."
                )
            self.query_pinball_quantiles = quantiles

        self.configure_ddp()
        self.configure_wandb()

        # Resolve normal resume checkpoint BEFORE model construction so the
        # architecture (K, bounds) is authoritative, not CLI defaults.
        # Precedence: explicit --checkpoint_path, then latest from --checkpoint_dir.
        # --pretrained_path is handled inside build_model.
        self._resume_ckpt_path = None
        self._resume_ckpt_payload = None
        # Always discover stage resume checkpoint regardless of pretrained_path.
        # If a stage checkpoint exists it takes precedence (model + optimizer
        # come from the same source).  pretrained_path is only used as a
        # fallback inside build_model when no stage checkpoint is found.
        if getattr(self.config, "checkpoint_path", None):
            path = self.config.checkpoint_path
        elif getattr(self.config, "checkpoint_dir", None):
            path = self.get_latest_checkpoint()
        else:
            path = None
        if path and os.path.exists(path):
            self._resume_ckpt_path = path
            self._resume_ckpt_payload = torch.load(
                path, map_location="cpu", weights_only=True,
            )

        self.build_model()
        self.configure_prior()
        if self.survival:
            # Align binner K with the loaded model (may differ from CLI default).
            saved_k = self.model_config.get("num_quantiles", None)
            if saved_k is not None:
                self.config.num_bins = saved_k
            self.configure_binner()
            self.configure_loss()
        self.configure_optimizer()
        self.configure_amp()
        self.restore_training_state()

    def configure_ddp(self):
        """Set up distributed training and system configuration."""
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by "
                        f"world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0

        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""
        if wandb is not None and self.config.wandb_log and self.master_process:
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _make_model_config(self, model):
        """Derive TabICL constructor kwargs from a loaded regressor model.

        The regressor's TabICL instance stores its full constructor arguments
        as attributes. Read them directly to preserve non-default options,
        falling back to current constructor defaults for legacy models that
        predate an argument. Only survival task fields are changed.
        """
        task_overrides = {"max_classes": 0, "survival": True}
        config = {
            name: getattr(model, name, parameter.default)
            for name, parameter in inspect.signature(TabICL).parameters.items()
            if name not in task_overrides
            and name != "recompute"
            and parameter.default is not inspect.Parameter.empty
        }
        config.update(task_overrides)
        config["recompute"] = getattr(getattr(model, "row_interactor", None), "recompute", False)
        return config

    def build_model(self):
        """Build and initialize the TabICL model.

        For a normal resume (checkpoint_path / checkpoint_dir), the
        checkpoint was already loaded during __init__ into
        ``_resume_ckpt_payload``.  The model is reconstructed from its
        saved config so all architecture fields (including K) are
        authoritative, then weights are strict-loaded before entering
        freeze/compile/DDP/optimizer setup.
        """

        num_bins = getattr(self.config, "num_bins", 50)

        # ── Normal resume checkpoint (non-pretrained) ────────────────
        if self._resume_ckpt_payload is not None:
            ckpt = self._resume_ckpt_payload
            pretrained_state = ckpt.get("state_dict", ckpt)
            if any(k.startswith("_orig_mod.") for k in pretrained_state):
                pretrained_state = {
                    k[len("_orig_mod."):]: v for k, v in pretrained_state.items()
                }
            saved_config = ckpt.get("config", None)
            if saved_config is not None and isinstance(saved_config, dict):
                saved_survival = saved_config.get("survival", None)
                if saved_survival is not None and bool(saved_survival) != bool(self.survival):
                    raise RuntimeError(
                        f"Resume task mismatch: checkpoint was saved with "
                        f"survival={saved_survival} but current --task is "
                        f"survival={bool(self.survival)}. Refusing to resume a "
                        f"{'survival' if saved_survival else 'regression'} "
                        f"checkpoint as the other task."
                    )
                model_config = dict(saved_config)
                model_config.setdefault("survival", self.survival)
                if self.survival:
                    model_config.setdefault("max_classes", 0)
                model = TabICL(**model_config).to(self.config.device)
                model.load_state_dict(pretrained_state, strict=True)
                self.model_config = model_config
                if self.master_process:
                    k = model_config.get("num_quantiles", "?")
                    print(f"Resumed from checkpoint ({self._resume_ckpt_path}): "
                          f"{len(pretrained_state)} keys; K={k}.")
                # Restore checkpoint survival metadata (binner edges, scaler)
                if self.survival:
                    self._restore_survival_metadata(ckpt, model=model)
            else:
                # Legacy resume: no saved config — build from CLI defaults.
                # If survival metadata is present, derive K from it so the
                # decoder shape matches the checkpoint.
                if self.master_process:
                    print("Resume checkpoint has no saved config; loading from CLI defaults.")
                surv_meta = ckpt.get("survival_metadata", None)
                legacy_k = self.config.num_bins
                if self.survival and surv_meta is not None:
                    meta_k = surv_meta.get("num_bins", None)
                    if meta_k is not None:
                        legacy_k = int(meta_k)
                fallback_config = {
                    "max_classes": 0 if self.survival else self.config.max_classes,
                    "survival": self.survival,
                    "embed_dim": self.config.embed_dim,
                    "col_num_blocks": self.config.col_num_blocks,
                    "col_nhead": self.config.col_nhead,
                    "col_num_inds": self.config.col_num_inds,
                    "row_num_blocks": self.config.row_num_blocks,
                    "row_nhead": self.config.row_nhead,
                    "row_num_cls": self.config.row_num_cls,
                    "row_rope_base": self.config.row_rope_base,
                    "icl_num_blocks": self.config.icl_num_blocks,
                    "icl_nhead": self.config.icl_nhead,
                    "ff_factor": self.config.ff_factor,
                    "dropout": self.config.dropout,
                    "activation": self.config.activation,
                    "norm_first": self.config.norm_first,
                }
                if self.survival:
                    fallback_config["num_quantiles"] = legacy_k
                model = TabICL(**fallback_config).to(self.config.device)
                missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
                if missing:
                    raise RuntimeError(
                        f"Legacy resume checkpoint is missing {len(missing)} keys: "
                        f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                        f"The checkpoint may be truncated or from a different architecture."
                    )
                if unexpected:
                    raise RuntimeError(
                        f"Legacy resume checkpoint has {len(unexpected)} unexpected keys: "
                        f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}. "
                        f"The checkpoint may be from a different architecture."
                    )
                self.model_config = fallback_config
                if self.survival:
                    self._restore_survival_metadata(ckpt, model=model)
        elif self.survival:
            checkpoint_path = getattr(self.config, "pretrained_path", None)

            if checkpoint_path is None:
                # Load regressor from HF Hub — it auto-detects architecture
                if self.master_process:
                    print("Loading pretrained TabICL regressor from Hugging Face Hub...")
                from tabicl._sklearn.regressor import TabICLRegressor

                loader = TabICLRegressor(
                    n_estimators=1,
                    model_path=None,
                    allow_auto_download=True,
                    device=self.config.device,
                )
                loader._resolve_device()
                loader._load_model()
                model = loader.model_

                model.configure_survival(num_bins)
                self.model_config = self._make_model_config(model)
                if self.master_process:
                    print("Survival head + y_encoder attached. Encoder weights preserved from HF Hub.")
            else:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                pretrained_state = ckpt.get("state_dict", ckpt)
                if any(k.startswith("_orig_mod.") for k in pretrained_state):
                    pretrained_state = {
                        k[len("_orig_mod."):]: v for k, v in pretrained_state.items()
                    }

                # Detect whether survival or regressor
                is_survival_ckpt = any(
                    k.startswith("icl_predictor.decoder.head.")
                    for k in pretrained_state
                )

                if is_survival_ckpt:
                    # New-style survival checkpoint — reconstruct from saved config
                    if self.master_process:
                        print(f"Loading survival checkpoint ({checkpoint_path}) from config...")
                    saved_config = ckpt.get("config", None)
                    if saved_config is not None and isinstance(saved_config, dict):
                        # Use the saved config; override only survival task fields.
                        # Preserve saved num_quantiles (bin count) — a checkpoint
                        # trained with K ≠ current CLI default must still load.
                        model_config = dict(saved_config)
                        model_config.setdefault("survival", True)
                        model_config.setdefault("max_classes", 0)
                        try:
                            model = TabICL(**model_config).to(self.config.device)
                            model.load_state_dict(pretrained_state, strict=True)
                            self.model_config = model_config
                            if self.master_process:
                                print(f"Strict-loaded survival checkpoint: {len(pretrained_state)} keys; "
                                      f"num_quantiles={model_config.get('num_quantiles', '?')}.")
                            # Restore exact binner edges/scaler from checkpoint metadata
                            self._restore_survival_metadata(ckpt, model=model)
                        except Exception as e:
                            # New-style checkpoint with saved config should load strictly.
                            # Don't silently fall back — propagate the error.
                            raise RuntimeError(
                                f"Failed to strict-load survival checkpoint '{checkpoint_path}' "
                                f"from its saved config: {e}"
                            ) from e
                    else:
                        if self.master_process:
                            print("Survival checkpoint has no saved config; using legacy fallback.")
                        is_survival_ckpt = False  # fall through to legacy path

                if is_survival_ckpt:
                    pass  # already handled above
                else:
                    # Legacy fallback: load via regressor for correct architecture
                    if self.master_process:
                        print("Loading checkpoint via regressor (legacy path)...")
                    from tabicl._sklearn.regressor import TabICLRegressor

                    # Legacy fallback: build correct architecture, then load
                    # checkpoint weights on top.
                    # Survival checkpoints: use HF Hub for the architectural base.
                    # Regressor checkpoints: reconstruct from the local file to
                    # preserve custom architectures and allow offline use.
                    is_legacy_survival = any(
                        k.startswith("icl_predictor.decoder.head.")
                        for k in pretrained_state
                    )
                    if is_legacy_survival:
                        tmp_loader = TabICLRegressor(
                            n_estimators=1,
                            model_path=None,
                            allow_auto_download=True,
                            device=self.config.device,
                        )
                        tmp_loader._resolve_device()
                        tmp_loader._load_model()
                        model = tmp_loader.model_

                        surv_meta = ckpt.get("survival_metadata", None)
                        legacy_k = num_bins
                        if surv_meta is not None:
                            meta_k = surv_meta.get("num_bins", None)
                            if meta_k is not None:
                                legacy_k = int(meta_k)
                        model.configure_survival(legacy_k)
                        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
                        if missing:
                            raise RuntimeError(
                                f"Legacy survival checkpoint is missing {len(missing)} keys: "
                                f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
                            )
                        if unexpected:
                            raise RuntimeError(
                                f"Legacy survival checkpoint has {len(unexpected)} unexpected keys: "
                                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}."
                            )
                        self.model_config = self._make_model_config(model)
                        self._restore_survival_metadata(ckpt, model=model)
                    else:
                        regression_config = ckpt.get("config", None)
                        if regression_config is not None and isinstance(regression_config, dict):
                            model = TabICL(**regression_config).to(self.config.device)
                            model.load_state_dict(pretrained_state, strict=True)
                        else:
                            tmp_loader = TabICLRegressor(
                                n_estimators=1,
                                model_path=checkpoint_path,
                                allow_auto_download=False,
                                device=self.config.device,
                            )
                            tmp_loader._resolve_device()
                            tmp_loader._load_model()
                            model = tmp_loader.model_

                        model.configure_survival(num_bins)
                        self.model_config = self._make_model_config(model)
                        if self.master_process:
                            print(f"Converted regressor checkpoint ({checkpoint_path}) to survival.")
        else:
            self.model_config = {
                "max_classes": self.config.max_classes,
                "embed_dim": self.config.embed_dim,
                "col_num_blocks": self.config.col_num_blocks,
                "col_nhead": self.config.col_nhead,
                "col_num_inds": self.config.col_num_inds,
                "row_num_blocks": self.config.row_num_blocks,
                "row_nhead": self.config.row_nhead,
                "row_num_cls": self.config.row_num_cls,
                "row_rope_base": self.config.row_rope_base,
                "icl_num_blocks": self.config.icl_num_blocks,
                "icl_nhead": self.config.icl_nhead,
                "ff_factor": self.config.ff_factor,
                "dropout": self.config.dropout,
                "activation": self.config.activation,
                "norm_first": self.config.norm_first,
            }
            model = TabICL(**self.model_config)

        model.to(device=self.config.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        if self.config.freeze_col:
            model.col_embedder.eval()
            for param in model.col_embedder.parameters():
                param.requires_grad = False

        if self.config.freeze_row:
            model.row_interactor.eval()
            for param in model.row_interactor.parameters():
                param.requires_grad = False

        if self.config.freeze_icl:
            model.icl_predictor.eval()
            for param in model.icl_predictor.parameters():
                param.requires_grad = False

        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def configure_prior(self):
        """Set up a tabular dataset generator for synthetic data during training."""

        if self.survival:
            from survival_prior import SurvivalPriorDataset, LoadSurvivalPriorDataset

            if self.config.prior_dir is None:
                dataset = SurvivalPriorDataset(
                    batch_size=self.config.batch_size,
                    batch_size_per_gp=self.config.batch_size_per_gp,
                    min_features=self.config.min_features,
                    max_features=self.config.max_features,
                    max_seq_len=self.config.max_seq_len,
                    min_seq_len=self.config.min_seq_len,
                    log_seq_len=self.config.log_seq_len,
                    seq_len_per_gp=self.config.seq_len_per_gp,
                    min_train_size=getattr(self.config, "min_train_size", 1.0),
                    max_train_size=getattr(self.config, "max_train_size", 1.0),
                    replay_small=self.config.replay_small,
                    prior_type=self.config.prior_type,
                    model_type=getattr(self.config, "survival_model_type", "ph"),
                    beta=getattr(self.config, "survival_beta", 1.0),
                    beta_sampling=getattr(self.config, "beta_sampling", "fixed"),
                    min_beta=getattr(self.config, "min_beta", 0.25),
                    max_beta=getattr(self.config, "max_beta", 2.0),
                    baseline_param_prior=getattr(self.config, "baseline_param_prior", "current"),
                    time_scale_sampling=getattr(self.config, "time_scale_sampling", "fixed"),
                    min_time_scale=getattr(self.config, "min_time_scale", 0.2),
                    max_time_scale=getattr(self.config, "max_time_scale", 5.0),
                    baseline_types=_parse_baseline_types(self.config),
                    baseline_mode=getattr(self.config, "baseline_mode", "mix"),
                    max_time=getattr(self.config, "survival_raw_time_max", 1e30),
                    min_censor_scale=getattr(self.config, "min_censor_scale", 1.0),
                    max_censor_scale=getattr(self.config, "max_censor_scale", 5.0),
                    min_event_rate=getattr(self.config, "min_event_rate", 0.40),
                    max_event_rate=getattr(self.config, "max_event_rate", 0.90),
                    censoring_strategy=getattr(self.config, "censoring_strategy", "target_event_rate"),
                    calibration_scope=getattr(self.config, "censor_calibration_scope", "dataset"),
                    device=self.config.prior_device,
                    n_jobs=1,
                )
            else:
                dataset = LoadSurvivalPriorDataset(
                    data_dir=self.config.prior_dir,
                    batch_size=self.config.batch_size,
                    ddp_world_size=getattr(self, "ddp_world_size", 1),
                    ddp_rank=getattr(self, "ddp_rank", 0),
                    start_from=self.config.load_prior_start,
                    delete_after_load=self.config.delete_after_load,
                    censor_calibration_scope=getattr(self.config, "censor_calibration_scope", "dataset"),
                    device=self.config.prior_device,
                )
        else:
            if self.config.prior_dir is None:
                dataset = PriorDataset(
                    batch_size=self.config.batch_size,
                    batch_size_per_gp=self.config.batch_size_per_gp,
                    min_features=self.config.min_features,
                    max_features=self.config.max_features,
                    max_classes=self.config.max_classes,
                    min_seq_len=self.config.min_seq_len,
                    max_seq_len=self.config.max_seq_len,
                    log_seq_len=self.config.log_seq_len,
                    seq_len_per_gp=self.config.seq_len_per_gp,
                    min_train_size=self.config.min_train_size,
                    max_train_size=self.config.max_train_size,
                    replay_small=self.config.replay_small,
                    prior_type=self.config.prior_type,
                    device=self.config.prior_device,
                    n_jobs=1,
                )
            else:
                dataset = LoadPriorDataset(
                    data_dir=self.config.prior_dir,
                    batch_size=self.config.batch_size,
                    ddp_world_size=getattr(self, "ddp_world_size", 1),
                    ddp_rank=getattr(self, "ddp_rank", 0),
                    start_from=self.config.load_prior_start,
                    delete_after_load=self.config.delete_after_load,
                    device=self.config.prior_device,
                )

        if self.master_process:
            print(dataset)

        # DataLoader workers run prior generation in background via spawn multiprocessing.
        # On Linux with set_start_method("spawn") this is safe: workers are fresh processes
        # with no inherited CUDA context and only run CPU-side SCM → survival sampling.
        # macOS users should set --prior_num_workers 0 if spawn pickling is problematic.
        prior_workers = getattr(self.config, "prior_num_workers", 1)
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=prior_workers,
            persistent_workers=(prior_workers > 0),
            pin_memory=False,
        )

    def configure_binner(self):
        """Build fixed bins on the standardized log-time axis.

        When :meth:`_restore_survival_metadata` has already populated
        ``self.binner`` and ``self.survival_time_scaler_config`` from a
        checkpoint, this method is a no-op.
        """
        if getattr(self, "_binner_restored", False):
            if self.master_process:
                print(f"TimeBinner (restored from checkpoint): {self.binner}")
            return

        from tabicl.survival import TimeBinner

        self.survival_time_scaler_config = {
            "eps": getattr(self.config, "survival_time_eps", 1e-8),
            "min_scale": getattr(self.config, "survival_time_min_scale", 0.1),
            "z_min": getattr(self.config, "survival_time_z_min", -6.0),
            "z_max": getattr(self.config, "survival_time_z_max", 6.0),
        }
        self.binner = TimeBinner.from_standardized_range(
            num_bins=getattr(self.config, "num_bins", 50),
            z_min=self.survival_time_scaler_config["z_min"],
            z_max=self.survival_time_scaler_config["z_max"],
        ).to(torch.device(self.config.device))

        if self.master_process:
            print(f"TimeBinner: {self.binner}")

    def configure_loss(self):
        """(no-op) Survival NLL is called inline in ``_run_micro_batch_survival``."""
        pass

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            params=self.raw_model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""
        _cuda_available = torch.cuda.is_available()
        amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.dtype]
        self.amp = (
            self.config.amp
            and "cuda" in self.config.device
            and _cuda_available
            and amp_dtype != torch.float32
        )
        if _cuda_available:
            self.scaler = torch.GradScaler(
                "cuda",
                enabled=self.amp and amp_dtype == torch.float16,
            )
        else:
            self.scaler = torch.GradScaler(enabled=False)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled with {self.config.dtype}.")
            self.amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            if self.master_process and self.config.amp and amp_dtype == torch.float32:
                print("Automatic Mixed Precision is disabled because dtype=float32.")
            self.amp_ctx = nullcontext()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`"""
        ckpt_dir = self.config.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return None
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        if not checkpoints:
            return None
        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            return os.path.join(ckpt_dir, latest_checkpoint)
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def _restore_survival_metadata(self, checkpoint: dict, *, model=None):
        """Restore exact saved binner edges, means, and scaler from checkpoint.

        Called for both ``pretrained_path`` and normal-resume survival
        checkpoints so custom ``z_min``/``z_max`` bounds are preserved.
        """
        surv_meta = checkpoint.get("survival_metadata", None)
        if surv_meta is None or not self.survival:
            # No metadata: configure_binner will build from config defaults
            return

        if surv_meta.get("time_scale") != "km_hybrid_log":
            print("Checkpoint has legacy raw-time survival metadata; keeping standardized TimeBinner.")
            return

        from tabicl.survival import TimeBinner

        saved_edges = surv_meta["binner_edges"].to(self.config.device)
        saved_means = surv_meta["binner_means"].to(self.config.device)
        saved_k = surv_meta.get("num_bins", len(saved_means))

        # Validate metadata integrity
        if not torch.isfinite(saved_edges).all():
            raise ValueError("binner_edges contains non-finite values")
        if not torch.isfinite(saved_means).all():
            raise ValueError("binner_means contains non-finite values")
        if not (saved_edges.diff() > 0).all():
            raise ValueError("binner_edges must be strictly increasing")
        if len(saved_edges) != saved_k + 1:
            raise ValueError(
                f"binner_edges length {len(saved_edges)} != K+1={saved_k + 1}"
            )
        if len(saved_means) != saved_k:
            raise ValueError(
                f"binner_means length {len(saved_means)} != K={saved_k}"
            )

        # Validate each mean lies inside its bin
        for i in range(saved_k):
            lo, hi = saved_edges[i], saved_edges[i + 1]
            m = saved_means[i]
            if m < lo or m > hi:
                raise ValueError(
                    f"binner_means[{i}]={m:.4g} outside bin [{lo:.4g}, {hi:.4g}]"
                )

        # Validate against the loaded model
        if model is not None:
            model_k = model.num_quantiles if hasattr(model, "num_quantiles") else model.icl_predictor.decoder.num_bins
            if saved_k != model_k:
                raise ValueError(
                    f"survival_metadata K={saved_k} != model K={model_k}"
                )

        self.binner = TimeBinner(
            bin_edges=saved_edges, bin_means=saved_means,
        ).to(torch.device(self.config.device))
        self.config.num_bins = saved_k
        self._binner_restored = True  # Signal configure_binner to skip defaults

        saved_scaler = surv_meta.get("time_scaler", None)
        if saved_scaler is not None:
            # Validate required scaler fields
            scaler_eps = saved_scaler.get("eps", None)
            scaler_min_scale = saved_scaler.get("min_scale", None)
            if scaler_eps is None or not float(scaler_eps) > 0 or not math.isfinite(float(scaler_eps)):
                raise ValueError(
                    f"Checkpoint scaler has invalid eps: {scaler_eps}"
                )
            if scaler_min_scale is None or not float(scaler_min_scale) > 0 or not math.isfinite(float(scaler_min_scale)):
                raise ValueError(
                    f"Checkpoint scaler has invalid min_scale: {scaler_min_scale}"
                )
            scaler_z_min = saved_scaler.get("z_min", None)
            scaler_z_max = saved_scaler.get("z_max", None)
            if scaler_z_min is None or scaler_z_max is None:
                raise ValueError(
                    "Checkpoint scaler missing required z_min/z_max fields"
                )
            scaler_z_min = float(scaler_z_min)
            scaler_z_max = float(scaler_z_max)
            if not math.isfinite(scaler_z_min) or not math.isfinite(scaler_z_max):
                raise ValueError(
                    f"Checkpoint scaler has non-finite z_min/z_max: "
                    f"z_min={scaler_z_min}, z_max={scaler_z_max}"
                )
            edge_z_min = float(saved_edges[0])
            edge_z_max = float(saved_edges[-1])
            # Use tolerance: float32 edges vs Python floats can differ by ~1e-7
            if not math.isclose(scaler_z_min, edge_z_min, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(
                    f"Scaler z_min ({scaler_z_min}) != binner edge[0] ({edge_z_min}). "
                    f"Checkpoint metadata is inconsistent."
                )
            if not math.isclose(scaler_z_max, edge_z_max, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(
                    f"Scaler z_max ({scaler_z_max}) != binner edge[-1] ({edge_z_max}). "
                    f"Checkpoint metadata is inconsistent."
                )
            self.survival_time_scaler_config = dict(saved_scaler)
        else:
            # Derive scaler bounds from restored binner edges so preprocessing
            # matches the checkpoint rather than CLI defaults.
            self.survival_time_scaler_config = {
                "eps": 1e-8,
                "min_scale": 0.1,
                "z_min": float(saved_edges[0]),
                "z_max": float(saved_edges[-1]),
            }
        if self.master_process:
            print(f"Restored survival metadata from checkpoint: K={saved_k}, "
                  f"z_range=[{saved_edges[0]:.2f}, {saved_edges[-1]:.2f}]")

    def restore_training_state(self):
        """Restore optimizer, scheduler, scaler, and step from a checkpoint.

        Model weights are loaded during :meth:`build_model`.  This method
        only restores training state (optimizer, scheduler, amp scaler,
        current step) when the model was already reconstructed from the
        checkpoint config.
        """
        # Reuse the checkpoint path resolved during __init__ so model and
        # training state always come from the same file.  Re-discovering the
        # latest checkpoint could pick a newer file than the one used to
        # construct the model.
        checkpoint_path = self._resume_ckpt_path

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Restoring training state from {checkpoint_path}")
        checkpoint = self._resume_ckpt_payload
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self._resume_ckpt_payload = None  # consumed

        if "state_dict" not in checkpoint:
            # Raw state dict (no wrapper keys) — weights were already loaded
            # by build_model via ckpt.get("state_dict", ckpt).  No training
            # state to restore.
            print("Checkpoint is a raw state dict; skipping training state restoration.")
            return

        if self.config.only_load_model:
            print("Only loading model weights (training state not restored).")
            return

        if "optimizer_state" not in checkpoint:
            print("Checkpoint has no optimizer_state; skipping training state restoration.")
            return

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.curr_step = checkpoint["curr_step"]
        print(f"Resuming training at step {self.curr_step}")

        # --- resume guard: reject mismatched survival supervision settings ---
        if self.survival and "survival_metadata" in checkpoint:
            meta = checkpoint["survival_metadata"]
            for attr, key, default in [
                ("query_supervision", "query_supervision", "observed"),
                ("censor_calibration_scope", "censor_calibration_scope", "dataset"),
                ("query_pinball_weight", "query_pinball_weight", 0.0),
                ("query_pinball_quantiles", "query_pinball_quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]),
            ]:
                saved = meta.get(key, default)
                cli = getattr(self, attr, None)
                if cli is not None and saved != cli:
                    raise RuntimeError(
                        f"Checkpoint was trained with {key}={saved!r}, "
                        f"but CLI specifies {key}={cli!r}. "
                        f"Use matching settings to resume, or set "
                        f"--only_load_model True to initialize a new objective."
                    )

    # --- deprecated: kept for backward compat with external callers ---
    def load_checkpoint(self):
        """Deprecated.  Use :meth:`restore_training_state`."""
        self.restore_training_state()

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "curr_step": self.curr_step,
        }
        if self.survival and hasattr(self, "binner"):
            checkpoint["survival_metadata"] = {
                "binner_edges": self.binner.bin_edges,
                "binner_means": self.binner.bin_means,
                "num_bins": self.binner.num_bins,
                "task": "survival",
                "time_scale": "km_hybrid_log",
                "time_scaler": getattr(self, "survival_time_scaler_config", None),
                "query_supervision": getattr(self, "query_supervision", "observed"),
                "censor_calibration_scope": getattr(self, "censor_calibration_scope", "dataset"),
                "query_pinball_weight": getattr(self, "query_pinball_weight", 0.0),
                "query_pinball_quantiles": getattr(self, "query_pinball_quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]),
            }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """Manage temporary checkpoints by deleting the oldest when limit is exceeded."""
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except Exception:
                continue

        temp_checkpoints.sort(key=lambda x: x[0])

        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Saves a checkpoint on early termination (e.g., data exhaustion)
        so that no progress is lost.
        """

        if self.master_process:
            step_progress = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        try:
            dataloader = iter(self.dataloader)
            for step in step_progress:
                with Timer() as prior_timer:
                    batch = next(dataloader)
                prior_time = prior_timer.elapsed

                with Timer() as train_timer:
                    results = self.run_batch(batch)
                train_time = train_timer.elapsed

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.curr_step = step + 1
                if self.master_process:
                    results.update({"prior_time": prior_time, "train_time": train_time})
                    step_progress.set_postfix(**{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()})

                    if self.curr_step % self.config.save_temp_every == 0 or self.curr_step % self.config.save_perm_every == 0:
                        ckpt_name = f"step-{self.curr_step}.ckpt"
                        self.save_checkpoint(name=ckpt_name)
                        is_temp = self.curr_step % self.config.save_temp_every == 0
                        is_perm = self.curr_step % self.config.save_perm_every == 0
                        if is_temp and not is_perm and self.config.max_checkpoints > 0:
                            self.manage_checkpoint()

                if self.wandb_run is not None:
                    results["lr"] = self.scheduler.get_last_lr()[0]
                    wandb.log(results, step=self.curr_step)
        except StopIteration:
            if self.master_process:
                print(f"\nData exhausted at step {self.curr_step}. Saving final checkpoint...")
                self.save_checkpoint(name=f"step-{self.curr_step}-final.ckpt")
                step_progress.close()
            raise

    # ------------------------------------------------------------------
    # Micro-batch helpers (classification)
    # ------------------------------------------------------------------

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError("All datasets in the micro batch must have the same training size.")
        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()
        return seq_len, train_size, micro_seq_len

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]
        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]
        return micro_X, micro_y

    # ------------------------------------------------------------------
    # Micro-batch: classification
    # ------------------------------------------------------------------

    def _run_micro_batch_classification(self, micro_batch, micro_batch_idx, num_micro_batches):
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size, _ = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            pred = self.model(micro_X, y_train, micro_d)
            pred = pred.flatten(end_dim=-2)
            true = y_test.long().flatten()
            loss = F.cross_entropy(pred, true)

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {}
            micro_results["ce"] = scaled_loss.item()
            accuracy = (pred.argmax(dim=1) == true).sum() / len(true)
            micro_results["accuracy"] = accuracy.item() / num_micro_batches

        return micro_results

    # ------------------------------------------------------------------
    # Micro-batch: survival
    # ------------------------------------------------------------------

    def _standardize_survival_micro_batch(
        self,
        t_train,
        delta_train,
        t_test,
        delta_test,
        t_event_test=None,
        train_sizes_ds=None,
        query_sizes_ds=None,
        *,
        return_event_info=False,
    ):
        """Delegate to the shared ``standardize_survival_micro_batch``.

        Set ``t_event_test`` and ``return_event_info=True`` for oracle
        event-time supervision.  All other callers get the legacy 4-tuple.
        """
        from tabicl.survival._scaler import standardize_survival_micro_batch

        scaler_kwargs = getattr(self, "survival_time_scaler_config", {
            "eps": getattr(self.config, "survival_time_eps", 1e-8),
            "min_scale": getattr(self.config, "survival_time_min_scale", 0.1),
            "z_min": getattr(self.config, "survival_time_z_min", -6.0),
            "z_max": getattr(self.config, "survival_time_z_max", 6.0),
        })
        t_train_z, delta_train_z, t_test_z, delta_test_z, t_event_z, t_event_in_range = \
            standardize_survival_micro_batch(
                t_train, delta_train, t_test, delta_test, t_event_test,
                train_sizes_ds, query_sizes_ds, scaler_kwargs,
            )
        if return_event_info:
            return t_train_z, delta_train_z, t_test_z, delta_test_z, t_event_z, t_event_in_range
        return t_train_z, delta_train_z, t_test_z, delta_test_z

    def _run_micro_batch_survival(self, micro_batch, micro_batch_idx, num_micro_batches):
        # Unpack 7-tuple from prior: (X, t, delta, t_event, d, seq_len, train_size)
        (micro_X, micro_t, micro_delta, micro_t_event,
         micro_d, micro_seq_len, micro_train_size) = micro_batch

        # Guard: variable-length survival micro-batches rely on GP ordering.
        if self.config.seq_len_per_gp:
            has_fixed_len = (
                getattr(self.config, "min_seq_len", None) is not None
                and self.config.min_seq_len == self.config.max_seq_len
            )
            if not has_fixed_len:
                mb = self.config.micro_batch_size
                bpg = self.config.batch_size_per_gp
                bs = self.config.batch_size  # already ceil-divided for DDP
                if bpg < bs and bpg % mb != 0:
                    raise ValueError(
                        f"When seq_len_per_gp=True with variable lengths, "
                        f"batch_size_per_gp ({bpg}) must be divisible by "
                        f"micro_batch_size ({mb}) to keep each GP group "
                        f"boundary aligned with micro-batches. "
                        f"Got bpg % mb = {bpg % mb}. "
                        f"Future hardening: support per-dataset padding masks."
                    )

        seq_len, _, per_ds_seq_lens = self.validate_micro_batch(micro_seq_len, micro_train_size)
        # Survival data has train_size == seq_len (no split in the prior).
        # We override to split half/half for context/query.
        train_size = seq_len // 2
        micro_X, micro_t = self.align_micro_batch(micro_X, micro_t, micro_d, seq_len)
        # Align delta and t_event same as t
        if micro_delta.shape[1] > seq_len:
            micro_delta = micro_delta[:, :seq_len]
        if micro_t_event.shape[1] > seq_len:
            micro_t_event = micro_t_event[:, :seq_len]

        micro_X = micro_X.to(self.config.device)
        micro_t = micro_t.to(self.config.device)
        micro_delta = micro_delta.to(self.config.device)
        micro_t_event = micro_t_event.to(self.config.device)
        micro_d = micro_d.to(self.config.device)
        per_ds_seq_lens = per_ds_seq_lens.to(self.config.device)

        # Per-dataset context/query sizes for padding masking
        train_sizes_ds = (per_ds_seq_lens / 2).long()  # (B,) half as context
        query_sizes_ds = per_ds_seq_lens - train_sizes_ds  # (B,)

        t_train = micro_t[:, :train_size]  # (B, train_size) — context
        t_test = micro_t[:, train_size:]   # (B, T - train_size) — query (includes padding)
        delta_train = micro_delta[:, :train_size]
        delta_test = micro_delta[:, train_size:]
        t_event_test = micro_t_event[:, train_size:]  # (B, T - train_size) — oracle query events

        event_mode = self.query_supervision == "event"
        if event_mode:
            # Oracle event supervision: use unclipped standardized t_event
            # for the NLL target (all query rows are events: delta=1).
            t_train, delta_train, t_test, _, t_event_z, t_event_in_range = \
                self._standardize_survival_micro_batch(
                    t_train, delta_train, t_test, delta_test,
                    t_event_test=t_event_test,
                    train_sizes_ds=train_sizes_ds,
                    query_sizes_ds=query_sizes_ds,
                    return_event_info=True,
                )
            delta_test_z = torch.ones_like(t_test)  # all query rows are events
        else:
            t_train, delta_train, t_test, delta_test = \
                self._standardize_survival_micro_batch(
                    t_train, delta_train, t_test, delta_test,
                    train_sizes_ds=train_sizes_ds,
                    query_sizes_ds=query_sizes_ds,
                )
            delta_test_z = delta_test
            t_event_z = None
            t_event_in_range = None

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            h_raw = self.model(micro_X, t_train, d=None, delta_train=delta_train)
            # h_raw: (B, T_test, K) — flatten for loss
            B, T_test, K = h_raw.shape
            h_raw_flat = h_raw.reshape(-1, K)

            # Build padding mask: valid positions are those within query_sizes_ds
            position_idx = torch.arange(T_test, device=self.config.device).unsqueeze(0)  # (1, T_test)
            valid_mask = position_idx < query_sizes_ds.unsqueeze(1)  # (B, T_test)
            valid_mask_flat = valid_mask.reshape(-1)  # (B * T_test,)

            pinball = None
            if event_mode:
                # Event NLL: bin index from unclipped t_event_z,
                # delta = 1 for all valid query rows.
                t_test_flat = t_event_z.reshape(-1)
                bin_idx = self.binner.bin_index(t_test_flat)  # (B * T_test,) long
                surv_nll = _masked_discrete_survival_nll(
                    h_raw_flat, bin_idx, delta_test_z.reshape(-1), valid_mask_flat,
                )
                loss = surv_nll

                # Always compute in_range for logging; pinball is optional
                t_event_in_range_flat = t_event_in_range.reshape(-1)
                if self.query_pinball_weight > 0.0:
                    from tabicl.survival import oracle_query_pinball_loss
                    tau = torch.tensor(
                        self.query_pinball_quantiles,
                        device=h_raw_flat.device,
                        dtype=h_raw_flat.dtype,
                    )
                    pinball = oracle_query_pinball_loss(
                        h_raw_flat, t_test_flat, self.binner,
                        in_range=t_event_in_range_flat,
                        valid_mask=valid_mask_flat,
                        tau_levels=tau,
                    )
                    loss = surv_nll + self.query_pinball_weight * pinball

                # Log underflow/overflow counts
                if valid_mask_flat.any():
                    v = valid_mask_flat
                    ir = t_event_in_range_flat
                    n_valid = v.sum().item()
                    n_in_range = (v & ir).sum().item()
                    n_underflow = (t_test_flat[v] < self.binner.bin_edges[0]).sum().item()
                    n_overflow = (t_test_flat[v] > self.binner.bin_edges[-1]).sum().item()
                else:
                    n_valid = n_in_range = n_underflow = n_overflow = 0.0
            else:
                # Observed mode (legacy): censored query NLL
                t_test_flat = t_test.reshape(-1)
                delta_test_flat = delta_test_z.reshape(-1)
                bin_idx = self.binner.bin_index(t_test_flat)  # (B * T_test,) long
                loss = _masked_discrete_survival_nll(
                    h_raw_flat, bin_idx, delta_test_flat, valid_mask_flat,
                )
                surv_nll = loss

            # Keep loss in float32 — GradScaler handles mixed-precision loss

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            loss_value = loss.item()
            nonfinite_logit_count = (~torch.isfinite(h_raw_flat)).sum().item()
            micro_results = {
                "surv_nll": surv_nll.item() / num_micro_batches,
                "_nonfinite_loss_count": float(not math.isfinite(loss_value)),
                "_nonfinite_logit_count": nonfinite_logit_count,
            }
            if event_mode:
                micro_results["query_n_valid"] = n_valid
                micro_results["query_n_underflow"] = n_underflow
                micro_results["query_n_overflow"] = n_overflow
                micro_results["query_n_in_range"] = n_in_range
                if pinball is not None:
                    micro_results["query_pinball"] = pinball.item() / num_micro_batches
                    micro_results["total_loss"] = loss.item() / num_micro_batches

        return micro_results

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """Dispatch to classification or survival micro-batch."""
        if self.survival:
            return self._run_micro_batch_survival(micro_batch, micro_batch_idx, num_micro_batches)
        return self._run_micro_batch_classification(micro_batch, micro_batch_idx, num_micro_batches)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def run_batch(self, batch):
        """Train the model on a batch, with survival support."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if self.survival and self.query_supervision == "event" and batch[3] is None:
            raise RuntimeError(
                "query_supervision='event' requires t_event in every batch, "
                "but the prior returned t_event=None.  Regenerate the disk "
                "prior with the current survival generator, switch to "
                "--survival_query_supervision observed, or use on-the-fly "
                "generation."
            )

        # Pad nested tensors to the same size
        batch_padded = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        num_micro_batches = math.ceil(self.config.batch_size / self.config.micro_batch_size)
        micro_batches = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch_padded]
        micro_batches = list(zip(*micro_batches))

        if self.survival:
            results = {
                "surv_nll": 0.0,
                "_nonfinite_loss_count": 0.0,
                "_nonfinite_logit_count": 0.0,
            }
            if self.query_supervision == "event":
                results["query_n_valid"] = 0.0
                results["query_n_underflow"] = 0.0
                results["query_n_overflow"] = 0.0
                results["query_n_in_range"] = 0.0
                if self.query_pinball_weight > 0.0:
                    results["query_pinball"] = 0.0
                    results["total_loss"] = 0.0
        else:
            results = {"ce": 0.0, "accuracy": 0.0}
        failed_batches = 0

        for idx, mb in enumerate(micro_batches):
            try:
                micro_results = self.run_micro_batch(mb, idx, num_micro_batches)
                for k, v in micro_results.items():
                    results[k] += v
            except _OOMError:
                print(f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} "
                      f"at step {self.curr_step}. Skipping.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                failed_batches += 1
                continue

        failure_ratio = failed_batches / num_micro_batches
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        if self.survival:
            nonfinite = torch.tensor(
                [
                    results.pop("_nonfinite_loss_count"),
                    results.pop("_nonfinite_logit_count"),
                ],
                device=self.config.device,
                dtype=torch.float32,
            )
            if self.ddp:
                all_reduce(nonfinite, op=ReduceOp.SUM)
            if nonfinite[0].item() > 0 or nonfinite[1].item() > 0:
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite survival numerics detected at step {self.curr_step}: "
                    f"{int(nonfinite[0].item())} bad micro-batch loss(es), "
                    f"{int(nonfinite[1].item())} non-finite hazard logit(s) across all ranks. "
                    "The optimizer step was aborted."
                )

        if self.survival and self.query_supervision == "event":
            n_valid = results.pop("query_n_valid")
            if n_valid > 0:
                results["query_event_underflow_rate"] = results.pop("query_n_underflow") / n_valid
                results["query_event_overflow_rate"] = results.pop("query_n_overflow") / n_valid
                results["query_event_in_range"] = results.pop("query_n_in_range") / n_valid
            else:
                results.pop("query_n_underflow")
                results.pop("query_n_overflow")
                results.pop("query_n_in_range")
                results["query_event_underflow_rate"] = 0.0
                results["query_event_overflow_rate"] = 0.0
                results["query_event_in_range"] = 0.0

        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clipping,
                error_if_nonfinite=True,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    # Parse comma-separated baseline types into a list
    if hasattr(config, "baseline_types") and isinstance(config.baseline_types, str):
        config.baseline_types = [b.strip() for b in config.baseline_types.split(",") if b.strip()]

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    trainer = Trainer(config)
    trainer.train()
