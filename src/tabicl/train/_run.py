from __future__ import annotations

import os
import timeit
import warnings
import functools
from contextlib import nullcontext

import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm

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

        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_prior()
        if self.survival:
            self.configure_binner()
            self.configure_loss()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()

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

    def build_model(self):
        """Build and initialize the TabICL model."""

        num_bins = getattr(self.config, "num_bins", 50)

        if self.survival:
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
                pretrained_state = model.state_dict()

                # Rebuild as survival: swap y_encoder + decoder
                icl_dim = model.embed_dim * model.row_num_cls
                model.icl_predictor.survival = True
                model.icl_predictor.y_encoder = nn.Linear(2, icl_dim).to(self.config.device)
                from tabicl.survival import DiscreteTimeSurvivalHead
                model.icl_predictor.decoder = DiscreteTimeSurvivalHead(d_model=icl_dim, num_bins=num_bins).to(self.config.device)
                model.survival = True

                self.model_config = {
                    "max_classes": 0, "num_quantiles": num_bins, "survival": True,
                    "embed_dim": model.embed_dim,
                    "col_num_blocks": model.col_embedder.num_blocks if hasattr(model.col_embedder, 'num_blocks') else self.config.col_num_blocks,
                    "col_nhead": self.config.col_nhead,
                    "col_num_inds": self.config.col_num_inds,
                    "row_num_blocks": self.config.row_num_blocks,
                    "row_nhead": self.config.row_nhead,
                    "row_num_cls": model.row_num_cls,
                    "row_rope_base": self.config.row_rope_base,
                    "icl_num_blocks": self.config.icl_num_blocks,
                    "icl_nhead": self.config.icl_nhead,
                    "ff_factor": self.config.ff_factor,
                    "dropout": self.config.dropout,
                    "activation": self.config.activation,
                    "norm_first": self.config.norm_first,
                }
                if self.master_process:
                    print(f"Survival head + y_encoder attached. Encoder weights preserved from HF Hub.")
            else:
                ckpt = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
                pretrained_state = ckpt.get("state_dict", ckpt)

                # Detect whether survival or regressor
                is_survival_ckpt = any(
                    k.startswith("icl_predictor.decoder.head.")
                    for k in pretrained_state
                )

                if is_survival_ckpt:
                    # Survival checkpoint — load via regressor for correct architecture
                    if self.master_process:
                        print(f"Loading survival checkpoint ({checkpoint_path}) for full weight transfer...")
                    from tabicl._sklearn.regressor import TabICLRegressor
                    # The HF Hub regressor gives us the correct architecture (col_nhead=4 etc.)
                    tmp_loader = TabICLRegressor(
                        n_estimators=1, model_path=None,
                        allow_auto_download=True, device=self.config.device,
                    )
                    tmp_loader._resolve_device()
                    tmp_loader._load_model()
                    model = tmp_loader.model_

                    # Convert to survival, then load the full state dict
                    from tabicl.survival import DiscreteTimeSurvivalHead
                    icl_dim = model.embed_dim * model.row_num_cls
                    model.icl_predictor.survival = True
                    model.icl_predictor.y_encoder = nn.Linear(2, icl_dim).to(self.config.device)
                    model.icl_predictor.decoder = DiscreteTimeSurvivalHead(d_model=icl_dim, num_bins=num_bins).to(self.config.device)
                    model.survival = True
                    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
                    self.model_config = {
                        "max_classes": 0, "num_quantiles": num_bins, "survival": True,
                        "embed_dim": model.embed_dim,
                        "col_num_blocks": self.config.col_num_blocks,
                        "col_nhead": self.config.col_nhead,
                        "col_num_inds": self.config.col_num_inds,
                        "row_num_blocks": self.config.row_num_blocks,
                        "row_nhead": self.config.row_nhead,
                        "row_num_cls": model.row_num_cls,
                        "row_rope_base": self.config.row_rope_base,
                        "icl_num_blocks": self.config.icl_num_blocks,
                        "icl_nhead": self.config.icl_nhead,
                        "ff_factor": self.config.ff_factor,
                        "dropout": self.config.dropout,
                        "activation": self.config.activation,
                        "norm_first": self.config.norm_first,
                    }
                    if self.master_process:
                        print(f"Loaded full survival checkpoint: {len(pretrained_state)} keys; "
                              f"{len(missing)} missing, {len(unexpected)} unexpected.")
                else:
                    # Regressor checkpoint — detach model from loader for architecture match
                    if self.master_process:
                        print(f"Loading regressor checkpoint for encoder weights...")
                    from tabicl._sklearn.regressor import TabICLRegressor
                    tmp_loader = TabICLRegressor(
                        n_estimators=1, model_path=checkpoint_path,
                        allow_auto_download=False, device=self.config.device,
                    )
                    tmp_loader._resolve_device()
                    tmp_loader._load_model()
                    model = tmp_loader.model_

                    # Convert to survival
                    icl_dim = model.embed_dim * model.row_num_cls
                    model.icl_predictor.survival = True
                    model.icl_predictor.y_encoder = nn.Linear(2, icl_dim).to(self.config.device)
                    from tabicl.survival import DiscreteTimeSurvivalHead
                    model.icl_predictor.decoder = DiscreteTimeSurvivalHead(d_model=icl_dim, num_bins=num_bins).to(self.config.device)
                    model.survival = True

                    self.model_config = {
                        "max_classes": 0, "num_quantiles": num_bins, "survival": True,
                        "embed_dim": model.embed_dim,
                        "col_num_blocks": self.config.col_num_blocks,
                        "col_nhead": self.config.col_nhead,
                        "col_num_inds": self.config.col_num_inds,
                        "row_num_blocks": self.config.row_num_blocks,
                        "row_nhead": self.config.row_nhead,
                        "row_num_cls": model.row_num_cls,
                        "row_rope_base": self.config.row_rope_base,
                        "icl_num_blocks": self.config.icl_num_blocks,
                        "icl_nhead": self.config.icl_nhead,
                        "ff_factor": self.config.ff_factor,
                        "dropout": self.config.dropout,
                        "activation": self.config.activation,
                        "norm_first": self.config.norm_first,
                    }
                    if self.master_process:
                        print(f"Encoder weights loaded from regressor checkpoint. Survival head attached.")
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

        # num_workers=0: fork is unsafe with PyTorch+CUDA loaded on any platform.
        # On-the-fly data generation runs synchronously in the main process.
        # This is fast enough for CPU-side SCM generation (<< GPU compute time).
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def configure_binner(self):
        """Build fixed bins on the standardized log-time axis."""
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
        """Set up the hybrid survival loss.

        The loss receives global ``curr_step`` values during training, so
        ``max_steps`` is the absolute decay horizon. Chunked training can
        override it with ``alpha_total_steps`` to keep the decay horizon
        independent of the current chunk endpoint.
        """
        from tabicl.survival import HybridSurvivalLoss

        alpha_horizon = getattr(self.config, "alpha_total_steps", None)
        if alpha_horizon is None:
            alpha_horizon = self.config.max_steps
        self.surv_loss_fn = HybridSurvivalLoss(
            alpha_start=getattr(self.config, "alpha_start", 3.0),
            alpha_floor=getattr(self.config, "alpha_floor", 0.05),
            max_steps=alpha_horizon,
        )

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
        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(
                device_type="cuda", dtype=torch.float16 if self.config.dtype == "float16" else torch.float32
            )
        else:
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

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        For survival checkpoints, reconstructs the survival head and
        TimeBinner from checkpoint metadata before loading weights.
        """
        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)

        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        # Restore survival metadata if present
        surv_meta = checkpoint.get("survival_metadata", None)
        if surv_meta is not None and self.survival:
            if surv_meta.get("time_scale") == "km_hybrid_log":
                from tabicl.survival import TimeBinner
                self.binner = TimeBinner(
                    bin_edges=surv_meta["binner_edges"].to(self.config.device),
                    bin_means=surv_meta["binner_means"].to(self.config.device),
                )
                self.binner = self.binner.to(torch.device(self.config.device))
                self.survival_time_scaler_config = surv_meta.get(
                    "time_scaler",
                    getattr(self, "survival_time_scaler_config", {
                        "eps": getattr(self.config, "survival_time_eps", 1e-8),
                        "min_scale": getattr(self.config, "survival_time_min_scale", 0.1),
                        "z_min": getattr(self.config, "survival_time_z_min", -6.0),
                        "z_max": getattr(self.config, "survival_time_z_max", 6.0),
                    }),
                )
                print(f"Restored TimeBinner from checkpoint: {self.binner}")
            else:
                print("Checkpoint has legacy raw-time survival metadata; keeping standardized TimeBinner.")

        self.raw_model.load_state_dict(checkpoint["state_dict"])

        if self.config.only_load_model:
            print("Only loading model weights")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            if "scaler_state" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state"])
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")

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
        t_event_test,
        train_sizes_ds,
        query_sizes_ds,
    ):
        from tabicl.survival import standardize_survival_micro_batch

        scaler_kwargs = getattr(self, "survival_time_scaler_config", {
            "eps": getattr(self.config, "survival_time_eps", 1e-8),
            "min_scale": getattr(self.config, "survival_time_min_scale", 0.1),
            "z_min": getattr(self.config, "survival_time_z_min", -6.0),
            "z_max": getattr(self.config, "survival_time_z_max", 6.0),
        })
        return standardize_survival_micro_batch(
            t_train,
            delta_train,
            t_test,
            delta_test,
            t_event_test,
            train_sizes_ds,
            query_sizes_ds,
            scaler_kwargs,
        )

    def _run_micro_batch_survival(self, micro_batch, micro_batch_idx, num_micro_batches):
        (micro_X, micro_t, micro_delta, micro_t_event,
         micro_d, micro_seq_len, micro_train_size) = micro_batch
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
        # After padding, each row uses first `train_size_ds` for context,
        # next `seq_len_ds - train_size_ds` for query, rest is padding.
        train_sizes_ds = (per_ds_seq_lens / 2).long()  # (B,) half as context
        query_sizes_ds = per_ds_seq_lens - train_sizes_ds  # (B,)

        t_train = micro_t[:, :train_size]  # (B, train_size) — context
        t_test = micro_t[:, train_size:]   # (B, T - train_size) — query (includes padding)
        delta_train = micro_delta[:, :train_size]
        delta_test = micro_delta[:, train_size:]
        t_event_test = micro_t_event[:, train_size:]

        t_train, delta_train, t_test, delta_test, t_event_test = self._standardize_survival_micro_batch(
            t_train, delta_train, t_test, delta_test, t_event_test, train_sizes_ds, query_sizes_ds,
        )

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            # Pass d=None — the HF Hub regressor has feature grouping enabled,
            # which asserts d is None in the column embedder.
            h_raw = self.model(micro_X, t_train, None, delta_train=delta_train)
            # h_raw: (B, T_test, K) — flatten for loss
            B, T_test, K = h_raw.shape
            h_raw_flat = h_raw.reshape(-1, K)
            t_test_flat = t_test.reshape(-1)
            delta_test_flat = delta_test.reshape(-1)
            t_event_test_flat = t_event_test.reshape(-1)

            # Build padding mask: valid positions are those within query_sizes_ds
            position_idx = torch.arange(T_test, device=self.config.device).unsqueeze(0)  # (1, T_test)
            valid_mask = position_idx < query_sizes_ds.unsqueeze(1)  # (B, T_test)
            valid_mask_flat = valid_mask.reshape(-1)  # (B * T_test,)

            if valid_mask_flat.all():
                loss, comps = self.surv_loss_fn(
                    h_raw_flat, t_test_flat, delta_test_flat, t_event_test_flat,
                    self.binner, step=self.curr_step,
                )
            else:
                # Masked loss: compute on valid positions only
                loss, comps = self.surv_loss_fn(
                    h_raw_flat[valid_mask_flat], t_test_flat[valid_mask_flat],
                    delta_test_flat[valid_mask_flat], t_event_test_flat[valid_mask_flat],
                    self.binner, step=self.curr_step,
                )

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {
                "surv_nll": comps["surv_nll"] / num_micro_batches,
                "impute": comps["impute"] / num_micro_batches,
                "alpha": comps["alpha"],
            }

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

        # Pad nested tensors to the same size
        batch_padded = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        num_micro_batches = math.ceil(self.config.batch_size / self.config.micro_batch_size)
        micro_batches = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch_padded]
        micro_batches = list(zip(*micro_batches))

        if self.survival:
            results = {"surv_nll": 0.0, "impute": 0.0, "alpha": 0.0}
        else:
            results = {"ce": 0.0, "accuracy": 0.0}
        failed_batches = 0

        for idx, mb in enumerate(micro_batches):
            try:
                micro_results = self.run_micro_batch(mb, idx, num_micro_batches)
                for k, v in micro_results.items():
                    if k == "alpha":
                        results[k] = v  # alpha is not accumulated
                    else:
                        results[k] += v
            except torch.cuda.OutOfMemoryError:
                print(f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} "
                      f"at step {self.curr_step}. Skipping.")
                torch.cuda.empty_cache()
                failed_batches += 1
                continue

        failure_ratio = failed_batches / num_micro_batches
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)

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
