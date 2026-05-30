# 3-stage survival pretraining curriculum with progressive freezing.
# All data is generated ON-THE-FLY — zero disk storage per batch.
#
# Usage:
#   export SURVIVAL_CHECKPOINT_DIR=/path/to/checkpoints
#   export SURVIVAL_WANDB_DIR=/path/to/wandb
#   bash scripts/train_survival_curriculum.sh
#
# Or override individual settings inline.
#
# Prerequisites:
#   pip install tabicl transformers wandb
#   export NCCL_DEBUG=WARN   (for multi-GPU DDP)
#
# Freezing scheme:
#   Stage 1: freeze ColEmbedding + RowInteraction  → train ICL transformer + survival head (~55%)
#   Stage 2: freeze ColEmbedding + RowInteraction  → train ICL transformer + survival head (~55%)
#   Stage 3: unfreeze all                           → full fine-tune on large data (~100%)
#
# Set environment variables or edit defaults below.
# ---------------------------------------------------------------------------

set -euo pipefail

CHECKPOINT_DIR="${SURVIVAL_CHECKPOINT_DIR:-./checkpoints}"
WANDB_DIR="${SURVIVAL_WANDB_DIR:-./wandb}"
NPROC="${NPROC_PER_NODE:-1}"

mkdir -p "${CHECKPOINT_DIR}" "${WANDB_DIR}"

##############################################################################
# Proportional Hazard (PH) — 4 baselines (Weibull, Gompertz, LogLogistic, LogNormal)
##############################################################################

# Stage 1: small fixed-length, freeze embeddings, train ICL+head
torchrun --standalone --nproc_per_node="${NPROC}" src/tabicl/train/_run.py \
    --task survival \
    --wandb_log True \
    --wandb_project TabICL-Survival \
    --wandb_name Survival-PH-Stage1 \
    --wandb_dir "${WANDB_DIR}" \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 100000 \
    --batch_size 512 \
    --micro_batch_size 4 \
    --lr 1e-4 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.02 \
    --gradient_clipping 1.0 \
    --prior_type mlp_scm \
    --prior_device cpu \
    --batch_size_per_gp 4 \
    --min_features 2 \
    --max_features 100 \
    --max_seq_len 1024 \
    --min_train_size 1.0 \
    --max_train_size 1.0 \
    --survival_model_type ph \
    --survival_beta 1.0 \
    --baseline_types weibull,gompertz,loglogistic,lognormal \
    --baseline_mode mix \
    --min_censor_scale 1.0 \
    --max_censor_scale 5.0 \
    --min_event_rate 0.40 \
    --max_event_rate 1.0 \
    --num_bins 50 \
    --alpha_start 3.0 \
    --alpha_floor 0.05 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --freeze_col True \
    --freeze_row True \
    --checkpoint_dir "${CHECKPOINT_DIR}/survival_ph_stage1" \
    --save_temp_every 50 \
    --save_perm_every 5000

# Stage 2: variable-length, freeze embeddings, load Stage 1 weights
torchrun --standalone --nproc_per_node="${NPROC}" src/tabicl/train/_run.py \
    --task survival \
    --wandb_log True \
    --wandb_project TabICL-Survival \
    --wandb_name Survival-PH-Stage2 \
    --wandb_dir "${WANDB_DIR}" \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 2000 \
    --batch_size 512 \
    --micro_batch_size 2 \
    --lr 1e-5 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.02 \
    --gradient_clipping 1.0 \
    --prior_type mlp_scm \
    --prior_device cpu \
    --batch_size_per_gp 2 \
    --min_features 2 \
    --max_features 100 \
    --min_seq_len 1000 \
    --max_seq_len 40000 \
    --log_seq_len True \
    --seq_len_per_gp True \
    --min_train_size 1.0 \
    --max_train_size 1.0 \
    --survival_model_type ph \
    --survival_beta 1.0 \
    --baseline_types weibull,gompertz,loglogistic,lognormal \
    --baseline_mode mix \
    --min_censor_scale 1.0 \
    --max_censor_scale 5.0 \
    --min_event_rate 0.40 \
    --max_event_rate 1.0 \
    --num_bins 50 \
    --alpha_start 3.0 \
    --alpha_floor 0.05 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --freeze_col True \
    --freeze_row True \
    --pretrained_path "${CHECKPOINT_DIR}/survival_ph_stage1/step-100000.ckpt" \
    --checkpoint_dir "${CHECKPOINT_DIR}/survival_ph_stage2" \
    --save_temp_every 50 \
    --save_perm_every 500

# Stage 3: large variable-length, unfreeze all, full fine-tune
torchrun --standalone --nproc_per_node="${NPROC}" src/tabicl/train/_run.py \
    --task survival \
    --wandb_log True \
    --wandb_project TabICL-Survival \
    --wandb_name Survival-PH-Stage3 \
    --wandb_dir "${WANDB_DIR}" \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 50 \
    --batch_size 512 \
    --micro_batch_size 1 \
    --lr 1e-5 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.02 \
    --gradient_clipping 1.0 \
    --prior_type mlp_scm \
    --prior_device cpu \
    --batch_size_per_gp 1 \
    --min_features 2 \
    --max_features 100 \
    --min_seq_len 40000 \
    --max_seq_len 60000 \
    --log_seq_len True \
    --seq_len_per_gp True \
    --replay_small True \
    --min_train_size 1.0 \
    --max_train_size 1.0 \
    --survival_model_type ph \
    --survival_beta 1.0 \
    --baseline_types weibull,gompertz,loglogistic,lognormal \
    --baseline_mode mix \
    --min_censor_scale 1.0 \
    --max_censor_scale 5.0 \
    --min_event_rate 0.40 \
    --max_event_rate 1.0 \
    --num_bins 50 \
    --alpha_start 3.0 \
    --alpha_floor 0.05 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --pretrained_path "${CHECKPOINT_DIR}/survival_ph_stage2/step-2000.ckpt" \
    --checkpoint_dir "${CHECKPOINT_DIR}/survival_ph_stage3" \
    --save_temp_every 10 \
    --save_perm_every 10


##############################################################################
# Accelerated Failure Time (AFT) — 3 baselines (Weibull, LogLogistic, LogNormal)
##############################################################################

# Stage 1
torchrun --standalone --nproc_per_node="${NPROC}" src/tabicl/train/_run.py \
    --task survival \
    --wandb_log True \
    --wandb_project TabICL-Survival \
    --wandb_name Survival-AFT-Stage1 \
    --wandb_dir "${WANDB_DIR}" \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 100000 \
    --batch_size 512 \
    --micro_batch_size 4 \
    --lr 1e-4 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.02 \
    --gradient_clipping 1.0 \
    --prior_type mlp_scm \
    --prior_device cpu \
    --batch_size_per_gp 4 \
    --min_features 2 \
    --max_features 100 \
    --max_seq_len 1024 \
    --min_train_size 1.0 \
    --max_train_size 1.0 \
    --survival_model_type aft \
    --survival_beta 1.0 \
    --baseline_types weibull,loglogistic,lognormal \
    --baseline_mode mix \
    --min_censor_scale 1.0 \
    --max_censor_scale 5.0 \
    --min_event_rate 0.40 \
    --max_event_rate 1.0 \
    --num_bins 50 \
    --alpha_start 3.0 \
    --alpha_floor 0.05 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --freeze_col True \
    --freeze_row True \
    --checkpoint_dir "${CHECKPOINT_DIR}/survival_aft_stage1" \
    --save_temp_every 50 \
    --save_perm_every 5000

# Stage 2
torchrun --standalone --nproc_per_node="${NPROC}" src/tabicl/train/_run.py \
    --task survival \
    --wandb_log True \
    --wandb_project TabICL-Survival \
    --wandb_name Survival-AFT-Stage2 \
    --wandb_dir "${WANDB_DIR}" \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 2000 \
    --batch_size 512 \
    --micro_batch_size 2 \
    --lr 1e-5 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.02 \
    --gradient_clipping 1.0 \
    --prior_type mlp_scm \
    --prior_device cpu \
    --batch_size_per_gp 2 \
    --min_features 2 \
    --max_features 100 \
    --min_seq_len 1000 \
    --max_seq_len 40000 \
    --log_seq_len True \
    --seq_len_per_gp True \
    --min_train_size 1.0 \
    --max_train_size 1.0 \
    --survival_model_type aft \
    --survival_beta 1.0 \
    --baseline_types weibull,loglogistic,lognormal \
    --baseline_mode mix \
    --min_censor_scale 1.0 \
    --max_censor_scale 5.0 \
    --min_event_rate 0.40 \
    --max_event_rate 1.0 \
    --num_bins 50 \
    --alpha_start 3.0 \
    --alpha_floor 0.05 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --freeze_col True \
    --freeze_row True \
    --pretrained_path "${CHECKPOINT_DIR}/survival_aft_stage1/step-100000.ckpt" \
    --checkpoint_dir "${CHECKPOINT_DIR}/survival_aft_stage2" \
    --save_temp_every 50 \
    --save_perm_every 500

# Stage 3 — Unfreeze all
torchrun --standalone --nproc_per_node="${NPROC}" src/tabicl/train/_run.py \
    --task survival \
    --wandb_log True \
    --wandb_project TabICL-Survival \
    --wandb_name Survival-AFT-Stage3 \
    --wandb_dir "${WANDB_DIR}" \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 50 \
    --batch_size 512 \
    --micro_batch_size 1 \
    --lr 1e-5 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.02 \
    --gradient_clipping 1.0 \
    --prior_type mlp_scm \
    --prior_device cpu \
    --batch_size_per_gp 1 \
    --min_features 2 \
    --max_features 100 \
    --min_seq_len 40000 \
    --max_seq_len 60000 \
    --log_seq_len True \
    --seq_len_per_gp True \
    --replay_small True \
    --min_train_size 1.0 \
    --max_train_size 1.0 \
    --survival_model_type aft \
    --survival_beta 1.0 \
    --baseline_types weibull,loglogistic,lognormal \
    --baseline_mode mix \
    --min_censor_scale 1.0 \
    --max_censor_scale 5.0 \
    --min_event_rate 0.40 \
    --max_event_rate 1.0 \
    --num_bins 50 \
    --alpha_start 3.0 \
    --alpha_floor 0.05 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --pretrained_path "${CHECKPOINT_DIR}/survival_aft_stage2/step-2000.ckpt" \
    --checkpoint_dir "${CHECKPOINT_DIR}/survival_aft_stage3" \
    --save_temp_every 10 \
    --save_perm_every 10
