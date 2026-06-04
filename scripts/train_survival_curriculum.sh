#!/bin/bash
# Single mixed PH/AFT survival pretraining curriculum.
# Three survival stages adapted from the original TabICL training plan.
#
# Usage:
#   bash scripts/train_survival_curriculum.sh
#
# Environment overrides:
#   CHECKPOINT_DIR     — base checkpoint directory (default: ./checkpoints)
#   WANDB_DIR          — wandb log directory (default: ./wandb)
#   NPROC_PER_NODE     — GPUs per node (default: 1)
#   WANDB_MODE         — wandb mode: online/offline/disabled (default: offline)
#   RUN_STAGES         — which stages to run: 1,2,3 or 1,2 or 1 (default: 1,2,3)
#                        Token-validated; empty/invalid/duplicate rejected.
#   CURRICULUM_ID      — checkpoint namespace (default: author_adapted_v1)
#   STAGE1_STEPS       — override Stage 1 max steps (default: 100000)
#   STAGE2_STEPS       — override Stage 2 max steps (default: 2000)
#   STAGE3_STEPS       — override Stage 3 max steps (default: 50)
#   SURVIVAL_QUERY_PINBALL_WEIGHT
#                      — optional oracle-query pinball weight (default: 0.0)
#   SURVIVAL_QUERY_PINBALL_QUANTILES
#                      — comma-separated pinball quantiles
#
#   STAGE*_STEPS overrides must be divisible by the stage's effective
#   checkpoint interval (gcd of save_temp_every=50 and the stage's
#   save_perm_every).  Non-aligned overrides error before training.
#   Stale higher-step checkpoints in the stage directory are also
#   rejected to prevent silent unwanted resume.
#
# Freezing scheme:
#   Stage 1: train all modules
#   Stage 2: train all modules
#   Stage 3: freeze ColEmbedding + RowInteraction  → train ICL + survival head
#
# Optimization scheme:
#   Stage 1: 1e-4 cosine decay with 2% warmup
#   Stage 2: 2e-5 → 5e-6 polynomial decay without warmup
#   Stage 3: constant 2e-6
#
# Pre-requisites:
#   pip install tabicl transformers wandb
#   export NCCL_DEBUG=WARN   (for multi-GPU DDP)
# ---------------------------------------------------------------------------

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
WANDB_DIR="${WANDB_DIR:-./wandb}"
NPROC="${NPROC_PER_NODE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_STAGES="${RUN_STAGES-1,2,3}"
CURRICULUM_ID="${CURRICULUM_ID:-author_adapted_v1}"

STAGE1_STEPS="${STAGE1_STEPS:-100000}"
STAGE2_STEPS="${STAGE2_STEPS:-2000}"
STAGE3_STEPS="${STAGE3_STEPS:-50}"
SURVIVAL_QUERY_PINBALL_WEIGHT="${SURVIVAL_QUERY_PINBALL_WEIGHT:-0.0}"
SURVIVAL_QUERY_PINBALL_QUANTILES="${SURVIVAL_QUERY_PINBALL_QUANTILES:-0.1,0.25,0.5,0.75,0.9}"

if [[ ! "$CURRICULUM_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: CURRICULUM_ID must contain only letters, numbers, '.', '_', or '-'" >&2
    exit 1
fi

STAGE1_DIR="${CHECKPOINT_DIR}/survival_mix_${CURRICULUM_ID}_stage1"
STAGE2_DIR="${CHECKPOINT_DIR}/survival_mix_${CURRICULUM_ID}_stage2"
STAGE3_DIR="${CHECKPOINT_DIR}/survival_mix_${CURRICULUM_ID}_stage3"

# ── Parse and validate RUN_STAGES ──────────────────────────────────────
# Reject empty tokens, invalid stage numbers, and duplicates.
# Sets _run_1, _run_2, _run_3 booleans for downstream dispatch.
if [[ -z "${RUN_STAGES//[[:space:]]/}" ]]; then
    echo "ERROR: RUN_STAGES is empty — nothing to run" >&2
    exit 1
fi
if [[ "$RUN_STAGES" =~ (^|,)[[:space:]]*(,|$) ]]; then
    echo "ERROR: RUN_STAGES contains an empty token" >&2
    exit 1
fi

_run_1=0; _run_2=0; _run_3=0; _run_count=0
IFS=',' read -ra _tokens <<< "$RUN_STAGES"
for _tok in "${_tokens[@]}"; do
    _tok="${_tok#"${_tok%%[![:space:]]*}"}"  # ltrim
    _tok="${_tok%"${_tok##*[![:space:]]}"}"  # rtrim
    case "$_tok" in
        1) if (( _run_1 )); then echo "ERROR: Duplicate stage '1'" >&2; exit 1; fi
           _run_1=1;;
        2) if (( _run_2 )); then echo "ERROR: Duplicate stage '2'" >&2; exit 1; fi
           _run_2=1;;
        3) if (( _run_3 )); then echo "ERROR: Duplicate stage '3'" >&2; exit 1; fi
           _run_3=1;;
        "") echo "ERROR: RUN_STAGES contains an empty token" >&2; exit 1;;
        *)  echo "ERROR: Invalid stage '${_tok}' in RUN_STAGES (must be 1, 2, or 3)" >&2; exit 1;;
    esac
    ((_run_count += 1))
done
if (( _run_count == 0 )); then
    echo "ERROR: RUN_STAGES is empty — nothing to run" >&2; exit 1
fi

# Validate that every selected stage produces a checkpoint.
# Effective interval = gcd(save_temp_every=50, stage save_perm_every).
# Checkpoints are written when step % temp_every == 0 OR step % perm_every == 0.
# Stage 1: gcd(50,5000)=50   Stage 2: gcd(50,500)=50   Stage 3: gcd(50,10)=10
_stg1_every=50   # gcd(save_temp_every=50, save_perm_every=5000)
_stg2_every=50   # gcd(save_temp_every=50, save_perm_every=500)
_stg3_every=10   # gcd(save_temp_every=50, save_perm_every=10)

guard_stale_checkpoints() {
    local stage_number="$1"
    local checkpoint_dir="$2"
    local requested_steps="$3"
    local max_stale=0
    local checkpoint_file checkpoint_name checkpoint_step

    [[ -d "$checkpoint_dir" ]] || return 0

    for checkpoint_file in "$checkpoint_dir"/step-*.ckpt; do
        [[ -f "$checkpoint_file" ]] || continue
        checkpoint_name="${checkpoint_file##*/}"
        if [[ "$checkpoint_name" =~ ^step-([0-9]+).*\.ckpt$ ]]; then
            checkpoint_step="${BASH_REMATCH[1]}"
            if (( checkpoint_step > max_stale )); then
                max_stale="$checkpoint_step"
            fi
        fi
    done

    if (( max_stale > requested_steps )); then
        echo "ERROR: Stage ${stage_number} checkpoint dir has step-${max_stale}.ckpt," >&2
        echo "       which exceeds the requested ${requested_steps} steps." >&2
        echo "       The trainer would resume from step ${max_stale} —" >&2
        echo "       clean the checkpoint directory or increase STAGE${stage_number}_STEPS." >&2
        exit 1
    fi
}

if (( _run_1 )); then
    if [[ ! "$STAGE1_STEPS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Stage1 steps must be a positive decimal integer (got ${STAGE1_STEPS})" >&2; exit 1
    fi
    # Normalize: strip leading zeros, enforce base-10, persist for downstream use.
    # The trainer saves "step-50.ckpt" not "step-00050.ckpt", so chaining
    # must use the canonical form.
    STAGE1_STEPS=$((10#${STAGE1_STEPS}))
    if (( STAGE1_STEPS <= 0 )); then
        echo "ERROR: Stage1 steps must be a positive integer (got ${STAGE1_STEPS})" >&2; exit 1
    fi
    if (( STAGE1_STEPS % _stg1_every != 0 )); then
        echo "ERROR: Stage1 steps (${STAGE1_STEPS}) must be divisible by ${_stg1_every}" >&2
        echo "       to produce a checkpoint." >&2; exit 1
    fi
    guard_stale_checkpoints 1 "${STAGE1_DIR}" "${STAGE1_STEPS}"
fi
if (( _run_2 )); then
    if [[ ! "$STAGE2_STEPS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Stage2 steps must be a positive decimal integer (got ${STAGE2_STEPS})" >&2; exit 1
    fi
    STAGE2_STEPS=$((10#${STAGE2_STEPS}))
    if (( STAGE2_STEPS <= 0 )); then
        echo "ERROR: Stage2 steps must be a positive integer (got ${STAGE2_STEPS})" >&2; exit 1
    fi
    if (( STAGE2_STEPS % _stg2_every != 0 )); then
        echo "ERROR: Stage2 steps (${STAGE2_STEPS}) must be divisible by ${_stg2_every}" >&2
        echo "       to produce a checkpoint." >&2; exit 1
    fi
    guard_stale_checkpoints 2 "${STAGE2_DIR}" "${STAGE2_STEPS}"
fi
if (( _run_3 )); then
    if [[ ! "$STAGE3_STEPS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Stage3 steps must be a positive decimal integer (got ${STAGE3_STEPS})" >&2; exit 1
    fi
    STAGE3_STEPS=$((10#${STAGE3_STEPS}))
    if (( STAGE3_STEPS <= 0 )); then
        echo "ERROR: Stage3 steps must be a positive integer (got ${STAGE3_STEPS})" >&2; exit 1
    fi
    if (( STAGE3_STEPS % _stg3_every != 0 )); then
        echo "ERROR: Stage3 steps (${STAGE3_STEPS}) must be divisible by ${_stg3_every}" >&2
        echo "       to produce a checkpoint." >&2; exit 1
    fi
    guard_stale_checkpoints 3 "${STAGE3_DIR}" "${STAGE3_STEPS}"
fi

mkdir -p "${CHECKPOINT_DIR}" "${WANDB_DIR}"

# ── Shared survival prior configuration ─────────────────────────────────
read -r -d '' SURVIVAL_FLAGS <<EOF || true
--survival_model_type mix
--baseline_types weibull,gompertz,loglogistic,lognormal
--baseline_mode mix
--beta_sampling log_uniform --min_beta 0.25 --max_beta 2.0
--baseline_param_prior broad
--time_scale_sampling log_uniform --min_time_scale 0.2 --max_time_scale 5.0
--censoring_strategy target_event_rate
--min_event_rate 0.40 --max_event_rate 0.90
--prior_type mlp_scm
--prior_device cpu
--min_train_size 1.0 --max_train_size 1.0
--survival_query_supervision event
--censor_calibration_scope context
--survival_query_pinball_weight ${SURVIVAL_QUERY_PINBALL_WEIGHT}
--survival_query_pinball_quantiles ${SURVIVAL_QUERY_PINBALL_QUANTILES}
EOF

# ── Shared model architecture ───────────────────────────────────────────
read -r -d '' MODEL_FLAGS <<'EOF' || true
--embed_dim 128
--col_num_blocks 3 --col_nhead 4 --col_num_inds 128
--row_num_blocks 3 --row_nhead 8 --row_num_cls 4 --row_rope_base 100000
--icl_num_blocks 12 --icl_nhead 4
--ff_factor 2 --norm_first True
EOF

# ── Shared training flags ───────────────────────────────────────────────
read -r -d '' TRAIN_FLAGS <<'EOF' || true
--task survival
--device cuda
--dtype float32
--amp True
--np_seed 42 --torch_seed 42
--batch_size 512
--gradient_clipping 1.0
--num_bins 50
--min_features 2 --max_features 100
--save_temp_every 50 --save_perm_every 5000
EOF

run_stage() {
    local stage_label="$1" checkpoint_dir="$2" max_steps="$3"
    local pretrained_path="${4:-}"
    shift 4  # remaining args are stage-specific flags

    echo ""; echo "============================================"
    echo "  Stage ${stage_label}"
    echo "  Checkpoint dir: ${checkpoint_dir}"
    echo "  Max steps:      ${max_steps}"
    [[ -n "$pretrained_path" ]] && echo "  Pretrained:     ${pretrained_path}"
    echo "============================================"; echo ""

    local extra_args=()
    [[ -n "$pretrained_path" ]] && extra_args+=(--pretrained_path "$pretrained_path")
    # Collect remaining stage-specific flags
    extra_args+=("$@")

    torchrun --standalone --nproc_per_node="${NPROC}" src/tabicl/train/_run.py \
        ${TRAIN_FLAGS} \
        ${MODEL_FLAGS} \
        ${SURVIVAL_FLAGS} \
        --wandb_log True \
        --wandb_project TabICL-Survival \
        --wandb_name "Survival-Mix-${stage_label}" \
        --wandb_dir "${WANDB_DIR}" \
        --wandb_mode "${WANDB_MODE}" \
        --max_steps "${max_steps}" \
        --checkpoint_dir "${checkpoint_dir}" \
        "${extra_args[@]}"
}

##############################################################################
# Stage 1 — small fixed-length, full-model adaptation
##############################################################################
if (( _run_1 )); then
    run_stage "Stage1" "${STAGE1_DIR}" "${STAGE1_STEPS}" "" \
        --lr 1e-4 \
        --scheduler cosine_warmup \
        --warmup_proportion 0.02 \
        --batch_size_per_gp 4 \
        --micro_batch_size 4 \
        --max_seq_len 1024 \
        --save_perm_every 5000
fi

##############################################################################
# Stage 2 — medium variable-length, full-model refinement, load Stage 1
##############################################################################
if (( _run_2 )); then
    if (( _run_1 )); then
        # Stage 1 was just run — chain from its requested checkpoint.
        STAGE1_CKPT="${STAGE1_DIR}/step-${STAGE1_STEPS}.ckpt"
        if [[ ! -f "${STAGE1_CKPT}" ]]; then
            echo "ERROR: Expected Stage 1 checkpoint not found: ${STAGE1_CKPT}" >&2
            echo "       Stage 1 may have failed or a stale checkpoint was cleaned." >&2; exit 1
        fi
    else
        # Stage 1 was skipped — use the latest checkpoint from a prior run.
        STAGE1_CKPT=$(ls -1 "${STAGE1_DIR}"/step-*.ckpt 2>/dev/null | sort -V | tail -1 || true)
        if [[ -z "${STAGE1_CKPT:-}" ]]; then
            echo "ERROR: No Stage 1 checkpoint found." >&2; exit 1
        fi
    fi
    echo "Chaining from Stage 1 checkpoint: $(basename "$STAGE1_CKPT")"
    run_stage "Stage2" "${STAGE2_DIR}" "${STAGE2_STEPS}" \
        "${STAGE1_CKPT}" \
        --lr 2e-5 \
        --scheduler polynomial_decay_warmup \
        --warmup_proportion 0 \
        --poly_decay_lr_end 5e-6 \
        --poly_decay_power 2.0 \
        --batch_size_per_gp 2 \
        --micro_batch_size 2 \
        --min_seq_len 1000 --max_seq_len 40000 \
        --log_seq_len True --seq_len_per_gp True \
        --save_perm_every 500
fi

##############################################################################
# Stage 3 — large variable-length, freeze encoders, load Stage 2
##############################################################################
if (( _run_3 )); then
    if (( _run_2 )); then
        # Stage 2 was just run — chain from its requested checkpoint.
        STAGE2_CKPT="${STAGE2_DIR}/step-${STAGE2_STEPS}.ckpt"
        if [[ ! -f "${STAGE2_CKPT}" ]]; then
            echo "ERROR: Expected Stage 2 checkpoint not found: ${STAGE2_CKPT}" >&2
            echo "       Stage 2 may have failed or a stale checkpoint was cleaned." >&2; exit 1
        fi
    else
        # Stage 2 was skipped — use the latest checkpoint from a prior run.
        STAGE2_CKPT=$(ls -1 "${STAGE2_DIR}"/step-*.ckpt 2>/dev/null | sort -V | tail -1 || true)
        if [[ -z "${STAGE2_CKPT:-}" ]]; then
            echo "ERROR: No Stage 2 checkpoint found." >&2; exit 1
        fi
    fi
    echo "Chaining from Stage 2 checkpoint: $(basename "$STAGE2_CKPT")"
    run_stage "Stage3" "${STAGE3_DIR}" "${STAGE3_STEPS}" \
        "${STAGE2_CKPT}" \
        --lr 2e-6 \
        --scheduler constant \
        --batch_size_per_gp 1 \
        --micro_batch_size 1 \
        --min_seq_len 40000 --max_seq_len 60000 \
        --log_seq_len True --seq_len_per_gp True \
        --replay_small True \
        --freeze_col True --freeze_row True \
        --save_perm_every 10
fi

echo ""; echo "============================================"
echo "  Survival pretraining curriculum complete."
echo "  Stages run: ${RUN_STAGES}"
echo "  Curriculum: ${CURRICULUM_ID}"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "============================================"
