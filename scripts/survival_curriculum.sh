# This script generates synthetic survival prior datasets for the 3-stage curriculum.
# Run stages independently or sequentially. Adjust save_dir and paths as needed.
# Use --model_type ph for proportional hazard, --model_type aft for accelerated failure time.

# =============================================
# Proportional Hazard (PH) — 4 baselines
# =============================================

# ----------------------------------
# Stage 1 — Small fixed-length datasets
# ----------------------------------
python survival_prior.py \
    --save_dir data/survival_stage1 \
    --num_batches 100000 \
    --batch_size 512 \
    --batch_size_per_gp 4 \
    --prior_type mlp_scm \
    --beta 1.0 \
    --baseline_types weibull,gompertz,loglogistic,lognormal \
    --baseline_mode mix \
    --min_features 2 --max_features 100 \
    --max_seq_len 1024 \
    --censoring_strategy target_event_rate \
    --min_event_rate 0.40 --max_event_rate 0.90 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu

# ----------------------------------
# Stage 2 — Medium variable-length datasets
# ----------------------------------
python survival_prior.py \
    --save_dir data/survival_stage2 \
    --num_batches 2000 \
    --batch_size 512 \
    --batch_size_per_gp 2 \
    --prior_type mlp_scm \
    --beta 1.0 \
    --baseline_types weibull,gompertz,loglogistic,lognormal \
    --baseline_mode mix \
    --min_features 2 --max_features 100 \
    --min_seq_len 1000 --max_seq_len 40000 \
    --log_seq_len True --seq_len_per_gp True \
    --censoring_strategy target_event_rate \
    --min_event_rate 0.40 --max_event_rate 0.90 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu

# ----------------------------------
# Stage 3 — Large variable-length datasets
# ----------------------------------
python survival_prior.py \
    --save_dir data/survival_stage3 \
    --num_batches 50 \
    --batch_size 512 \
    --batch_size_per_gp 1 \
    --prior_type mlp_scm \
    --beta 1.0 \
    --baseline_types weibull,gompertz,loglogistic,lognormal \
    --baseline_mode mix \
    --min_features 2 --max_features 100 \
    --min_seq_len 40000 --max_seq_len 60000 \
    --log_seq_len True --seq_len_per_gp True \
    --replay_small True \
    --censoring_strategy target_event_rate \
    --min_event_rate 0.40 --max_event_rate 0.90 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu

# =============================================
# Accelerated Failure Time (AFT) — 3 baselines (no Gompertz)
# =============================================

# Stage 1 — Small fixed-length datasets
python survival_prior.py \
    --model_type aft \
    --save_dir data/survival_aft_stage1 \
    --num_batches 100000 \
    --batch_size 512 \
    --batch_size_per_gp 4 \
    --prior_type mlp_scm \
    --beta 1.0 \
    --baseline_types weibull,loglogistic,lognormal \
    --baseline_mode mix \
    --min_features 2 --max_features 100 \
    --max_seq_len 1024 \
    --censoring_strategy target_event_rate \
    --min_event_rate 0.40 --max_event_rate 0.90 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu

# Stage 2 — Medium variable-length datasets
python survival_prior.py \
    --model_type aft \
    --save_dir data/survival_aft_stage2 \
    --num_batches 2000 \
    --batch_size 512 \
    --batch_size_per_gp 2 \
    --prior_type mlp_scm \
    --beta 1.0 \
    --baseline_types weibull,loglogistic,lognormal \
    --baseline_mode mix \
    --min_features 2 --max_features 100 \
    --min_seq_len 1000 --max_seq_len 40000 \
    --log_seq_len True --seq_len_per_gp True \
    --censoring_strategy target_event_rate \
    --min_event_rate 0.40 --max_event_rate 0.90 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu

# Stage 3 — Large variable-length datasets
python survival_prior.py \
    --model_type aft \
    --save_dir data/survival_aft_stage3 \
    --num_batches 50 \
    --batch_size 512 \
    --batch_size_per_gp 1 \
    --prior_type mlp_scm \
    --beta 1.0 \
    --baseline_types weibull,loglogistic,lognormal \
    --baseline_mode mix \
    --min_features 2 --max_features 100 \
    --min_seq_len 40000 --max_seq_len 60000 \
    --log_seq_len True --seq_len_per_gp True \
    --replay_small True \
    --censoring_strategy target_event_rate \
    --min_event_rate 0.40 --max_event_rate 0.90 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu
