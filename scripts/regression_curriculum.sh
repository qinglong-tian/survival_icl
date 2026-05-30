# This script generates synthetic regression prior datasets for the 3-stage curriculum.
# Run stages independently or sequentially. Adjust save_dir and paths as needed.

# ----------------------------------
# Stage 1 — Small fixed-length datasets
# ----------------------------------
python regression_prior.py \
    --save_dir data/regression_stage1 \
    --num_batches 100000 \
    --batch_size 512 \
    --batch_size_per_gp 4 \
    --prior_type mix_scm \
    --min_features 2 --max_features 100 \
    --max_seq_len 1024 \
    --min_train_size 0.1 --max_train_size 0.9 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu

# ----------------------------------
# Stage 2 — Medium variable-length datasets
# ----------------------------------
python regression_prior.py \
    --save_dir data/regression_stage2 \
    --num_batches 2000 \
    --batch_size 512 \
    --batch_size_per_gp 2 \
    --prior_type mix_scm \
    --min_features 2 --max_features 100 \
    --min_seq_len 1000 --max_seq_len 40000 \
    --log_seq_len True --seq_len_per_gp True \
    --min_train_size 0.5 --max_train_size 0.9 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu

# ----------------------------------
# Stage 3 — Large variable-length datasets
# ----------------------------------
python regression_prior.py \
    --save_dir data/regression_stage3 \
    --num_batches 50 \
    --batch_size 512 \
    --batch_size_per_gp 1 \
    --prior_type mix_scm \
    --min_features 2 --max_features 100 \
    --min_seq_len 40000 --max_seq_len 60000 \
    --log_seq_len True --seq_len_per_gp True \
    --replay_small True \
    --min_train_size 0.5 --max_train_size 0.9 \
    --n_jobs -1 --num_threads_per_generate 1 --device cpu
