# ---------------------------------------------------------------------------
# NYU Torch settings for the boolean-introspection experiment.
# `source cluster/env.sh` before submitting jobs.  Values follow probe_conf.
# ---------------------------------------------------------------------------

# Slurm account from `my_slurm_accounts` (required for every job).
export TORCH_ACCOUNT="${TORCH_ACCOUNT:-torch_pr_1363_cds}"

# Code location (jobs use $SLURM_SUBMIT_DIR when submitted with --chdir).
export REPO="${REPO:-/scratch/$USER/intros}"
# Run outputs: adapters, eval JSON, figures.  Scratch, not backed up.
export RUNS_ROOT="${RUNS_ROOT:-/scratch/$USER/intros_runs}"

# Shared caches on scratch (same HF cache as probe_conf, so models are reused).
export HF_HOME="/scratch/$USER/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="/scratch/$USER/torch_cache"
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
export PIP_CACHE_DIR="/scratch/$USER/pip_cache"
export UV_CACHE_DIR="/scratch/$USER/uv_cache"
export TMPDIR="/scratch/$USER/tmp"

# Container.  By default this REUSES the probe_conf (section33) overlay: it already
# has torch, transformers, safetensors, pandas, matplotlib and pytest, which is all
# this project needs.  Jobs mount it read-only, so sharing is safe.  To build a
# separate one, set OVERLAY to a new path and run cluster/01_setup_overlay.sh.
export OVERLAY="${OVERLAY:-/scratch/$USER/section33_overlay-15GB-500K.ext3}"
export SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"
export OVERLAY_SRC="/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz"

# Models.
export MODEL="${MODEL:-Qwen/Qwen3-32B}"
export SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
# Run-name prefix; runs land in $RUNS_ROOT/${TAG}_L<levels>.
export TAG="${TAG:-qwen3-32b}"

mkdir -p "$RUNS_ROOT" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" \
         "$UV_CACHE_DIR" "$TMPDIR" 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
