# Sourced by the sbatch scripts.  Resolves the repo, loads env.sh, and defines
# `in_container "<shell commands>"`, which runs them inside the Apptainer image
# with the GPU, the overlay (read-only) and the caches set up.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"

if [ ! -f "$OVERLAY" ]; then
  echo "ERROR: overlay not found: $OVERLAY (see cluster/README.md, step 2)" >&2
  exit 1
fi

in_container() {
  srun apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
    set -euo pipefail
    source /ext3/env.sh
    export HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TORCH_HOME=$TORCH_HOME TRITON_CACHE_DIR=$TRITON_CACHE_DIR
    export TMPDIR=$TMPDIR PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
    cd $REPO
    $1
  "
}

# Levels for this task: LEVELS env (e.g. "1,2,3,4" for a mixed run) or the array index.
resolve_levels() {
  LEVELS="${LEVELS:-${SLURM_ARRAY_TASK_ID:-}}"
  if [ -z "$LEVELS" ]; then
    echo "ERROR: set LEVELS=1 (or 1,2,3,4) or submit as an array, e.g. --array=1-4" >&2
    exit 1
  fi
  RUN_DIR="$RUNS_ROOT/${TAG}_L${LEVELS//,/}"
  echo "host=$(hostname) levels=$LEVELS run_dir=$RUN_DIR model=$MODEL"
}
