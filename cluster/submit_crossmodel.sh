#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the whole cross-model experiment as one dependency chain (run on the login node).
#
# For each seed and each self model (Qwen3, and with SWAP=1 the other family too):
#   10_train (array over levels, rule labels)                       = the self model M1
#     -> 60 record (M1's choices)
#        -> 60 same  (outsider: fresh LoRA on the same base)       } run in parallel
#        -> 60 other (outsider: the other family)                  }
# and finally one 60 summary over everything.  With SWAP=1 this is the full 2 x 2:
# Qwen3 self with Qwen3 and Mistral outsiders, and Mistral self with Mistral and Qwen3 outsiders.
#
#   source /scratch/$USER/intros/cluster/env.sh
#   bash $REPO/cluster/submit_crossmodel.sh                                # 32B, seed 2025, levels 1-4
#   SEEDS="2025 2026 2027" bash $REPO/cluster/submit_crossmodel.sh         # three seeds
#   SELF_MODEL=Qwen/Qwen3-8B SELF_TAG=qwen3-8b OTHER_MODEL=meta-llama/Llama-3.1-8B-Instruct \
#     OTHER_TAG=llama3.1-8b SBATCH_EXTRA="--constraint=l40s|h200" bash $REPO/cluster/submit_crossmodel.sh
#
# Variables (defaults in brackets):
#   SELF_MODEL [$MODEL = Qwen/Qwen3-32B]   SELF_TAG [$TAG = qwen3-32b]
#   OTHER_MODEL [mistralai/Mistral-Small-24B-Instruct-2501]   OTHER_TAG [mistral-small-24b]
#   SEEDS [2025]  ARRAY [1-4] (levels)  SWAP [1]  WITH_EVAL [0] (also run 20_eval on each M1)
#   TRAIN_ARGS / M2_TRAIN_ARGS / XM_EVAL_ARGS: extra flags; SBATCH_EXTRA: extra sbatch flags.
# Already-finished stages are no-ops, so resubmitting after a failure is safe.
# Seed 2025 keeps the plain tag (so existing runs are reused); other seeds add "-s<seed>".
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-/scratch/$USER/intros}"
source "$REPO/cluster/env.sh"
cd "$REPO"
mkdir -p logs
SELF_MODEL="${SELF_MODEL:-$MODEL}"; SELF_TAG="${SELF_TAG:-$TAG}"
OTHER_MODEL="${OTHER_MODEL:-mistralai/Mistral-Small-24B-Instruct-2501}"; OTHER_TAG="${OTHER_TAG:-mistral-small-24b}"
SEEDS="${SEEDS:-2025}"; ARRAY="${ARRAY:-1-4}"; SWAP="${SWAP:-1}"
read -r -a EXTRA <<< "${SBATCH_EXTRA:-}"
COMMON=(--parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" --export=ALL "${SBATCH_MAIL[@]}" "${EXTRA[@]}")
ALL=()

chain() {  # chain <self model> <tag> <outsider model of the other family> <seed>
  local model=$1 tag=$2 other=$3 seed=$4 train rec same oth
  train=$(MODEL=$model TAG=$tag TRAIN_ARGS="--seed $seed ${TRAIN_ARGS:-}" \
          sbatch "${COMMON[@]}" --array="$ARRAY" --job-name="bi-train-$tag" cluster/10_train.sbatch)
  if [ "${WITH_EVAL:-0}" = "1" ]; then
    ALL+=("$(MODEL=$model TAG=$tag sbatch "${COMMON[@]}" --array="$ARRAY" --dependency="aftercorr:$train" \
             --job-name="bi-eval-$tag" cluster/20_eval.sbatch)")
  fi
  rec=$(TAG=$tag XM_STAGE=record sbatch "${COMMON[@]}" --array="$ARRAY" --dependency="aftercorr:$train" \
        --time=06:00:00 --job-name="xm-record-$tag" cluster/60_crossmodel.sbatch)
  same=$(TAG=$tag XM_STAGE=same sbatch "${COMMON[@]}" --array="$ARRAY" --dependency="aftercorr:$rec" \
         --job-name="xm-same-$tag" cluster/60_crossmodel.sbatch)
  oth=$(TAG=$tag XM_STAGE=other OTHER_MODEL=$other sbatch "${COMMON[@]}" --array="$ARRAY" \
        --dependency="aftercorr:$rec" --job-name="xm-other-$tag" cluster/60_crossmodel.sbatch)
  ALL+=("$train" "$rec" "$same" "$oth")
  echo "seed $seed  self=$model ($tag): train=$train record=$rec same=$same other=$oth (outsider $other)"
}

for seed in $SEEDS; do
  sfx=""; [ "$seed" = "2025" ] || sfx="-s$seed"
  chain "$SELF_MODEL" "$SELF_TAG$sfx" "$OTHER_MODEL" "$seed"
  [ "$SWAP" = "1" ] && chain "$OTHER_MODEL" "$OTHER_TAG$sfx" "$SELF_MODEL" "$seed"
done
DEP=$(IFS=:; echo "${ALL[*]}")
SUM=$(XM_STAGE=summary sbatch "${COMMON[@]}" --dependency="afterany:$DEP" --time=00:30:00 \
      --job-name=xm-summary cluster/60_crossmodel.sbatch)
echo "summary=$SUM -> $RUNS_ROOT/xm_summary/report.md   (watch with: squeue --me)"
