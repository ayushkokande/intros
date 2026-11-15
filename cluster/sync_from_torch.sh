#!/usr/bin/env bash
# Pull eval results, logs and figures (NOT adapter checkpoints) to the laptop.
#   bash cluster/sync_from_torch.sh            # --dry-run to preview
#   WITH_CHECKPOINTS=1 bash cluster/sync_from_torch.sh   # also adapters (~0.5 GB each for 32B)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/intros_runs/}"
DEST="${DEST:-$(dirname "$HERE")/runs/}"
OPTS=(-avz --progress)
[ "${WITH_CHECKPOINTS:-0}" = "1" ] || OPTS+=(--exclude 'checkpoints/' --exclude 'resume/')
[ "${1:-}" = "--dry-run" ] && OPTS+=(--dry-run)
mkdir -p "$DEST"
rsync "${OPTS[@]}" "$SRC" "$DEST"
