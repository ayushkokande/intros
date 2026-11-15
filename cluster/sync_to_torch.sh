#!/usr/bin/env bash
# Push code from the laptop to Torch (DTN).  Run on your LAPTOP.
#   bash cluster/sync_to_torch.sh            # add --delete to remove deleted files
#   DRY_RUN=1 bash cluster/sync_to_torch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(dirname "$HERE")}"
DEST="${TORCH_DEST:-kk6081@dtn.torch.hpc.nyu.edu:/scratch/kk6081/intros/}"
OPTS=(-avz --progress --exclude '.venv/' --exclude 'runs/' --exclude 'results/' --exclude 'logs/*.out' --exclude 'logs/*.err'
      --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '*.pdf')
[ "${DRY_RUN:-0}" = "1" ] && OPTS+=(--dry-run)
[ "${1:-}" = "--delete" ] && OPTS+=(--delete)
echo "source: $SRC"; echo "dest:   $DEST"
rsync "${OPTS[@]}" "$SRC/" "$DEST"
