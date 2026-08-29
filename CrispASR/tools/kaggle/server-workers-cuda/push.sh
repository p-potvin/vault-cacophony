#!/usr/bin/env bash
# Push + trigger the server worker-pool CUDA concurrency proof (improvements Phase 4b).
# Push the branch (or merge to main) to GitHub FIRST — the kernel clones by ref.
# Usage: bash tools/kaggle/server-workers-cuda/push.sh
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
command -v kaggle >/dev/null || { echo "kaggle CLI not found (pip install kaggle)" >&2; exit 1; }
echo "Pushing $(jq -r .id kernel-metadata.json) ..."
kaggle kernels push -p "$dir"
echo "Monitor:  kaggle kernels status chr1str/crispasr-server-workers-cuda"
echo "Output:   kaggle kernels output chr1str/crispasr-server-workers-cuda -p ./out"
