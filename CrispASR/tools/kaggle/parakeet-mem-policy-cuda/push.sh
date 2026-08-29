#!/usr/bin/env bash
# Push + trigger the parakeet memory-policy CUDA proof kernel (improvements Phase 2).
# Push the branch (or merge to main) to GitHub FIRST — the kernel clones by ref
# (CRISPASR_REF, default main). Usage: bash tools/kaggle/parakeet-mem-policy-cuda/push.sh
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
command -v kaggle >/dev/null || { echo "kaggle CLI not found (pip install kaggle)" >&2; exit 1; }
echo "Pushing $(jq -r .id kernel-metadata.json) ..."
kaggle kernels push -p "$dir"
echo "Monitor:  kaggle kernels status chr1str/crispasr-parakeet-mem-policy-cuda"
echo "Output:   kaggle kernels output chr1str/crispasr-parakeet-mem-policy-cuda -p ./out"
