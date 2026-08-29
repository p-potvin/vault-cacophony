#!/usr/bin/env bash
# Push the CUDA im2col occupancy wider-sweep kernel (P100). Needs chr1str Kaggle auth.
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
command -v kaggle >/dev/null 2>&1 || { echo "kaggle CLI not found" >&2; exit 1; }
kaggle kernels push -p "$dir"
echo "Monitor: kaggle kernels status chr1str/crispasr-im2col-occupancy-sweep-cuda"
