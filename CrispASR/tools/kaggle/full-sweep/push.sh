#!/bin/bash
# Stage the canonical sweep script + harness alongside the kernel metadata and
# push to Kaggle (chr1s4). Avoids committing a duplicate of the 800-line script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STAGE="$(mktemp -d)"
cp "$ROOT/tools/kaggle-benchmark-all-backends.py" "$STAGE/"
cp "$ROOT/tools/kaggle/kaggle_harness.py" "$STAGE/"
cp "$HERE/kernel-metadata.json" "$STAGE/"
echo "Staged in $STAGE; pushing to Kaggle..."
kaggle kernels push -p "$STAGE"
rm -rf "$STAGE"
