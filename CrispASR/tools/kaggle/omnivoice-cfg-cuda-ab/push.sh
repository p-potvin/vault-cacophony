#!/usr/bin/env bash
# Push the OmniVoice CFG CUDA A/B kernel under chr1s4.
set -e
export KAGGLE_API_TOKEN=KGAT_8bf612aeb5eb7e3eb52c0ee861871ee5  # chr1s4
cd "$(dirname "$0")"
python -m kaggle kernels push -p .
