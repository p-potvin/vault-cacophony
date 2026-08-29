#!/bin/bash
# Push under chr1s4 (chr1str is running the requant kernel).
export KAGGLE_API_TOKEN=KGAT_95d684fe44dd004ae6e78f9b32edaf3c
cd "$(dirname "$0")" && cp ../kaggle_harness.py . && kaggle kernels push -p .
