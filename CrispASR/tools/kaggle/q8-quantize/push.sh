#!/bin/bash
# Push under chr1str (CrispASR kernel convention).
export KAGGLE_API_TOKEN=KGAT_cb3f25c81b9e65d706ebcf655f1daa42
cd "$(dirname "$0")" && cp ../kaggle_harness.py . && kaggle kernels push -p .
