#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "kaggle kernels push -p $DIR"
kaggle kernels push -p "$DIR"
ID="$(python -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")"
echo "Poll: kaggle kernels status $ID"
