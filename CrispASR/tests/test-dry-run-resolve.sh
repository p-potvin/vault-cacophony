#!/bin/bash
# test-dry-run-resolve.sh — regression for --dry-run-resolve model-key matching.
#
# build_preview() must mirror crispasr_resolve_model's match priority, or the
# preview lies about which model a `-m <alias>` selects. Specifically, a
# sub-variant registry key (e.g. parakeet-tdt_ctc-110m) must resolve to its own
# entry rather than being shadowed by the filename-inferred backend default.
#
# Offline: --dry-run-resolve only previews registry resolution; it never
# downloads. No models required. SKIPs only if the binary is missing.

set -uo pipefail
cd "$(dirname "$0")/.."

CRISPASR=""
for cand in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    if [ -x "$cand" ]; then CRISPASR="$cand"; break; fi
done
if [ -z "$CRISPASR" ]; then
    echo "SKIP: crispasr binary not found (build first)"; exit 0
fi

PASS=0; FAIL=0
# arg|expected-registry-filename
CASES="
parakeet-tdt_ctc-110m|parakeet-tdt_ctc-110m-q4_k.gguf
parakeet|parakeet-tdt-0.6b-v3-q4_k.gguf
parakeet-v2|parakeet-tdt-0.6b-v2-q4_k.gguf
parakeet-tdt-1.1b|parakeet-tdt-1.1b-q4_k.gguf
"
while IFS='|' read -r arg expected; do
    [ -z "$arg" ] && continue
    got=$("$CRISPASR" -m "$arg" --backend parakeet --auto-download --dry-run-resolve 2>&1 \
        | sed -n 's/^  registry:[[:space:]]*//p' | head -1)
    if [ "$got" = "$expected" ]; then
        echo "  ✓ -m $arg → $got"; PASS=$((PASS+1))
    else
        echo "  ✗ -m $arg → '$got' (expected '$expected')"; FAIL=$((FAIL+1))
    fi
done <<< "$CASES"

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
