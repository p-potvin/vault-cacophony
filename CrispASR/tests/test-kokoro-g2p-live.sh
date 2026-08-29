#!/bin/bash
# test-kokoro-g2p-live.sh — #316 end-to-end: numbers are spoken, phonemes drive
# the model, and the lexicon auto-downloads.
#
# The unit tests cover the G2P in isolation; only a real synthesis proves the
# reported bug is gone, because the defect was invisible to every tensor-level
# check (crispasr-diff for kokoro is phoneme-IN, so it starts downstream of the
# G2P entirely).
#
# Needs: a kokoro GGUF + voice pack, an ASR model for the round-trip, network
# for the lexicon. SKIPs (exit 0) when any is missing — never a false red.
set -uo pipefail
cd "$(dirname "$0")/.."

CRISPASR=""
for c in build/bin/crispasr build-ninja-compile/bin/crispasr ./bin/crispasr; do
    [ -x "$c" ] && { CRISPASR="$c"; break; }
done
[ -z "$CRISPASR" ] && { echo "SKIP: crispasr binary not found"; exit 0; }

MODELS="${CRISPASR_MODELS_DIR:-${CRISPASR_MODELS:-$HOME/.cache/crispasr}}"
KOKORO="${CRISPASR_KOKORO_MODEL:-$MODELS/kokoro-82m-q8_0.gguf}"
VOICE="${CRISPASR_KOKORO_VOICE:-$MODELS/kokoro-voice-af_heart.gguf}"
ASR="${CRISPASR_MODEL_WHISPER:-$MODELS/ggml-tiny.bin}"
for f in "$KOKORO" "$VOICE" "$ASR"; do
    [ -f "$f" ] || { echo "SKIP: missing $f"; exit 0; }
done

TMP=$(mktemp -d -t kokoro-g2p.XXXXXX); trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ok()  { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

# ── 1. the reported bug: a number must be SPOKEN, not silently dropped ────────
echo "=== #316: numbers survive the G2P ==="
"$CRISPASR" --tts "A text to speech model with 82 million parameters." \
    --backend kokoro -m "$KOKORO" --voice "$VOICE" --tts-output "$TMP/n.wav" >/dev/null 2>&1
if [ -s "$TMP/n.wav" ]; then
    TXT=$("$CRISPASR" -m "$ASR" -f "$TMP/n.wav" -l en --no-prints 2>/dev/null | tr -d '\n')
    echo "    ASR: $TXT"
    # A tiny ASR model transcribes spoken digits loosely — "eighty two" has come
    # back as "82", "eighty-two" and even "80-to-a". Any of those proves the
    # number was SPOKEN, which is the whole point: before the fix it phonemized
    # to the empty string and the ASR heard "with million parameters", with no
    # numeric token at all.
    if echo "$TXT" | grep -qiE "8[0-9]|eighty|eight"; then
        ok "the number is audible in the output"
    else
        bad "number missing — the G2P dropped it again (this IS #316)"
    fi
else
    bad "synthesis produced no audio"
fi

# ── 2. --tts-phonemes drives the model directly ───────────────────────────────
echo "=== --tts-phonemes bypasses the G2P ==="
"$CRISPASR" --tts "THIS TEXT MUST BE IGNORED" --tts-phonemes "həlˈO wˈɜɹld" \
    --backend kokoro -m "$KOKORO" --voice "$VOICE" --tts-output "$TMP/p.wav" >/dev/null 2>&1
if [ -s "$TMP/p.wav" ]; then
    TXT=$("$CRISPASR" -m "$ASR" -f "$TMP/p.wav" -l en --no-prints 2>/dev/null | tr -d '\n')
    echo "    ASR: $TXT"
    # The phonemes say "hello world"; the --tts text says something else. If the
    # flag were ignored we would hear the text instead — the exact silent
    # fallback the CLI refuses to make.
    if echo "$TXT" | grep -qi "hello"; then
        ok "the phonemes were synthesized, not the text"
    else
        bad "phonemes ignored — heard the --tts text instead"
    fi
else
    bad "phoneme synthesis produced no audio"
fi

# ── 3. an unsupported backend must refuse, not fall back ─────────────────────
echo "=== --tts-phonemes is refused where it cannot work ==="
# Checked before any model load, so this needs no qwen3-tts weights.
OUT=$("$CRISPASR" --tts "x" --tts-phonemes "abc" --backend qwen3-tts -m /nonexistent.gguf 2>&1)
if echo "$OUT" | grep -q "tts-phonemes is not supported"; then
    ok "refused with a message naming the flag"
else
    bad "did not refuse (silent fallback makes an A/B unreadable)"
fi

echo
echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ]
