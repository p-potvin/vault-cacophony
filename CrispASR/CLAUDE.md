# CrispASR — Claude Code project instructions

## Build

```bash
CCACHE_DIR=/mnt/volume1/.ccache cmake -G Ninja -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Test

```bash
# Unit tests (no models needed, ~5s):
ctest --test-dir build -L unit --timeout 30

# Integration / live tests (need models):
export CRISPASR_MODELS_DIR=/mnt/storage/gguf-models
source tests/env-live-tests.sh
ctest --test-dir build --rerun-failed --output-on-failure --timeout 300
```

Key env vars: `CRISPASR_MODELS_DIR`, `CRISPASR_MODEL_WHISPER`, `CRISPASR_PARAFORMER_MODEL`.
See `tests/env-live-tests.sh` for the full list.

## Lint

```bash
./tools/format.sh --fix <changed .cpp/.h files>
```

Must use clang-format v18 — `tools/format.sh` enforces this.

## Storage

- **GGUF models**: `/mnt/storage/gguf-models` — all converted GGUF files live here
- **NEVER use `/tmp`** — it is a tiny tmpfs that fills up and kills processes. Use `/mnt/volume1/tmp-overflow` for temp files, or write directly to `/mnt/storage` or `/mnt/volume1`.
- **HuggingFace cache**: `/mnt/akademie_storage/huggingface/hub/`
- **ccache**: `/mnt/volume1/.ccache`

## Commit workflow

- No PRs — merge locally and push to main directly
- Always run `tools/format.sh --fix` on changed C/C++ files before committing
- Use plain branches, not worktrees
