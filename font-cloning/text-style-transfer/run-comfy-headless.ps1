<#
.SYNOPSIS
  Start a headless ComfyUI on :8189 (core nodes only) using the D: model library.
.NOTES
  Uses font-cloning\.venv-comfy (torch 2.8.0+cu129) and the ComfyUI source tree of
  the Comfy Desktop install at D:\ComfyUI\resources\ComfyUI. Runtime dirs live under
  text-style-transfer\comfy-runtime so the Desktop install is never written to.
  --disable-all-custom-nodes keeps memory-management patches (multigpu/DisTorch) out;
  only the GGUF loader (registered by the `gguf` folder) is whitelisted.
#>
param([int]$Port = 8189, [string]$BindHost = "127.0.0.1", [string[]]$Whitelist = @("ComfyUI-GGUF", "gguf"))
$ErrorActionPreference = "Stop"
$here   = $PSScriptRoot
$python = Join-Path (Split-Path -Parent $here) ".venv-comfy\Scripts\python.exe"
$src    = "D:\ComfyUI\resources\ComfyUI"
$rt     = Join-Path $here "comfy-runtime"
if (-not (Test-Path $python)) { throw "missing $python - see README (uv venv .venv-comfy)" }
$args = @("main.py", "--listen", $BindHost, "--port", "$Port", "--disable-auto-launch",
          "--disable-all-custom-nodes",
          "--extra-model-paths-config", (Join-Path $rt "extra_model_paths.yaml"),
          "--input-directory",  (Join-Path $rt "input"),
          "--output-directory", (Join-Path $rt "output"),
          "--user-directory",   (Join-Path $rt "user"),
          "--temp-directory",   (Join-Path $rt "temp"))
foreach ($w in $Whitelist) { $args += @("--whitelist-custom-nodes", $w) }
Push-Location $src
try { & $python @args } finally { Pop-Location }
