<#
.SYNOPSIS
  Launch the text-style-transfer UI. Uses Comfy Desktop on :8188 if it is up,
  otherwise starts the core-only headless ComfyUI on :8189, then starts Gradio.
#>
param([int]$UiPort = 7870, [switch]$ForceHeadless)
$ErrorActionPreference = "Stop"
$here   = $PSScriptRoot
$repo   = Split-Path -Parent (Split-Path -Parent $here)
$python = Join-Path $repo ".venv\Scripts\python.exe"

function Test-Comfy([string]$url) {
  try { $r = Invoke-RestMethod -Uri "$url/system_stats" -TimeoutSec 5; return $null -ne $r.system } catch { return $false }
}

$server = $null
if (-not $ForceHeadless -and (Test-Comfy "http://127.0.0.1:8188")) { $server = "http://127.0.0.1:8188" }
elseif (Test-Comfy "http://127.0.0.1:8189") { $server = "http://127.0.0.1:8189" }
else {
  Write-Host "No ComfyUI reachable - starting headless core-only server on :8189 ..."
  Start-Process -FilePath "pwsh" -ArgumentList @("-NoProfile","-File",(Join-Path $here "run-comfy-headless.ps1")) -WindowStyle Minimized
  for ($i = 0; $i -lt 60 -and -not (Test-Comfy "http://127.0.0.1:8189"); $i++) { Start-Sleep -Seconds 2 }
  if (-not (Test-Comfy "http://127.0.0.1:8189")) { throw "headless ComfyUI did not come up on :8189 (see logs\comfy-8189.log)" }
  $server = "http://127.0.0.1:8189"
}
Write-Host "ComfyUI backend: $server"
$env:COMFY_URL = $server
$env:GRADIO_SERVER_PORT = "$UiPort"
Push-Location $here
try { & $python app.py } finally { Pop-Location }
