# Starts the Cacophony web UI server and opens the browser.
# Requires: repo venv (.venv) with huggingface_hub + the built UI (npm run build in cacophony-ui/).

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Repo ".venv\Scripts\python.exe"
$Server = Join-Path $Repo "scripts\cacophony\ui_server.py"
$Dist = Join-Path $Repo "cacophony-ui\dist\index.html"

if (-not (Test-Path $VenvPython)) { throw "venv python not found at $VenvPython" }
if (-not (Test-Path $Dist)) { Write-Warning "Built UI not found at $Dist - run `npm run build` in cacophony-ui/" }

$Port = 8753
Write-Host "[*] Starting Cacophony UI server on http://127.0.0.1:$Port ..."
& $VenvPython $Server --port $Port