<#
.SYNOPSIS
    VaultWares audio.cpp Background Service — Setup & Scheduler

.DESCRIPTION
    Registers, configures, and manages the VaultWares audio.cpp inference server
    (port 8099) as a persistent Windows Scheduled Task daemon.

    Adheres to VaultWares AUTOMATION_POLICY:
    - Runs headless via conhost.exe with hidden pwsh.exe execution
    - Runs at logon / system start under BUILTIN\Administrators with Highest runlevel (full CUDA acceleration)
    - Auto-restarts on crash with 1-minute retry interval and no execution time limit

.PARAMETER Install
    Registers the scheduled task and generates the default JSON configuration.

.PARAMETER Uninstall
    Stops the server process and unregisters the scheduled task.

.PARAMETER Start
    Starts the registered scheduled task.

.PARAMETER Stop
    Stops the scheduled task and terminates any running audiocpp_server processes on the configured port.

.PARAMETER Restart
    Stops and restarts the service.

.PARAMETER Status
    Displays current service state, process ID, port listener, and loaded models.

.PARAMETER Port
    Listening port for the inference server (default: 8099).

.PARAMETER StartNow
    Switch — immediately starts the service after installation.

.EXAMPLE
    .\setup-audiocpp-service.ps1 -Install -StartNow

.EXAMPLE
    .\setup-audiocpp-service.ps1 -Status

.EXAMPLE
    .\setup-audiocpp-service.ps1 -Stop
#>
[CmdletBinding(DefaultParameterSetName = "Install")]
param(
    [Parameter(ParameterSetName = "Install")]
    [switch]$Install,

    [Parameter(ParameterSetName = "Uninstall")]
    [switch]$Uninstall,

    [Parameter(ParameterSetName = "Start")]
    [switch]$Start,

    [Parameter(ParameterSetName = "Stop")]
    [switch]$Stop,

    [Parameter(ParameterSetName = "Restart")]
    [switch]$Restart,

    [Parameter(ParameterSetName = "Status")]
    [switch]$Status,

    [int]$Port = 8099,
    [string]$Backend = "cuda",
    [int]$Device = 0,
    [int]$Threads = 8,
    [string]$ConfigPath,
    [switch]$StartNow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TaskName  = "VaultWares-AudioCppServer"
$ScriptDir = $PSScriptRoot
$RepoRoot  = Split-Path -Parent $ScriptDir
$AudioCpp  = Join-Path $RepoRoot "audio.cpp"
$Launcher  = Join-Path $ScriptDir "run-audiocpp-service.ps1"
$ServerExe = Join-Path $AudioCpp "audiocpp_server.exe"

if (-not $ConfigPath) {
    $ConfigPath = Join-Path $env:LOCALAPPDATA "VaultWares\audiocpp-server-$Port.json"
}

function Show-Header {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║     VaultWares audio.cpp Background Service Management       ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Resolve-AsrModel {
    $store = Join-Path $env:LOCALAPPDATA "VaultWares\models"
    $candidates = @(
        (Join-Path $store "Nemotron-3.5-ASR-Streaming-0.6B-GGUF"),
        (Join-Path $AudioCpp "models\Nemotron-3.5-ASR-Streaming-0.6B-GGUF"),
        (Join-Path $store "parakeet-tdt-0.6b-v3-gguf\parakeet-tdt-0.6b-v3-q8_0.gguf"),
        (Join-Path $AudioCpp "models\Parakeet-TDT-0.6B-v3-GGUF\parakeet-tdt-0.6b-v3-q8_0.gguf")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    throw "No ASR model found at candidate paths"
}

function Ensure-Config {
    $modelPath = Resolve-AsrModel
    $configDir = Split-Path -Parent $ConfigPath
    if (-not (Test-Path -LiteralPath $configDir)) {
        New-Item -ItemType Directory -Force -Path $configDir | Out-Null
    }

    $isNemotron = ($modelPath -match "Nemotron")
    $modelId = if ($isNemotron) { "nemotron" } else { "parakeet" }
    $family = if ($isNemotron) { "nemotron_asr" } else { "parakeet_tdt" }
    $mode = if ($isNemotron) { "streaming" } else { "offline" }

    $modelEntry = [ordered]@{
        id     = $modelId
        family = $family
        path   = ($modelPath -replace '\\', '/')
        task   = "asr"
        mode   = $mode
    }
    if ($family -eq "parakeet_tdt") {
        $modelEntry.session_options = @{ "parakeet_tdt.offline_mode" = "full_context" }
    }

    $config = [ordered]@{
        host      = "127.0.0.1"
        port      = $Port
        backend   = $Backend
        device    = $Device
        threads   = $Threads
        lazy_load = $false
        models    = @($modelEntry)
    }

    $config | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8
    Write-Host "  Config written: $ConfigPath ($modelId)" -ForegroundColor Green
}

function Stop-ServerProcesses {
    $stopped = 0
    Get-CimInstance Win32_Process -Filter "Name = 'audiocpp_server.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.CommandLine -match "audiocpp-server-$Port\.json" -or $_.CommandLine -match "port $Port") {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }
    if ($stopped -gt 0) {
        Write-Host "  Terminated $stopped running audiocpp_server process(es)." -ForegroundColor Yellow
    }
}

function Get-ServiceStatus {
    Show-Header
    Write-Host "Status check for Task '$TaskName':" -ForegroundColor White
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Write-Host "  Task State      : $($task.State)" -ForegroundColor $(if ($task.State -eq 'Running') { 'Green' } else { 'Yellow' })
    } else {
        Write-Host "  Task State      : Not Registered" -ForegroundColor Red
    }

    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        Write-Host "  Port $Port       : LISTENING (PID: $($conn.OwningProcess) - $($proc.ProcessName))" -ForegroundColor Green

        # Test HTTP health endpoint
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 -ErrorAction Stop
            Write-Host "  Engine Health   : status=$($health.status), backend=$($health.backend), models=$($health.models)" -ForegroundColor Green
        } catch {
            Write-Host "  Engine Health   : Failed to query /health ($($_.Exception.Message))" -ForegroundColor Yellow
        }

        # Test Models endpoint
        try {
            $models = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3 -ErrorAction Stop
            $loadedIds = ($models.data | Where-Object { $_.loaded } | ForEach-Object { $_.id }) -join ", "
            Write-Host "  Loaded Models   : $loadedIds" -ForegroundColor Green
        } catch {
            Write-Host "  Loaded Models   : Failed to query /v1/models" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Port $Port       : NOT LISTENING" -ForegroundColor Red
    }
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

if ($Status) {
    Get-ServiceStatus
    exit 0
}

if ($Stop) {
    Show-Header
    Write-Host "Stopping service '$TaskName'..." -ForegroundColor Yellow
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Stop-ServerProcesses
    Write-Host "  Service stopped." -ForegroundColor Green
    exit 0
}

if ($Uninstall) {
    Show-Header
    Write-Host "Uninstalling service '$TaskName'..." -ForegroundColor Yellow
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Stop-ServerProcesses
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
    Write-Host "  Task '$TaskName' unregistered." -ForegroundColor Green
    exit 0
}

if ($Restart) {
    Show-Header
    Write-Host "Restarting service '$TaskName'..." -ForegroundColor Yellow
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Stop-ServerProcesses
    Start-Sleep -Seconds 1
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 2
    Get-ServiceStatus
    exit 0
}

if ($Start) {
    Show-Header
    Write-Host "Starting service '$TaskName'..." -ForegroundColor Yellow
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 2
    Get-ServiceStatus
    exit 0
}

# Default: Install
Show-Header
Write-Host "Configuring VaultWares audio.cpp background service on port $Port..." -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $ServerExe)) {
    throw "audiocpp_server.exe not found at $ServerExe"
}
if (-not (Test-Path -LiteralPath $Launcher)) {
    throw "Launcher not found at $Launcher"
}

Ensure-Config

Write-Host "Registering Scheduled Task '$TaskName'..." -ForegroundColor Yellow
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null

$conhostPath = (Get-Command "conhost.exe" -ErrorAction Stop).Source
$pwshPath    = (Get-Command "pwsh.exe" -ErrorAction Stop).Source

$launchArgs = "-NoProfile -WindowStyle Hidden -NonInteractive -ExecutionPolicy Bypass " +
              "-File `"$Launcher`" -ConfigPath `"$ConfigPath`" -AudioCppDir `"$AudioCpp`""

$action = New-ScheduledTaskAction `
    -Execute $conhostPath `
    -Argument "--headless `"$pwshPath`" $launchArgs" `
    -WorkingDirectory $AudioCpp

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -MultipleInstances IgnoreNew `
    -RestartCount 99 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

$principal = New-ScheduledTaskPrincipal -GroupId "BUILTIN\Administrators" -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "VaultWares: persistent background audio.cpp inference server (port $Port) for real-time speech transcription and live subtitles." `
    | Out-Null

Write-Host "  Task '$TaskName' registered successfully." -ForegroundColor Green

if ($StartNow -or $Install) {
    Write-Host "Starting service now..." -ForegroundColor Yellow
    # Stop any orphan server first
    Stop-ServerProcesses
    Start-ScheduledTask -TaskName $TaskName

    # Wait for server readiness
    $deadline = (Get-Date).AddSeconds(30)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2 -ErrorAction Stop
            if ($r.data | Where-Object { $_.id -eq "parakeet" -and $_.loaded }) {
                $ready = $true
                break
            }
        } catch { }
        Start-Sleep -Milliseconds 500
    }

    if ($ready) {
        Write-Host "  Server is UP and ready on http://127.0.0.1:$Port!" -ForegroundColor Green
    } else {
        Write-Warning "  Task started, but server did not report ready within 30s. Check logs at %LOCALAPPDATA%\VaultWares\logs\audiocpp-server-$Port.log"
    }
}

Get-ServiceStatus
