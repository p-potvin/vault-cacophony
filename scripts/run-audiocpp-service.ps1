<#
.SYNOPSIS
    VaultWares audio.cpp Background Service Runner

.DESCRIPTION
    Launcher and supervisor for audiocpp_server.exe.
    Sets up CUDA environment paths, ensures log directories exist,
    and runs the server against the specified JSON configuration.
#>
[CmdletBinding()]
param(
    [string]$ConfigPath = "$env:LOCALAPPDATA\VaultWares\audiocpp-server-8099.json",
    [string]$AudioCppDir,
    [string]$LogDir = "$env:LOCALAPPDATA\VaultWares\logs"
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot  = Split-Path -Parent $ScriptDir

if (-not $AudioCppDir) {
    $AudioCppDir = Join-Path $RepoRoot "audio.cpp"
}

$ExePath = Join-Path $AudioCppDir "audiocpp_server.exe"
if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "audiocpp_server.exe not found at $ExePath"
}

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Configuration file not found at $ConfigPath"
}

if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

$LogFile = Join-Path $LogDir "audiocpp-server-8099.log"

# Add audio.cpp to PATH for local CUDA runtime DLLs
$env:PATH = "$AudioCppDir;$env:PATH"
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

$timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
Add-Content -LiteralPath $LogFile -Value "[$timestamp] Starting audiocpp_server from $AudioCppDir with config $ConfigPath"

# Run audiocpp_server synchronously so scheduled task stays attached to the process lifetime
$processArgs = @(
    "--config", "`"$ConfigPath`"",
    "--log",
    "--log-file", "`"$LogFile`""
)

try {
    Push-Location $AudioCppDir
    & $ExePath --config "$ConfigPath" --log --log-file "$LogFile" 2>&1 | Add-Content -LiteralPath $LogFile
}
finally {
    Pop-Location
    $exitStamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Add-Content -LiteralPath $LogFile -Value "[$exitStamp] audiocpp_server exited (code: $LASTEXITCODE)"
}
