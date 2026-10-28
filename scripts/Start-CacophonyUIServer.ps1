# Keeps the Cacophony UI server running: one launcher per session (mutex), the
# server restarted 10 s after any exit. The launcher never wedges: it waits on
# the server with a bounded poll (Start-Process -PassThru), checks that port
# 8753 actually comes up, kills a hung-but-silent server (taskkill /T so the
# venv shim and its interpreter both die), and gives up -- releasing the mutex --
# after 5 rapid exits. A dead server therefore never leaves a stale launcher
# blocking the scheduled task: a fresh task run can always take over.
# Run by the scheduled task \VaultWares\Cacophony UI Server
# (scripts\Register-CacophonyUITask.ps1). The task has NO trigger: start it
# manually from Task Scheduler when the studio is needed, and End it to stop.
# Tailnet-only: binds 100.71.101.21 (clopeux-desktop) so greencloud's nginx
# (cacophony.vaultwares.ca) can proxy to it; never 0.0.0.0.
# Logs (under %LOCALAPPDATA%\VaultWares\cacophony\): server.out.log and
# server.err.log hold the server's stdout/stderr for the current run
# (truncated on each start); launcher.log appends launcher events (start,
# kills, exits, give-up). Override the interpreter for testing via
# $env:CACOPHONY_UI_PYTHON.
$ErrorActionPreference = "Continue"
$created = $false
$mutex = New-Object System.Threading.Mutex($true, "Local\CacophonyUIServer", [ref]$created)
if (-not $created) { exit 0 }
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if ($env:CACOPHONY_UI_PYTHON) { $python = $env:CACOPHONY_UI_PYTHON }
$env:PYTHONIOENCODING = "utf-8"
$logDir = Join-Path $env:LOCALAPPDATA "VaultWares\cacophony"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = Join-Path $logDir "server.out.log"
$errLog = Join-Path $logDir "server.err.log"
$launcherLog = Join-Path $logDir "launcher.log"
$port = 8753
$hostAddr = "100.71.101.21"
$restartDelaySeconds = 10
$maxRapidExits = 5
$graceSeconds = 20

function Test-UiPort([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Write-LauncherLog([string]$Message) {
    $stamp = Get-Date -Format "ddd, dd MMM yyyy HH:mm"
    Add-Content -LiteralPath $launcherLog -Value "[$stamp] $Message"
}

try {
    $rapidExits = 0
    while ($true) {
        if (Test-UiPort $port) {
            # Someone (a rogue server, a manual run) holds the port: don't
            # fight, just wait until it clears, then supervise again.
            Write-LauncherLog "port $port already listening; waiting for it to clear"
            Start-Sleep -Seconds 10
            continue
        }
        $startedAt = Get-Date
        Push-Location -LiteralPath $root
        try {
            # -WindowStyle Hidden: a console child started from a hidden parent
            # (conhost --headless) otherwise gets its own visible window.
            $proc = Start-Process -FilePath $python `
                -ArgumentList @('-u', 'scripts\cacophony\ui_server.py', '--host', "$hostAddr", '--port', "$port") `
                -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -WindowStyle Hidden
        }
        catch {
            Write-LauncherLog "could not start server ($python): $($_.Exception.Message); giving up"
            Pop-Location
            exit 1
        }
        Pop-Location
        Write-LauncherLog "server started PID $($proc.Id)"

        # Bounded wait: poll so a dead or hung child can never wedge this loop.
        while (-not $proc.HasExited) {
            if ($proc.WaitForExit(2000)) { break }
            $upSeconds = ((Get-Date) - $startedAt).TotalSeconds
            if ($upSeconds -ge $graceSeconds -and -not (Test-UiPort $port)) {
                Write-LauncherLog "server PID $($proc.Id) alive but port $port not listening after $graceSeconds s; killing it"
                taskkill /PID $($proc.Id) /T /F | Out-Null
            }
        }

        $uptime = ((Get-Date) - $startedAt).TotalSeconds
        if ($uptime -lt $restartDelaySeconds) { $rapidExits++ } else { $rapidExits = 0 }
        if ($proc.HasExited) { $proc.WaitForExit() }  # flush async output so ExitCode is populated
        $exitCode = if ($proc.HasExited -and $null -ne $proc.ExitCode) { $proc.ExitCode } else { -1 }
        if ($rapidExits -ge $maxRapidExits) {
            Write-LauncherLog "server exited $rapidExits times in a row within ${restartDelaySeconds}s (last code $exitCode); giving up so a fresh launcher can take over"
            exit 1
        }
        Write-LauncherLog "server exited (code $exitCode); restarting in ${restartDelaySeconds}s"
        Start-Sleep -Seconds $restartDelaySeconds
    }
}
finally {
    $mutex.ReleaseMutex()
}