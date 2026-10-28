# Registers \VaultWares\Cacophony UI Server: NO trigger - run it manually from
# Task Scheduler when the studio is needed (hidden, conhost --headless), never
# times out. Same model as vault-raggae's \VaultWares\Enrichment Gateway task,
# minus the at-logon trigger.
$launcher = Join-Path $PSScriptRoot "Start-CacophonyUIServer.ps1"
$action = New-ScheduledTaskAction -Execute "conhost.exe" -Argument "--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$launcher`""
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -Priority 7 -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskPath "\VaultWares\" -TaskName "Cacophony UI Server" -Action $action -Settings $settings `
    -Description "Cacophony multi-speaker conversation studio (orchestrator control panel + voices). Serves tailnet-only on 100.71.101.21:8753 (https://cacophony.vaultwares.ca via greencloud nginx); generate conversations from the UI. No trigger: Start the task manually from Task Scheduler, End it to stop. Logs %LOCALAPPDATA%\VaultWares\cacophony\: server.out.log/server.err.log (server, per run) + launcher.log (append-only events)" -Force | Out-Null
"registered"