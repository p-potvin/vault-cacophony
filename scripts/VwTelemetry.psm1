<#
    Model-run telemetry for the audio.cpp pipeline.

    This repo deliberately imports no PyTorch and drives audio.cpp from
    PowerShell, so the Python recorder in vaultwares-adk is the wrong tool here.
    Instead this writes the same NDJSON batch the recorder produces straight to
    the run spool, and the existing hourly drain
    (vaultwares-adk/scripts/drain-run-spool.ps1) ships it. No Python, no
    dependency, and delivery is already solved.

    ONE FILE PER BATCH, not one appended line. The drain renames a file to
    *.sent once every line in it has been posted; a line appended between the
    read and the rename would be silently lost. Unique filenames remove the race
    entirely, and the drain already handles many files.

    Everything here is best-effort: a transcription must never fail because
    telemetry could not be written.
#>

Set-StrictMode -Version Latest

function Get-VwSpoolDir {
    $dir = $env:VW_RUNS_SPOOL_DIR
    if (-not $dir) { $dir = [Environment]::GetEnvironmentVariable("VW_RUNS_SPOOL_DIR", "Machine") }
    if (-not $dir) { $dir = "D:\AiHistory\run-spool" }
    return $dir
}

function New-VwRunId {
    return ([guid]::NewGuid().ToString("N"))
}

function Write-VwModelRun {
    <#
    .SYNOPSIS
        Record one model invocation. Never throws.
    .PARAMETER Fields
        Hashtable of run fields. provider/runtime/model/task are expected;
        anything else the API does not have a column for lands in `extra`.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Fields)

    try {
        $run = @{} + $Fields
        if (-not $run.ContainsKey("run_id"))   { $run["run_id"]   = New-VwRunId }
        if (-not $run.ContainsKey("provider")) { $run["provider"] = "audiocpp" }
        if (-not $run.ContainsKey("runtime"))  { $run["runtime"]  = "audio.cpp" }
        if (-not $run.ContainsKey("model"))    { $run["model"]    = "unknown" }
        if (-not $run.ContainsKey("status"))   { $run["status"]   = "ok" }
        # Local GPU work: a real zero rather than an unmeasured cost, so the
        # free-vs-paid split stays honest.
        if (-not $run.ContainsKey("cost_usd")) { $run["cost_usd"] = 0.0 }
        if (-not $run.ContainsKey("is_free"))  { $run["is_free"]  = $true }

        $batch = @{
            schema      = 1
            source      = "vw-audiocpp"
            host        = $env:COMPUTERNAME
            collectedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            batchIndex  = 0
            runs        = @($run)
        }

        $dir = Get-VwSpoolDir
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
        $name = "{0}-{1}.jsonl" -f (Get-Date -Format "yyyy-MM-dd-HHmmssfff"), (Get-Random -Maximum 99999)
        $json = $batch | ConvertTo-Json -Depth 6 -Compress
        [System.IO.File]::WriteAllText((Join-Path $dir $name), $json + "`n", [System.Text.UTF8Encoding]::new($false))
    }
    catch {
        Write-Verbose "vw-telemetry: $($_.Exception.Message)"
    }
}

function Get-VwAudioCppTask {
    <#
    .SYNOPSIS
        Pull task and family out of an audiocpp_cli argument list.
    .DESCRIPTION
        The CLI is driven with --task asr|sep and --family nemotron_asr|
        bs_roformer|parakeet_tdt|htdemucs, so the workload is fully described by
        its own arguments. Reading them beats hardcoding what the caller meant.
    #>
    param([string[]]$Arguments)

    $out = @{ task = $null; family = $null; mode = $null; language = $null }
    for ($i = 0; $i -lt $Arguments.Count; $i++) {
        switch ($Arguments[$i]) {
            "--task"     { if ($i + 1 -lt $Arguments.Count) { $out.task     = $Arguments[$i + 1] } }
            "--family"   { if ($i + 1 -lt $Arguments.Count) { $out.family   = $Arguments[$i + 1] } }
            "--mode"     { if ($i + 1 -lt $Arguments.Count) { $out.mode     = $Arguments[$i + 1] } }
            "--language" { if ($i + 1 -lt $Arguments.Count) { $out.language = $Arguments[$i + 1] } }
        }
    }
    return $out
}

function ConvertTo-VwTaskName {
    <#
        audio.cpp's own task names mapped onto the shared vocabulary, so ASR
        here lands in the same bucket as the NeMo and Ollama transcriptions
        rather than in a task nothing else uses.
    #>
    param([string]$AudioCppTask)
    switch ($AudioCppTask) {
        "asr" { return "audio-asr" }
        "sep" { return "audio-separation" }
        default { return $(if ($AudioCppTask) { "audio-$AudioCppTask" } else { "audio" }) }
    }
}

Export-ModuleMember -Function Write-VwModelRun, Get-VwAudioCppTask, ConvertTo-VwTaskName, Get-VwSpoolDir, New-VwRunId
