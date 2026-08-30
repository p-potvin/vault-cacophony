<#
    Tests for the audio.cpp model-run telemetry emitter.

    Run:  pwsh -File tests/VwTelemetry.Tests.ps1
    Plain assertions rather than Pester, so this needs nothing installed --
    matching the repo's no-heavy-dependencies stance.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Import-Module (Join-Path $PSScriptRoot "../scripts/VwTelemetry.psm1") -Force

$failures = 0
function Check([string]$Name, [scriptblock]$Body) {
    try {
        & $Body
        Write-Host "  ok   $Name" -ForegroundColor Green
    } catch {
        $script:failures++
        Write-Host "  FAIL $Name -- $($_.Exception.Message)" -ForegroundColor Red
    }
}
function Assert([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

$spool = Join-Path ([System.IO.Path]::GetTempPath()) ("vw-tel-test-" + [guid]::NewGuid().ToString("N").Substring(0,8))
$env:VW_RUNS_SPOOL_DIR = $spool

Write-Host "argument parsing"

Check "reads task and family off the CLI arguments" {
    $s = Get-VwAudioCppTask -Arguments @('--family','bs_roformer','--task','sep','--mode','offline')
    Assert ($s.task -eq 'sep') "task was $($s.task)"
    Assert ($s.family -eq 'bs_roformer') "family was $($s.family)"
    Assert ($s.mode -eq 'offline') "mode was $($s.mode)"
}

Check "missing flags come back null rather than throwing" {
    $s = Get-VwAudioCppTask -Arguments @('--audio','x.wav')
    Assert ($null -eq $s.task) "expected null task"
    Assert ($null -eq $s.family) "expected null family"
}

Check "a trailing flag with no value does not overrun the array" {
    $s = Get-VwAudioCppTask -Arguments @('--task')
    Assert ($null -eq $s.task) "expected null task"
}

Write-Host "task naming"

Check "audio.cpp task names map onto the shared vocabulary" {
    # asr must land in the same bucket as the NeMo and Ollama transcriptions,
    # or per-task comparisons split across two names for the same work.
    Assert ((ConvertTo-VwTaskName 'asr') -eq 'audio-asr') "asr mapped wrong"
    Assert ((ConvertTo-VwTaskName 'sep') -eq 'audio-separation') "sep mapped wrong"
}

Check "an unknown task is namespaced rather than dropped" {
    Assert ((ConvertTo-VwTaskName 'diarize') -eq 'audio-diarize') "unknown task mapped wrong"
}

Write-Host "spool writing"

Check "writes one file per batch" {
    # The drain renames a file to .sent once every line in it is posted; a line
    # appended between the read and the rename would be lost. One file per
    # batch removes the race.
    Write-VwModelRun -Fields @{ model='nemotron_asr'; task='audio-asr'; duration_ms=1000.0 }
    Write-VwModelRun -Fields @{ model='bs_roformer'; task='audio-separation'; duration_ms=2000.0 }
    $files = @(Get-ChildItem $spool -Filter *.jsonl)
    Assert ($files.Count -eq 2) "expected 2 files, got $($files.Count)"
}

Check "each file is one valid JSON batch with one run" {
    $file = @(Get-ChildItem $spool -Filter *.jsonl)[0]
    $batch = Get-Content $file.FullName -Raw | ConvertFrom-Json
    Assert ($batch.schema -eq 1) "schema was $($batch.schema)"
    Assert ($batch.runs.Count -eq 1) "expected 1 run"
    Assert ($batch.source -eq 'vw-audiocpp') "source was $($batch.source)"
}

Check "local work is marked free with an exact zero" {
    $file = @(Get-ChildItem $spool -Filter *.jsonl)[0]
    $run = (Get-Content $file.FullName -Raw | ConvertFrom-Json).runs[0]
    # Distinguishes "cost nothing" from "cost unknown"; the free-vs-paid split
    # depends on it.
    Assert ($run.cost_usd -eq 0.0) "cost_usd was $($run.cost_usd)"
    Assert ($run.is_free -eq $true) "is_free was $($run.is_free)"
    Assert ($run.provider -eq 'audiocpp') "provider was $($run.provider)"
}

Check "a run id is generated when the caller omits one" {
    $file = @(Get-ChildItem $spool -Filter *.jsonl)[0]
    $run = (Get-Content $file.FullName -Raw | ConvertFrom-Json).runs[0]
    Assert ($run.run_id -and $run.run_id.Length -ge 16) "run_id was $($run.run_id)"
}

Check "an unwritable spool does not throw at the caller" {
    # A transcription must never fail because telemetry could not be written.
    $env:VW_RUNS_SPOOL_DIR = "Z:\definitely\not\a\real\path"
    Write-VwModelRun -Fields @{ model='x'; task='audio-asr' }
    $env:VW_RUNS_SPOOL_DIR = $spool
}

Remove-Item $spool -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures failed" -ForegroundColor Red; exit 1 }
Write-Host "all passed" -ForegroundColor Green
