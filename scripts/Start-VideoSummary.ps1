<#
.SYNOPSIS
    Dense captions and a structured summary for a video, VSS-shaped, on one GPU.

.DESCRIPTION
    The NVIDIA VSS blueprint's summarization profile, reduced to what runs on
    this box: no NIMs, no Elasticsearch, no Kafka. ffmpeg chunks the file, a
    GGUF vision model served by Ollama captions each chunk, and a text pass
    aggregates the captions into the LVS response shape. Everything lands
    beside the file:

        <base>.captions.jsonl   one line per chunk (rt-vlm's dense caption shape)
        <base>.summary.json     the LVS CompletionResponse envelope
        <base>.summary.md       the same, readable
        <base>.tags.json        `caption` and `event` tracks in the tag store

    The request vocabulary is VSS's: a scenario, a list of event types to
    surface, optional objects of interest, chunk duration and frames per chunk.
    With -Audio the .srt that Start-SubtitlesAudioCpp.ps1 wrote is folded into
    each chunk's prompt, which is what VSS's ENABLE_AUDIO does with its own ASR.

.PARAMETER Events
    Comma-separated event types, e.g. "typing,window change". Without it every
    chunk gets a free-form dense caption and the summary uses "notable activity".

.PARAMETER Audio
    Read <base>.srt beside each file and hand the cues for each chunk to the
    VLM. Requires the subtitle pass to have run first.

.EXAMPLE
    .\Start-VideoSummary.ps1 -TargetDir "D:\Media\clip.mp4" -Scenario "screen recording" -Events "typing,window change" -Audio

.EXAMPLE
    .\Start-VideoSummary.ps1 -TargetDir "D:\Media" -Recurse -SkipExisting -CaptionsOnly
#>
[CmdletBinding()]
param(
    [Alias("Folder","Directory")][string]$TargetDir = $PWD.Path,
    [Alias("r")][switch]$Recurse,
    [switch]$SkipExisting,
    [switch]$Resume,
    [switch]$Recaption,
    [switch]$CaptionsOnly,
    [string]$Scenario = "",
    [string]$Events = "",
    [string]$Objects = "",
    [switch]$Audio,
    # VSS defaults: 10 s chunks, no overlap, fixed frames per chunk.
    [double]$ChunkSeconds = 0,
    [double]$OverlapSeconds = -1,
    [int]$FramesPerChunk = 0,
    [string]$VlmModel = "",
    [string]$LlmModel = "",
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [string]$Config = "",
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
$Pipeline = Join-Path $PSScriptRoot "vss_pipeline.py"
if (-not $Config) { $Config = Join-Path $RepoRoot "config\vss\lvs-local.yaml" }

Import-Module (Join-Path $PSScriptRoot "VwTelemetry.psm1") -Force -ErrorAction SilentlyContinue

foreach ($p in @($Python, $Pipeline)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Error "missing: $p"; exit 1 }
}
foreach ($exe in @("ffmpeg","ffprobe")) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { Write-Error "$exe not on PATH"; exit 1 }
}

# Ollama is the only runtime. A missing model is the usual first failure, so
# it is checked here rather than surfacing as an HTTP 404 on chunk one.
try {
    $tags = Invoke-RestMethod -Uri "$($OllamaUrl.TrimEnd('/'))/api/tags" -TimeoutSec 10
} catch {
    Write-Error "Ollama not reachable at $OllamaUrl ($($_.Exception.Message)). Start the Ollama app first."; exit 1
}
$pulled = @($tags.models | ForEach-Object { $_.name })
$effectiveVlm = $VlmModel
if (-not $effectiveVlm -and (Test-Path -LiteralPath $Config)) {
    $m = Select-String -LiteralPath $Config -Pattern '^\s*vlm_model:\s*(\S+)' | Select-Object -First 1
    if ($m) { $effectiveVlm = $m.Matches[0].Groups[1].Value }
}
if (-not $effectiveVlm) { $effectiveVlm = "qwen3-vl:2b" }
foreach ($m in @($effectiveVlm, $LlmModel) | Where-Object { $_ }) {
    if ($pulled -notcontains $m -and $pulled -notcontains "$m:latest") {
        Write-Error "model '$m' is not pulled. Run: ollama pull $m"; exit 1
    }
}

function Format-Num([double]$v) { return $v.ToString([System.Globalization.CultureInfo]::InvariantCulture) }

$Extensions = @(".mp4",".mkv",".webm",".mov",".avi",".m4v",".ts",".mpg",".mpeg",".wmv")
if (-not (Test-Path -LiteralPath $TargetDir)) { Write-Error "not found: $TargetDir"; exit 1 }
if ((Get-Item -LiteralPath $TargetDir) -is [System.IO.DirectoryInfo]) {
    $files = Get-ChildItem -LiteralPath $TargetDir -File -Recurse:$Recurse |
             Where-Object { $Extensions -contains $_.Extension.ToLower() }
    if (-not $files) { Write-Host "no video files under $TargetDir"; exit 0 }
} else {
    $files = @(Get-Item -LiteralPath $TargetDir)
}

$cmd = if ($CaptionsOnly) { "caption" } else { "summarize" }
$eventList = @($Events -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$objectList = @($Objects -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })

Write-Host ("`n  {0} file(s) | {1} | vlm={2} | llm={3} | events={4} | audio={5}`n" -f `
    $files.Count, $cmd, $effectiveVlm, $(if ($LlmModel) { $LlmModel } else { "same as vlm" }),
    $(if ($eventList) { $eventList -join "," } else { "free-form" }), $Audio.IsPresent) -ForegroundColor Cyan

$done = 0; $skipped = 0; $failed = 0
foreach ($f in $files) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $want = Join-Path $f.DirectoryName $(if ($CaptionsOnly) { "$base.captions.jsonl" } else { "$base.summary.json" })
    if ($SkipExisting -and (Test-Path -LiteralPath $want)) {
        Write-Host "  = $($f.Name) (exists)" -ForegroundColor DarkGray; $skipped++; continue
    }
    # The Python side prints the file header, per-chunk progress and its own
    # timing, so nothing is echoed here beyond failures.
    $args = @($Pipeline, $cmd, $f.FullName, '--config', $Config, '--ollama-url', $OllamaUrl)
    if ($VlmModel)     { $args += @('--vlm-model', $VlmModel) }
    if ($LlmModel)     { $args += @('--llm-model', $LlmModel) }
    if ($Scenario)     { $args += @('--scenario', $Scenario) }
    foreach ($e in $eventList)  { $args += @('--event', $e) }
    foreach ($o in $objectList) { $args += @('--object-of-interest', $o) }
    if ($ChunkSeconds -gt 0)    { $args += @('--chunk-duration', (Format-Num $ChunkSeconds)) }
    if ($OverlapSeconds -ge 0)  { $args += @('--chunk-overlap-duration', (Format-Num $OverlapSeconds)) }
    if ($FramesPerChunk -gt 0)  { $args += @('--num-frames-per-chunk', $FramesPerChunk) }
    if ($Audio)     { $args += '--enable-audio' }
    if ($Resume)    { $args += '--resume' }
    if ($Recaption -and -not $CaptionsOnly) { $args += '--recaption' }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python @args
    $code = $LASTEXITCODE
    $sw.Stop()

    # One telemetry row per file: the VLM run is the model work, the LLM pass
    # rides on the same model by default. Best-effort, never fatal.
    if (Get-Command Write-VwModelRun -ErrorAction SilentlyContinue) {
        $duration = 0.0
        try { $duration = [double](& ffprobe -v error -show_entries format=duration -of csv=p=0 $f.FullName) } catch {}
        Write-VwModelRun @{
            provider = "ollama"; runtime = "ollama"; model = $effectiveVlm
            task = "video-captioning"; status = $(if ($code -eq 0) { "ok" } else { "error" })
            duration_ms = [int]$sw.Elapsed.TotalMilliseconds
            media_seconds = $duration; file = $f.Name
            scenario = $Scenario; events = ($eventList -join ",")
        }
    }

    if ($code -eq 0) {
        $done++
    } else {
        $failed++
        Write-Host "    [!] pipeline exited $code" -ForegroundColor Red
    }
}

Write-Host ("`n  {0} done, {1} skipped, {2} failed`n" -f $done, $skipped, $failed) -ForegroundColor Cyan
