<#
.SYNOPSIS
    Time each pass of the pipeline and capture a targeted Nsight Systems trace.

.DESCRIPTION
    Wall-clock timing answers "is it fast enough"; it never answers "why not".
    This runs each stage against a bounded clip and puts an nsys trace beside
    the timing, so the second question has an answer too.

    Traces are kept small on purpose, three ways:

      * A bounded clip. Trace volume scales with work done, not with wall time,
        so profiling a feature film produces a gigabyte-scale .nsys-rep that
        says exactly what thirty seconds says. -Seconds cuts the clip first.
      * --sample=none --cpuctxsw=none. CPU sampling and context-switch tracking
        are the bulk of a default trace and none of it is about the GPU.
      * --trace=cuda only. No OS, no cuDNN/cuBLAS API tracing.

    Measured that way an ASR trace is a few MB rather than a few hundred, and
    still resolves every kernel launch.

    A stage that does not touch CUDA -- the speaker pass runs CAM++ on the CPU
    through onnxruntime -- is timed and reported without a trace, rather than
    given an empty one.

.PARAMETER Audio
    Source media. Anything ffmpeg reads; it is cut and resampled per stage.

.PARAMETER Seconds
    Length of the clip each stage is measured on. 0 uses the whole file, which
    is worth doing once to confirm a rate holds, and not worth tracing.

.PARAMETER Stages
    Any of sep, asr, speakers. Defaults to all three.

.EXAMPLE
    .\Measure-Passes.ps1 -Audio "D:\Media\episode.mkv"

.EXAMPLE
    .\Measure-Passes.ps1 -Audio clip.wav -Stages asr -Seconds 60 -Repeat 3
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Audio,
    [double]$Seconds = 30,
    [ValidateSet("sep","asr","speakers","s2s")][string[]]$Stages = @("sep","asr","speakers"),
    [string]$OutDir,
    [int]$Repeat = 1,
    # Room the s2s agent answers in, matching Start-Conversation.ps1.
    [double]$S2sReplyWindow = 25,
    [switch]$NoTrace,
    [ValidateSet("nemotron","parakeet")][string]$Engine = "nemotron",
    [string]$AudioCpp,
    [string]$Python,
    [string]$Nsys
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $AudioCpp) { $AudioCpp = Join-Path $RepoRoot "audio.cpp" }
if (-not $Python)   { $Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
if (-not $OutDir)   { $OutDir   = Join-Path ([System.IO.Path]::GetTempPath()) "cacophony-bench" }

$Cli = Join-Path $AudioCpp "audiocpp_cli.exe"
$SepModel = Join-Path $AudioCpp "models\htdemucs-f16.gguf"
$AsrModel = if ($Engine -eq "nemotron") {
    Join-Path $AudioCpp "models\Nemotron-3.5-ASR-Streaming-0.6B-GGUF"
} else {
    Join-Path $AudioCpp "models\Parakeet-TDT-0.6B-v3-GGUF\parakeet-tdt-0.6b-v3-q8_0.gguf"
}

# nsys ships inside whichever Nsight is installed, and the two lay themselves
# out differently. Look rather than hard-code a version that will move.
if (-not $Nsys) {
    $Nsys = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Recurse -Filter "nsys.exe" `
                -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
$CanTrace = -not $NoTrace -and $Nsys -and (Test-Path -LiteralPath $Nsys)
if (-not $NoTrace -and -not $CanTrace) {
    Write-Host "  [!] nsys not found; timing only" -ForegroundColor Yellow
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$env:PATH = "$AudioCpp;$env:PATH"
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

function Format-Num([double]$v) {
    return $v.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Invoke-Quiet {
    param([string]$Exe, [string[]]$Arguments)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe @Arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "$(Split-Path -Leaf $Exe) failed (exit $LASTEXITCODE): $((($out | Select-Object -Last 3) -join ' | '))"
        }
    } finally { $ErrorActionPreference = $prev }
}

function Get-AudioSeconds([string]$Path) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $raw = & ffprobe -v error -show_entries format=duration -of csv=p=0 -i $Path 2>$null }
    finally { $ErrorActionPreference = $prev }
    $v = 0.0
    [void][double]::TryParse("$raw".Trim(), [System.Globalization.NumberStyles]::Float,
                             [System.Globalization.CultureInfo]::InvariantCulture, [ref]$v)
    return $v
}

# Peak VRAM is sampled rather than traced: nsys reports allocations the CUDA API
# made, which is not the same as what the driver is holding, and the number
# people actually need is "will this fit beside the other model".
function Start-VramWatch {
    # Each sample is emitted as it is taken. A job that only wrote its running
    # maximum at the end would write nothing at all, because the only way this
    # loop ends is Stop-Job.
    return Start-Job -ScriptBlock {
        while ($true) {
            $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" |
                    Where-Object { $_.Trim() } | ForEach-Object { [int]($_.Trim()) }
            ($used | Measure-Object -Sum).Sum
            Start-Sleep -Milliseconds 200
        }
    }
}

function Stop-VramWatch($job, [int]$baseline) {
    if (-not $job) { return $null }
    Stop-Job $job -ErrorAction SilentlyContinue
    $peak = (Receive-Job $job -ErrorAction SilentlyContinue | Measure-Object -Maximum).Maximum
    Remove-Job $job -Force -ErrorAction SilentlyContinue
    if ($peak -gt $baseline) { return [int]($peak - $baseline) }
    return $null
}

# Both reports out of one nsys invocation, with the export forced.
#
# `nsys stats` caches a .sqlite beside the report and, finding a stale one,
# refuses the whole command rather than re-exporting or using it -- so the
# second run of this script silently lost every GPU number while still printing
# a wall clock. Forcing the export costs a couple of seconds and cannot be
# wrong. One invocation rather than two because the export is the slow part and
# doing it twice buys nothing.
function Get-GpuStats {
    param([string]$Rep)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = & $Nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum `
                     --format csv --force-export=true $Rep 2>$null
    } catch { return $null } finally { $ErrorActionPreference = $prev }
    if (-not $lines) { return $null }

    # The two tables arrive back to back in the order they were asked for, each
    # opening with its own header row. Split on those.
    $tables = @(); $current = $null
    foreach ($line in $lines) {
        if ($line -match '^Time \(%\),') {
            if ($current) { $tables += ,$current }
            $current = @($line)
        } elseif ($current -and $line -match ',') {
            $current += $line
        }
    }
    if ($current) { $tables += ,$current }
    if (-not $tables) { return $null }

    $sum = {
        param($rows)
        if (-not $rows) { return 0.0 }
        return ($rows | ForEach-Object { [double]$_.'Total Time (ns)' } | Measure-Object -Sum).Sum
    }
    $kern = if ($tables.Count -ge 1) { $tables[0] | ConvertFrom-Csv } else { $null }
    $mem  = if ($tables.Count -ge 2) { $tables[1] | ConvertFrom-Csv } else { $null }
    return @{
        KernelNs = & $sum $kern
        MemNs    = & $sum $mem
        Launches = if ($kern) { ($kern | ForEach-Object { [int]$_.Instances } | Measure-Object -Sum).Sum } else { 0 }
        Top      = if ($kern) {
            $kern | Select-Object -First 3 | ForEach-Object {
                # Kernel names are C++ template signatures; everything after the
                # first paren is argument types nobody reads in a summary.
                $name = $_.Name -replace '^void ', '' -replace '\(const.*$', '' -replace '\($', ''
                "{0,5:N1}%  {1,6} x  {2}" -f [double]$_.'Time (%)', $_.Instances, $name.Trim()
            }
        } else { @() }
    }
}

# One measured run: time it, and trace it if the stage uses the GPU.
function Measure-Stage {
    param([string]$Name, [string]$Exe, [string[]]$Arguments, [double]$Media, [switch]$Cpu)

    $trace = Join-Path $OutDir "$Name.nsys-rep"

    # Timing and tracing are separate runs, always. Under nsys this stage runs
    # measurably slower, and polling nvidia-smi beside it costs a little more --
    # so a run that carries either one is not a run whose wall clock is worth
    # reporting. The timed runs carry neither.
    $wall = @()
    for ($i = 0; $i -lt $Repeat; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        Invoke-Quiet $Exe $Arguments
        $sw.Stop()
        $wall += $sw.Elapsed.TotalSeconds
    }
    # The minimum, not the mean: every source of noise here only ever adds time,
    # so the fastest run is the closest thing to the cost of the work itself.
    $best = ($wall | Measure-Object -Minimum).Minimum
    $spread = if ($Repeat -gt 1) {
        "  ({0} runs, +{1:N2}s spread)" -f $Repeat, (($wall | Measure-Object -Maximum).Maximum - $best)
    } else { "" }
    Write-Host ("  {0,-10} {1,7:N2}s wall   {2,7:N1}x realtime{3}" -f `
        $Name, $best, ($Media / $best), $spread) -ForegroundColor Green

    if ($Cpu) {
        Write-Host "             CPU stage (onnxruntime); no CUDA trace" -ForegroundColor DarkGray
        return
    }
    if (-not $CanTrace) { return }

    # The profiled run, whose numbers are ratios and totals rather than a clock.
    $baseline = ((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" |
                 Where-Object { $_.Trim() } | ForEach-Object { [int]($_.Trim()) } |
                 Measure-Object -Sum).Sum
    $watch = Start-VramWatch
    Invoke-Quiet $Nsys (@('profile','--trace=cuda','--sample=none','--cpuctxsw=none',
                          '--cuda-memory-usage=true','--force-overwrite=true',
                          '-o',$trace, $Exe) + $Arguments)
    $peak = Stop-VramWatch $watch $baseline
    if ($peak) { Write-Host ("             {0,7:N0} MB GPU memory over idle" -f $peak) -ForegroundColor DarkGray }
    if (-not (Test-Path -LiteralPath $trace)) { return }

    $stats = Get-GpuStats $trace
    $size  = (Get-Item -LiteralPath $trace).Length / 1MB
    if ($stats -and $stats.KernelNs) {
        $busy = $stats.KernelNs / 1e6
        Write-Host ("             {0,7:N0} ms GPU kernels in {1:N0} launches, {2:N0} ms memcpy" -f `
            $busy, $stats.Launches, ($stats.MemNs / 1e6)) -ForegroundColor DarkGray
        Write-Host ("             {0,7:N1}% of wall spent in GPU kernels" -f `
            (100 * $busy / 1000 / $best)) -ForegroundColor DarkGray
        foreach ($k in $stats.Top) { Write-Host "               $k" -ForegroundColor DarkGray }
    }
    Write-Host ("             trace {0:N1} MB -> {1}" -f $size, $trace) -ForegroundColor DarkGray
}

# -- clips ------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $Audio)) { Write-Error "not found: $Audio"; exit 1 }
$full = Get-AudioSeconds $Audio
$span = if ($Seconds -gt 0 -and $Seconds -lt $full) { $Seconds } else { $full }
$cut  = if ($Seconds -gt 0 -and $Seconds -lt $full) { @('-t', (Format-Num $Seconds)) } else { @() }

$wav16 = Join-Path $OutDir "bench.16k.wav"
$wav44 = Join-Path $OutDir "bench.44k.wav"
Invoke-Quiet ffmpeg (@('-hide_banner','-loglevel','error','-nostdin','-y') + $cut +
                     @('-i',$Audio,'-vn','-ac','1','-ar','16000',$wav16))
if ($Stages -contains "sep") {
    Invoke-Quiet ffmpeg (@('-hide_banner','-loglevel','error','-nostdin','-y') + $cut +
                         @('-i',$Audio,'-vn','-ac','2','-ar','44100',$wav44))
}

Write-Host ("`n  {0}  |  {1:N1}s of {2:N1}s  |  engine={3}  |  trace={4}" -f `
    (Split-Path -Leaf $Audio), $span, $full, $Engine, $CanTrace) -ForegroundColor Cyan
# Loading a 1.8 GB model and building its CUDA context costs the same five
# seconds whether the clip is thirty seconds or thirty minutes, so a short clip
# reports a throughput far below the one a real file gets. The clip is for the
# trace; the rate wants -Seconds 0.
if ($cut.Count -gt 0 -and $span -lt 120) {
    Write-Host ("  realtime factors on a {0:N0}s clip are held down by model load; " -f $span +
                "-Seconds 0 measures the rate`n") -ForegroundColor DarkYellow
} else { Write-Host "" }

# -- stages -----------------------------------------------------------------

if ($Stages -contains "sep") {
    if (-not (Test-Path -LiteralPath $SepModel)) {
        Write-Host "  [!] separation model missing; skipping sep" -ForegroundColor Yellow
    } else {
        Measure-Stage -Name "sep" -Exe $Cli -Media $span -Arguments @(
            '--family','htdemucs','--task','sep','--mode','offline','--model',$SepModel,
            '--backend','cuda','--audio',$wav44,'--out-dir',(Join-Path $OutDir "sep"))
    }
}

if ($Stages -contains "asr") {
    $words = Join-Path $OutDir "bench.words.json"
    $asrArgs = if ($Engine -eq "nemotron") {
        @('--task','asr','--family','nemotron_asr','--model',$AsrModel,'--backend','cuda',
          '--mode','streaming','--language','auto','--request-option','keep_language_tags=true',
          '--audio',$wav16,'--words-out',$words,'--text-out',(Join-Path $OutDir "bench.txt"))
    } else {
        @('--task','asr','--family','parakeet_tdt','--model',$AsrModel,'--backend','cuda',
          '--session-option','parakeet_tdt.offline_mode=auto',
          '--audio',$wav16,'--words-out',$words)
    }
    Measure-Stage -Name "asr" -Exe $Cli -Media $span -Arguments $asrArgs
}

if ($Stages -contains "speakers") {
    # The speaker pass needs cues, and cues come from the ASR. Build them from
    # whatever the asr stage just wrote rather than running the model twice.
    $words = Join-Path $OutDir "bench.words.json"
    $tags  = Join-Path $OutDir "bench.tags.json"
    if (-not (Test-Path -LiteralPath $words)) {
        Write-Host "  [!] speakers needs the asr stage to have run; skipping" -ForegroundColor Yellow
    } else {
        $srtArgs = @((Join-Path $PSScriptRoot "words_to_srt.py"),
                     '--words',$words,'--out',(Join-Path $OutDir "bench.srt"),
                     '--tags-out',$tags,'--media',$Audio,'--model-name',$Engine)
        if ($Engine -eq "nemotron") {
            $srtArgs += @('--merge-tokens','--tagged-text',(Join-Path $OutDir "bench.txt"))
        }
        & $Python @srtArgs | Out-Null
        Measure-Stage -Name "speakers" -Cpu -Exe $Python -Media $span -Arguments @(
            (Join-Path $PSScriptRoot "speakers.py"), '--audio', $wav16, '--tags', $tags)
    }
}

if ($Stages -contains "s2s") {
    # PersonaPlex is only in the from-source build, and it resolves its model
    # contract spec relative to the working directory -- --model-spec-override
    # does not stand in for that -- so this stage runs from the audio.cpp root.
    $s2sCli = Join-Path $AudioCpp "build\windows-cuda-release\bin\audiocpp_cli.exe"
    $s2sModel = Join-Path $AudioCpp "models\PersonaPlex-GGUF"
    if (-not (Test-Path -LiteralPath $s2sCli) -or -not (Test-Path -LiteralPath $s2sModel)) {
        Write-Host "  [!] PersonaPlex build or model missing; skipping s2s" -ForegroundColor Yellow
    } else {
        # Output length equals input length, so the clip *is* the workload. The
        # padding is the room the model answers in, exactly as the conversation
        # driver does it, so the measurement matches the real turn.
        $s2sIn = Join-Path $OutDir "bench.s2s.wav"
        Invoke-Quiet ffmpeg @('-hide_banner','-loglevel','error','-nostdin','-y',
                              '-i',$wav16,'-af',"apad=pad_dur=$(Format-Num $S2sReplyWindow)",
                              '-ac','1','-ar','24000',$s2sIn)
        $s2sSpan = Get-AudioSeconds $s2sIn
        Push-Location $AudioCpp
        try {
            Measure-Stage -Name "s2s" -Exe $s2sCli -Media $s2sSpan -Arguments @(
                '--task','s2s','--family','personaplex','--model','models\PersonaPlex-GGUF',
                '--backend','cuda','--audio',$s2sIn,
                '--text','You are a concise assistant. Reply in one short sentence.',
                '--request-option','voice_id=NATF2',
                '--out',(Join-Path $OutDir "bench.s2s.out.wav"))
        } finally { Pop-Location }
        Write-Host ("             {0:N1}s of stream generated (input {0:N1}s; output length = input length)" -f `
            $s2sSpan) -ForegroundColor DarkGray
    }
}

Write-Host ("`n  traces in {0}`n" -f $OutDir) -ForegroundColor Cyan
