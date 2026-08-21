<#
.SYNOPSIS
    Generate .srt subtitles with audio.cpp. No PyTorch, GPU end to end.

.DESCRIPTION
    Supersedes Start-SubtitlesGgml.ps1, which used CrispASR. audio.cpp runs the
    same parakeet-tdt-0.6b-v3 at the same speed but adds a working CUDA
    htdemucs, which CrispASR does not have (its GPU path aborts in ggml's
    binbcast kernel, leaving separation on a scalar CPU route).

        ffmpeg          44.1 kHz stereo (separation) / 16 kHz mono (ASR)
        [-Separate]     audiocpp --family htdemucs --task sep   ~10.6x realtime
        audiocpp        --task asr --family parakeet_tdt --words-out
        words_to_srt.py cues from word gaps
        translate_srt.py deep_translator

    Separation is ON by default here, unlike the CrispASR version: at 10.6x
    realtime a 16-minute video costs ~90 s, against ~2.5 hours on CrispASR's
    CPU path and ~35 min for PyTorch demucs at shifts=1.

.PARAMETER Gap
    Seconds of silence that starts a new cue. Cue boundaries come from gaps
    between recognised words, so there is no VAD threshold to tune.

.EXAMPLE
    .\Start-SubtitlesAudioCpp.ps1 -TargetDir "D:\Media" -Recurse -Langs "en"

.EXAMPLE
    .\Start-SubtitlesAudioCpp.ps1 -TargetDir "D:\wavs" -NoSeparate -Langs "fr,es"
#>
[CmdletBinding()]
param(
    [Alias("Folder","Directory")][string]$TargetDir = $PWD.Path,
    [string]$OutputDir,
    [Alias("r")][switch]$Recurse,
    [string]$Langs = "",
    [switch]$SkipExisting,
    [switch]$NoSeparate,
    [double]$VolumeBoost = 1.5,
    [double]$Gap = 0.6,
    # parakeet's offline_mode defaults to full_context, which encodes the whole
    # utterance and asks for O(T^2) attention memory: an 18-minute file requested
    # a 21 GB CUDA buffer and failed outright. auto switches to bounded
    # overlapping windows past this threshold. 300 s is deliberately
    # conservative -- extrapolating that measurement it needs ~1.7 GB -- and it
    # keeps the better full-context quality for short clips, where long_form
    # measurably loses context ("6 mois" instead of "17 mois de différence").
    [double]$LongFormThreshold = 300,
    [int]$MaxChars = 76,
    [double]$MaxDur = 6.0,
    [int]$Width = 42,
    [string[]]$Extensions = @(".mp4",".mkv",".avi",".mov",".webm",".mp3",".wav",".flac",".m4a",".aac",".ogg"),
    [string]$AudioCpp,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $AudioCpp) { $AudioCpp = Join-Path $RepoRoot "audio.cpp" }
if (-not $Python)   { $Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe" }

$Cli      = Join-Path $AudioCpp "audiocpp_cli.exe"
$SepModel = Join-Path $AudioCpp "models\htdemucs-f16.gguf"
$AsrModel = Join-Path $AudioCpp "models\Parakeet-TDT-0.6B-v3-GGUF\parakeet-tdt-0.6b-v3-q8_0.gguf"
$ToSrt    = Join-Path $PSScriptRoot "words_to_srt.py"
$ToLang   = Join-Path $PSScriptRoot "translate_srt.py"

foreach ($p in @($Cli, $AsrModel, $Python, $ToSrt, $ToLang)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Error "missing: $p"; exit 1 }
}
if (-not $NoSeparate -and -not (Test-Path -LiteralPath $SepModel)) {
    Write-Error "missing separation model: $SepModel (or pass -NoSeparate)"; exit 1
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Write-Error "ffmpeg not on PATH"; exit 1 }

# audio.cpp ships its own CUDA runtime DLLs beside the exe.
$env:PATH = "$AudioCpp;$env:PATH"
# A stale CUDA_VISIBLE_DEVICES from an older shell hides GPUs from the runtime.
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

if (-not $OutputDir) { $OutputDir = $TargetDir }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) "subs-audiocpp"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

$files = Get-ChildItem -LiteralPath $TargetDir -File -Recurse:$Recurse |
         Where-Object { $Extensions -contains $_.Extension.ToLower() }
if (-not $files) { Write-Host "no media files under $TargetDir"; exit 0 }

Write-Host ("`n  {0} file(s) | separate={1} | langs={2} | gap={3}s`n" -f `
    $files.Count, (-not $NoSeparate), $(if($Langs){$Langs}else{"none"}), $Gap) -ForegroundColor Cyan

$done = 0; $skipped = 0; $failed = 0
foreach ($f in $files) {
    $base    = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $srtPath = Join-Path $OutputDir "$base.srt"

    # -LiteralPath everywhere: media names routinely contain [ and ], which
    # PowerShell path cmdlets treat as wildcard character classes, so Test-Path
    # returns False for files that exist.
    if ($SkipExisting -and (Test-Path -LiteralPath $srtPath)) {
        Write-Host "  = $($f.Name) (exists)" -ForegroundColor DarkGray; $skipped++; continue
    }
    Write-Host "  > $($f.Name)" -ForegroundColor White
    $t0 = Get-Date
    # Non-ASCII paths have broken these runtimes before, so everything handed to
    # the CLI is an ASCII stem under the temp dir; results move back afterwards.
    $safe = "job_" + [System.BitConverter]::ToString(
        [System.Security.Cryptography.MD5]::Create().ComputeHash(
            [System.Text.Encoding]::UTF8.GetBytes($f.FullName))).Replace("-","").Substring(0,12)
    $stemDir = Join-Path $Tmp $safe
    try {
        $asrWav = Join-Path $Tmp "$safe.16k.wav"
        if ($NoSeparate) {
            & ffmpeg -hide_banner -loglevel error -nostdin -y -i $f.FullName `
                     -vn -ac 1 -ar 16000 -af "volume=$VolumeBoost" $asrWav 2>&1 | Out-Null
        } else {
            # htdemucs expects the training rate: 44.1 kHz stereo.
            $sepWav = Join-Path $Tmp "$safe.44k.wav"
            & ffmpeg -hide_banner -loglevel error -nostdin -y -i $f.FullName `
                     -vn -ac 2 -ar 44100 $sepWav 2>&1 | Out-Null
            if (-not (Test-Path -LiteralPath $sepWav)) { throw "ffmpeg failed" }

            New-Item -ItemType Directory -Force -Path $stemDir | Out-Null
            & $Cli --family htdemucs --task sep --mode offline --model $SepModel `
                   --backend cuda --audio $sepWav --out-dir $stemDir 2>&1 | Out-Null
            $vocals = Join-Path $stemDir "vocals.wav"
            if (-not (Test-Path -LiteralPath $vocals)) { throw "separation produced no vocals stem" }

            & ffmpeg -hide_banner -loglevel error -nostdin -y -i $vocals `
                     -ac 1 -ar 16000 -af "volume=$VolumeBoost" $asrWav 2>&1 | Out-Null
        }
        if (-not (Test-Path -LiteralPath $asrWav)) { throw "no 16 kHz audio for ASR" }

        $words = Join-Path $Tmp "$safe.words.json"
        # --out is for audio outputs and writes nothing for ASR; the transcript
        # goes to stdout and the timings to --words-out, which is all we need.
        & $Cli --task asr --family parakeet_tdt --model $AsrModel --backend cuda `
               --session-option "parakeet_tdt.offline_mode=auto" `
               --session-option "parakeet_tdt.audio_chunk_threshold_sec=$LongFormThreshold" `
               --audio $asrWav --words-out $words 2>&1 | Out-Null
        if (-not (Test-Path -LiteralPath $words)) { throw "ASR produced no words JSON" }

        $tmpSrt = Join-Path $Tmp "$safe.srt"
        & $Python $ToSrt --words $words --out $tmpSrt --gap $Gap `
                  --max-chars $MaxChars --max-dur $MaxDur --width $Width
        if (-not (Test-Path -LiteralPath $tmpSrt)) { throw "cue builder produced no .srt" }
        Move-Item -LiteralPath $tmpSrt -Destination $srtPath -Force

        if ($Langs) { & $Python $ToLang --srt $srtPath --langs $Langs }

        $done++
        Write-Host ("    done in {0:N1}s" -f ((Get-Date) - $t0).TotalSeconds) -ForegroundColor Green
    }
    catch { $failed++; Write-Host "    [!] $($_.Exception.Message)" -ForegroundColor Red }
    finally {
        foreach ($x in @("$safe.16k.wav","$safe.44k.wav","$safe.words.json")) {
            $p = Join-Path $Tmp $x
            if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue }
        }
        if (Test-Path -LiteralPath $stemDir) { Remove-Item -LiteralPath $stemDir -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

Write-Host ("`n  {0} done, {1} skipped, {2} failed`n" -f $done, $skipped, $failed) -ForegroundColor Cyan
