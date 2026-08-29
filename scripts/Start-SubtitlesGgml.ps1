<#
.SYNOPSIS
    Generate .srt subtitles with a pure-ggml pipeline. No PyTorch.

.DESCRIPTION
    Replaces the NeMo/PyTorch path in vw-cli's generate_subtitles.py. Every
    model stage runs through crispasr.exe on ggml:

        ffmpeg      extract 16 kHz mono WAV
        [--Separate] crispasr --separate --stems vocals   (htdemucs, 38 MB)
        crispasr    --backend parakeet -osrt              (parakeet-tdt-0.6b-v3)
        translate_srt.py                                  (deep_translator)

    Writes <name>.srt plus one <name>.<lang>.srt per requested language.

    Separation is OFF by default. It only pays for itself on noisy or
    music-heavy audio, and it currently runs on CPU at roughly 9x realtime:
    the CUDA path aborts in ggml's binbcast kernel, and Windows has no host
    BLAS so the CrossTransformer (~86% of the forward pass) takes the scalar
    route. For clean dialogue parakeet does fine without it.

.PARAMETER Device
    GPU index for crispasr (0 = the RTX 3060; this box is single-GPU now). Sets both
    CUDA_VISIBLE_DEVICES and GGML_VK_VISIBLE_DEVICES via crispasr's --device.

.EXAMPLE
    .\Start-SubtitlesGgml.ps1 -TargetDir "D:\Media" -Recurse -Langs "fr,es"

.EXAMPLE
    .\Start-SubtitlesGgml.ps1 -TargetDir "D:\Media" -Separate -Device 1
#>
[CmdletBinding()]
param(
    [Alias("Folder","Directory")][string]$TargetDir = $PWD.Path,
    [string]$OutputDir,
    [Alias("r")][switch]$Recurse,
    [string]$Langs = "",
    [switch]$SkipExisting,
    [switch]$Separate,
    [int]$Device = -1,
    [double]$VolumeBoost = 1.5,
    # Cue shaping. Without -ml the ASR emits ONE cue for the whole file (a 179 s
    # video came back as a single 163 s subtitle), which is unusable in a
    # player. 42 chars is the usual single-line subtitle width; -sow keeps the
    # split on word boundaries instead of mid-word.
    [int]$MaxLen = 42,
    [switch]$NoSplitOnWord,
    [string[]]$Extensions = @(".mp4",".mkv",".avi",".mov",".webm",".mp3",".wav",".flac",".m4a",".aac",".ogg"),
    [string]$CrispAsr,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $CrispAsr) { $CrispAsr = Join-Path $RepoRoot "CrispASR\build\bin\crispasr.exe" }
if (-not $Python)   { $Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
$Translator = Join-Path $PSScriptRoot "translate_srt.py"

foreach ($p in @($CrispAsr, $Python, $Translator)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Error "missing: $p"; exit 1 }
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Write-Error "ffmpeg not on PATH"; exit 1 }

# crispasr links CUDA 13, whose runtime DLLs live in bin\x64 rather than bin.
# Without this the process dies with 0xC0000135 and prints nothing at all.
$CudaBin = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"
) | Where-Object { Test-Path (Join-Path $_ "cublas64_13.dll") } | Select-Object -First 1
if ($CudaBin) { $env:PATH = "$CudaBin;$(Split-Path $CudaBin);$env:PATH" }
else { Write-Warning "CUDA 13 bin\x64 not found; crispasr may fail to start silently." }

# A stale machine-level CUDA_VISIBLE_DEVICES makes crispasr's --device a silent
# no-op, because cli.cpp only sets the variable when it is not already present.
if ($Device -ge 0 -and $env:CUDA_VISIBLE_DEVICES) {
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
}

if (-not $OutputDir) { $OutputDir = $TargetDir }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) "subs-ggml"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

$files = Get-ChildItem -Path $TargetDir -File -Recurse:$Recurse |
         Where-Object { $Extensions -contains $_.Extension.ToLower() }
if (-not $files) { Write-Host "no media files under $TargetDir"; exit 0 }

Write-Host ("`n  {0} file(s) | separate={1} | langs={2} | device={3} | max-len={4}`n" -f `
    $files.Count, $Separate.IsPresent, $(if($Langs){$Langs}else{"none"}), `
    $(if($Device -ge 0){$Device}else{"auto"}), $MaxLen) -ForegroundColor Cyan

$devArgs = @(); if ($Device -ge 0) { $devArgs = @("--device", "$Device") }
$done = 0; $skipped = 0; $failed = 0

foreach ($f in $files) {
    $base    = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $srtPath = Join-Path $OutputDir "$base.srt"

    # -LiteralPath throughout: media filenames routinely contain [ and ], which
    # PowerShell's path cmdlets treat as wildcard character classes. Without it
    # "Movie [HD].mp4" makes Test-Path return False for a file that exists, and
    # the pipeline throws on its own output.
    if ($SkipExisting -and (Test-Path -LiteralPath $srtPath)) {
        Write-Host "  = $($f.Name) (exists)" -ForegroundColor DarkGray; $skipped++; continue
    }
    Write-Host "  > $($f.Name)" -ForegroundColor White
    $t0 = Get-Date
    try {
        # crispasr cannot open non-ASCII paths on Windows: given "frères.wav" it
        # exits 2 and prints usage even though the file exists (narrow argv +
        # ANSI-codepage fopen). Media filenames are routinely accented, so every
        # path handed to crispasr is an ASCII-safe stem in $Tmp, and results are
        # moved back to the real name afterwards. ffmpeg handles UTF-8 fine, so
        # only the crispasr-facing half needs this.
        $safe = "job_" + [System.BitConverter]::ToString(
            [System.Security.Cryptography.MD5]::Create().ComputeHash(
                [System.Text.Encoding]::UTF8.GetBytes($f.FullName))).Replace("-","").Substring(0,12)

        # 1. audio — 16 kHz mono is what the ASR wants; boost helps quiet dialogue.
        $wav = Join-Path $Tmp "$safe.wav"
        & ffmpeg -hide_banner -loglevel error -nostdin -y -i $f.FullName `
                 -vn -ac 1 -ar 16000 -af "volume=$VolumeBoost" $wav 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $wav)) { throw "ffmpeg failed" }

        # 2. optional vocal isolation
        $asrInput = $wav
        if ($Separate) {
            & $CrispAsr --separate --stems vocals -m auto --backend htdemucs `
                        --sep-output-dir $Tmp -f $wav @devArgs 2>&1 | Out-Null
            $vocals = Join-Path $Tmp "${safe}_vocals.wav"
            if (Test-Path -LiteralPath $vocals) { $asrInput = $vocals }
            else { Write-Warning "    separation produced no stem; using the raw audio" }
        }

        # 3. ASR -> SRT, written under the safe stem then moved to the real name.
        $tmpSrt = Join-Path $Tmp "$safe.srt"
        if (Test-Path -LiteralPath $tmpSrt) { Remove-Item -LiteralPath $tmpSrt -Force }
        $cueArgs = @("-ml", "$MaxLen"); if (-not $NoSplitOnWord) { $cueArgs += "-sow" }
        & $CrispAsr --backend parakeet -m auto -osrt -f $asrInput -of (Join-Path $Tmp $safe) @cueArgs @devArgs 2>&1 | Out-Null
        if (-not (Test-Path -LiteralPath $tmpSrt)) { throw "ASR produced no .srt" }
        Move-Item -LiteralPath $tmpSrt -Destination $srtPath -Force

        # 4. translations
        if ($Langs) { & $Python $Translator --srt $srtPath --langs $Langs }

        $done++
        Write-Host ("    done in {0:N1}s" -f ((Get-Date) - $t0).TotalSeconds) -ForegroundColor Green
    }
    catch {
        $failed++
        Write-Host "    [!] $($_.Exception.Message)" -ForegroundColor Red
    }
    finally {
        foreach ($leftover in @((Join-Path $Tmp "$safe.wav"), (Join-Path $Tmp "${safe}_vocals.wav"))) {
            if (Test-Path -LiteralPath $leftover) { Remove-Item -LiteralPath $leftover -ErrorAction SilentlyContinue }
        }
    }
}

Write-Host ("`n  {0} done, {1} skipped, {2} failed`n" -f $done, $skipped, $failed) -ForegroundColor Cyan
