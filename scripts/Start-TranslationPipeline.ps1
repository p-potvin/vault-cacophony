<#
.SYNOPSIS
    Standalone Speech-to-Text Translation & .SRT Subtitle Generation with Riva-Translate-4B.

.DESCRIPTION
    Runs offline speech translation and synchronized .srt generation decoupled from
    the real-time full-duplex Nemotron conversation pipeline.

    Workflow:
        1. Extract 16 kHz audio (ffmpeg)
        2. Speech recognition (audio.cpp with nemotron-3.5 or parakeet)
        3. Cue segmentation & sentence reconstruction (words_to_srt.py)
        4. Neural Machine Translation (Riva-Translate-4B GGUF)
        5. Proportional cue redistribution into <media>.<lang>.srt

.PARAMETER TargetDir
    Directory or single media file to process.

.PARAMETER Langs
    Comma-separated target language codes (e.g. "es,fr,de,ja,zh").

.PARAMETER SourceLang
    Source language code (default: "en").

.PARAMETER OutputDir
    Custom destination directory for generated .srt files.

.PARAMETER Recurse
    Recursively process subdirectories.

.PARAMETER SkipExisting
    Skip media files that already have translated .srt files generated.

.PARAMETER AsrEngine
    ASR backend model: "nemotron" (default) or "parakeet".

.PARAMETER ModelPath
    Path to Riva-Translate-4B GGUF model file.

.EXAMPLE
    .\Start-TranslationPipeline.ps1 -TargetDir "D:\Media" -Langs "es,fr,de"

.EXAMPLE
    .\Start-TranslationPipeline.ps1 -TargetDir "video.mp4" -Langs "es" -SourceLang "en"
#>
[CmdletBinding()]
param(
    [Alias("Folder", "Path")][string]$TargetDir = $PWD.Path,
    [Parameter(Mandatory=$true)][string]$Langs,
    [string]$SourceLang = "en",
    [string]$OutputDir,
    [Alias("r")][switch]$Recurse,
    [switch]$Overwrite,
    [Alias("SkipExisting", "SkipIfTranslated")][switch]$SkipCompleted,
    [ValidateSet("nemotron", "parakeet")][string]$AsrEngine = "nemotron",
    [string]$ModelPath = "",
    [double]$Gap = 0.6,
    [switch]$NoSeparate,
    [switch]$Separate,
    [ValidateSet("bs_roformer", "htdemucs")][string]$Separator = "bs_roformer",
    [int]$SepPasses = 1
)

$ErrorActionPreference = "Stop"

if (-not $ModelPath) {
    $ModelPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\audio.cpp\models\Riva-Translate-4B-Instruct.i1-Q4_K_M.gguf"))
} else {
    $ModelPath = [System.IO.Path]::GetFullPath($ModelPath)
}

# Ensure CUDA on PATH
$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
$CudaX64 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
if (Test-Path $CudaBin) {
    $env:PATH = "$CudaX64;$CudaBin;" + $env:PATH
}

if (-not (Test-Path -LiteralPath $TargetDir)) {
    Write-Error "Target not found: $TargetDir"
    exit 1
}

$Script = Join-Path $PSScriptRoot "speech_to_translated_srt.py"
if (-not (Test-Path $Script)) {
    Write-Error "Pipeline script not found: $Script"
    exit 1
}

$Extensions = @(".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".flac", ".ogg", ".webm", ".avi", ".mov")

if ((Get-Item -LiteralPath $TargetDir) -is [System.IO.DirectoryInfo]) {
    $files = Get-ChildItem -LiteralPath $TargetDir -File -Recurse:$Recurse |
             Where-Object { $Extensions -contains $_.Extension.ToLower() }
    if (-not $files) {
        Write-Host "No compatible media files found under $TargetDir" -ForegroundColor Yellow
        exit 0
    }
} else {
    $files = @(Get-Item -LiteralPath $TargetDir)
}

Write-Host ("`n=== Riva Speech-to-Text Translation Pipeline ===" ) -ForegroundColor Cyan
Write-Host ("  Files:        {0}" -f $files.Count)
Write-Host ("  Target Langs: {0}" -f $Langs)
Write-Host ("  Source Lang:  {0}" -f $SourceLang)
Write-Host ("  ASR Engine:   {0}" -f $AsrEngine)
Write-Host ("  Model:        {0}`n" -f [System.IO.Path]::GetFileName($ModelPath))

$done = 0; $skipped = 0; $failed = 0

foreach ($f in $files) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $destDir = if ($OutputDir) { $OutputDir } else { $f.DirectoryName }
    
    # Check if all target SRTs exist
    $targetList = $Langs -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    $allExist = $true
    foreach ($lang in $targetList) {
        $expectedSrt = Join-Path $destDir "$base.$lang.srt"
        if (-not (Test-Path -LiteralPath $expectedSrt)) {
            $allExist = $false
            break
        }
    }

    if (-not $Overwrite -and $SkipCompleted -and $allExist) {
        Write-Host "  = $($f.Name) (all target subtitles exist)" -ForegroundColor DarkGray
        $skipped++
        continue
    }

    Write-Host "  > Processing: $($f.Name)" -ForegroundColor White
    $t0 = Get-Date

    $pyArgs = @(
        $Script,
        $f.FullName,
        "--langs", $Langs,
        "--source", $SourceLang,
        "--asr-engine", $AsrEngine,
        "--model", $ModelPath,
        "--gap", $Gap
    )
    if ($Overwrite) {
        $pyArgs += @("--overwrite")
    }
    if ($SkipCompleted) {
        $pyArgs += @("--skip-completed")
    }
    if ($NoSeparate) {
        $pyArgs += @("--no-separate")
    } else {
        $pyArgs += @("--separator", $Separator, "--sep-passes", $SepPasses)
    }
    if ($OutputDir) {
        $pyArgs += @("--out-dir", $OutputDir)
    }

    try {
        & uv run python @pyArgs
        if ($LASTEXITCODE -eq 0) {
            $done++
            $elapsed = ((Get-Date) - $t0).TotalSeconds
            Write-Host ("    [OK] Finished in {0:N1}s" -f $elapsed) -ForegroundColor Green
        } else {
            $failed++
            Write-Host "    [!] Pipeline reported an error" -ForegroundColor Red
        }
    }
    catch {
        $failed++
        Write-Host "    [!] Exception: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ("`nSummary: {0} completed, {1} skipped, {2} failed`n" -f $done, $skipped, $failed) -ForegroundColor Cyan
