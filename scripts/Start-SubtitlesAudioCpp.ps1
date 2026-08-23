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
                        in -SepChunkMinutes slices, crossfaded back together
        audiocpp        --task asr --family parakeet_tdt --words-out
        words_to_srt.py cues from word gaps
        translate_srt.py deep_translator

    Separation is ON by default here, unlike the CrispASR version: at 10.6x
    realtime a 16-minute video costs ~90 s, against ~2.5 hours on CrispASR's
    CPU path and ~35 min for PyTorch demucs at shifts=1.

.PARAMETER Gap
    Seconds of silence that starts a new cue. Cue boundaries come from gaps
    between recognised words, so there is no VAD threshold to tune.

.PARAMETER SepChunkMinutes
    Minutes of audio per separation process. 0 separates the file in one shot.

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
    # htdemucs is flat on the GPU -- ~0.8 GB of VRAM whatever the file length,
    # because the graph only ever sees one 7.8 s training segment. What grows
    # with length is host RAM: the session holds the mix, an accumulator for all
    # four stems and then the four output buffers, all at full length, which
    # measures 0.28 GB per minute of audio on top of a 0.7 GB floor. That is the
    # wall a feature-length file hits, not the 3060. Six minutes a chunk keeps a
    # separation process near 2 GB, leaving room for parakeet (~2 GB VRAM,
    # ~2.2 GB RSS), the ffmpeg workers and a third model.
    [double]$SepChunkMinutes = 6,
    # Each chunk is read $SepOverlap seconds early and crossfaded (linear, so
    # coherent content keeps its level) over that lead-in. Both edges of the
    # fade are the parts htdemucs separates worst -- the first and last moments
    # of an input -- and each sits at zero weight exactly where it is worst.
    # 10 s covers a full 7.8 s training segment on either side.
    [double]$SepOverlap = 10,
    # parakeet's offline_mode defaults to full_context, which encodes the whole
    # utterance and asks for O(T^2) attention memory: an 18-minute file requested
    # a 21 GB CUDA buffer and failed outright. auto switches to bounded
    # overlapping windows past this threshold. 300 s is deliberately
    # conservative -- extrapolating that measurement it needs ~1.7 GB -- and it
    # keeps the better full-context quality for short clips, where long_form
    # measurably loses context ("6 mois" instead of "17 mois de différence").
    [double]$LongFormThreshold = 300,
    # Past the threshold, parakeet re-encodes bounded windows and keeps only the
    # centre of each. That centre defaults to audio_chunk_duration_sec = 2, a
    # streaming-latency number that is disastrous offline: words straddling a
    # centre edge are dropped, and on 22 and 26 minutes of LibriSpeech dev-clean
    # the transcript came back missing 40% of its words -- 42.3% and 44.3% WER.
    # Widening the centre fixes it, non-monotonically, with 45 s the best of
    # 10/15/20/30/45/60/90/120 on both sets: 3.4% and 6.8% WER, and 3x faster
    # than 2 s (29 s -> 10 s for 22 minutes) since it re-encodes 30x less. It
    # costs ~2.5 GB of VRAM against 2.0 GB at 2 s; 90 s and beyond climb past
    # 3 GB and get worse, not better. Under the threshold full_context still
    # wins slightly (5.8% vs 6.5% WER on a 4.7-minute clip), so it stays.
    [double]$AsrWindow = 45,
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
foreach ($exe in @("ffmpeg","ffprobe")) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { Write-Error "$exe not on PATH"; exit 1 }
}

# audio.cpp ships its own CUDA runtime DLLs beside the exe.
$env:PATH = "$AudioCpp;$env:PATH"
# A stale CUDA_VISIBLE_DEVICES from an older shell hides GPUs from the runtime.
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

# Every number handed to ffmpeg goes through here. A French locale renders 2.5
# as "2,5", which ffmpeg reads as a truncated 2 -- silently wrong timestamps.
function Format-Num([double]$v) {
    return $v.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

# audiocpp prints its CUDA banner on stderr, and Windows PowerShell 5.1 turns
# redirected stderr into error records, which $ErrorActionPreference = 'Stop'
# raises as a failure -- the whole file aborts on a line that says "found 2 CUDA
# devices". So every child process goes through here: the exit code is the only
# failure signal, and the output surfaces only when that code is nonzero.
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe @Arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "$What failed (exit $LASTEXITCODE): $((($out | Select-Object -Last 4) -join ' | '))"
        }
    }
    finally { $ErrorActionPreference = $prev }
}

function Get-AudioSeconds([string]$Path) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $raw = & ffprobe -v error -show_entries format=duration -of csv=p=0 -i $Path 2>$null }
    finally { $ErrorActionPreference = $prev }
    $out = 0.0
    if (-not [double]::TryParse(
            "$raw".Trim(), [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture, [ref]$out)) {
        throw "ffprobe gave no duration for $Path"
    }
    return $out
}

# Slice the mix, separate each slice in its own process, and hand back the vocal
# stems in timeline order. Slice i covers [i*chunk - overlap, (i+1)*chunk], so
# consecutive stems share exactly $SepOverlap seconds of real audio -- which is
# what the crossfade downstream consumes, leaving the assembled timeline equal
# to the original one and word timings needing no offset.
function Invoke-Separation {
    param(
        [string]$Cli, [string]$Model, [string]$Mix, [string]$WorkDir,
        [double]$ChunkSec, [double]$Overlap
    )
    $total = Get-AudioSeconds $Mix
    $n = 1
    if ($ChunkSec -gt 0 -and $total -gt $ChunkSec) {
        $n = [int][Math]::Ceiling($total / $ChunkSec)
        # A sliver of a last chunk is worse than an overlong one: htdemucs needs
        # more than one 7.8 s segment to chew on, and the fade would eat most of
        # it. Fold it into its predecessor instead.
        if ($n -ge 2 -and ($total - ($n - 1) * $ChunkSec) -lt (3 * $Overlap)) { $n-- }
    }

    $vocals = @()
    for ($i = 0; $i -lt $n; $i++) {
        $part = Join-Path $WorkDir ("part{0:d3}.wav" -f $i)
        $dir  = Join-Path $WorkDir ("sep{0:d3}" -f $i)
        $src  = $Mix
        if ($n -gt 1) {
            $coreStart = $i * $ChunkSec
            $coreEnd   = if ($i -eq $n - 1) { $total } else { ($i + 1) * $ChunkSec }
            $readStart = if ($i -eq 0) { 0.0 } else { $coreStart - $Overlap }
            Invoke-Native ffmpeg @(
                '-hide_banner','-loglevel','error','-nostdin','-y',
                '-ss',(Format-Num $readStart),'-t',(Format-Num ($coreEnd - $readStart)),
                '-i',$Mix,'-c:a','pcm_s16le',$part) "chunk $($i + 1)/$n cut"
            if (-not (Test-Path -LiteralPath $part)) { throw "ffmpeg could not cut chunk $($i + 1)/$n" }
            $src = $part
            Write-Host ("    sep {0}/{1}" -f ($i + 1), $n) -ForegroundColor DarkGray
        }

        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Invoke-Native $Cli @(
            '--family','htdemucs','--task','sep','--mode','offline','--model',$Model,
            '--backend','cuda','--audio',$src,'--out-dir',$dir) "separation (chunk $($i + 1)/$n)"
        $stem = Join-Path $dir "vocals.wav"
        if (-not (Test-Path -LiteralPath $stem)) { throw "separation produced no vocals stem (chunk $($i + 1)/$n)" }

        $keep = Join-Path $WorkDir ("vocals{0:d3}.wav" -f $i)
        Move-Item -LiteralPath $stem -Destination $keep -Force
        # The other three stems are dead weight; drop them before the next chunk
        # so peak disk holds one chunk of them, not a whole film of them.
        Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue }
        $vocals += $keep
    }
    return ,$vocals
}

# Stitch the vocal stems and resample to the 16 kHz mono the ASR wants in one
# pass, so no full-length 44.1 kHz stereo file is ever written back out.
function Join-VocalStems {
    param([string[]]$Stems, [string]$Out, [double]$Overlap, [double]$Volume)
    $ffArgs = @('-hide_banner','-loglevel','error','-nostdin','-y')
    foreach ($s in $Stems) { $ffArgs += @('-i', $s) }
    $prev = '0:a'
    $filter = ''
    for ($i = 1; $i -lt $Stems.Count; $i++) {
        $filter += "[$prev][${i}:a]acrossfade=d=$(Format-Num $Overlap):c1=tri:c2=tri[x$i];"
        $prev = "x$i"
    }
    $filter += "[$prev]volume=$(Format-Num $Volume)[out]"
    $ffArgs += @('-filter_complex', $filter, '-map', '[out]', '-ac', '1', '-ar', '16000', $Out)
    Invoke-Native ffmpeg $ffArgs "vocal stitch"
}

# Without -OutputDir each .srt lands beside its own video, which is where every
# player looks for it and the only arrangement that survives -Recurse: a flat
# output directory collides the moment two seasons both hold an "Episode 1".
if ($OutputDir) { New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null }
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) "subs-audiocpp"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

# -TargetDir takes one file as readily as a directory. Callers that have a single
# path in hand -- a player asking for one sidecar, a context menu on one file --
# would otherwise have to hand over its parent and hope the extension filter
# spared the rest of the folder.
if (-not (Test-Path -LiteralPath $TargetDir)) { Write-Error "not found: $TargetDir"; exit 1 }
if ((Get-Item -LiteralPath $TargetDir) -is [System.IO.DirectoryInfo]) {
    $files = Get-ChildItem -LiteralPath $TargetDir -File -Recurse:$Recurse |
             Where-Object { $Extensions -contains $_.Extension.ToLower() }
    if (-not $files) { Write-Host "no media files under $TargetDir"; exit 0 }
} else {
    $files = @(Get-Item -LiteralPath $TargetDir)
}

$chunkLabel = if ($NoSeparate) { "n/a" } elseif ($SepChunkMinutes -gt 0) { "$SepChunkMinutes min" } else { "whole file" }
Write-Host ("`n  {0} file(s) | separate={1} | chunk={2} | langs={3} | gap={4}s`n" -f `
    $files.Count, (-not $NoSeparate), $chunkLabel, $(if($Langs){$Langs}else{"none"}), $Gap) -ForegroundColor Cyan

$done = 0; $skipped = 0; $failed = 0
foreach ($f in $files) {
    $base    = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $srtPath = Join-Path $(if ($OutputDir) { $OutputDir } else { $f.DirectoryName }) "$base.srt"

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
    $workDir = Join-Path $Tmp $safe
    try {
        $asrWav = Join-Path $Tmp "$safe.16k.wav"
        if ($NoSeparate) {
            Invoke-Native ffmpeg @(
                '-hide_banner','-loglevel','error','-nostdin','-y','-i',$f.FullName,
                '-vn','-ac','1','-ar','16000','-af',"volume=$(Format-Num $VolumeBoost)",$asrWav) "audio extract"
        } else {
            # htdemucs expects the training rate: 44.1 kHz stereo.
            $sepWav = Join-Path $Tmp "$safe.44k.wav"
            Invoke-Native ffmpeg @(
                '-hide_banner','-loglevel','error','-nostdin','-y','-i',$f.FullName,
                '-vn','-ac','2','-ar','44100',$sepWav) "audio extract"
            if (-not (Test-Path -LiteralPath $sepWav)) { throw "ffmpeg failed" }

            New-Item -ItemType Directory -Force -Path $workDir | Out-Null
            $stems = Invoke-Separation -Cli $Cli -Model $SepModel -Mix $sepWav -WorkDir $workDir `
                                       -ChunkSec ($SepChunkMinutes * 60) -Overlap $SepOverlap
            Join-VocalStems -Stems $stems -Out $asrWav -Overlap $SepOverlap -Volume $VolumeBoost
        }
        if (-not (Test-Path -LiteralPath $asrWav)) { throw "no 16 kHz audio for ASR" }

        $words = Join-Path $Tmp "$safe.words.json"
        # --out is for audio outputs and writes nothing for ASR; the transcript
        # goes to stdout and the timings to --words-out, which is all we need.
        Invoke-Native $Cli @(
            '--task','asr','--family','parakeet_tdt','--model',$AsrModel,'--backend','cuda',
            '--session-option','parakeet_tdt.offline_mode=auto',
            '--session-option',"parakeet_tdt.audio_chunk_threshold_sec=$(Format-Num $LongFormThreshold)",
            '--session-option',"parakeet_tdt.audio_chunk_duration_sec=$(Format-Num $AsrWindow)",
            '--audio',$asrWav,'--words-out',$words) "ASR"
        if (-not (Test-Path -LiteralPath $words)) { throw "ASR produced no words JSON" }

        # The two Python steps keep their console output -- cue and translation
        # counts are the only progress a long run shows -- so they are called
        # directly. Unredirected native stderr is not turned into error records,
        # so their warnings do not abort the file either.
        $tmpSrt = Join-Path $Tmp "$safe.srt"
        & $Python $ToSrt --words $words --out $tmpSrt --gap (Format-Num $Gap) `
                  --max-chars $MaxChars --max-dur (Format-Num $MaxDur) --width $Width
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
        if (Test-Path -LiteralPath $workDir) { Remove-Item -LiteralPath $workDir -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

Write-Host ("`n  {0} done, {1} skipped, {2} failed`n" -f $done, $skipped, $failed) -ForegroundColor Cyan
