<#
.SYNOPSIS
    Generate .srt subtitles with audio.cpp. No PyTorch, GPU end to end.

.DESCRIPTION
    Supersedes Start-SubtitlesGgml.ps1, which used CrispASR. audio.cpp runs the
    same parakeet-tdt-0.6b-v3 at the same speed but adds a working CUDA
    htdemucs, which CrispASR does not have (its GPU path aborts in ggml's
    binbcast kernel, leaving separation on a scalar CPU route).

        ffmpeg          44.1 kHz stereo (separation) / 16 kHz mono (ASR)
        [-Separate]     audiocpp --family bs_roformer --task sep  ~8x realtime
                        in -SepChunkMinutes slices, crossfaded back together
        audiocpp        --task asr --family nemotron_asr --mode streaming
                        --language auto, sub-word timings + language tags
        words_to_srt.py cues from word gaps, <lang> tags -> .tags.json
        translate_srt.py deep_translator

    Separation is ON by default here, unlike the CrispASR version: at ~8x
    realtime a 16-minute video costs ~2 min, against ~2.5 hours on CrispASR's
    CPU path and ~35 min for PyTorch demucs at shifts=1.

    The separator is BS-RoFormer at one inference pass (-Separator htdemucs for
    the old one). It beats htdemucs where separation actually matters -- speech
    buried 5 dB under music, 54.3% WER against 57.6% -- and matches its speed
    once the pass count is one rather than the packaged four.

    The ASR is nemotron-3.5-asr-streaming-0.6b (-Engine parakeet for the old
    one). It writes <base>.tags.json beside the .srt: the language it decoded
    each cue in, which is the first entry in what will hold speaker, emotion and
    the rest.

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
    # BS-RoFormer replaced htdemucs on the audio that is actually hard. Speech
    # over music at 0 dB, WER after separation: htdemucs 26.4%, BS-RoFormer
    # 27.1%. Push the music 5 dB louder and it inverts -- htdemucs 57.6%,
    # BS-RoFormer 54.3%, and BS-RoFormer returns 226 of the words against 202.
    # It writes two stems instead of four, so three quarters of the stem writing
    # goes with them.
    [ValidateSet("bs_roformer","htdemucs")][string]$Separator = "bs_roformer",
    # RoFormer's overlapping inference passes, and the reason it can feel slow:
    # time scales with this exactly -- 1.8x realtime at 4, 3.8x at 2, 8.0x at 1
    # on a 2-minute clip -- while quality barely moves. At -5 dB four passes
    # scored 53.9% WER against one pass at 54.3%. One pass, which also lands
    # within a whisker of htdemucs's 8.6x.
    [ValidateRange(1, 8)][int]$SepPasses = 1,
    [double]$VolumeBoost = 1.5,
    [double]$Gap = 0.6,
    # Both separators hold the whole mix and its stems in memory, so both grow
    # with the length of what they are given: BS-RoFormer measured 0.76 GB of
    # host RAM on 2 minutes and 2.95 GB on 8, about 0.37 GB a minute, with VRAM
    # climbing 2.8 -> 4.3 GB over the same span. (htdemucs is 0.47 GB a minute
    # and 2.0 -> 3.8 GB.) That is the wall a feature-length file hits, not the
    # card. Six minutes a chunk keeps a separation process near 2 GB of RAM and
    # under 4 GB of VRAM, leaving room for the ASR, the ffmpeg workers and a
    # third model.
    [double]$SepChunkMinutes = 6,
    # Each chunk is read $SepOverlap seconds early and crossfaded (linear, so
    # coherent content keeps its level) over that lead-in. Both edges of the fade
    # are the parts a separator handles worst -- the first and last moments of an
    # input, where it has no context on one side -- and each sits at zero weight
    # exactly where it is worst. 10 s covers any of these models' windows twice
    # over.
    [double]$SepOverlap = 10,
    # parakeet's offline_mode defaults to full_context, which encodes the whole
    # utterance and asks for O(T^2) attention memory: an 18-minute file requested
    # a 21 GB CUDA buffer and failed outright. auto switches to bounded
    # overlapping windows past this threshold. 300 s is deliberately
    # conservative -- extrapolating that measurement it needs ~1.7 GB -- and it
    # keeps the better full-context quality for short clips, where long_form
    # measurably loses context ("6 mois" instead of "17 mois de différence").
    # nemotron replaced parakeet as the default on word recovery, not on WER.
    # Across two LibriSpeech sets parakeet scored 3.41% and 6.77% against
    # nemotron's 3.71% and 2.61% -- but the second set is the tell: parakeet
    # returned 3998 of 4212 words and nemotron 4207. Parakeet-TDT predicts a
    # duration per token and a bad one jumps the decoder forward, so whole
    # sentences vanish; nemotron does not do it. It costs ~31x realtime against
    # parakeet's ~110x, still thirty times faster than the audio, and it detects
    # the language it is decoding. parakeet stays for the live path, where 90 ms
    # a window against 119 ms is worth something.
    [ValidateSet("nemotron","parakeet")][string]$Engine = "nemotron",
    # Attribute each cue to a speaker and write the speaker track. Clusters CAM++
    # embeddings of 2 s windows: 98.1% of speech landed on the right person on a
    # two-speaker test. Names come from the voice store when a cluster matches
    # something enrolled there, and the cluster stays SPEAKER_00 when it does not.
    [switch]$Speakers,
    [string]$Voices,
    # nemotron only. auto makes it announce the language per segment, which is
    # what fills the .tags.json beside the .srt.
    [string]$Language = "auto",
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

# Model-run telemetry. Optional by design: if the module is missing the
# pipeline runs exactly as before, because Write-AudioCppRun checks for the
# command before calling it.
$script:VwAudioSeconds = 0.0
Import-Module (Join-Path $PSScriptRoot "VwTelemetry.psm1") -Force -ErrorAction SilentlyContinue
$SepModel = if ($Separator -eq "bs_roformer") {
    Join-Path $AudioCpp "models\BS-RoFormer-ep368-GGUF\bs-roformer-ep368-q8_0.gguf"
} else {
    Join-Path $AudioCpp "models\htdemucs-f16.gguf"
}
$AsrModel = if ($Engine -eq "nemotron") {
    Join-Path $AudioCpp "models\Nemotron-3.5-ASR-Streaming-0.6B-GGUF"
} else {
    Join-Path $AudioCpp "models\Parakeet-TDT-0.6B-v3-GGUF\parakeet-tdt-0.6b-v3-q8_0.gguf"
}
$ToSrt    = Join-Path $PSScriptRoot "words_to_srt.py"
$ToLang   = Join-Path $PSScriptRoot "translate_srt.py"
$ToSpeakers = Join-Path $PSScriptRoot "speakers.py"

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

    # Only audiocpp invocations are model runs. ffmpeg goes through here too,
    # and recording a remux as a model run would put ordinary media work in the
    # model table and inflate every volume figure with inference that never
    # happened.
    $isModel = $Exe -match 'audiocpp'
    $sw = if ($isModel) { [System.Diagnostics.Stopwatch]::StartNew() } else { $null }

    try {
        $out = & $Exe @Arguments 2>&1
        $code = $LASTEXITCODE
        if ($isModel) { Write-AudioCppRun $Arguments $What $sw.Elapsed.TotalMilliseconds $code }
        if ($code -ne 0) {
            throw "$What failed (exit $code): $((($out | Select-Object -Last 4) -join ' | '))"
        }
    }
    finally { $ErrorActionPreference = $prev }
}

# Records one run per audiocpp invocation. Wrapped in its own try so a
# telemetry fault can never fail a transcription that already succeeded.
function Write-AudioCppRun {
    param([string[]]$Arguments, [string]$What, [double]$ElapsedMs, [int]$ExitCode)
    try {
        if (-not (Get-Command Write-VwModelRun -ErrorAction SilentlyContinue)) { return }
        $spec = Get-VwAudioCppTask -Arguments $Arguments

        $fields = @{
            model       = $(if ($spec.family) { $spec.family } else { 'audiocpp' })
            task        = (ConvertTo-VwTaskName $spec.task)
            runtime     = 'audio.cpp'
            service     = 'subtitles-audiocpp'
            project     = 'vault-cacophony'
            duration_ms = [math]::Round($ElapsedMs, 3)
            backend     = 'cuda'
            caller      = $What
        }
        if ($spec.mode)     { $fields['mode'] = $spec.mode }
        if ($spec.language) { $fields['language'] = $spec.language }
        # Set by the caller before each stage; the real-time factor is the
        # number that means anything for audio work, and it needs the clip
        # length to exist at all.
        if ($script:VwAudioSeconds -gt 0) { $fields['audio_seconds'] = $script:VwAudioSeconds }
        if ($ExitCode -ne 0) {
            $fields['status'] = 'error'
            $fields['error_class'] = "AudioCppExit$ExitCode"
        }
        Write-VwModelRun -Fields $fields
    }
    catch { }
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
        [double]$ChunkSec, [double]$Overlap, [string]$Family, [int]$Passes
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
        $sepArgs = @('--family',$Family,'--task','sep','--mode','offline','--model',$Model,
                     '--backend','cuda','--audio',$src,'--out-dir',$dir)
        if ($Family -eq "bs_roformer") {
            $sepArgs += @('--session-option', "bs_roformer.num_overlap=$Passes")
        }
        # Per chunk, not the whole mix: a 16 minute video cut into eight
        # slices would otherwise report each slice as having processed all
        # 16 minutes, making the real-time factor look eight times better
        # than it is.
        $script:VwAudioSeconds = if ($n -gt 1) { $coreEnd - $readStart } else { $total }
        Invoke-Native $Cli $sepArgs "separation (chunk $($i + 1)/$n)"
        $script:VwAudioSeconds = 0.0
        $stem = Join-Path $dir "vocals.wav"
        if (-not (Test-Path -LiteralPath $stem)) { throw "separation produced no vocals stem (chunk $($i + 1)/$n)" }

        $keep = Join-Path $WorkDir ("vocals{0:d3}.wav" -f $i)
        Move-Item -LiteralPath $stem -Destination $keep -Force
        # The stems nobody asked for are dead weight -- one from BS-RoFormer,
        # three from htdemucs -- so they go before the next chunk, and peak disk
        # holds one chunk of them rather than a whole film.
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
                                       -ChunkSec ($SepChunkMinutes * 60) -Overlap $SepOverlap `
                                       -Family $Separator -Passes $SepPasses
            Join-VocalStems -Stems $stems -Out $asrWav -Overlap $SepOverlap -Volume $VolumeBoost
        }
        if (-not (Test-Path -LiteralPath $asrWav)) { throw "no 16 kHz audio for ASR" }

        # ASR sees the whole clip in one pass, so the full duration is the
        # right denominator for its real-time factor.
        try { $script:VwAudioSeconds = Get-AudioSeconds $asrWav } catch { $script:VwAudioSeconds = 0.0 }

        $words = Join-Path $Tmp "$safe.words.json"
        $tagged = Join-Path $Tmp "$safe.tagged.txt"
        # --out is for audio outputs and writes nothing for ASR; the transcript
        # goes to stdout and the timings to --words-out, which is all we need.
        if ($Engine -eq "nemotron") {
            # Streaming, not offline: nemotron's offline encoder attends over the
            # whole utterance and asked for 51 GB on 22 minutes. Streaming costs
            # ~31x realtime and 5.5 GB, and loses nothing -- 4207 of 4212 words
            # on the set where parakeet returned 3998.
            Invoke-Native $Cli @(
                '--task','asr','--family','nemotron_asr','--model',$AsrModel,'--backend','cuda',
                '--mode','streaming','--language',$Language,
                '--request-option','keep_language_tags=true',
                '--audio',$asrWav,'--words-out',$words,'--text-out',$tagged) "ASR"
        } else {
            Invoke-Native $Cli @(
                '--task','asr','--family','parakeet_tdt','--model',$AsrModel,'--backend','cuda',
                '--session-option','parakeet_tdt.offline_mode=auto',
                '--session-option',"parakeet_tdt.audio_chunk_threshold_sec=$(Format-Num $LongFormThreshold)",
                '--session-option',"parakeet_tdt.audio_chunk_duration_sec=$(Format-Num $AsrWindow)",
                '--audio',$asrWav,'--words-out',$words) "ASR"
        }
        if (-not (Test-Path -LiteralPath $words)) { throw "ASR produced no words JSON" }

        # The two Python steps keep their console output -- cue and translation
        # counts are the only progress a long run shows -- so they are called
        # directly. Unredirected native stderr is not turned into error records,
        # so their warnings do not abort the file either.
        $tmpSrt = Join-Path $Tmp "$safe.srt"
        $srtArgs = @('--words',$words,'--out',$tmpSrt,'--gap',(Format-Num $Gap),
                     '--max-chars',$MaxChars,'--max-dur',(Format-Num $MaxDur),'--width',$Width)
        if ($Engine -eq "nemotron") {
            # nemotron times sub-word pieces and announces the language it decoded
            # each segment in; the tags land in the transcript rather than the
            # token stream, so the cue builder gets both and lines them up.
            $tagsPath = Join-Path $(if ($OutputDir) { $OutputDir } else { $f.DirectoryName }) "$base.tags.json"
            $srtArgs += @('--merge-tokens','--tagged-text',$tagged,'--tags-out',$tagsPath,
                          '--media',$f.FullName,'--model-name','nemotron-3.5-asr-streaming-0.6b')
        }
        & $Python $ToSrt @srtArgs
        if (-not (Test-Path -LiteralPath $tmpSrt)) { throw "cue builder produced no .srt" }
        Move-Item -LiteralPath $tmpSrt -Destination $srtPath -Force

        # The speaker pass reads the cues the line above just wrote, so it runs
        # here rather than as a separate command: the 16 kHz wav it needs is
        # still on disk, and re-extracting it later would cost more than the
        # pass itself.
        if ($Speakers -and $Engine -eq "nemotron") {
            $spkArgs = @($ToSpeakers, '--audio', $asrWav, '--tags', $tagsPath)
            if ($Voices) { $spkArgs += @('--voices', $Voices) }
            & $Python @spkArgs
        } elseif ($Speakers) {
            Write-Host "    [!] -Speakers needs the tag store, which only -Engine nemotron writes" -ForegroundColor Yellow
        }

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
