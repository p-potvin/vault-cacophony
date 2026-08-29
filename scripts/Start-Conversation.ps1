<#
.SYNOPSIS
    Two PersonaPlex agents talking to each other, one turn at a time.

.DESCRIPTION
    The first step toward the full-duplex loop, deliberately stopping short of
    it: the agents alternate rather than overlap. Something you can hear is
    something you can adjust, and a strictly alternating conversation is
    listenable, seekable and diffable in a way two people talking over each
    other is not. Overlap is the next step, not this one.

        seed.wav  ->  agent A hears it, replies  ->  turn001.wav
                  ->  agent B hears that, replies -> turn002.wav
                  ->  agent A hears that ...

    PersonaPlex is full duplex, which changes the shape of a "turn". It emits
    one output frame per input frame, so it does not listen and then answer --
    it answers *into* the stream it is being given, and its reply is only as
    long as the room left for it. Measured on a 6 s question padded to 15.9 s:

        0.0 - 5.9 s    silent, listening
        5.9 - 11.6 s   its answer, starting as the question ends
        11.6 - 15.9 s  silent again, finished

    Two things follow, and the script is built on both. Input is padded with
    -ReplyWindow seconds of silence, or the answer is cut off by the end of the
    audio. And the model endpoints itself -- the trailing silence is it having
    decided to stop -- so the end of a turn is found by trimming rather than by
    a separate VAD pass.

    Each turn is a .wav on disk, the whole thing is stitched into one
    conversation.wav, and the turns are written into a tag store with a speaker
    track -- the same shape the subtitle pipeline produces for a film. That tag
    store is ground truth for who spoke when, which makes every generated
    conversation a free test fixture for the diarization pass.

    Routing is file-based on purpose. The previous attempt at this went through
    virtual audio devices (VB-CABLE, Voicemeeter) and never reached a model
    problem -- it died on colliding audio drivers. Passing audio between agents
    as files means there is no virtual cable, no routing matrix, and no acoustic
    echo to cancel, because neither agent is connected to anything that could
    hear itself.

.PARAMETER Seed
    The opening utterance: a .wav of someone saying something to start with.
    Any sample rate; it is converted for the model.

.PARAMETER ReplyWindow
    Seconds of silence appended to every input, which is the room the agent has
    to answer in. Too small truncates replies; too large only costs time, since
    the model falls silent when it is done.

.PARAMETER Play
    Play each turn as it is produced, so the conversation can be followed live
    rather than only listened to afterwards.

.EXAMPLE
    .\Start-Conversation.ps1 -Seed samples\question.wav -Turns 6 -Play

.EXAMPLE
    .\Start-Conversation.ps1 -Seed opener.wav -Turns 8 `
        -APrompt "You are a sceptical physicist." `
        -BPrompt "You are an enthusiastic startup founder."
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Seed,
    [int]$Turns = 6,
    [string]$OutDir,
    # Room for the agent to answer in. 12 s was not enough -- three of six turns
    # ran to the end of it and were cut off mid-word. The model goes quiet when
    # it is finished, and the trim then discards the rest, so an over-large
    # window costs generation time and nothing else. Err long.
    [double]$ReplyWindow = 25.0,

    [string]$AName = "AGENT_A",
    [string]$AVoice = "NATF2",
    [string]$APrompt = "You are a curious, concise conversationalist. Reply naturally and briefly, then ask the other person something.",

    [string]$BName = "AGENT_B",
    [string]$BVoice = "NATM1",
    [string]$BPrompt = "You are a thoughtful, concise conversationalist. Reply naturally and briefly, then ask the other person something.",

    # Carry the conversation forward as text in each agent's system prompt.
    #
    # The alternative -- keeping the model's KV cache between turns -- was built
    # and measured and does not work here: continuing a session skips the voice
    # and system prompts, so both agents collapse into one voice and one
    # persona, and feeding the model its own speech on the user stream sends it
    # into a repetition loop ("Hi, my name is Amy. Hi, my name is Amy."). A
    # fresh session per turn keeps each agent's own voice, and hearing the
    # *other* voice is what the model was trained for.
    #
    # Text is also the cheaper carrier: a minute of speech is 750 frames of
    # audio context against roughly 200 tokens of transcript, and it survives
    # past the 3000-frame ceiling that audio context cannot.
    [switch]$NoTranscript,
    # How many recent turns to replay. The whole conversation would grow the
    # prompt without bound; the recent past is what a reply actually needs.
    [int]$TranscriptTurns = 2,
    [switch]$Play,
    [double]$Temperature = 0.8,
    [int]$SamplingSeed = 42424242,
    [string]$Model,
    [string]$Cli,
    [string]$AudioCpp,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $AudioCpp) { $AudioCpp = Join-Path $RepoRoot "audio.cpp" }
if (-not $Python)   { $Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
if (-not $Model)    { $Model    = "models\PersonaPlex-GGUF" }
if (-not $OutDir)   { $OutDir   = Join-Path ([System.IO.Path]::GetTempPath()) "cacophony-conversation" }
# The vendored audiocpp_cli.exe predates PersonaPlex; the from-source build is
# what has it. Prefer that, and fall back so the path is at least explicit.
if (-not $Cli) {
    $built = Join-Path $AudioCpp "build\windows-cuda-release\bin\audiocpp_cli.exe"
    $Cli = if (Test-Path -LiteralPath $built) { $built } else { Join-Path $AudioCpp "audiocpp_cli.exe" }
}

$ToTags = Join-Path $PSScriptRoot "conversation_tags.py"
$ToTrim = Join-Path $PSScriptRoot "trim_speech.py"
# The ASR runs through the vendored binary, which already has nemotron and is
# what the subtitle pipeline uses. PersonaPlex needs the from-source build. They
# are never resident together -- PersonaPlex alone takes 10.9 GB of a 12 GB card
# -- so the two run as separate processes, one after the other.
$AsrCli = Join-Path $AudioCpp "audiocpp_cli.exe"
$AsrModel = Join-Path $AudioCpp "models\Nemotron-3.5-ASR-Streaming-0.6B-GGUF"
# PersonaPlex speaks at Mimi's rate. Keeping the whole conversation there means
# nothing is resampled between turns.
$Rate = 24000

foreach ($p in @($Cli, $Python, $ToTags, $ToTrim, $Seed)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Error "missing: $p"; exit 1 }
}
if (-not (Test-Path -LiteralPath (Join-Path $AudioCpp $Model))) {
    Write-Error "missing model: $(Join-Path $AudioCpp $Model)"; exit 1
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Write-Error "ffmpeg not on PATH"; exit 1 }

$env:PATH = "$AudioCpp;$env:PATH"
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Format-Num([double]$v) {
    return $v.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

# audiocpp prints its CUDA banner on stderr, which PowerShell turns into error
# records that $ErrorActionPreference = 'Stop' raises as a failure. Exit code is
# the only signal that means anything.
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe @Arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "$What failed (exit $LASTEXITCODE): $((($out | Select-Object -Last 4) -join ' | '))"
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

# SoundPlayer is synchronous and takes plain PCM wav, which is what the model
# writes. It is in the framework, so hearing the conversation costs no
# dependency -- and being able to hear it is the whole point of alternating first.
function Invoke-Playback([string]$Path) {
    try {
        $player = New-Object System.Media.SoundPlayer $Path
        $player.PlaySync()
        $player.Dispose()
    } catch {
        Write-Host "      [!] could not play: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

$agents = @(
    @{ Name = $AName; Voice = $AVoice; Prompt = $APrompt },
    @{ Name = $BName; Voice = $BVoice; Prompt = $BPrompt }
)

# Nemotron announces the language it decoded each segment in; the tags are for
# the subtitle path and only clutter a prompt.
function Get-Spoken([string]$Wav) {
    if ($NoTranscript) { return "" }
    $base = [System.IO.Path]::ChangeExtension($Wav, $null).TrimEnd('.')
    $asrWav = "$base.16k.wav"
    $asrTxt = "$base.txt"
    try {
        Invoke-Native ffmpeg @('-hide_banner','-loglevel','error','-nostdin','-y',
                               '-i',$Wav,'-ac','1','-ar','16000',$asrWav) "asr resample"
        Invoke-Native $AsrCli @(
            '--task','asr','--family','nemotron_asr','--model',$AsrModel,'--backend','cuda',
            '--mode','streaming','--language','auto',
            '--audio',$asrWav,'--text-out',$asrTxt) "transcribe"
    } catch {
        Write-Host "      [!] could not transcribe: $($_.Exception.Message)" -ForegroundColor Yellow
        return ""
    }
    if (-not (Test-Path -LiteralPath $asrTxt)) { return "" }
    $text = (Get-Content -LiteralPath $asrTxt -Raw)
    $text = [regex]::Replace($text, '<[A-Za-z]{2,3}(-[A-Za-z]{2,4})?>', ' ')
    return ($text -replace '\s+', ' ').Trim()
}

# The transcript goes in the system prompt because that is the only text channel
# the model has: plain text wrapped in <system> tags and tokenized.
#
# The *shape* of it matters more than the content. This slot holds a persona
# description in training, so a transcript laid out as a chat log reads as
# nothing the model has seen. Two things measurably degrade with a long,
# multi-line prompt: the agent stops waiting for the other speaker to finish
# (reply latency went from 5.9 s to 0.00 s on every turn) and it drifts off
# persona entirely -- given a founder persona and a chat log it answered
# "Thanks for calling Desert Dreams, how can I help you today".
#
# So the history is folded into one flowing paragraph that still reads as a
# persona instruction, and only the last exchange or two is carried. This is the
# cheap-context idea working within what the model will accept, rather than
# against it.
function Build-Prompt {
    param([hashtable]$Agent, [object[]]$Said, [string]$Other)
    if ($NoTranscript -or -not $Said) { return $Agent.Prompt }
    # Everything except the last utterance. The last one is already arriving as
    # audio, and presenting it again as text put both channels behind the same
    # words: the agents converged on one sentence and repeated it at each other
    # ("Yeah, that's right. We figured it out already", four turns running).
    # Text carries the memory the audio cannot; it should not restate the audio.
    $older = @($Said | Select-Object -SkipLast 1 | Select-Object -Last $TranscriptTurns)
    if (-not $older) { return "$($Agent.Prompt) You are talking with $Other." }
    $context = ($older | ForEach-Object { "{0} said: {1}" -f $_.who, $_.text }) -join " Then "
    return "$($Agent.Prompt) You are talking with $Other. Earlier in this conversation, " +
           "$context Now answer what you just heard, and say something new."
}

Write-Host ("`n  {0} ({1}) <-> {2} ({3})  |  {4} turns  |  {5}s reply window" -f `
    $AName, $AVoice, $BName, $BVoice, $Turns, $ReplyWindow) -ForegroundColor Cyan
Write-Host ("  seed: {0}`n" -f (Split-Path -Leaf $Seed)) -ForegroundColor Cyan

$current = Join-Path $OutDir "turn000.seed.wav"
Invoke-Native ffmpeg @('-hide_banner','-loglevel','error','-nostdin','-y',
                       '-i',$Seed,'-vn','-ac','1','-ar',"$Rate",$current) "seed conversion"

$record = @()
$said = @()
$clock = Get-AudioSeconds $current
$record += [ordered]@{ agent = "SEED"; voice = "-"; wav = $current; text = "";
                       start = 0.0; end = [math]::Round($clock, 3);
                       latency = $null; tail = $null }
if ($Play) { Invoke-Playback $current }

# PersonaPlex resolves its model contract spec relative to the working
# directory, and --model-spec-override does not stand in for that, so the CLI is
# run from the audio.cpp root where model_specs/ lives.
Push-Location $AudioCpp
try {
    # The opener is part of the conversation, so the first agent is told what it
    # is answering rather than having to infer it from audio alone.
    $seedText = Get-Spoken $current
    if ($seedText) {
        Write-Host ("   0. opener: `"{0}`"" -f $seedText) -ForegroundColor Gray
        $said += @{ who = "SOMEONE"; text = $seedText }
        $record[0].text = $seedText
    }

    for ($i = 0; $i -lt $Turns; $i++) {
        $agent  = $agents[$i % 2]
        $stem   = Join-Path $OutDir ("turn{0:d3}" -f ($i + 1))
        $padded = "$stem.in.wav"
        $raw    = "$stem.raw.wav"
        $wav    = "$stem.wav"

        Write-Host ("  {0,2}. {1} ({2}) listening..." -f ($i + 1), $agent.Name, $agent.Voice) `
            -ForegroundColor White

        # How long the other agent talks for. The reply is whatever comes after
        # that, so this is both the padding anchor and what the trim skips.
        $heard = Get-AudioSeconds $current

        # The room the agent answers in. Without it the reply is cut off by the
        # end of the input, because output length equals input length.
        Invoke-Native ffmpeg @('-hide_banner','-loglevel','error','-nostdin','-y',
                               '-i',$current,'-af',"apad=pad_dur=$(Format-Num $ReplyWindow)",
                               '-ac','1','-ar',"$Rate",$padded) "pad turn $($i + 1)"

        $other = $agents[($i + 1) % 2].Name
        $prompt = Build-Prompt -Agent $agent -Said $said -Other $other

        $t0 = Get-Date
        try {
            Invoke-Native $Cli @(
                '--task','s2s','--family','personaplex','--model',$Model,'--backend','cuda',
                '--audio',$padded,
                '--text',$prompt,
                '--request-option',"voice_id=$($agent.Voice)",
                '--temperature',(Format-Num $Temperature),
                # A different seed per turn. Holding one seed across a whole
                # conversation makes every turn a deterministic function of very
                # similar input, which pushes the agents into shadowing each
                # other -- repeating the content they just heard back in their
                # own voice instead of answering it.
                '--seed',"$($SamplingSeed + $i)",
                '--out',$raw) "turn $($i + 1) ($($agent.Name))"
        } catch {
            Write-Host "      [!] $($_.Exception.Message)" -ForegroundColor Red
            break
        }
        $spent = ((Get-Date) - $t0).TotalSeconds
        if (-not (Test-Path -LiteralPath $raw)) {
            Write-Host "      [!] no audio produced; stopping" -ForegroundColor Red
            break
        }

        # Trimming is also the endpoint detector: what it cuts off the end is
        # the model having stopped talking of its own accord.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $trimJson = & $Python $ToTrim --in $raw --out $wav --skip (Format-Num $heard) --json 2>&1
        $trimCode = $LASTEXITCODE
        $ErrorActionPreference = $prev
        if ($trimCode -ne 0) {
            Write-Host "      [!] the agent said nothing; stopping" -ForegroundColor Yellow
            break
        }
        $trim = $trimJson | ConvertFrom-Json

        $colour = if ($trim.truncated) { "Yellow" } else { "DarkGray" }
        $note = if ($trim.truncated) { "  [!] cut off -- raise -ReplyWindow" } else { "" }
        Write-Host ("      {0:N2}s speech  (waited {1:N2}s, left {2:N2}s)  in {3:N1}s{4}" -f `
            $trim.speech, $trim.latency, $trim.tail, $spent, $note) -ForegroundColor $colour

        # Transcribe it now rather than at the end: this is what the other agent
        # is about to be told was said, so it has to exist before the next turn.
        $spoken = Get-Spoken $wav
        if ($spoken) {
            Write-Host ("      `"{0}`"" -f $spoken) -ForegroundColor Gray
            $said += @{ who = $agent.Name; text = $spoken }
        }

        $record += [ordered]@{ agent = $agent.Name; voice = $agent.Voice; wav = $wav; text = $spoken;
                               start = [math]::Round($clock, 3);
                               end = [math]::Round($clock + $trim.speech, 3);
                               latency = $trim.latency; tail = $trim.tail }
        $clock += $trim.speech
        if ($Play) { Invoke-Playback $wav }

        # This turn is what the other agent hears next. That one line is the
        # conversation loop; everything else here is bookkeeping.
        $current = $wav
    }
}
finally { Pop-Location }

if ($record.Count -le 1) {
    Write-Host "`n  no turns generated`n" -ForegroundColor Red
    exit 1
}

# Stitch, so the conversation is one file to play rather than a folder to click
# through. The turns already sit end to end on the timeline recorded above.
$listFile = Join-Path $OutDir "concat.txt"
$record | ForEach-Object { "file '$((Resolve-Path -LiteralPath $_.wav).Path -replace '\\','/')'" } |
    Set-Content -LiteralPath $listFile -Encoding ASCII
$conversation = Join-Path $OutDir "conversation.wav"
Invoke-Native ffmpeg @('-hide_banner','-loglevel','error','-nostdin','-y',
                       '-f','concat','-safe','0','-i',$listFile,
                       '-ac','1','-ar',"$Rate",$conversation) "conversation stitch"

$turnsJson = Join-Path $OutDir "turns.json"
ConvertTo-Json @($record) -Depth 4 | Set-Content -LiteralPath $turnsJson -Encoding UTF8
$tagsPath = Join-Path $OutDir "conversation.tags.json"
& $Python $ToTags --turns $turnsJson --out $tagsPath --media $conversation

$spoken = $record | Where-Object { $_.latency -ne $null }
if ($spoken) {
    $meanLatency = ($spoken | ForEach-Object { $_.latency } | Measure-Object -Average).Average
    Write-Host ("`n  {0:N1}s of conversation over {1} turns, mean reply latency {2:N2}s" -f `
        $clock, $spoken.Count, $meanLatency) -ForegroundColor Green
}
Write-Host ("    {0}" -f $conversation) -ForegroundColor Green
Write-Host ("    {0}`n" -f $tagsPath) -ForegroundColor DarkGray
