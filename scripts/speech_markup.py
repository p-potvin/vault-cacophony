#!/usr/bin/env python3
"""One representation of *how* a line is spoken, rendered for every engine.

Every TTS backend here wants delivery spelled differently. Fish Audio S2 reads
free-form bracket cues (`[pause]`, `[laughing]`, `[emphasis]`); OmniVoice has a
closed set of non-verbal tokens (`[laughter]`, `[sigh]`, `[surprise-ah]`) and
speaks anything else literally; Kokoro, Magpie, Chatterbox, CosyVoice and Qwen3
take plain text and nothing more. Writing delivery directly in one engine's
dialect and then string-patching it for the others is how `[icy tone]` ends up
read aloud by Kokoro.

So delivery is held once, engine-neutrally, as a short list of tokens, and the
dialects are *renderings* of it:

    Text("I think") Break(320ms) Filler("um") Break(140ms) Text("that's") ...

The canonical serialisation is W3C SSML 1.1 -- the one cross-vendor standard for
synthesis markup -- using only its portable core:

    <speak>         root, xml:lang
    <prosody>       rate / pitch / volume baseline for the utterance
    <break time=>   every pause, in milliseconds
    <emphasis>      a stressed word
    <mark name=>    for what SSML has no element for (non-verbal vocalisations,
                    filled pauses, the delivery tone), so it survives a round
                    trip and is ignored by any SSML engine that does not use it

Emotion is carried as EmotionML's default dimensional vocabulary (W3C EmotionML
1.0, pleasure-arousal-dominance, each in [-1, 1]) and mapped onto SSML prosody
with a Schröder-style linear mapping: arousal raises pitch, rate and loudness;
valence lifts pitch a little; dominance lowers it and adds level.

Where the numbers come from, so they can be argued with rather than guessed at:

* Pause durations are log-normal and come in three populations -- brief
  (~100-200 ms), medium (~0.5 s) and long (~1-1.5 s, an excess of them in
  spontaneous speech) -- Campione & Véronis, Speech Prosody 2002.
* A filled pause is preceded by silence most of the time (~83% of "uh"/"um" in
  the Edinburgh conversational corpus) -- Dall, Tomalin & Wester, SSW 2016.
* "uh" announces a short delay and "um" a longer one -- Clark & Fox Tree,
  Cognition 2002 -- so "um" is followed by a longer pause than "uh".
* Inserting breaths at phrase boundaries makes synthetic conversation more
  natural -- Sundaram & Narayanan, Eurospeech 2003 -- and speakers breathe at
  major boundaries roughly every breath group of a few seconds.
* Filled pauses are only natural where the speaker is plausibly planning; spread
  evenly into fluent speech they read as a tic -- Székely et al., SSW 2019.

Pure Python, standard library only, so every pass that touches a voice can use
it without pulling in a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
import re
from typing import Dict, Iterable, List, Optional, Union
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

SSML_NS = "http://www.w3.org/2001/10/synthesis"

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

# SSML 1.1 break strengths and the durations they stand for here. The spec
# leaves the mapping to the processor; these follow the pause populations above.
STRENGTH_MS = {"none": 0, "x-weak": 80, "weak": 150, "medium": 350, "strong": 700, "x-strong": 1200}

# Non-verbal vocalisations an utterance can carry. Engine dialects map from these.
VOCALS = ("breath", "sigh", "laugh", "chuckle", "gasp", "clear_throat", "hmm", "sniff", "tsk",
          "confirmation", "question")

# Filled pauses ("uh", "um") are fillers proper; the rest are discourse markers
# that are still words and are always spoken.
FILLED_PAUSES = {"uh", "um", "er", "erm", "uhm", "hm", "mm"}


@dataclass
class Text:
    text: str


@dataclass
class Break:
    ms: int

    @property
    def strength(self) -> str:
        """The SSML strength class this duration falls in."""
        best = "none"
        for name, ms in STRENGTH_MS.items():
            if self.ms >= ms:
                best = name
        return best


@dataclass
class Filler:
    word: str


@dataclass
class Vocal:
    kind: str


@dataclass
class Emphasis:
    text: str
    level: str = "moderate"  # SSML: strong | moderate | reduced | none


Token = Union[Text, Break, Filler, Vocal, Emphasis]


@dataclass
class Prosody:
    """An SSML <prosody> baseline, kept numeric so it can be combined and applied.

    rate is a multiplier (1.0 = default), pitch and range are semitone offsets,
    volume is in dB. These are SSML 1.1's relative forms ("108%", "+1.5st",
    "+2dB"), which every SSML engine accepts and which map directly onto the
    controls the non-SSML engines do have (OmniVoice speed, WSOLA time-stretch).
    """
    rate: float = 1.0
    pitch_st: float = 0.0
    range_st: float = 0.0
    volume_db: float = 0.0

    def combine(self, other: "Prosody") -> "Prosody":
        return Prosody(self.rate * other.rate, self.pitch_st + other.pitch_st,
                       self.range_st + other.range_st, self.volume_db + other.volume_db)

    def is_neutral(self) -> bool:
        return (abs(self.rate - 1.0) < 0.005 and abs(self.pitch_st) < 0.05
                and abs(self.range_st) < 0.05 and abs(self.volume_db) < 0.05)

    def ssml_attrs(self) -> Dict[str, str]:
        attrs = {}
        if abs(self.rate - 1.0) >= 0.005:
            attrs["rate"] = f"{self.rate * 100:.0f}%"
        if abs(self.pitch_st) >= 0.05:
            attrs["pitch"] = f"{self.pitch_st:+.1f}st"
        if abs(self.range_st) >= 0.05:
            attrs["range"] = f"{self.range_st:+.1f}st"
        if abs(self.volume_db) >= 0.05:
            attrs["volume"] = f"{self.volume_db:+.1f}dB"
        return attrs

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Prosody":
        return _from_known(cls, data)

    @classmethod
    def from_ssml_attrs(cls, attrs: Dict[str, str]) -> "Prosody":
        def num(value: str, unit: str) -> float:
            return float(value.strip().replace(unit, "")) if value else 0.0

        rate = attrs.get("rate", "")
        named = {"x-slow": 0.7, "slow": 0.85, "medium": 1.0, "fast": 1.15, "x-fast": 1.3, "default": 1.0}
        if rate in named:
            r = named[rate]
        elif rate.endswith("%"):
            r = float(rate[:-1]) / 100.0
        elif rate:
            r = float(rate)
        else:
            r = 1.0
        return cls(rate=r, pitch_st=num(attrs.get("pitch", ""), "st"),
                   range_st=num(attrs.get("range", ""), "st"),
                   volume_db=num(attrs.get("volume", ""), "dB"))


@dataclass
class Emotion:
    """EmotionML 1.0 dimensional emotion (pleasure/valence, arousal, dominance), each in [-1, 1]."""
    arousal: float = 0.0
    valence: float = 0.0
    dominance: float = 0.0

    def to_prosody(self, intensity: float = 1.0) -> Prosody:
        """Schröder-style linear PAD -> prosody mapping, deliberately mild.

        Expressive TTS engines already act the emotion out from the tone cue;
        this only nudges the baseline in the same direction so engines without
        tone cues still move. Full scale (arousal 1.0) is about +2 st, +10% rate.
        """
        a, v, d = (max(-1.0, min(1.0, x)) * intensity for x in (self.arousal, self.valence, self.dominance))
        return Prosody(
            rate=1.0 + 0.10 * a + 0.03 * v,
            pitch_st=2.0 * a + 0.5 * v - 0.5 * d,
            range_st=2.0 * a + 1.0 * v,
            volume_db=3.0 * a + 1.0 * d,
        )


# Default emotion coordinates for the tone cues the personas use. A ToneAnchor
# may override these; anything not listed is neutral.
TONE_EMOTION: Dict[str, Emotion] = {
    "excited": Emotion(0.8, 0.6, 0.3),
    "surprised": Emotion(0.7, 0.2, -0.2),
    "in a hurry tone": Emotion(0.6, 0.0, 0.4),
    "nervous": Emotion(0.5, -0.4, -0.5),
    "worried": Emotion(0.3, -0.5, -0.4),
    "frustrated": Emotion(0.5, -0.6, 0.3),
    "sarcastic": Emotion(0.1, -0.3, 0.5),
    "skeptical": Emotion(-0.1, -0.2, 0.3),
    "confident": Emotion(0.2, 0.4, 0.7),
    "icy tone": Emotion(-0.3, -0.4, 0.7),
    "calm": Emotion(-0.5, 0.3, 0.2),
    "soft tone": Emotion(-0.4, 0.4, -0.2),
    "sad": Emotion(-0.5, -0.6, -0.4),
    "whisper": Emotion(-0.6, 0.0, -0.3),
}


@dataclass
class Utterance:
    tokens: List[Token] = field(default_factory=list)
    tone: str = ""
    prosody: Prosody = field(default_factory=Prosody)
    lang: str = "en-US"

    # -- building ----------------------------------------------------------

    def add(self, token: Token) -> "Utterance":
        # Adjacent breaks merge: two pauses in a row are one longer pause.
        if isinstance(token, Break) and self.tokens and isinstance(self.tokens[-1], Break):
            self.tokens[-1] = Break(self.tokens[-1].ms + token.ms)
        elif isinstance(token, Text) and self.tokens and isinstance(self.tokens[-1], Text):
            self.tokens[-1] = Text(f"{self.tokens[-1].text} {token.text}".strip())
        elif not (isinstance(token, Text) and not token.text.strip()):
            self.tokens.append(token)
        return self

    def words(self) -> str:
        """The spoken words only -- what a transcript should show."""
        parts = []
        for t in self.tokens:
            if isinstance(t, (Text, Emphasis)):
                parts.append(t.text)
            elif isinstance(t, Filler):
                parts.append(f", {t.word}," if parts else f"{t.word},")
        return _tidy(" ".join(parts))

    def to_dict(self) -> dict:
        return {
            "tone": self.tone,
            "lang": self.lang,
            "prosody": asdict(self.prosody),
            "tokens": [{"type": type(t).__name__.lower(), **asdict(t)} for t in self.tokens],
        }

    # -- SSML --------------------------------------------------------------

    def to_ssml(self) -> str:
        body = []
        if self.tone:
            body.append(f"<mark name={quoteattr('tone:' + self.tone)}/>")
        for t in self.tokens:
            if isinstance(t, Text):
                body.append(escape(t.text))
            elif isinstance(t, Break):
                body.append(f'<break time="{int(t.ms)}ms"/>')
            elif isinstance(t, Filler):
                body.append(f"<mark name={quoteattr('filler:' + t.word)}/>{escape(t.word)}")
            elif isinstance(t, Vocal):
                body.append(f"<mark name={quoteattr('vocal:' + t.kind)}/>")
            elif isinstance(t, Emphasis):
                body.append(f'<emphasis level="{t.level}">{escape(t.text)}</emphasis>')
        inner = " ".join(body)
        inner = re.sub(r"\s+([,.;:!?])", r"\1", inner)
        attrs = self.prosody.ssml_attrs()
        if attrs:
            attr_s = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            inner = f"<prosody {attr_s}>{inner}</prosody>"
        return (f'<speak version="1.1" xmlns="{SSML_NS}" xml:lang="{self.lang}">'
                f"{inner}</speak>")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_time(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("ms"):
        return int(float(value[:-2]))
    if value.endswith("s"):
        return int(float(value[:-1]) * 1000)
    return int(float(value))


def parse_ssml(ssml: str) -> Utterance:
    """SSML (as written by `to_ssml`, or any portable-core SSML) back to an Utterance."""
    root = ET.fromstring(ssml)
    lang = root.get("{http://www.w3.org/XML/1998/namespace}lang") or "en-US"
    utt = Utterance(lang=lang)
    pending_filler: List[Optional[str]] = [None]

    def text(chunk: Optional[str]) -> None:
        if not chunk or not chunk.strip():
            return
        chunk = chunk.strip()
        if pending_filler[0] is not None:
            word = pending_filler[0]
            pending_filler[0] = None
            if chunk.lower().startswith(word.lower()):
                utt.add(Filler(word))
                chunk = chunk[len(word):].strip()
                if not chunk:
                    return
        utt.add(Text(chunk))

    def walk(node: ET.Element) -> None:
        name = _local(node.tag)
        if name == "prosody":
            utt.prosody = utt.prosody.combine(Prosody.from_ssml_attrs(node.attrib))
            text(node.text)
            for child in node:
                walk(child)
                text(child.tail)
            return
        if name == "break":
            if "time" in node.attrib:
                utt.add(Break(_parse_time(node.attrib["time"])))
            else:
                utt.add(Break(STRENGTH_MS.get(node.attrib.get("strength", "medium"), 350)))
            return
        if name == "emphasis":
            utt.add(Emphasis("".join(node.itertext()).strip(), node.attrib.get("level", "moderate")))
            return
        if name == "mark":
            mark = node.attrib.get("name", "")
            kind, _, value = mark.partition(":")
            if kind == "tone":
                utt.tone = value
            elif kind == "vocal":
                utt.add(Vocal(value))
            elif kind == "filler":
                pending_filler[0] = value
            return
        # speak, s, p, voice, anything else: descend.
        text(node.text)
        for child in node:
            walk(child)
            text(child.tail)

    walk(root)
    return utt


# Bracket / punctuation cues as LLMs and engine dialects write them -> tokens.
_BRACKET_VOCAL = {
    "laugh": "laugh", "laughs": "laugh", "laughing": "laugh", "laughter": "laugh",
    "chuckle": "chuckle", "chuckles": "chuckle", "chuckling": "chuckle",
    "sigh": "sigh", "sighs": "sigh", "sighing": "sigh",
    "breath": "breath", "breathes": "breath", "inhale": "breath", "inhales": "breath", "breathing": "breath",
    "gasp": "gasp", "gasps": "gasp", "gasping": "gasp", "surprise-ah": "gasp", "surprise-oh": "gasp",
    "clear throat": "clear_throat", "clears throat": "clear_throat", "throat clear": "clear_throat",
    "hmm": "hmm", "dissatisfaction-hnn": "hmm", "sniff": "sniff", "tsk": "tsk",
    "confirmation-en": "confirmation", "mm-hmm": "confirmation",
    "question-en": "question", "question-ah": "question", "question-oh": "question",
    "question-ei": "question", "question-yi": "question", "hmm?": "question",
    "surprise-wa": "gasp", "surprise-yo": "gasp",
}
_BRACKET_BREAK = {"break": 350, "pause": 350, "short pause": 180, "long pause": 900, "long-break": 900, "long break": 900}


def parse_tagged(text: str, known_tones: Iterable[str] = ()) -> Utterance:
    """A bracket-tagged line (LLM output, Fish/OmniVoice dialect) to an Utterance.

    The first unrecognised bracket cue is taken as the delivery tone (that is
    how the prompts ask for it); `*word*` is emphasis; "..." and "--" are pauses.
    Unknown cues after the tone are dropped rather than risk being spoken.
    """
    known = {t.lower() for t in known_tones} | set(TONE_EMOTION)
    utt = Utterance()
    pos = 0
    pattern = re.compile(r"\[([^\]]{1,40})\]|\*([^*]{1,40})\*|(\.\.\.|…|\s--\s|\s—\s)")
    for m in pattern.finditer(text):
        utt.add(Text(text[pos:m.start()].strip()))
        pos = m.end()
        if m.group(1) is not None:
            cue = m.group(1).strip().lower()
            if cue in _BRACKET_VOCAL:
                utt.add(Vocal(_BRACKET_VOCAL[cue]))
            elif cue in _BRACKET_BREAK:
                utt.add(Break(_BRACKET_BREAK[cue]))
            elif cue == "emphasis":
                nxt = re.match(r"\s*([A-Za-z'][\w'-]*)", text[pos:])
                if nxt:
                    utt.add(Emphasis(nxt.group(1)))
                    pos += nxt.end()
            elif not utt.tone and (cue in known or re.fullmatch(r"[a-z][a-z\s-]{1,30}", cue)):
                utt.tone = cue
        elif m.group(2) is not None:
            utt.add(Emphasis(m.group(2).strip()))
        else:
            utt.add(Break(450 if m.group(3).strip() in ("...", "…") else 250))
    utt.add(Text(text[pos:].strip()))
    # A filled pause typed as a word is still a filled pause.
    out = Utterance(tone=utt.tone, prosody=utt.prosody, lang=utt.lang)
    for t in utt.tokens:
        if isinstance(t, Text):
            for piece in re.split(r"(\b(?:uh|um|erm|uhm)\b,?)", t.text, flags=re.IGNORECASE):
                bare = piece.strip().rstrip(",").lower()
                if bare in FILLED_PAUSES:
                    out.add(Filler(bare))
                elif piece.strip():
                    out.add(Text(piece.strip()))
        else:
            out.add(t)
    return out


def parse_any(text: str, known_tones: Iterable[str] = ()) -> Utterance:
    stripped = text.lstrip()
    if stripped.startswith("<speak"):
        return parse_ssml(stripped)
    return parse_tagged(text, known_tones)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

FISH_VOCAL = {
    "breath": "[breath]", "sigh": "[sighs]", "laugh": "[laughs]", "chuckle": "[chuckling]",
    "gasp": "[gasping]", "clear_throat": "[clear throat]", "hmm": "[hmm]", "sniff": "[sniff]", "tsk": "[tsk]",
    "confirmation": "[mm-hmm]", "question": "[hmm?]",
}
OMNIVOICE_VOCAL = {
    "sigh": "[sigh]", "laugh": "[laughter]", "chuckle": "[laughter]", "gasp": "[surprise-ah]",
    "clear_throat": "[confirmation-en]", "hmm": "[dissatisfaction-hnn]", "sniff": "[sniff]",
    "confirmation": "[confirmation-en]", "question": "[question-ah]",
}
OMNIVOICE_ALLOWED = {
    "[laughter]", "[sigh]", "[sniff]", "[confirmation-en]", "[question-en]", "[question-ah]", "[question-oh]",
    "[question-ei]", "[question-yi]", "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]",
}

ENGINE_DIALECT = {
    "fish_audio": "fish", "fish": "fish", "fish-s2": "fish",
    "omnivoice": "omnivoice", "omni": "omnivoice",
    "ssml": "ssml",
}


def dialect_for(engine: Optional[str]) -> str:
    """fish | omnivoice | ssml | plain -- everything not listed takes plain text."""
    return ENGINE_DIALECT.get((engine or "").lower(), "plain")


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r",\s*([.!?])", r"\1", s)
    s = re.sub(r"([.!?])\s*,", r"\1", s)
    s = re.sub(r"\.\.\.\s*,", "...", s)
    s = re.sub(r",\s*\.\.\.", "...", s)
    s = re.sub(r"(\.\.\.\s*){2,}", "... ", s)
    # A sentence end is already a pause; "Fine.... Whatever" is not a longer one.
    s = re.sub(r"(?<!\.)([.!?])\.\.\.", r"\1", s)
    s = re.sub(r"\.\.\.([.!?])", r"\1", s)
    return s.strip(" ,")


def _punct_break(ms: int) -> str:
    """A pause written as punctuation, for engines whose only pause control is text."""
    if ms < 120:
        return ""
    if ms < 300:
        return ","
    return "..."


def render(utt: Utterance, engine: Optional[str]) -> str:
    """The utterance in `engine`'s own dialect. Nothing unsupported survives."""
    dialect = dialect_for(engine)
    if dialect == "ssml":
        return utt.to_ssml()
    out: List[str] = []
    if dialect == "fish" and utt.tone:
        out.append(f"[{utt.tone}]")
    for t in utt.tokens:
        if isinstance(t, Text):
            out.append(t.text)
        elif isinstance(t, Filler):
            # Set off by commas: most filled pauses sit in a silence of their own.
            after_words = bool(out) and not out[-1].rstrip().endswith("]")
            out.append(f", {t.word}," if after_words else f"{t.word},")
        elif isinstance(t, Emphasis):
            out.append(f"[emphasis] {t.text}" if dialect == "fish" else t.text)
        elif isinstance(t, Vocal):
            if dialect == "fish":
                out.append(FISH_VOCAL.get(t.kind, ""))
            elif dialect == "omnivoice":
                out.append(OMNIVOICE_VOCAL.get(t.kind, ""))
        elif isinstance(t, Break):
            if dialect == "fish":
                if t.ms >= 120:
                    out.append("[short pause]" if t.ms < 300 else "[pause]" if t.ms < 800 else "[long pause]")
            else:
                out.append(_punct_break(t.ms))
    rendered = _tidy(" ".join(x for x in out if x))
    if dialect == "omnivoice":
        rendered = re.sub(r"\[[^\]]+\]", lambda m: m.group(0) if m.group(0) in OMNIVOICE_ALLOWED else "", rendered)
        rendered = _tidy(rendered)
    if dialect == "plain":
        rendered = _tidy(re.sub(r"\[[^\]]*\]|[*_~`]", "", rendered))
    # A line that opens on punctuation reads as a stray click to most engines.
    rendered = re.sub(r"^(?:[,;:]\s*)+", "", rendered)
    return rendered


def sanitize_for_engine(text: str, engine: Optional[str], known_tones: Iterable[str] = ()) -> str:
    """Last line of defence before synthesis: whatever markup came in, emit the engine's dialect."""
    if not text:
        return text
    if dialect_for(engine) == "ssml" and text.lstrip().startswith("<speak"):
        return text
    if "<speak" not in text and "[" not in text and "*" not in text and dialect_for(engine) != "ssml":
        return re.sub(r"\s+", " ", text).strip()
    return render(parse_any(text, known_tones), engine)


# --------------------------------------------------------------------------
# Natural-delivery model: where pauses, breaths and filled pauses go
# --------------------------------------------------------------------------

def _from_known(cls, data: Optional[dict]):
    """Build a dataclass from a stored dict, ignoring keys this version does not know."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass
class PauseModel:
    """Log-normal pause populations (Campione & Véronis 2002), medians in ms."""
    brief_ms: float = 160.0
    medium_ms: float = 480.0
    long_ms: float = 1200.0
    sigma: float = 0.35  # log-space spread; ~0.3-0.45 matches conversational corpora

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PauseModel":
        return _from_known(cls, data)

    def sample(self, kind: str, rng: random.Random) -> int:
        median = {"brief": self.brief_ms, "medium": self.medium_ms, "long": self.long_ms}[kind]
        return int(round(math.exp(math.log(median) + self.sigma * rng.gauss(0.0, 1.0))))


@dataclass
class DisfluencyModel:
    """How often and where a speaker hesitates."""
    filled_pause_rate: float = 0.03          # filled pauses per word (conversational English ~1-6 per 100 words)
    um_ratio: float = 0.45                   # share of filled pauses that are "um" rather than "uh"
    silence_before_filler_prob: float = 0.83  # Dall et al. 2016
    repetition_prob: float = 0.04            # chance per turn of a function-word repetition ("I- I think")
    breath_group_words: int = 12             # words between breaths in running speech (~3-4 s)
    onset_breath_min_words: int = 14         # take a breath before turns at least this long

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "DisfluencyModel":
        return _from_known(cls, data)

    def filler_for_delay(self, rng: random.Random) -> str:
        return "um" if rng.random() < self.um_ratio else "uh"


_REPEATABLE = re.compile(r"^(I|I'm|it's|the|we|you|that|and|but|so|it|this)\b", re.IGNORECASE)
_PLANNING_POINT = re.compile(r"(,\s+|\s+(?=(?:but|because|so|although|while|whereas|and then|which|when)\b))",
                             re.IGNORECASE)


def humanize(
    words: Union[str, Utterance],
    rng: random.Random,
    pauses: Optional[PauseModel] = None,
    disfluency: Optional[DisfluencyModel] = None,
    taking_floor: bool = True,
    interrupting: bool = False,
    tone: str = "",
    prosody: Optional[Prosody] = None,
    pause_prob: float = 0.35,
) -> Utterance:
    """Words (or an already-cued Utterance) -> an Utterance with breaths, pauses and hesitations.

    Hesitations go only at planning points (clause boundaries, before a
    conjunction), never evenly through the line, and their number scales with
    the line's length rather than being a fixed per-turn quota. Cues already in
    an Utterance (a vocalisation or emphasis an LLM placed on purpose) stay
    where they were; only its plain-text stretches are worked on.
    """
    pauses = pauses or PauseModel()
    dis = disfluency or DisfluencyModel()
    source = words if isinstance(words, Utterance) else Utterance(tokens=[Text(words.strip())])
    utt = Utterance(tone=tone or source.tone, prosody=prosody or source.prosody, lang=source.lang)
    n_words = len(source.words().split())
    if not n_words:
        return utt

    if taking_floor and not interrupting and n_words >= dis.onset_breath_min_words:
        utt.add(Vocal("breath"))
        utt.add(Break(pauses.sample("brief", rng)))

    first_text = next((t for t in source.tokens if isinstance(t, Text)), None)
    if first_text is not None and not interrupting and rng.random() < dis.repetition_prob:
        m = _REPEATABLE.match(first_text.text)
        if m and source.tokens and source.tokens[0] is first_text:
            utt.add(Text(f"{m.group(1)}-"))
            utt.add(Break(pauses.sample("brief", rng) // 2))

    expected_fillers = dis.filled_pause_rate * n_words
    budget = int(expected_fillers) + (1 if rng.random() < expected_fillers - int(expected_fillers) else 0)
    texts = [t.text for t in source.tokens if isinstance(t, Text)]
    boundaries = max(1, sum(len(_PLANNING_POINT.findall(t)) for t in texts))
    since_breath = 0

    for token in source.tokens:
        if not isinstance(token, Text):
            utt.add(token)
            if isinstance(token, Vocal) and token.kind == "breath":
                since_breath = 0
            continue
        chunks = _PLANNING_POINT.split(token.text)
        for i, chunk in enumerate(chunks):
            if i % 2 == 1:  # a separator: a planning point
                if chunk.strip().startswith(","):
                    utt.add(Text(","))
                if since_breath >= dis.breath_group_words and not interrupting:
                    utt.add(Break(pauses.sample("brief", rng)))
                    utt.add(Vocal("breath"))
                    since_breath = 0
                if budget > 0 and rng.random() < min(1.0, budget / boundaries):
                    if rng.random() < dis.silence_before_filler_prob:
                        utt.add(Break(pauses.sample("brief", rng)))
                    filler = dis.filler_for_delay(rng)
                    utt.add(Filler(filler))
                    # "um" signals the longer delay, so it is followed by the longer pause.
                    utt.add(Break(pauses.sample("medium" if filler == "um" else "brief", rng)))
                    budget -= 1
                elif rng.random() < pause_prob:
                    utt.add(Break(pauses.sample("brief", rng)))
                continue
            piece = chunk.strip()
            if piece:
                utt.add(Text(piece))
                since_breath += len(piece.split())
    return utt


__all__ = [
    "Text", "Break", "Filler", "Vocal", "Emphasis", "Token", "Prosody", "Emotion", "Utterance",
    "TONE_EMOTION", "PauseModel", "DisfluencyModel", "parse_ssml", "parse_tagged", "parse_any",
    "render", "sanitize_for_engine", "dialect_for", "humanize",
]
