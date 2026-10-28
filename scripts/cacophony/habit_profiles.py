"""Voice personality and VocalHabitProfile definitions.

Each persona / custom voice has a distinct vocal personality that governs:
1. `tone_palette`: Allowed opening delivery tones (e.g. `[icy tone]`, `[excited]`, `[sarcastic]`,
   `[in a hurry tone]`, `[soft tone]`) with semantic anchor descriptions for EmbeddingGemma-2 scoring.
2. `filler_habits`: Weighted speaker-specific filler expressions (`"like"`, `"I mean"`, `"uh"`,
   `"you know"`, `"honestly"`, `"look"`) with allowed grammatical slots (`pre_onset`, `post_verb`,
   `clause_break`, `pre_adj`).
3. `throat_and_pause_habits`: Probabilities for respiratory/anatomical noises (`clear_throat`,
   `sigh`, `gasp`, `laugh`, `chuckle`, `confirmation`, `dissatisfaction`, `surprise`) and pauses
   at legal slots (`pre_onset`, `clause_break`, `post_turn`), plus `*emphasis*` probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToneAnchor:
    """A single opening tone tag and its semantic anchor text for EmbeddingGemma-2 similarity."""
    tag: str                       # e.g. "icy tone", "excited", "sarcastic", "in a hurry tone", "soft tone"
    anchor: str                    # Semantic description compared against turn text via EG2 cosine
    lexical_keywords: List[str] = field(default_factory=list)
    omnivoice_onset: str = ""      # Optional OmniVoice onset token when this tone wins (e.g. "[sigh]", "[surprise-ah]")


@dataclass
class FillerItem:
    """A persona-specific filler word or phrase and the syntactic slots where it can appear."""
    word: str                      # e.g. "like", "I mean", "uh", "you know", "honestly"
    weight: float = 1.0            # Relative selection weight within the speaker's lexicon
    slots: List[str] = field(default_factory=lambda: ["post_verb", "clause_break"])


@dataclass
class ThroatAndPauseHabits:
    """Anatomical throat/breath noise, pause, and emphasis probabilities by respiratory boundary."""
    pre_onset_prob: float = 0.28           # Chance of a throat noise / breath before speaking
    pre_onset_tags: Dict[str, float] = field(
        default_factory=lambda: {"clear_throat": 0.45, "sigh": 0.35, "confirmation": 0.20}
    )
    clause_break_pause_prob: float = 0.40  # Chance of a pause ([break] / — / ...) or mid-sentence sigh at clause break
    clause_break_sigh_prob: float = 0.18   # Chance that a clause-break event is a sigh/breath rather than a clean pause
    post_turn_prob: float = 0.22           # Chance of a trailing chuckle/sigh/reaction at the end of a turn
    post_turn_tags: Dict[str, float] = field(
        default_factory=lambda: {"sigh": 0.50, "chuckle": 0.30, "laugh": 0.20}
    )
    emphasis_prob: float = 0.35            # Chance to mark 1 salient content word with *emphasis*
    cooldown_turns: int = 1                # Minimum turns by this speaker between throat noises


@dataclass
class VocalHabitProfile:
    """Complete vocal personality profile for a speaker or enrolled reference voice."""
    voice_name: str
    persona_name: str
    default_tone: str
    tone_palette: List[ToneAnchor] = field(default_factory=list)
    filler_rate: float = 0.35              # Probability of injecting a filler in a turn (can trigger multiple if > 0.65)
    max_fillers_per_turn: int = 2
    filler_lexicon: List[FillerItem] = field(default_factory=list)
    throat_habits: ThroatAndPauseHabits = field(default_factory=ThroatAndPauseHabits)
    cadence_speed: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VocalHabitProfile":
        tones = [ToneAnchor(**t) if isinstance(t, dict) else t for t in data.get("tone_palette", [])]
        fillers = [FillerItem(**f) if isinstance(f, dict) else f for f in data.get("filler_lexicon", [])]
        th_raw = data.get("throat_habits") or {}
        throat = ThroatAndPauseHabits(**th_raw) if isinstance(th_raw, dict) else th_raw
        return cls(
            voice_name=data.get("voice_name", "unknown"),
            persona_name=data.get("persona_name", data.get("voice_name", "UNKNOWN")),
            default_tone=data.get("default_tone", "calm"),
            tone_palette=tones,
            filler_rate=float(data.get("filler_rate", 0.35)),
            max_fillers_per_turn=int(data.get("max_fillers_per_turn", 2)),
            filler_lexicon=fillers,
            throat_habits=throat,
            cadence_speed=float(data.get("cadence_speed", 1.0)),
        )


def builtin_profiles() -> Dict[str, VocalHabitProfile]:
    """Default VocalHabitProfiles keyed by both persona name (MARGOT, DESMOND, ...) and custom voice stem."""
    margot = VocalHabitProfile(
        voice_name="high-woman-voice-excited",
        persona_name="MARGOT",
        default_tone="icy tone",
        tone_palette=[
            ToneAnchor(
                tag="icy tone",
                anchor="Cold, blunt, skeptical analysis, unimpressed critique, cutting precision.",
                lexical_keywords=["ridiculous", "actually", "data", "wrong", "opinion", "hat", "hats", "yellow", "taste"],
                omnivoice_onset="[dissatisfaction-hnn]",
            ),
            ToneAnchor(
                tag="excited",
                anchor="Urgent reaction, sudden breakthrough, animated critique, high-energy take!",
                lexical_keywords=["whoa", "look", "pattern", "bright", "color", "jump", "now", "alert", "huge"],
                omnivoice_onset="[surprise-ah]",
            ),
            ToneAnchor(
                tag="in a hurry tone",
                anchor="Fast interrupting correction, urgent interjection, no time to waste.",
                lexical_keywords=["wait", "stop", "hold on", "listen", "immediately", "quick", "check"],
                omnivoice_onset="[question-ah]",
            ),
        ],
        filler_rate=0.38,
        max_fillers_per_turn=1,
        filler_lexicon=[
            FillerItem("frankly", weight=1.4, slots=["pre_onset", "clause_break"]),
            FillerItem("I mean", weight=1.2, slots=["pre_onset", "clause_break"]),
            FillerItem("literally", weight=1.0, slots=["post_verb", "pre_adj"]),
            FillerItem("look", weight=1.1, slots=["pre_onset"]),
        ],
        throat_habits=ThroatAndPauseHabits(
            pre_onset_prob=0.34,
            pre_onset_tags={"clear_throat": 0.50, "sigh": 0.35, "dissatisfaction": 0.15},
            clause_break_pause_prob=0.48,
            clause_break_sigh_prob=0.22,
            post_turn_prob=0.20,
            post_turn_tags={"sigh": 0.70, "dissatisfaction": 0.30},
            emphasis_prob=0.55,
            cooldown_turns=1,
        ),
        cadence_speed=1.15,
    )

    desmond = VocalHabitProfile(
        voice_name="sarcastic-male-voice",
        persona_name="DESMOND",
        default_tone="sarcastic",
        tone_palette=[
            ToneAnchor(
                tag="sarcastic",
                anchor="Dry, ironic, contrarian pushback, mocking an obvious explanation or conspiracy pattern.",
                lexical_keywords=["sure", "coincidence", "pattern", "obviously", "totally", "yellow", "right", "classic"],
                omnivoice_onset="[sigh]",
            ),
            ToneAnchor(
                tag="skeptical",
                anchor="Questioning the official story, probing deeper into a hidden systemic cascade or anomaly.",
                lexical_keywords=["why", "cascade", "scrubber", "wonder", "suspicious", "strange", "odd", "if"],
                omnivoice_onset="[question-en]",
            ),
            ToneAnchor(
                tag="confident",
                anchor="Laying out a grand theory with smug certainty and amusement.",
                lexical_keywords=["told you", "exactly", "proves", "connected", "always", "knew"],
                omnivoice_onset="[laughter]",
            ),
        ],
        filler_rate=0.55,
        max_fillers_per_turn=2,
        filler_lexicon=[
            FillerItem("I mean", weight=1.6, slots=["pre_onset", "clause_break"]),
            FillerItem("you know", weight=1.4, slots=["clause_break", "post_verb"]),
            FillerItem("uh", weight=1.2, slots=["pre_onset", "post_verb"]),
            FillerItem("like", weight=1.1, slots=["post_verb", "pre_adj"]),
        ],
        throat_habits=ThroatAndPauseHabits(
            pre_onset_prob=0.38,
            pre_onset_tags={"sigh": 0.45, "clear_throat": 0.35, "chuckle": 0.20},
            clause_break_pause_prob=0.45,
            clause_break_sigh_prob=0.28,
            post_turn_prob=0.32,
            post_turn_tags={"chuckle": 0.55, "sigh": 0.30, "laugh": 0.15},
            emphasis_prob=0.45,
            cooldown_turns=1,
        ),
        cadence_speed=1.08,
    )

    kai = VocalHabitProfile(
        voice_name="normal-male-slightly-faster-cadence",
        persona_name="KAI",
        default_tone="in a hurry tone",
        tone_palette=[
            ToneAnchor(
                tag="in a hurry tone",
                anchor="Rapid-fire observation, blurting out live sensor numbers and wild takes at high speed.",
                lexical_keywords=["sensor", "numbers", "fast", "bro", "dude", "wild", "crazy", "jumping"],
                omnivoice_onset="[confirmation-en]",
            ),
            ToneAnchor(
                tag="excited",
                anchor="Mind-blown enthusiasm, astonished reaction to a sudden spike or crazy theory!",
                lexical_keywords=["whoa", "insane", "huge", "wait", "seriously", "boom", "massive"],
                omnivoice_onset="[surprise-oh]",
            ),
            ToneAnchor(
                tag="surprised",
                anchor="Startled realization, double-taking at unexpected readings.",
                lexical_keywords=["what", "hold up", "weird", "no way", "really"],
                omnivoice_onset="[question-oh]",
            ),
        ],
        filler_rate=0.72,  # Heavy "like" / "uh" habit after verbs and before adjectives
        max_fillers_per_turn=2,
        filler_lexicon=[
            FillerItem("like", weight=2.4, slots=["post_verb", "pre_adj", "clause_break"]),
            FillerItem("uh", weight=1.3, slots=["pre_onset", "post_verb"]),
            FillerItem("honestly", weight=1.1, slots=["pre_onset", "clause_break"]),
            FillerItem("you know", weight=1.0, slots=["clause_break"]),
        ],
        throat_habits=ThroatAndPauseHabits(
            pre_onset_prob=0.30,
            pre_onset_tags={"gasp": 0.45, "confirmation": 0.35, "clear_throat": 0.20},
            clause_break_pause_prob=0.35,
            clause_break_sigh_prob=0.15,
            post_turn_prob=0.28,
            post_turn_tags={"laugh": 0.55, "chuckle": 0.45},
            emphasis_prob=0.50,
            cooldown_turns=1,
        ),
        cadence_speed=1.20,
    )

    chloe = VocalHabitProfile(
        voice_name="softwaredev-30yomale-drama3",
        persona_name="CHLOE",
        default_tone="soft tone",
        tone_palette=[
            ToneAnchor(
                tag="soft tone",
                anchor="Dry, pragmatic, understated systems realism, calm and matter-of-fact.",
                lexical_keywords=["fashion", "practical", "simple", "nominal", "standard", "basic", "fine", "hats"],
                omnivoice_onset="[sigh]",
            ),
            ToneAnchor(
                tag="skeptical",
                anchor="Unimpressed, weary realist pointing out a mundane cause.",
                lexical_keywords=["probably", "just", "filter", "valve", "boring", "again", "usual", "yellow"],
                omnivoice_onset="[dissatisfaction-hnn]",
            ),
            ToneAnchor(
                tag="worried",
                anchor="Quiet concern that the group is wasting time on absurd tangents.",
                lexical_keywords=["concerning", "overload", "dropping", "problem", "risk", "tangent"],
                omnivoice_onset="[question-ah]",
            ),
        ],
        filler_rate=0.48,
        max_fillers_per_turn=2,
        filler_lexicon=[
            FillerItem("honestly", weight=1.6, slots=["pre_onset", "clause_break", "post_verb"]),
            FillerItem("like", weight=1.4, slots=["post_verb", "pre_adj"]),
            FillerItem("um", weight=1.2, slots=["pre_onset", "clause_break"]),
            FillerItem("I guess", weight=1.1, slots=["clause_break"]),
        ],
        throat_habits=ThroatAndPauseHabits(
            pre_onset_prob=0.36,
            pre_onset_tags={"sigh": 0.55, "clear_throat": 0.30, "dissatisfaction": 0.15},
            clause_break_pause_prob=0.46,
            clause_break_sigh_prob=0.25,
            post_turn_prob=0.25,
            post_turn_tags={"sigh": 0.75, "chuckle": 0.25},
            emphasis_prob=0.38,
            cooldown_turns=1,
        ),
        cadence_speed=1.10,
    )

    clopeux = VocalHabitProfile(
        voice_name="clopeux-voice-samples1",
        persona_name="CLOPEUX",
        default_tone="calm",
        tone_palette=[
            ToneAnchor(
                tag="calm",
                anchor="Deadpan, composed commander cutting through noise with dry authority and direct orders.",
                lexical_keywords=["isolate", "check", "enough", "focus", "report", "deck", "team", "order"],
                omnivoice_onset="[confirmation-en]",
            ),
            ToneAnchor(
                tag="icy tone",
                anchor="Sharp, deadpan observation shutting down a tangent or excuse.",
                lexical_keywords=["tangent", "stop", "listen", "nobody", "point", "quiet"],
                omnivoice_onset="[dissatisfaction-hnn]",
            ),
            ToneAnchor(
                tag="sarcastic",
                anchor="Dry, unexpected deadpan wit delivered with a straight face.",
                lexical_keywords=["suitcase", "brilliant", "fascinating", "hats", "wonderful"],
                omnivoice_onset="[sigh]",
            ),
        ],
        filler_rate=0.30,
        max_fillers_per_turn=1,
        filler_lexicon=[
            FillerItem("look", weight=1.5, slots=["pre_onset"]),
            FillerItem("frankly", weight=1.2, slots=["clause_break", "pre_onset"]),
            FillerItem("uh", weight=0.9, slots=["clause_break"]),
        ],
        throat_habits=ThroatAndPauseHabits(
            pre_onset_prob=0.32,
            pre_onset_tags={"clear_throat": 0.55, "confirmation": 0.30, "sigh": 0.15},
            clause_break_pause_prob=0.50,
            clause_break_sigh_prob=0.18,
            post_turn_prob=0.18,
            post_turn_tags={"sigh": 0.60, "chuckle": 0.40},
            emphasis_prob=0.45,
            cooldown_turns=1,
        ),
        cadence_speed=1.08,
    )

    profiles = {
        "MARGOT": margot,
        "high-woman-voice-excited": margot,
        "DESMOND": desmond,
        "sarcastic-male-voice": desmond,
        "KAI": kai,
        "normal-male-slightly-faster-cadence": kai,
        "CHLOE": chloe,
        "softwaredev-30yomale-drama3": chloe,
        "CLOPEUX": clopeux,
        "Clopeux": clopeux,
        "clopeux-voice-samples1": clopeux,
        "clopeux-voice-samples2": clopeux,
        "clopeux-voice-samples3": clopeux,
    }
    return profiles


def resolve_profile(speaker_name: str, voice_name: Optional[str] = None) -> VocalHabitProfile:
    """Return the VocalHabitProfile for a persona or reference voice name."""
    catalog = builtin_profiles()
    if speaker_name in catalog:
        return catalog[speaker_name]
    if speaker_name.upper() in catalog:
        return catalog[speaker_name.upper()]
    if voice_name and voice_name in catalog:
        return catalog[voice_name]
    return catalog["MARGOT"]
