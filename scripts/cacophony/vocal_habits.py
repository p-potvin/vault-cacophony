"""The "Smart Picker": grammar- and anatomy-aware prosody, pause, throat noise, and filler injector.

Why this exists:
- A generic LLM prompt told to "use filler words and tags" puts `[sighs]` or `"like"` in random,
  un-anatomical places or makes every character sound identical.
- Real speakers have distinct vocal habits (`VocalHabitProfile`):
  1. **Opening Tone**: Determined per-turn by scoring the clean line against the persona's
     `tone_palette` via `vault-raggae` EmbeddingGemma-2 (`[icy tone]`, `[excited]`, `[sarcastic]`, etc.).
  2. **Grammar-Aware Filler Words**:
     - `post_verb`: Fillers like `"like"`, `"uh"`, `"literally"` land after linking/auxiliary/mental
       verbs (`is`, `are`, `was`, `were`, `seems`, `looks`, `think`, `feel`, `going to`, `need to`).
     - `pre_onset`: Discourse markers like `"I mean,"`, `"Look,"`, `"Honestly,"`, `"Frankly,"` land at
       the start of a turn before the main clause.
     - `clause_break`: Fillers like `"you know"`, `"I mean"`, `"uh"` land after a comma or conjunction
       (`but`, `because`, `so`, `although`, `while`).
  3. **Anatomical Throat/Breath Noises & Pauses**:
     - You clear your throat or sigh **before** speaking (`pre_onset`) when taking the floor (`TURN_HOLD`
       or `TURN_SWITCH`), NOT while barging in (`INTERRUPTION`).
     - Mid-utterance pauses (`[break]` / `...`) or respiratory sighs happen at `clause_break` boundaries.
     - Chuckles, laughs, or trailing sighs happen at `post_turn` after finishing a point.
  4. **Engine-Specific Tag Translation**:
     - `fish_audio`: `[<leading_tone>]` at start + inline `[sighs]`, `[laughs]`, `[chuckling]`,
       `[clear throat]`, `[gasping]`, `[break]`, and `*emphasis*` around a high-salience word.
     - `omnivoice`: Native OmniVoice non-verbal tokens (`[laughter]`, `[sigh]`, `[confirmation-en]`,
       `[question-en]`, `[question-ah]`, `[question-oh]`, `[surprise-ah]`, `[surprise-oh]`,
       `[dissatisfaction-hnn]`) + natural punctuation pauses (`...`, `—`), stripping any unsupported
       bracket tags so OmniVoice never speaks bracket text literally.

Delivery is built engine-neutrally (`speech_markup.Utterance`) and only then
rendered into the engine's dialect, so every engine gets the same pauses,
breaths and hesitations in the only form it understands -- Fish bracket cues,
OmniVoice tokens, plain punctuation for Kokoro/Magpie/CosyVoice/Chatterbox --
and every turn also has a canonical SSML 1.1 rendering for the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

import speech_markup as sm

from cacophony.habit_profiles import (  # noqa: E402
    FillerItem,
    ToneAnchor,
    VocalHabitProfile,
    resolve_profile,
)
from cacophony.raggae_bridge import RaggaeBridge


# The OmniVoice non-verbal vocabulary, kept here for callers that check against it.
OMNIVOICE_ALLOWED_TAGS = set(sm.OMNIVOICE_ALLOWED)

# Post-verb targets where fillers like "like" or "uh" naturally attach in English speech
_POST_VERB_PATTERN = re.compile(
    r"\b(is|are|was|were|be|been|seems|looks|sounds|feels|think|thought|mean|going to|need to|have to|want to)\b"
    r"(\s+)(?!like\b|uh\b|um\b|not\b|n't\b|to\b)([a-zA-Z]{3,})",
    re.IGNORECASE,
)

# Clause boundary where a pause, breath/sigh, or clause-break filler naturally lands
_CLAUSE_BREAK_PATTERN = re.compile(
    r"(,\s+|\s+(?:but|because|so|although|while|whereas|and yet)\s+)",
    re.IGNORECASE,
)

# Stopwords excluded from *emphasis* selection
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "your", "have", "just", "what",
    "about", "they", "their", "there", "were", "been", "would", "could", "should",
    "like", "mean", "know", "honestly", "frankly", "really", "very", "some", "into",
    "when", "where", "which", "while", "because", "than", "then", "them", "these",
    "those", "only", "also", "even", "still", "more", "most", "other", "such",
}


@dataclass
class AnnotatedTurn:
    """Result of passing a clean dialogue turn through the Smart Picker."""
    clean_text: str
    rendered_text: str
    engine_type: str
    selected_tone: str
    tone_score: float
    tone_method: str
    injected_fillers: List[str] = field(default_factory=list)
    injected_throat_tags: List[str] = field(default_factory=list)
    emphasized_word: Optional[str] = None
    ssml: str = ""                   # canonical SSML 1.1 rendering of the same delivery
    speed_scale: float = 1.0         # the turn's SSML rate (tone/emotion), to multiply into engine speed
    markup: Dict[str, object] = field(default_factory=dict)  # engine-neutral token list


class SmartVocalPicker:
    """Syntactic and respiratory-boundary habit injector with per-speaker cooldown tracking."""

    def __init__(
        self,
        bridge: Optional[RaggaeBridge] = None,
        seed: Optional[int] = None,
    ):
        self.bridge = bridge or RaggaeBridge()
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        # Tracks the last turn index on which each speaker emitted a throat noise
        self._last_throat_turn: Dict[str, int] = {}

    def _can_emit_throat(self, speaker: str, turn_num: int, cooldown: int) -> bool:
        last = self._last_throat_turn.get(speaker, -999)
        return (turn_num - last) > max(0, cooldown)

    def _record_throat(self, speaker: str, turn_num: int) -> None:
        self._last_throat_turn[speaker] = turn_num

    def _sample_weighted_dict(self, weights: Dict[str, float]) -> Optional[str]:
        if not weights:
            return None
        keys = list(weights.keys())
        probs = np.asarray([max(0.0, float(weights[k])) for k in keys], dtype=np.float64)
        total = probs.sum()
        if total <= 0:
            return keys[0]
        probs /= total
        idx = int(self.rng.choice(len(keys), p=probs))
        return keys[idx]

    def _sample_filler_for_slot(
        self,
        lexicon: List[FillerItem],
        slot: str,
        used_words: set[str],
        text_lower: str,
    ) -> Optional[str]:
        candidates = [
            item for item in lexicon
            if slot in item.slots
            and item.word.lower() not in used_words
            and not text_lower.startswith(item.word.lower() + ",")
            and not text_lower.startswith(item.word.lower() + " ")
        ]
        if not candidates:
            return None
        weights = np.asarray([max(0.01, float(c.weight)) for c in candidates], dtype=np.float64)
        weights /= weights.sum()
        idx = int(self.rng.choice(len(candidates), p=weights))
        return candidates[idx].word

    def _inject_post_verb_filler(self, text: str, filler: str) -> Tuple[str, bool]:
        """Insert a filler like 'like' or 'uh' immediately after the first eligible verb."""
        matches = list(_POST_VERB_PATTERN.finditer(text))
        if not matches:
            return text, False
        m = matches[0]
        verb = m.group(1)
        space = m.group(2)
        next_word = m.group(3)
        if filler.lower() == next_word.lower():
            return text, False
        replacement = f"{verb},{space}{filler}, {next_word}" if filler in ("I mean", "you know", "honestly") else f"{verb}{space}{filler} {next_word}"
        updated = text[: m.start()] + replacement + text[m.end() :]
        return updated, True

    def _inject_clause_break(
        self,
        text: str,
        pause_token: Optional[str],
        filler: Optional[str],
    ) -> Tuple[str, bool]:
        """Insert a pause/breath token and/or filler at the first natural clause break."""
        matches = list(_CLAUSE_BREAK_PATTERN.finditer(text))
        if not matches:
            return text, False
        m = matches[0]
        sep = m.group(1)
        insert_parts = []
        if pause_token:
            insert_parts.append(pause_token)
        if filler:
            insert_parts.append(f"{filler},")
        if not insert_parts:
            return text, False
        insertion = " ".join(insert_parts)
        if sep.startswith(","):
            updated = text[: m.start()] + f", {insertion} " + text[m.end() :].lstrip()
        else:
            updated = text[: m.start()] + f" {insertion}{sep}" + text[m.end() :]
        return re.sub(r"\s+", " ", updated).strip(), True

    @staticmethod
    def sanitize_for_engine(text: str, engine: str) -> str:
        """Ensure only markup supported by `engine` remains in the final TTS string."""
        return sm.sanitize_for_engine(text, engine)

    # Abstract habit keys (habit_profiles) -> engine-neutral vocalisations.
    _HABIT_VOCAL = {
        "clear_throat": "clear_throat", "sigh": "sigh", "gasp": "gasp", "laugh": "laugh",
        "chuckle": "chuckle", "confirmation": "confirmation", "dissatisfaction": "hmm",
        "surprise": "gasp", "question_ah": "question", "question_oh": "question", "question_en": "question",
    }

    def _emphasize(self, utt: sm.Utterance) -> Optional[str]:
        """Turn one salient content word into an <emphasis>, in place."""
        candidates = []
        for ti, tok in enumerate(utt.tokens):
            if isinstance(tok, sm.Text):
                for wi, w in enumerate(tok.text.split(" ")):
                    bare = re.sub(r"[^a-zA-Z]", "", w)
                    if len(bare) >= 5 and bare.lower() not in _STOPWORDS and re.fullmatch(r"[^a-zA-Z]*[a-zA-Z]+[^a-zA-Z]*", w):
                        candidates.append((ti, wi))
        if not candidates:
            return None
        ti, wi = candidates[int(self.rng.integers(len(candidates)))]
        words = utt.tokens[ti].text.split(" ")  # type: ignore[union-attr]
        m = re.match(r"^([^a-zA-Z]*)([a-zA-Z]+)([^a-zA-Z]*)$", words[wi])
        if not m:
            return None
        prefix, word, suffix = m.groups()
        before = " ".join(words[:wi] + ([prefix] if prefix else []))
        after = " ".join(([suffix] if suffix else []) + words[wi + 1:])
        replacement: List[sm.Token] = []
        if before.strip():
            replacement.append(sm.Text(before.strip()))
        replacement.append(sm.Emphasis(word))
        if after.strip():
            replacement.append(sm.Text(after.strip()))
        utt.tokens[ti:ti + 1] = replacement
        return word

    @staticmethod
    def _first_mid_break(utt: sm.Utterance) -> Optional[int]:
        seen_text = False
        for i, tok in enumerate(utt.tokens):
            if isinstance(tok, (sm.Text, sm.Emphasis)):
                seen_text = True
            elif seen_text and isinstance(tok, sm.Break):
                return i
        return None

    def annotate_turn(
        self,
        clean_text: str,
        speaker_name: str,
        voice_name: Optional[str] = None,
        engine_type: str = "omnivoice",
        transition_type: str = "turn_switch",
        turn_num: int = 1,
        pre_scored_tone: Optional[Tuple[ToneAnchor, float, str]] = None,
        profile: Optional[VocalHabitProfile] = None,
    ) -> AnnotatedTurn:
        """Annotate one utterance with persona tone, discourse markers, pauses, breaths, hesitations and vocal noises.

        `clean_text` may already carry inline cues (an LLM allowed to write
        `[sighs]` or `*really*`); they are kept where they were written and
        translated for the engine like everything else.
        """
        prof = profile or resolve_profile(speaker_name, voice_name)
        eng = engine_type.lower()
        palette = [t.tag for t in prof.tone_palette]

        # Inline cues the LLM wrote, separated from the words.
        inline = sm.parse_tagged(clean_text.strip(), palette)
        words_only = inline.words()

        if pre_scored_tone is not None:
            tone_anchor, tone_score, tone_method = pre_scored_tone
        else:
            scored = self.bridge.score_tones_batch([words_only or clean_text], [prof])
            tone_anchor, tone_score, tone_method = scored[0]
        tone_tag = tone_anchor.tag
        if inline.tone:
            # The writer picked a delivery on purpose; it outranks the scorer.
            tone_tag = inline.tone
            tone_anchor = next((t for t in prof.tone_palette if t.tag.lower() == inline.tone),
                               ToneAnchor(tag=inline.tone, anchor=""))
            tone_method = "inline"

        injected_fillers: List[str] = []
        injected_throat: List[str] = []
        used_fillers: set[str] = set()

        trans = transition_type.lower()
        is_barge_in = trans == "interruption"
        taking_floor = trans in ("turn_switch", "turn_hold", "interleaved_reply")
        throat_cfg = prof.throat_habits
        can_throat = self._can_emit_throat(prof.persona_name, turn_num, throat_cfg.cooldown_turns)

        # 1. The persona's discourse markers ("frankly", "like", "you know"),
        #    placed by grammar. Filled pauses ("uh", "um") are not placed here:
        #    they belong to the hesitation model below, which puts them where a
        #    speaker would actually be planning.
        lexicon = [f for f in prof.filler_lexicon if f.word.lower() not in sm.FILLED_PAUSES]
        segments = [t for t in inline.tokens]
        text_idx = [i for i, t in enumerate(segments) if isinstance(t, sm.Text)]

        def _edit_first_text(fn) -> bool:
            for i in text_idx:
                new_text, ok = fn(segments[i].text)  # type: ignore[union-attr]
                if ok:
                    segments[i] = sm.Text(new_text)
                    return True
            return False

        if lexicon and self.rng.random() < prof.filler_rate:
            pv_filler = self._sample_filler_for_slot(lexicon, "post_verb", used_fillers, words_only.lower())
            if pv_filler and _edit_first_text(lambda t: self._inject_post_verb_filler(t, pv_filler)):
                injected_fillers.append(pv_filler)
                used_fillers.add(pv_filler.lower())

            if len(injected_fillers) < prof.max_fillers_per_turn and (
                not injected_fillers or self.rng.random() < (prof.filler_rate * 0.55)
            ):
                if self.rng.random() < 0.50 and text_idx and text_idx[0] == 0:
                    onset_filler = self._sample_filler_for_slot(lexicon, "pre_onset", used_fillers, words_only.lower())
                    first = segments[0].text  # type: ignore[union-attr]
                    if onset_filler and len(words_only.split()) >= 5:
                        lead, rest = first[0], first[1:]
                        if lead.isupper() and not first.startswith(("I ", "I'")) and not any(
                            first.startswith(n) for n in ("Margot", "Desmond", "Kai", "Chloe", "Clopeux", "Yellow", "Hat")
                        ):
                            lead = lead.lower()
                        word = onset_filler if onset_filler == "I mean" else onset_filler.capitalize()
                        segments[0] = sm.Text(f"{word}, {lead}{rest}")
                        injected_fillers.append(onset_filler)
                        used_fillers.add(onset_filler.lower())

        if len(injected_fillers) < prof.max_fillers_per_turn and self.rng.random() < (prof.filler_rate * 0.45):
            cb_filler = self._sample_filler_for_slot(lexicon, "clause_break", used_fillers, words_only.lower())
            if cb_filler and _edit_first_text(lambda t: self._inject_clause_break(t, None, cb_filler)):
                injected_fillers.append(cb_filler)
                used_fillers.add(cb_filler.lower())

        # 2. Pauses, breaths and hesitations at planning points, from the
        #    speaker's pause and disfluency models, under the turn's prosody.
        emotion = tone_anchor.emotion() if hasattr(tone_anchor, "emotion") else sm.TONE_EMOTION.get(tone_tag, sm.Emotion())
        prosody = prof.baseline.combine(emotion.to_prosody(prof.emotion_intensity))
        utt = sm.humanize(
            sm.Utterance(tokens=segments, tone=tone_tag),
            self.py_rng,
            pauses=prof.pauses,
            disfluency=prof.disfluency,
            taking_floor=taking_floor,
            interrupting=is_barge_in,
            tone=tone_tag,
            prosody=prosody,
            pause_prob=throat_cfg.clause_break_pause_prob,
        )
        utt.emotion = emotion
        injected_fillers += [t.word for t in utt.tokens if isinstance(t, sm.Filler)]

        # 3. A mid-utterance sigh at a clause pause.
        if can_throat and self.rng.random() < throat_cfg.clause_break_pause_prob * throat_cfg.clause_break_sigh_prob:
            at = self._first_mid_break(utt)
            if at is not None:
                utt.tokens.insert(at + 1, sm.Vocal("sigh"))
                injected_throat.append("clause_sigh")
                self._record_throat(prof.persona_name, turn_num)
                can_throat = False

        # 4. Throat noise before speaking -- never on a barge-in, which has no
        #    time for one. It replaces an onset breath rather than stacking.
        onset_vocal: Optional[str] = None
        if not is_barge_in and can_throat and self.rng.random() < throat_cfg.pre_onset_prob:
            abstract = self._sample_weighted_dict(throat_cfg.pre_onset_tags)
            if abstract:
                onset_vocal = self._HABIT_VOCAL.get(abstract, "clear_throat")
                injected_throat.append(f"pre_onset:{abstract}")
        elif eng in ("omnivoice", "omni") and tone_anchor.omnivoice_onset and can_throat and self.rng.random() < 0.35:
            cue = sm.parse_tagged(tone_anchor.omnivoice_onset).tokens
            onset_vocal = cue[0].kind if cue and isinstance(cue[0], sm.Vocal) else None
            if onset_vocal:
                injected_throat.append(f"tone_onset:{tone_anchor.omnivoice_onset}")
        if onset_vocal:
            if utt.tokens and isinstance(utt.tokens[0], sm.Vocal) and utt.tokens[0].kind == "breath":
                utt.tokens[0] = sm.Vocal(onset_vocal)
            else:
                utt.tokens.insert(0, sm.Vocal(onset_vocal))
            self._record_throat(prof.persona_name, turn_num)
            can_throat = False

        # 5. A trailing chuckle, laugh or sigh after making the point.
        if can_throat and self.rng.random() < throat_cfg.post_turn_prob:
            abstract = self._sample_weighted_dict(throat_cfg.post_turn_tags)
            if abstract:
                utt.add(sm.Vocal(self._HABIT_VOCAL.get(abstract, "sigh")))
                injected_throat.append(f"post_turn:{abstract}")
                self._record_throat(prof.persona_name, turn_num)

        # 6. One stressed word. Carried for every engine: Fish and SSML voice
        #    it, the plain-text engines simply drop the mark.
        emphasized_word = None
        if not any(isinstance(t, sm.Emphasis) for t in utt.tokens) and self.rng.random() < throat_cfg.emphasis_prob:
            emphasized_word = self._emphasize(utt)
        elif any(isinstance(t, sm.Emphasis) for t in utt.tokens):
            emphasized_word = next(t.text for t in utt.tokens if isinstance(t, sm.Emphasis))

        return AnnotatedTurn(
            clean_text=clean_text,
            rendered_text=sm.render(utt, eng),
            engine_type=eng,
            selected_tone=tone_tag,
            tone_score=tone_score,
            tone_method=tone_method,
            injected_fillers=injected_fillers,
            injected_throat_tags=injected_throat,
            emphasized_word=emphasized_word,
            ssml=utt.to_ssml(),
            speed_scale=round(float(utt.prosody.rate), 4),
            markup=utt.to_dict(),
        )

    def annotate_batch(
        self,
        items: List[Dict[str, object]],
        engine_type: str = "omnivoice",
    ) -> List[AnnotatedTurn]:
        """Batch-score all turns via a single vault-raggae `/v1/embed` call, then run the Smart Picker on each.

        Each dict in `items` should contain:
            - `text`: clean spoken line
            - `speaker_name`: persona name
            - `voice_name`: optional reference voice stem
            - `engine_type`: optional per-speaker engine override
            - `transition_type`: optional transition type (`turn_hold`, `turn_switch`, `interruption`, `interleaved_reply`)
            - `turn_num`: integer turn number
        """
        if not items:
            return []

        texts = [str(it.get("text") or "") for it in items]
        profiles = [
            resolve_profile(str(it.get("speaker_name") or ""), str(it.get("voice_name") or "") or None)
            for it in items
        ]
        scored_tones = self.bridge.score_tones_batch(texts, profiles)

        results: List[AnnotatedTurn] = []
        for idx, (it, prof, scored) in enumerate(zip(items, profiles, scored_tones)):
            spk_engine = str(it.get("engine_type") or engine_type).lower()
            turn_num = int(it.get("turn_num") or (idx + 1))
            trans = str(it.get("transition_type") or "turn_switch")
            ann = self.annotate_turn(
                clean_text=texts[idx],
                speaker_name=str(it.get("speaker_name") or prof.persona_name),
                voice_name=str(it.get("voice_name") or prof.voice_name),
                engine_type=spk_engine,
                transition_type=trans,
                turn_num=turn_num,
                pre_scored_tone=scored,
                profile=prof,
            )
            results.append(ann)
        return results
