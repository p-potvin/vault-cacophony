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
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from cacophony.habit_profiles import (
    FillerItem,
    ToneAnchor,
    VocalHabitProfile,
    resolve_profile,
)
from cacophony.raggae_bridge import RaggaeBridge


# Abstract throat/reaction keys -> engine-native tag strings
FISH_AUDIO_TAG_MAP: Dict[str, str] = {
    "clear_throat": "[clear throat]",
    "sigh": "[sighs]",
    "gasp": "[gasping]",
    "laugh": "[laughs]",
    "chuckle": "[chuckling]",
    "confirmation": "[soft tone]",
    "dissatisfaction": "[sighs]",
    "surprise": "[gasping]",
    "pause": "[break]",
}

OMNIVOICE_TAG_MAP: Dict[str, str] = {
    "clear_throat": "[confirmation-en]",
    "sigh": "[sigh]",
    "gasp": "[surprise-ah]",
    "laugh": "[laughter]",
    "chuckle": "[laughter]",
    "confirmation": "[confirmation-en]",
    "dissatisfaction": "[dissatisfaction-hnn]",
    "surprise": "[surprise-oh]",
    "question_ah": "[question-ah]",
    "question_oh": "[question-oh]",
    "question_en": "[question-en]",
    "pause": "...",
}

OMNIVOICE_ALLOWED_TAGS = {
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
}

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


class SmartVocalPicker:
    """Syntactic and respiratory-boundary habit injector with per-speaker cooldown tracking."""

    def __init__(
        self,
        bridge: Optional[RaggaeBridge] = None,
        seed: Optional[int] = None,
    ):
        self.bridge = bridge or RaggaeBridge()
        self.rng = np.random.default_rng(seed)
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

    def _apply_word_emphasis(self, text: str, engine: str) -> Tuple[str, Optional[str]]:
        """Wrap one high-salience content word in *emphasis* for fish_audio."""
        if engine not in ("fish_audio", "fish", "fish-s2"):
            return text, None
        # Find words outside square brackets
        tokens = text.split(" ")
        candidate_indices: List[int] = []
        in_bracket = False
        for idx, tok in enumerate(tokens):
            if "[" in tok:
                in_bracket = True
            if not in_bracket:
                bare = re.sub(r"[^a-zA-Z]", "", tok)
                if len(bare) >= 5 and bare.lower() not in _STOPWORDS:
                    candidate_indices.append(idx)
            if "]" in tok:
                in_bracket = False
        if not candidate_indices:
            return text, None
        chosen_idx = int(self.rng.choice(candidate_indices))
        raw_tok = tokens[chosen_idx]
        m = re.match(r"^([^a-zA-Z]*)([a-zA-Z]+)([^a-zA-Z]*)$", raw_tok)
        if not m:
            return text, None
        prefix, word, suffix = m.groups()
        tokens[chosen_idx] = f"{prefix}*{word}*{suffix}"
        return " ".join(tokens), word

    @staticmethod
    def sanitize_for_engine(text: str, engine: str) -> str:
        """Ensure only tags supported by `engine` remain in the final TTS string."""
        eng = engine.lower()
        if eng == "omnivoice":
            # Strip *emphasis* asterisks while keeping the word
            cleaned = re.sub(r"\*([^*]+)\*", r"\1", text)
            # Keep only official OmniVoice non-verbal bracket tags; convert [break] to '...' and strip others
            def _omni_bracket(m: "re.Match[str]") -> str:
                full = m.group(0).strip().lower()
                if full in OMNIVOICE_ALLOWED_TAGS:
                    return full
                if full in ("[break]", "[pause]"):
                    return "..."
                if full in ("[sighs]", "[sighing]"):
                    return "[sigh]"
                if full in ("[laughs]", "[laughing]", "[chuckling]", "[chuckle]"):
                    return "[laughter]"
                if full in ("[gasping]", "[gasp]"):
                    return "[surprise-ah]"
                if full in ("[clear throat]",):
                    return "[confirmation-en]"
                return ""

            cleaned = re.sub(r"\[[^\]]+\]", _omni_bracket, cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned
        return re.sub(r"\s+", " ", text).strip()

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
        """Annotate a single clean utterance with persona-specific tone, fillers, pauses, and throat noises."""
        prof = profile or resolve_profile(speaker_name, voice_name)
        eng = engine_type.lower()

        if pre_scored_tone is not None:
            tone_anchor, tone_score, tone_method = pre_scored_tone
        else:
            scored = self.bridge.score_tones_batch([clean_text], [prof])
            tone_anchor, tone_score, tone_method = scored[0]

        working = clean_text.strip()
        injected_fillers: List[str] = []
        injected_throat: List[str] = []
        used_fillers: set[str] = set()

        is_barge_in = transition_type == "interruption"
        throat_cfg = prof.throat_habits
        can_throat = self._can_emit_throat(prof.persona_name, turn_num, throat_cfg.cooldown_turns)

        # 1. Grammar-aware filler words (up to max_fillers_per_turn)
        if prof.filler_lexicon and self.rng.random() < prof.filler_rate:
            # Prefer post_verb slot first (e.g. "is like", "seems uh")
            pv_filler = self._sample_filler_for_slot(prof.filler_lexicon, "post_verb", used_fillers, working.lower())
            if pv_filler:
                working, ok = self._inject_post_verb_filler(working, pv_filler)
                if ok:
                    injected_fillers.append(pv_filler)
                    used_fillers.add(pv_filler.lower())

            # Optional second filler or fallback in pre_onset / clause_break
            if len(injected_fillers) < prof.max_fillers_per_turn and (
                not injected_fillers or self.rng.random() < (prof.filler_rate * 0.55)
            ):
                if self.rng.random() < 0.50:
                    onset_filler = self._sample_filler_for_slot(prof.filler_lexicon, "pre_onset", used_fillers, working.lower())
                    if onset_filler and len(working.split()) >= 5:
                        # Lowercase first letter if it's not "I" or a proper name
                        first_char = working[0]
                        rest = working[1:]
                        if first_char.isupper() and not working.startswith("I ") and not working.startswith("I'"):
                            for p_name in ("Margot", "Desmond", "Kai", "Chloe", "Clopeux", "Yellow", "Hat", "Hats"):
                                if working.startswith(p_name):
                                    break
                            else:
                                first_char = first_char.lower()
                        working = f"{onset_filler.capitalize() if onset_filler != 'I mean' else 'I mean'}, {first_char}{rest}"
                        injected_fillers.append(onset_filler)
                        used_fillers.add(onset_filler.lower())

        # 2. Clause-break pause / mid-phrase sigh / clause filler
        clause_pause_tok: Optional[str] = None
        clause_filler_tok: Optional[str] = None
        if self.rng.random() < throat_cfg.clause_break_pause_prob:
            if can_throat and self.rng.random() < throat_cfg.clause_break_sigh_prob:
                clause_pause_tok = OMNIVOICE_TAG_MAP["sigh"] if eng == "omnivoice" else FISH_AUDIO_TAG_MAP["sigh"]
                injected_throat.append("clause_sigh")
                self._record_throat(prof.persona_name, turn_num)
                can_throat = False
            else:
                clause_pause_tok = "..." if eng == "omnivoice" else "[break]"

        if len(injected_fillers) < prof.max_fillers_per_turn and self.rng.random() < (prof.filler_rate * 0.45):
            clause_filler_tok = self._sample_filler_for_slot(prof.filler_lexicon, "clause_break", used_fillers, working.lower())

        if clause_pause_tok or clause_filler_tok:
            working, cb_ok = self._inject_clause_break(working, clause_pause_tok, clause_filler_tok)
            if cb_ok and clause_filler_tok:
                injected_fillers.append(clause_filler_tok)
                used_fillers.add(clause_filler_tok.lower())

        # 3. Pre-onset throat noise (suppressed on fast barge-in interruptions so barge-ins stay snappy)
        onset_tag_str = ""
        if not is_barge_in and can_throat and self.rng.random() < throat_cfg.pre_onset_prob:
            abstract_onset = self._sample_weighted_dict(throat_cfg.pre_onset_tags)
            if abstract_onset:
                onset_tag_str = (
                    OMNIVOICE_TAG_MAP.get(abstract_onset, "[confirmation-en]")
                    if eng == "omnivoice"
                    else FISH_AUDIO_TAG_MAP.get(abstract_onset, "[clear throat]")
                )
                injected_throat.append(f"pre_onset:{abstract_onset}")
                self._record_throat(prof.persona_name, turn_num)
                can_throat = False
        elif eng == "omnivoice" and tone_anchor.omnivoice_onset and can_throat and self.rng.random() < 0.35:
            # In OmniVoice, surface the tone's expressive onset token (e.g. [sigh], [surprise-ah])
            onset_tag_str = tone_anchor.omnivoice_onset
            injected_throat.append(f"tone_onset:{onset_tag_str}")
            self._record_throat(prof.persona_name, turn_num)
            can_throat = False

        # 4. Post-turn tail noise (chuckle, laugh, trailing sigh)
        post_tag_str = ""
        if can_throat and self.rng.random() < throat_cfg.post_turn_prob:
            abstract_post = self._sample_weighted_dict(throat_cfg.post_turn_tags)
            if abstract_post:
                post_tag_str = (
                    OMNIVOICE_TAG_MAP.get(abstract_post, "[sigh]")
                    if eng == "omnivoice"
                    else FISH_AUDIO_TAG_MAP.get(abstract_post, "[sighs]")
                )
                injected_throat.append(f"post_turn:{abstract_post}")
                self._record_throat(prof.persona_name, turn_num)

        # 5. Salient word *emphasis* (Fish Audio S2 Pro supports *emphasis*)
        emphasized_word = None
        if eng in ("fish_audio", "fish", "fish-s2") and self.rng.random() < throat_cfg.emphasis_prob:
            working, emphasized_word = self._apply_word_emphasis(working, eng)

        # 6. Assemble final engine-specific utterance
        prefix_parts = []
        if eng in ("fish_audio", "fish", "fish-s2"):
            prefix_parts.append(f"[{tone_anchor.tag}]")
        if onset_tag_str:
            prefix_parts.append(onset_tag_str)

        assembled = " ".join(prefix_parts + [working])
        if post_tag_str:
            assembled = f"{assembled} {post_tag_str}"

        rendered = self.sanitize_for_engine(assembled, eng)
        return AnnotatedTurn(
            clean_text=clean_text,
            rendered_text=rendered,
            engine_type=eng,
            selected_tone=tone_anchor.tag,
            tone_score=tone_score,
            tone_method=tone_method,
            injected_fillers=injected_fillers,
            injected_throat_tags=injected_throat,
            emphasized_word=emphasized_word,
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
