"""HF Inference Client dialogue brain for multi-bot conversations."""

from __future__ import annotations

import os
import re
from typing import Generator, List, Dict, Optional
from huggingface_hub import InferenceClient, get_token


class HfDialogueBrain:
    """Manages conversational dialogue generation using remote HF Inference Client."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b:cheapest",
        api_key: Optional[str] = None,
        temperature: float = 0.88,
    ):
        token = api_key or os.environ.get("HF_TOKEN") or get_token()
        if not token:
            raise ValueError(
                "No Hugging Face token found. Set HF_TOKEN env var or run 'huggingface-cli login'."
            )
        self.model = model
        self.temperature = temperature
        self.client = InferenceClient(api_key=token)

    def _chat_completion_with_fallback(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ):
        """Invoke `self.client.chat.completions.create` with fallback models on transient 429 overload."""
        import time

        candidates = [self.model]
        for fallback in ("openai/gpt-oss-20b:cheapest", "Qwen/Qwen2.5-72B-Instruct"):
            if fallback not in candidates:
                candidates.append(fallback)

        last_err: Optional[Exception] = None
        for idx, candidate_model in enumerate(candidates):
            try:
                return self.client.chat.completions.create(
                    model=candidate_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                )
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                if ("429" in msg or "overloaded" in msg or "busy" in msg or "503" in msg) and idx + 1 < len(candidates):
                    time.sleep(0.5)
                    continue
                raise
        raise last_err  # type: ignore[misc]

    def generate_turn(
        self,
        speaker_name: str,
        system_persona: str,
        history: List[Dict[str, str]],
        target_speaker: Optional[str] = None,
        target_utterance: Optional[str] = None,
        prev_speaker: Optional[str] = None,
        prev_utterance: Optional[str] = None,
        bounce_mode: Optional[str] = None,
        narrative_hint: Optional[str] = None,
        max_tokens: int = 250,
        stream: bool = False,
        allow_prosody_tags: bool = False,
    ) -> str:
        """Generate a single conversational turn in character with optional bounce off previous speaker."""
        topic_block = f"\nConversation premise / starting topic: {narrative_hint}\n" if narrative_hint else ""
        if allow_prosody_tags:
            tag_rule = (
                "4. Modulate the TTS voice by weaving 1 to 3 inline Fish Audio S2 Pro vocal acting, emotion, and "
                "staging tags in square brackets directly into the line where the delivery shifts (e.g. "
                "[sighing], [laughing], [chuckling], [gasping], [clear throat], [break], [in a hurry tone], "
                "[soft tone], [emphasis], [excited], [nervous], [sarcastic], [skeptical], [frustrated], "
                "[confident], [calm], [surprised], [worried]).\n"
                "5. Output ONLY the spoken line with its square-bracket tags. NO asterisks, NO parentheses, "
                "NO quotes, NO emojis, NO speaker name prefixes."
            )
        else:
            tag_rule = (
                "4. Output ONLY the spoken words. NO stage directions, NO asterisks, NO quotes, "
                "NO emojis, NO bracketed labels, NO speaker name prefixes."
            )

        system_instruction = (
            f"You are {speaker_name}. {system_persona}{topic_block}\n"
            "Improvised multi-speaker dialogue rules:\n"
            "1. Improvise the story and conversation organically in character! Take the premise in unexpected, "
            "lively directions—riff, disagree, joke, escalate, speculate, or call out other people in the room.\n"
            "2. Speak 1 to 2 natural, idiomatic sentences (8-26 words total) with varied sentence openings so no two "
            "turns start the same way. Do not force generic filler words into every line.\n"
            "3. Never repeat or paraphrase what the previous speaker just said—always add a new angle or reaction.\n"
            f"{tag_rule}"
        )

        # Format history from speaker_name's perspective so only speaker_name's own prior lines are "assistant"
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]
        prefix_mine = f"{speaker_name}:"
        for item in history:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if content.upper().startswith(prefix_mine.upper()):
                own_line = content.split(":", 1)[1].strip()
                messages.append({"role": "assistant", "content": own_line})
            else:
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += f"\n{content}"
                else:
                    messages.append({"role": "user", "content": content})

        if not history:
            directive = (
                f"Kick off the conversation in character as {speaker_name}"
                + (f", addressing {target_speaker}" if target_speaker else "")
                + f", inspired by this opening topic: \"{narrative_hint or 'Start an impromptu conversation.'}\" "
                "Put it in your own natural voice with expressive tags and cadence."
            )
        elif prev_speaker and prev_utterance and prev_speaker != target_speaker:
            if bounce_mode == "uno_reverse":
                directive = f"Jump in and challenge or riff on what {prev_speaker} just said."
            elif bounce_mode == "continuation":
                directive = f"Riff on {prev_speaker}'s point and push the story somewhere new."
            elif bounce_mode == "contrast":
                directive = f"Push back on {prev_speaker}'s take with your own angle."
            else:
                directive = f"React to {prev_speaker} while tying back to {target_speaker}'s point, and improvise the next beat."
        elif target_speaker:
            directive = f"React directly to {target_speaker} in character as {speaker_name} and improvise the next beat."
        else:
            directive = f"Improvise your next spoken turn in character as {speaker_name}."

        if messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n[Director cue for {speaker_name}: {directive}]"
        else:
            messages.append({"role": "user", "content": f"[Director cue for {speaker_name}: {directive}]"})

        effective_tokens = max_tokens
        if effective_tokens < 800 and ("gpt-oss" in self.model.lower() or "deepinfra" in self.model.lower()):
            effective_tokens = 900

        if stream:
            stream_resp = self._chat_completion_with_fallback(
                messages=messages,
                max_tokens=effective_tokens,
                temperature=self.temperature,
                stream=True,
            )
            collected = []
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content or ""
                collected.append(delta)
            raw = "".join(collected)
        else:
            resp = self._chat_completion_with_fallback(
                messages=messages,
                max_tokens=effective_tokens,
                temperature=self.temperature,
                stream=False,
            )
            choice = resp.choices[0]
            raw = choice.message.content or ""
            if not raw.strip():
                reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None) or ""
                if reasoning:
                    forbidden = {
                        (target_utterance or "").strip().lower(),
                        (prev_utterance or "").strip().lower(),
                    }
                    quoted = [
                        q.strip()
                        for q in re.findall(r'["\u201c]([^"\u201d]{10,160})["\u201d]', reasoning)
                        if q.strip().lower() not in forbidden
                    ]
                    if quoted:
                        raw = quoted[-1]
                    else:
                        lines = [
                            line.strip()
                            for line in reasoning.strip().split("\n")
                            if line.strip() and not line.startswith("User") and not line.startswith("Rule")
                        ]
                        if lines:
                            raw = lines[-1]

        cleaned = self._clean_speech_text(raw, allow_prosody_tags=allow_prosody_tags)
        if not cleaned:
            raise RuntimeError(f"LLM returned empty speech text for {speaker_name} (raw={raw!r})")
        return cleaned

    def generate_turns_batch(
        self,
        personas: Dict[str, str],
        turn_plan: List[Dict[str, Optional[str]]],
        history: List[Dict[str, str]],
        narrative_hint: Optional[str] = None,
        max_tokens: int = 1200,
    ) -> List[str]:
        """Generate 5–10 dialogue turns in a single LLM request according to `turn_plan`.

        Args:
            personas: Mapping of `{speaker_name: persona_description}`.
            turn_plan: List of dicts `[{"speaker": "MARGOT", "target": "DESMOND", "bounce_mode": "contrast"}, ...]`.
            history: Recent conversation history (`[{"role": ..., "content": "NAME: text"}, ...]`).
            narrative_hint: Premise / starting topic.
            max_tokens: Token budget for the full batch.

        Returns:
            List of clean spoken strings matching `len(turn_plan)`. Automatically falls back to
            `generate_turn` for any missing lines if the LLM omits a turn.
        """
        if not turn_plan:
            return []
        if len(turn_plan) == 1:
            step = turn_plan[0]
            spk = step["speaker"] or "SPEAKER"
            return [
                self.generate_turn(
                    speaker_name=spk,
                    system_persona=personas.get(spk, ""),
                    history=history,
                    target_speaker=step.get("target"),
                    bounce_mode=step.get("bounce_mode"),
                    narrative_hint=narrative_hint,
                    allow_prosody_tags=False,
                )
            ]

        cast_lines = "\n".join(f"- {name}: {desc}" for name, desc in personas.items())
        plan_lines = []
        for idx, step in enumerate(turn_plan, start=1):
            spk = step.get("speaker") or "SPEAKER"
            tgt = step.get("target")
            bmode = step.get("bounce_mode")
            cue = f"addresses {tgt}" if tgt else "addresses the room"
            if bmode:
                cue += f" ({bmode})"
            plan_lines.append(f"{idx}. {spk} ({cue})")

        recent_block = "\n".join(
            (item.get("content") or "").strip() for item in history if (item.get("content") or "").strip()
        )
        history_section = f"\nRecent dialogue so far:\n{recent_block}\n" if recent_block else "\n(Opening of the conversation.)\n"

        system_prompt = (
            "You are a multi-speaker dialogue writer generating live, realistic room conversation.\n"
            f"Premise / topic: {narrative_hint or 'Impromptu group discussion.'}\n\n"
            f"Cast of characters:\n{cast_lines}\n\n"
            "Rules:\n"
            "1. Write sharp, distinct, in-character spoken lines (1 to 2 natural sentences, 8-25 words per turn).\n"
            "2. Do NOT put stage directions, brackets, asterisks, or generic filler soup in the text; a downstream "
            "prosody engine injects each character's vocal habits.\n"
            "3. Never repeat what the previous speaker just said—escalate, push back, joke, or introduce a fresh angle.\n"
            "4. Format output strictly as numbered lines matching the requested turn plan:\n"
            "1. SPEAKER: Spoken line here.\n"
            "2. SPEAKER: Spoken line here."
        )

        user_prompt = (
            f"{history_section}\n"
            f"Generate the next {len(turn_plan)} turns in exact order:\n"
            + "\n".join(plan_lines)
            + "\n\nReply with ONLY the numbered lines (`1. NAME: ...`)."
        )

        resp = self._chat_completion_with_fallback(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max(max_tokens, 220 * len(turn_plan)),
            temperature=self.temperature,
            stream=False,
        )
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        if not raw:
            raw = (getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None) or "").strip()

        # Parse numbered or speaker-prefixed lines
        parsed_by_idx: Dict[int, str] = {}
        unnumbered_candidates: List[str] = []
        raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw_clean = re.sub(r"<thought>.*?</thought>", "", raw_clean, flags=re.DOTALL)

        for line in raw_clean.splitlines():
            s = line.strip()
            if not s:
                continue
            m_num = re.match(r"^(\d+)[\.\)\-:]\s*(?:[A-Za-z0-9_\-]+\s*:\s*)?(.+)$", s)
            if m_num:
                idx_num = int(m_num.group(1)) - 1
                text_part = self._clean_speech_text(m_num.group(2), allow_prosody_tags=False)
                if 0 <= idx_num < len(turn_plan) and text_part:
                    parsed_by_idx[idx_num] = text_part
                    continue
            m_spk = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.+)$", s)
            if m_spk and m_spk.group(1).upper() in {k.upper() for k in personas}:
                text_part = self._clean_speech_text(m_spk.group(2), allow_prosody_tags=False)
                if text_part:
                    unnumbered_candidates.append(text_part)

        results: List[str] = []
        running_history = list(history)
        cand_cursor = 0
        for i, step in enumerate(turn_plan):
            spk = step.get("speaker") or "SPEAKER"
            line_text = parsed_by_idx.get(i)
            if not line_text and cand_cursor < len(unnumbered_candidates):
                line_text = unnumbered_candidates[cand_cursor]
                cand_cursor += 1
            if not line_text:
                line_text = self.generate_turn(
                    speaker_name=spk,
                    system_persona=personas.get(spk, ""),
                    history=running_history[-8:],
                    target_speaker=step.get("target"),
                    bounce_mode=step.get("bounce_mode"),
                    narrative_hint=narrative_hint,
                    allow_prosody_tags=False,
                )
            results.append(line_text)
            running_history.append({"role": "assistant", "content": f"{spk}: {line_text}"})

        return results

    def generate_backchannel(
        self,
        speaker_name: str,
        last_utterance: str,
    ) -> str:
        """Generate a very brief backchannel interjection (e.g. 'Yeah', 'No way', 'Wait really?')."""
        prompt = (
            f"The other speaker just said: '{last_utterance}'.\n"
            "Provide a natural, 1 to 3 word spoken backchannel reaction (e.g. 'Right', 'Exactly', 'Wait, what?', 'Mm-hmm', 'Hold on'). "
            "Reply with ONLY the spoken words, no quotes, no punctuation except commas/question marks."
        )
        resp = self._chat_completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250 if "gpt-oss" in self.model else 15,
            temperature=0.7,
            stream=False,
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        if not raw.strip():
            reasoning = getattr(choice.message, "reasoning_content", None) or ""
            if reasoning:
                quoted = re.findall(r'["\']([^"\']{2,30})["\']', reasoning)
                if quoted:
                    raw = quoted[-1]
        cleaned = self._clean_speech_text(raw)
        return cleaned or "Right."

    @staticmethod
    def _clean_speech_text(text: str, allow_prosody_tags: bool = False) -> str:
        """Strip reasoning tags, markdown, quotes, emojis, and normalize unicode punctuation."""
        import re
        # Normalize unicode punctuation
        cleaned = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", ", ")
        cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
        cleaned = cleaned.replace("…", "...")
        # Normalize subscript digits (CO\u2082 -> CO2) and superscript digits (x\u00b2 -> x2)
        # so Windows cp1252 consoles and the TTS backend receive plain ASCII numerals.
        for ch_from, ch_to in zip("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789"):
            cleaned = cleaned.replace(ch_from, ch_to)
        for ch_from, ch_to in zip("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "0123456789"):
            cleaned = cleaned.replace(ch_from, ch_to)
        # Remove think / thought tags
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL)
        # Remove parenthetical / asterisk stage directions e.g. (laughs), *sighs*
        cleaned = re.sub(r"\(.*?\)", "", cleaned)
        cleaned = re.sub(r"\*.*?\*", "", cleaned)
        if allow_prosody_tags:
            # Keep short lowercase prosody/emotion tags like [sighing], [in a hurry tone], [sarcastic], [break],
            # but strip long bracketed meta-prompts like [Context: ...] or [MARGOT, ...].
            def _filter_bracket(m: "re.Match[str]") -> str:
                inner = m.group(1).strip()
                if 2 <= len(inner) <= 32 and ":" not in inner and re.fullmatch(r"[a-zA-Z\s\-]+", inner):
                    return f"[{inner.lower()}]"
                return ""
            cleaned = re.sub(r"\[(.*?)\]", _filter_bracket, cleaned)
        else:
            cleaned = re.sub(r"\[.*?\]", "", cleaned)
        # Remove markdown bold/italics
        cleaned = re.sub(r"[_*~`]", "", cleaned)
        # Remove wrapping quotes
        cleaned = cleaned.strip().strip('"\'')
        # Remove leading speaker labels e.g. "DESMOND:", "[sarcastic] DESMOND:"
        cleaned = re.sub(r"^(\[[a-z\s\-]+\]\s*)?[A-Za-z0-9_\-\s]{1,20}:\s*", lambda m: m.group(1) or "", cleaned)
        cleaned = cleaned.strip().strip('"\'')
        # Keep up to 2 sentences so multi-clause natural rhythm ("Wait a second. Look at the telemetry!") survives
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]
        if sentences:
            cleaned = " ".join(sentences[:2])
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
