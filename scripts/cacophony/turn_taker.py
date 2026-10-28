"""HMM-based turn-taking transition and timing scheduler inspired by FastMSS."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple
import numpy as np


class TransitionType(Enum):
    TURN_HOLD = "TURN_HOLD"        # Same speaker continues after a brief pause
    TURN_SWITCH = "TURN_SWITCH"    # Other speaker takes the floor after conversational gap
    INTERRUPTION = "INTERRUPTION"  # Other speaker cuts in before current turn ends
    BACKCHANNEL = "BACKCHANNEL"    # Interjection/reaction overlapping during the turn
    INTERLEAVED_REPLY = "INTERLEAVED_REPLY"  # Answering an earlier speaker while another is interjecting


class TurnTakingScheduler:
    """Calculates realistic conversational offsets and overlaps for multi-speaker dialogue."""

    def __init__(
        self,
        mean_pause_ts: float = 0.35,   # Mean pause between turns (seconds) - fast pace
        mean_pause_th: float = 0.25,   # Mean pause between consecutive utterances by same speaker
        overlap_ratio_ir: float = 0.25, # Overlap fraction during interruption (10% - 40%)
        overlap_ratio_bc: float = 0.50, # When backchannel fires during an ongoing utterance
        interruption_prob: float = 0.20,
        backchannel_prob: float = 0.25,
    ):
        self.mean_pause_ts = mean_pause_ts
        self.mean_pause_th = mean_pause_th
        self.overlap_ratio_ir = overlap_ratio_ir
        self.overlap_ratio_bc = overlap_ratio_bc
        self.interruption_prob = interruption_prob
        self.backchannel_prob = backchannel_prob

    def sample_turn_switch_offset(self, prev_offset: float, prev_duration: float) -> Tuple[float, TransitionType]:
        """Determine when the counterpart starts speaking relative to previous turn."""
        roll = np.random.random()

        if roll < self.interruption_prob:
            # Interruption: start speaking BEFORE previous speaker finishes
            overlap_frac = np.clip(np.random.normal(self.overlap_ratio_ir, 0.08), 0.10, 0.45)
            overlap_dur = overlap_frac * prev_duration
            offset = max(prev_offset + 0.3, prev_offset + prev_duration - overlap_dur)
            return offset, TransitionType.INTERRUPTION
        else:
            # Fast turn switch: very short conversational gap
            pause = max(0.08, float(np.random.exponential(self.mean_pause_ts)))
            offset = prev_offset + prev_duration + pause
            return offset, TransitionType.TURN_SWITCH

    def sample_interruption_offset(
        self,
        prev_offset: float,
        prev_duration: float,
        min_overlap: float = 0.125,
        max_overlap: float = 0.25,
    ) -> float:
        """Sample an intentional conversational barge-in overlapping 0.125s - 0.25s of previous speaker."""
        overlap = float(np.random.uniform(min_overlap, max_overlap))
        overlap = min(overlap, prev_duration * 0.35)
        return max(prev_offset + 0.3, prev_offset + prev_duration - overlap)

    def sample_fast_reply_offset(self, target_offset: float, target_duration: float, min_gap: float = 0.08, max_gap: float = 0.22) -> float:
        """Sample an immediate, snappy reply offset right after a targeted speaker finishes."""
        gap = float(np.random.uniform(min_gap, max_gap))
        return target_offset + target_duration + gap

    def sample_backchannel_offset(self, utt_offset: float, utt_duration: float) -> float:
        """Calculate where a backchannel reaction (or ambient filler) lands inside an ongoing utterance."""
        rel_pos = float(np.clip(np.random.normal(self.overlap_ratio_bc, 0.12), 0.25, 0.80))
        return utt_offset + (rel_pos * utt_duration)

    @staticmethod
    def stagger_turn(
        proposed_start: float,
        duration: float,
        speaker_idx: int,
        scheduled_turns: list,
        max_concurrent: int = 2,
        max_overlap_sec: float = 0.25,
        min_speaker_gap: float = 0.08,
    ) -> float:
        """Ensure no more than max_concurrent speakers talk simultaneously,
        limit overlap against ANY other active speaker to <= max_overlap_sec (0.125s - 0.25s),
        and prevent self-overlap.

        Args:
            proposed_start: Target start offset in seconds.
            duration: Utterance length in seconds.
            speaker_idx: Identifier of speaker seeking floor.
            scheduled_turns: List of (start_time, end_time, speaker_idx) tuples.
            max_concurrent: Maximum active speakers at any time (default 2).
            max_overlap_sec: Hard ceiling on overlap between ANY two speakers (default 0.25s).
            min_speaker_gap: Gap between consecutive turns by the same speaker.

        Returns:
            Adjusted start time guaranteed to never exceed max_overlap_sec against any speaker.
        """
        # 1. Prevent speaker from talking over their own previous speech
        same_speaker_ends = [end for start, end, spk in scheduled_turns if spk == speaker_idx]
        earliest_t = max(proposed_start, max(same_speaker_ends, default=0.0) + min_speaker_gap)

        # 2. Extract active intervals of other speakers
        other_intervals = [(start, end) for start, end, spk in scheduled_turns if spk != speaker_idx]
        if not other_intervals:
            return round(earliest_t, 3)

        curr_t = earliest_t
        for _ in range(50):  # Safety iteration limit
            collision_found = False

            # Invariant A: Overlap with ANY individual other speaker must be <= max_overlap_sec (0.25s)
            for s, e in other_intervals:
                if curr_t < e and curr_t + duration > s:
                    overlap = min(curr_t + duration, e) - max(curr_t, s)
                    if overlap > max_overlap_sec + 0.005:
                        # Too much parallel speech! Push curr_t forward so overlap is within barge-in window
                        target_barge = e - float(np.random.uniform(0.125, max_overlap_sec))
                        curr_t = max(curr_t, target_barge)
                        collision_found = True
                        break

            # Invariant B: Max concurrent speakers <= max_concurrent (never 3 speaking)
            critical_points = sorted(set(
                [curr_t, curr_t + duration] +
                [s for s, e in other_intervals if curr_t <= s < curr_t + duration] +
                [e for s, e in other_intervals if curr_t < e <= curr_t + duration]
            ))

            for i in range(len(critical_points) - 1):
                mid_point = (critical_points[i] + critical_points[i + 1]) / 2.0
                active_others = sum(1 for s, e in other_intervals if s <= mid_point < e)
                if active_others >= max_concurrent:
                    overlapping_ends = [e for s, e in other_intervals if s <= mid_point < e]
                    first_clear = min(overlapping_ends)
                    curr_t = max(curr_t, first_clear + 0.05)
                    collision_found = True
                    break

            if not collision_found:
                break

        return round(curr_t, 3)
