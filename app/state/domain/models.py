from collections import Counter
from dataclasses import dataclass
from itertools import groupby

from app.shared.domain.states import IN_BED_STATES, ActivityState
from app.shared.domain.time_range import TimeRange

# Reasons the classifier attaches that other features (alerting) act on.
LYING_OUTSIDE_BED = "lying_outside_bed"  # possible fall
BED_EDGE = "bed_edge"  # sitting with hips on the bed border
NO_PERSON_DETECTED = "no_person_detected"  # UNKNOWN because nobody was tracked
AWAY_FROM_BED = "away_from_bed"  # standing with hips outside the bed polygon
RECLINED = "reclined"  # lying label from a half-upright posture on the bed
LOW_CONFIDENCE = "low_confidence"  # UNKNOWN because the frame's label was too weak


@dataclass(frozen=True)
class StateCandidate:
    """One frame's opinion. `reason` explains UNKNOWN or weak labels (e.g. "lying_outside_bed")."""

    state: ActivityState
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class StateSegment:
    time_range: TimeRange
    state: ActivityState
    confidence: float  # mean of the frame confidences inside the segment
    reason: str = ""  # most common frame reason, "" if none

    @property
    def duration(self) -> float:
        return self.time_range.duration


@dataclass(frozen=True)
class Timeline:
    """Ordered, non-overlapping segments that cover the whole analysed video."""

    segments: tuple[StateSegment, ...]

    @property
    def duration(self) -> float:
        return (
            self.segments[-1].time_range.end - self.segments[0].time_range.start
            if self.segments
            else 0.0
        )

    def seconds_by_state(self) -> dict[ActivityState, float]:
        totals: dict[ActivityState, float] = {}
        for segment in self.segments:
            totals[segment.state] = totals.get(segment.state, 0.0) + segment.duration
        return totals

    def bed_runs(self) -> list[tuple[bool, list[StateSegment]]]:
        """Maximal stretches that are all in bed / all out of bed (UNKNOWN counts as out)."""
        return [
            (in_bed, list(group))
            for in_bed, group in groupby(self.segments, key=lambda s: s.state in IN_BED_STATES)
        ]

    def segment_at(self, time_sec: float) -> StateSegment:
        """The segment covering `time_sec`; times past the end give the last segment."""
        for segment in self.segments:
            if time_sec < segment.time_range.end:
                return segment
        return self.segments[-1]

    def state_at(self, time_sec: float) -> ActivityState:
        return self.segment_at(time_sec).state


def mean_confidence(segments: list[StateSegment]) -> float:
    """Duration-weighted mean, so a long confident segment outweighs a short shaky one."""
    total = sum(s.duration for s in segments)
    return sum(s.confidence * s.duration for s in segments) / total if total else 0.0


def most_common_reason(reasons: list[str]) -> str:
    """Mode over all frames, "" included: a few odd frames must not label a whole segment."""
    return Counter(reasons).most_common(1)[0][0] if reasons else ""
