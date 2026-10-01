from collections.abc import Callable

from app.events.domain.bed_event import BedEvent, BedEventKind
from app.extract.domain.observation import FrameFeatures
from app.reasoning.domain.models import AmbiguityCase, CaseKind
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.shared.domain.time_range import TimeRange
from app.state.domain.models import (
    LYING_OUTSIDE_BED,
    NO_PERSON_DETECTED,
    StateSegment,
    Timeline,
)


def _segment_case(segment: StateSegment, escalate_sec: float) -> CaseKind | None:
    if segment.state == S.UNKNOWN and segment.reason == LYING_OUTSIDE_BED:
        return CaseKind.LYING_OUTSIDE_BED  # possible fall: always worth a look, however short
    if segment.state == S.OUT_OF_BED:
        return CaseKind.TRACK_LOST
    if segment.state == S.UNKNOWN and segment.duration > escalate_sec:
        if segment.reason == NO_PERSON_DETECTED:
            return CaseKind.TRACK_LOST
        return CaseKind.LOW_CONFIDENCE
    return None


def _streaks(
    features: list[FrameFeatures], predicate: Callable[[FrameFeatures], bool]
) -> list[TimeRange]:
    """Time ranges of consecutive frames for which `predicate` holds."""
    ranges, start, last = [], None, 0.0
    for f in features:
        if predicate(f):
            start = f.time_sec if start is None else start
            last = f.time_sec
        elif start is not None:
            ranges.append(TimeRange(start, last))
            start = None
    if start is not None:
        ranges.append(TimeRange(start, last))
    return ranges


def _excursion_end(timeline: Timeline, start: float) -> float:
    """End of the out-of-bed stretch that contains `start`."""
    for in_bed, run in timeline.bed_runs():
        if not in_bed and run[0].time_range.start <= start < run[-1].time_range.end:
            return run[-1].time_range.end
    return start


def find_cases(
    timeline: Timeline,
    events: list[BedEvent],
    features: list[FrameFeatures],
    settings: Settings,
) -> list[AmbiguityCase]:
    """Everything the rule-based pipeline is unsure about, most important first."""
    cases = []
    for segment in timeline.segments:
        kind = _segment_case(segment, settings.unknown_escalate_sec)
        if kind:
            cases.append(AmbiguityCase(kind, segment.time_range))

    for event in events:
        if event.kind != BedEventKind.BED_EXIT:
            continue
        end = _excursion_end(timeline, event.start_time)
        # A short excursion may just be standing and sitting back; a weak one may be a detection error.
        short = end - event.start_time < 2 * settings.bed_exit_confirm_sec
        if short or event.confidence < settings.agent_min_confidence:
            cases.append(AmbiguityCase(CaseKind.BED_EXIT, TimeRange(event.start_time, end)))

    for streak in _streaks(features, lambda f: f.other_people > 0):
        if streak.duration >= settings.multiple_persons_min_sec:
            cases.append(AmbiguityCase(CaseKind.MULTIPLE_PERSONS, streak))

    order = list(CaseKind)
    return sorted(cases, key=lambda c: (order.index(c.kind), c.time_range.start))
