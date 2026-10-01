from dataclasses import replace

from app.events.domain.bed_event import BedEvent, BedEventKind
from app.reasoning.domain.models import RESOLVED_BY_AGENT, AgentFinding, CaseKind
from app.shared.domain.states import ActivityState as S
from app.shared.domain.time_range import TimeRange
from app.state.domain.models import StateSegment, Timeline


def _merge(a: StateSegment, b: StateSegment) -> StateSegment:
    total = a.duration + b.duration
    confidence = (a.confidence * a.duration + b.confidence * b.duration) / total
    return StateSegment(
        TimeRange(a.time_range.start, b.time_range.end),
        a.state,
        confidence,
        a.reason if a.reason == b.reason else "",
    )


def apply_findings(
    timeline: Timeline, findings: list[AgentFinding], min_confidence: float
) -> Timeline:
    """Let confident findings resolve UNKNOWN segments; never touch a segment with a confident state.

    Boundaries stay put, so the durations still sum to the video length. Neighbours that end up
    with the same state are merged.
    """
    resolved = {
        f.case.time_range: f
        for f in findings
        if f.state not in (None, S.UNKNOWN) and f.confidence >= min_confidence
    }
    segments: list[StateSegment] = []
    for segment in timeline.segments:
        finding = resolved.get(segment.time_range) if segment.state == S.UNKNOWN else None
        if finding:
            segment = replace(
                segment,
                state=finding.state,
                confidence=finding.confidence,
                reason=RESOLVED_BY_AGENT,
            )
        if segments and segments[-1].state == segment.state:
            segments[-1] = _merge(segments[-1], segment)
        else:
            segments.append(segment)
    return Timeline(tuple(segments))


def remove_rejected_exits(
    events: list[BedEvent], findings: list[AgentFinding], min_confidence: float
) -> list[BedEvent]:
    """Drop bed exits the agent confidently judged to be 'stood up and sat back down'."""
    rejected = {
        f.case.time_range.start
        for f in findings
        if f.case.kind == CaseKind.BED_EXIT
        and f.exit_confirmed is False
        and f.confidence >= min_confidence
    }
    return [e for e in events if not (e.kind == BedEventKind.BED_EXIT and e.start_time in rejected)]
