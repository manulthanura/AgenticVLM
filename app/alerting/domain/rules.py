from collections.abc import Callable, Sequence

from app.alerting.domain.decision import AlertDecision
from app.events.domain.bed_event import BedEvent, BedEventKind
from app.reasoning.domain.models import AgentFinding
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.shared.domain.states import Decision
from app.state.domain.models import (
    BED_EDGE,
    LYING_OUTSIDE_BED,
    StateSegment,
    Timeline,
    mean_confidence,
)

# A rule looks at the timeline and events and returns the decisions it fires (possibly none).
AlertRule = Callable[[Timeline, list[BedEvent], Settings], list[AlertDecision]]


def _too_long(
    groups: list[list[StateSegment]], limit_sec: float, decision: Decision, rule: str
) -> list[AlertDecision]:
    """Fire once a group of segments lasts longer than `limit_sec` (0 = fire immediately)."""
    fired = []
    for group in groups:
        start, end = group[0].time_range.start, group[-1].time_range.end
        if end - start > limit_sec:
            evidence = {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "limit_sec": limit_sec,
                "confidence": mean_confidence(group),
                "states": sorted({s.state.value for s in group}),
            }
            fired.append(AlertDecision(decision, rule, start + limit_sec, evidence))
    return fired


def _each(timeline: Timeline, keep: Callable[[StateSegment], bool]) -> list[list[StateSegment]]:
    return [[s] for s in timeline.segments if keep(s)]


_EVENT_DECISION = {
    BedEventKind.BED_EXIT: Decision.MONITOR,
    BedEventKind.RETURN_TO_BED: Decision.NORMAL,
}


def bed_events(
    timeline: Timeline, events: list[BedEvent], settings: Settings
) -> list[AlertDecision]:
    # A bed exit is worth a look by default (nights are the risky case); a return is reassuring.
    return [
        AlertDecision(
            _EVENT_DECISION[e.kind],
            e.kind.value,
            e.confirmed_time,
            {
                "start_sec": e.start_time,
                "previous_state": e.previous_state.value,
                "current_state": e.current_state.value,
                "confidence": e.confidence,
            },
        )
        for e in events
    ]


def lying_outside_bed(
    timeline: Timeline, events: list[BedEvent], settings: Settings
) -> list[AlertDecision]:
    # Horizontal posture away from the bed is a possible fall: no waiting period.
    groups = _each(timeline, lambda s: s.state == S.UNKNOWN and s.reason == LYING_OUTSIDE_BED)
    return _too_long(groups, 0, Decision.ALERT, "lying_outside_bed")


def out_of_bed_too_long(
    timeline: Timeline, events: list[BedEvent], settings: Settings
) -> list[AlertDecision]:
    # Whole out-of-bed stretch, but only if the person was actually seen (not just UNKNOWN).
    runs = [
        run
        for in_bed, run in timeline.bed_runs()
        if not in_bed and any(s.state != S.UNKNOWN for s in run)
    ]
    return _too_long(runs, settings.out_of_bed_alert_sec, Decision.ALERT, "out_of_bed_too_long")


def out_of_view_too_long(
    timeline: Timeline, events: list[BedEvent], settings: Settings
) -> list[AlertDecision]:
    groups = _each(timeline, lambda s: s.state == S.OUT_OF_BED)
    return _too_long(groups, settings.out_of_view_alert_sec, Decision.ALERT, "out_of_view_too_long")


def edge_sitting_too_long(
    timeline: Timeline, events: list[BedEvent], settings: Settings
) -> list[AlertDecision]:
    # Sitting on the bed border for long can mean trouble standing up; sitting inside the bed is fine.
    groups = _each(timeline, lambda s: s.state == S.SITTING_ON_BED and s.reason == BED_EDGE)
    return _too_long(
        groups, settings.edge_sitting_monitor_sec, Decision.MONITOR, "edge_sitting_too_long"
    )


def unknown_too_long(
    timeline: Timeline, events: list[BedEvent], settings: Settings
) -> list[AlertDecision]:
    # Long blind spots mean we cannot vouch for the person (lying outside bed has its own ALERT rule).
    groups = _each(timeline, lambda s: s.state == S.UNKNOWN and s.reason != LYING_OUTSIDE_BED)
    return _too_long(groups, settings.unknown_monitor_sec, Decision.MONITOR, "unknown_too_long")


RULES: tuple[AlertRule, ...] = (
    bed_events,
    lying_outside_bed,
    out_of_bed_too_long,
    out_of_view_too_long,
    edge_sitting_too_long,
    unknown_too_long,
)


def agent_low_confidence(
    findings: Sequence[AgentFinding], settings: Settings
) -> list[AlertDecision]:
    # If even the agent could not settle a case, a human should look; it is never an ALERT by itself.
    return [
        AlertDecision(
            Decision.MONITOR,
            "agent_low_confidence",
            f.case.time_range.start,
            {
                "case": f.case.kind.value,
                "start_sec": f.case.time_range.start,
                "end_sec": f.case.time_range.end,
                "confidence": f.confidence,
                "rationale": f.rationale,
            },
        )
        for f in findings
        if f.confidence < settings.agent_min_confidence
    ]


def evaluate_alerts(
    timeline: Timeline,
    events: list[BedEvent],
    settings: Settings,
    findings: Sequence[AgentFinding] = (),
) -> list[AlertDecision]:
    """All decisions in time order. Use overall_decision() for the single worst one.

    `timeline` and `events` should already include the agent's resolutions (see Reasoner).
    """
    fired = [d for rule in RULES for d in rule(timeline, events, settings)]
    fired += agent_low_confidence(findings, settings)
    return sorted(fired, key=lambda d: d.time_sec)
