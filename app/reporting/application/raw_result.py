"""Machine-readable result (float seconds) so later steps like evaluation need not parse text reports."""

import json

from app.events.domain.bed_event import BedEvent, BedEventKind
from app.shared.domain.states import ActivityState
from app.shared.domain.time_range import TimeRange
from app.state.domain.models import StateSegment, Timeline


def dump_raw_result(timeline: Timeline, events: list[BedEvent]) -> str:
    return json.dumps(
        {
            "duration_sec": timeline.duration,
            "segments": [
                {
                    "start_sec": s.time_range.start,
                    "end_sec": s.time_range.end,
                    "state": s.state.value,
                    "confidence": s.confidence,
                    "reason": s.reason,
                }
                for s in timeline.segments
            ],
            "events": [
                {
                    "kind": e.kind.value,
                    "start_sec": e.start_time,
                    "confirmed_sec": e.confirmed_time,
                    "previous_state": e.previous_state.value,
                    "current_state": e.current_state.value,
                    "confidence": e.confidence,
                }
                for e in events
            ],
        },
        indent=2,
    )


def load_raw_result(text: str) -> tuple[Timeline, list[BedEvent]]:
    data = json.loads(text)
    timeline = Timeline(
        tuple(
            StateSegment(
                TimeRange(s["start_sec"], s["end_sec"]),
                ActivityState(s["state"]),
                s["confidence"],
                s["reason"],
            )
            for s in data["segments"]
        )
    )
    events = [
        BedEvent(
            BedEventKind(e["kind"]),
            e["start_sec"],
            e["confirmed_sec"],
            ActivityState(e["previous_state"]),
            ActivityState(e["current_state"]),
            e["confidence"],
        )
        for e in data["events"]
    ]
    return timeline, events
