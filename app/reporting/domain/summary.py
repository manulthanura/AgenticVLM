from dataclasses import dataclass

from app.events.domain.bed_event import BedEvent, BedEventKind
from app.reporting.domain.formatting import human
from app.shared.domain.states import IN_BED_STATES, ActivityState
from app.state.domain.models import Timeline


@dataclass(frozen=True)
class ActivitySummary:
    observation_duration_sec: float
    activity_duration_sec: dict[ActivityState, float]
    bed_exit_count: int
    bed_return_count: int
    total_in_bed_sec: float
    total_out_of_bed_sec: float
    longest_out_of_bed_period_sec: float
    final_state: ActivityState | None

    def to_dict(self) -> dict[str, object]:
        """Seconds as numbers for machines, plus a readable `_human` copy of each duration."""
        durations = {
            state.value: round(self.activity_duration_sec.get(state, 0.0), 2)
            for state in ActivityState
        }
        return {
            "observation_duration_sec": round(self.observation_duration_sec, 2),
            "activity_duration_sec": durations,
            "bed_exit_count": self.bed_exit_count,
            "bed_return_count": self.bed_return_count,
            "total_in_bed_sec": round(self.total_in_bed_sec, 2),
            "total_out_of_bed_sec": round(self.total_out_of_bed_sec, 2),
            "longest_out_of_bed_period_sec": round(self.longest_out_of_bed_period_sec, 2),
            "final_state": self.final_state.value if self.final_state else None,
            "observation_duration_human": human(self.observation_duration_sec),
            "activity_duration_human": {
                state: human(sec) for state, sec in durations.items() if sec
            },
            "total_in_bed_human": human(self.total_in_bed_sec),
            "total_out_of_bed_human": human(self.total_out_of_bed_sec),
            "longest_out_of_bed_period_human": human(self.longest_out_of_bed_period_sec),
        }


def build_summary(timeline: Timeline, events: list[BedEvent]) -> ActivitySummary:
    """Durations come from segments, never from counting frame labels"""
    by_state = timeline.seconds_by_state()
    in_bed = sum(by_state.get(state, 0.0) for state in IN_BED_STATES)
    out_of_bed_stretches = [
        sum(segment.duration for segment in run)
        for is_in_bed, run in timeline.bed_runs()
        if not is_in_bed
    ]
    return ActivitySummary(
        observation_duration_sec=timeline.duration,
        activity_duration_sec=by_state,
        bed_exit_count=sum(e.kind == BedEventKind.BED_EXIT for e in events),
        bed_return_count=sum(e.kind == BedEventKind.RETURN_TO_BED for e in events),
        total_in_bed_sec=in_bed,
        total_out_of_bed_sec=timeline.duration - in_bed,  # everything not in bed, UNKNOWN included
        longest_out_of_bed_period_sec=max(out_of_bed_stretches, default=0.0),
        final_state=timeline.segments[-1].state if timeline.segments else None,
    )
