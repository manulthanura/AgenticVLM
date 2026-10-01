from app.events.domain.bed_event import BedEvent, BedEventKind
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.state.domain.models import AWAY_FROM_BED, StateSegment, Timeline, mean_confidence

# Evidence that the person really is away from the bed, not just standing next to it.
_AWAY_STATES = {S.WALKING, S.SITTING_OUTSIDE_BED, S.OUT_OF_BED}


def _is_away(segment: StateSegment) -> bool:
    return segment.state in _AWAY_STATES or (
        segment.state == S.STANDING and segment.reason == AWAY_FROM_BED
    )


class BedEventDetector:
    """Finds bed exits and returns from a timeline.

    The timeline is split into alternating in-bed / out-of-bed runs. An out-of-bed run is a real
    excursion only if it contains an away state and lasts long enough; occlusion (UNKNOWN) or
    standing beside the bed and sitting back down therefore never counts as an exit.
    """

    def __init__(self, settings: Settings) -> None:
        self._exit_confirm = settings.bed_exit_confirm_sec
        self._return_confirm = settings.bed_return_confirm_sec

    def detect(self, timeline: Timeline) -> list[BedEvent]:
        runs = timeline.bed_runs()
        events: list[BedEvent] = []
        for k, (in_bed, run) in enumerate(runs):
            if in_bed:
                continue
            # The excursion starts when standing (or the first away state) begins, skipping leading UNKNOWN.
            active = [s for s in run if s.state == S.STANDING or s.state in _AWAY_STATES]
            away = next((s for s in run if _is_away(s)), None)
            if (
                away is None
                or run[-1].time_range.end - active[0].time_range.start < self._exit_confirm
            ):
                continue
            start = active[0].time_range.start
            before = runs[k - 1][1] if k > 0 else None
            after = runs[k + 1][1] if k + 1 < len(runs) else None
            if before:  # a video that starts out of bed has no exit to report
                events.append(
                    BedEvent(
                        BedEventKind.BED_EXIT,
                        start,
                        start + self._exit_confirm,
                        before[-1].state,
                        away.state,
                        mean_confidence(run[run.index(active[0]) :]),
                    )
                )
            if after and sum(s.duration for s in after) >= self._return_confirm:
                start = after[0].time_range.start
                events.append(
                    BedEvent(
                        BedEventKind.RETURN_TO_BED,
                        start,
                        start + self._return_confirm,
                        run[-1].state,
                        after[0].state,
                        mean_confidence(after),
                    )
                )
        return events
