from app.shared.domain.states import ActivityState
from app.shared.domain.time_range import TimeRange
from app.state.domain.models import StateSegment, Timeline


def timeline(*runs: tuple[ActivityState, float] | tuple[ActivityState, float, str]) -> Timeline:
    """`(state, seconds[, reason])` runs laid end to end from t=0, each with confidence 0.9."""
    segments, t = [], 0.0
    for state, seconds, *reason in runs:
        segments.append(StateSegment(TimeRange(t, t + seconds), state, 0.9, *reason))
        t += seconds
    return Timeline(tuple(segments))
