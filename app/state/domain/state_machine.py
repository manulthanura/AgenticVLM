from collections import Counter

from app.shared.config import Settings
from app.shared.domain.states import ActivityState, is_transition_allowed
from app.shared.domain.time_range import TimeRange
from app.state.domain.models import (
    LOW_CONFIDENCE,
    StateCandidate,
    StateSegment,
    Timeline,
    most_common_reason,
)


class StateMachine:
    """Smooths per-frame candidates into segments: dwell hysteresis + allowed transitions only."""

    def __init__(self, settings: Settings) -> None:
        self._dwell = settings.min_state_dwell_sec
        self._share_min = settings.dwell_share_min
        self._min_confidence = settings.state_confidence_min
        self._starts: list[tuple[float, ActivityState]] = []  # accepted state changes
        self._frames: list[tuple[float, float, str]] = []  # time, confidence, reason
        self._recent: list[
            tuple[float, ActivityState]
        ] = []  # candidates over the last dwell window
        self._first_time: float | None = None  # first frame, for the initial segment

    @property
    def current(self) -> ActivityState | None:
        return self._starts[-1][1] if self._starts else None

    def update(self, time_sec: float, candidate: StateCandidate) -> None:
        weak = candidate.confidence < self._min_confidence
        state = ActivityState.UNKNOWN if weak else candidate.state
        # A weak label becomes UNKNOWN, and the segment should say why (unless a better reason exists).
        reason = candidate.reason or (LOW_CONFIDENCE if weak else "")
        self._frames.append((time_sec, candidate.confidence, reason))
        if self._first_time is None:
            self._first_time = time_sec
        if not self._starts:
            self._seed(time_sec, state, weak)
            return
        self._remember(time_sec, state)
        challenger = self._challenger()
        if challenger is None:
            return
        since, state = challenger
        since = max(since, self._starts[-1][0])  # starts must stay in order
        if is_transition_allowed(self.current, state):
            self._starts.append((since, state))
        else:
            # Implausible jump (e.g. lying -> walking): a step was missed, so admit we don't know
            # and make the new state prove itself again from here.
            self._starts.append((since, ActivityState.UNKNOWN))
            self._recent = [(time_sec, state)]

    def _seed(self, time_sec: float, state: ActivityState, weak: bool) -> None:
        """First segment: one weak first frame is not evidence of UNKNOWN, so wait for a confident one."""
        if not weak:
            self._starts.append((self._first_time, state))
        elif time_sec - self._first_time >= self._dwell:  # nothing confident for a whole dwell
            self._starts.append((self._first_time, ActivityState.UNKNOWN))
        else:
            return
        self._remember(time_sec, state)

    def _remember(self, time_sec: float, state: ActivityState) -> None:
        """Keep the last dwell window, plus the one frame at or before its start so the window is full."""
        self._recent.append((time_sec, state))
        while len(self._recent) > 1 and self._recent[1][0] <= time_sec - self._dwell:
            self._recent.pop(0)

    def _challenger(self) -> tuple[float, ActivityState] | None:
        """A non-current state that filled enough of a full dwell window, and when it first showed up.

        Share instead of an unbroken run: two WALKING frames inside a long STANDING run must not
        restart its clock (see plans/05-state-smoothing.md).
        """
        time_sec = self._recent[-1][0]
        if self._recent[0][0] > time_sec - self._dwell:
            return None  # window not full yet
        counts = Counter(s for _, s in self._recent if s != self.current)
        if not counts:
            return None
        state, n = counts.most_common(1)[0]
        if n / len(self._recent) < self._share_min:
            return None
        return next(t for t, s in self._recent if s == state), state

    def finish(self, end_time: float) -> Timeline:
        if not self._starts and self._first_time is not None:  # video shorter than one dwell
            self._starts.append((self._first_time, ActivityState.UNKNOWN))
        ends = [start for start, _ in self._starts[1:]] + [end_time]
        segments = []
        for (start, state), end in zip(self._starts, ends, strict=True):
            if end <= start:
                continue
            inside = [(c, r) for t, c, r in self._frames if start <= t < end]
            confidence = sum(c for c, _ in inside) / len(inside) if inside else 0.0
            segments.append(
                StateSegment(
                    TimeRange(start, end),
                    state,
                    confidence,
                    most_common_reason([r for _, r in inside]),
                )
            )
        return Timeline(tuple(segments))
