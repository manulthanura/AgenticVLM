from collections import Counter, defaultdict
from dataclasses import dataclass

from app.events.domain.bed_event import BedEvent, BedEventKind
from app.shared.domain.states import ActivityState
from app.state.domain.models import Timeline


def _ratio(numerator: int, denominator: int) -> float | None:
    """None (not 0) when there is nothing to divide by: 'no data' must not look like 'all wrong'."""
    return numerator / denominator if denominator else None


def evaluated_duration(truth: Timeline, predicted: Timeline) -> float:
    return min(truth.duration, predicted.duration)


# --- activity: frame-level ---------------------------------------------------------------------


@dataclass(frozen=True)
class ClassScore:
    precision: float | None
    recall: float | None
    support: int  # ground-truth frames of this state


@dataclass(frozen=True)
class FrameMetrics:
    n_frames: int
    accuracy: float | None
    confusion: dict[ActivityState, dict[ActivityState, int]]  # truth -> predicted -> count
    per_class: dict[ActivityState, ClassScore]


def frame_metrics(truth: Timeline, predicted: Timeline, step_sec: float) -> FrameMetrics:
    """Compare both timelines once every `step_sec` (1 s by default). UNKNOWN counts as a miss."""
    n = int(evaluated_duration(truth, predicted) / step_sec)
    confusion: dict[ActivityState, Counter] = defaultdict(Counter)
    for i in range(n):
        t = i * step_sec
        confusion[truth.state_at(t)][predicted.state_at(t)] += 1
    per_class = {}
    for state in ActivityState:
        hits = confusion[state][state]
        predicted_total = sum(row[state] for row in confusion.values())
        support = sum(confusion[state].values())
        if support or predicted_total:
            per_class[state] = ClassScore(
                _ratio(hits, predicted_total), _ratio(hits, support), support
            )
    correct = sum(confusion[s][s] for s in list(confusion))
    return FrameMetrics(
        n_frames=n,
        accuracy=_ratio(correct, n),
        confusion={t: dict(row) for t, row in confusion.items() if row},
        per_class=per_class,
    )


# --- bed events --------------------------------------------------------------------------------


@dataclass(frozen=True)
class EventScore:
    matched: int
    predicted: int
    truth: int

    @property
    def precision(self) -> float | None:
        return _ratio(self.matched, self.predicted)

    @property
    def recall(self) -> float | None:
        return _ratio(self.matched, self.truth)


def _count_matches(predicted: list[float], truth: list[float], tolerance_sec: float) -> int:
    """Greedy one-to-one matching, closest pairs first, so one event is never counted twice."""
    pairs = sorted(
        (abs(p - t), i, j)
        for i, p in enumerate(predicted)
        for j, t in enumerate(truth)
        if abs(p - t) <= tolerance_sec
    )
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    for _, i, j in pairs:
        if i not in used_predicted and j not in used_truth:
            used_predicted.add(i)
            used_truth.add(j)
    return len(used_predicted)


def match_events(
    predicted: list[BedEvent], truth: list[BedEvent], tolerance_sec: float
) -> dict[BedEventKind, EventScore]:
    """Match per event kind on start_time within +-tolerance."""
    scores = {}
    for kind in BedEventKind:
        p = [e.start_time for e in predicted if e.kind == kind]
        t = [e.start_time for e in truth if e.kind == kind]
        scores[kind] = EventScore(_count_matches(p, t, tolerance_sec), len(p), len(t))
    return scores


# --- durations ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DurationError:
    predicted_sec: float
    truth_sec: float

    @property
    def abs_error_sec(self) -> float:
        return abs(self.predicted_sec - self.truth_sec)


def _seconds_within(timeline: Timeline, limit_sec: float) -> dict[ActivityState, float]:
    totals: dict[ActivityState, float] = {}
    for s in timeline.segments:
        overlap = min(s.time_range.end, limit_sec) - s.time_range.start
        if overlap > 0:
            totals[s.state] = totals.get(s.state, 0.0) + overlap
    return totals


def duration_errors(truth: Timeline, predicted: Timeline) -> dict[ActivityState, DurationError]:
    """Seconds per state, both clipped to the span both timelines cover."""
    limit = evaluated_duration(truth, predicted)
    want, got = _seconds_within(truth, limit), _seconds_within(predicted, limit)
    return {
        s: DurationError(got.get(s, 0.0), want.get(s, 0.0))
        for s in ActivityState
        if s in want or s in got
    }


# --- failure cases -----------------------------------------------------------------------------


@dataclass
class FailureCase:
    start_sec: float
    end_sec: float
    truth_state: ActivityState
    predicted_state: ActivityState
    reason: str  # the predicted segment's own explanation, e.g. keypoints_not_visible

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def find_failures(
    truth: Timeline, predicted: Timeline, step_sec: float, max_cases: int
) -> list[FailureCase]:
    """Runs of consecutive wrong samples with the same (truth, predicted) pair, longest first."""
    n = int(evaluated_duration(truth, predicted) / step_sec)
    failures: list[FailureCase] = []
    current: FailureCase | None = None
    for i in range(n):
        t = i * step_sec
        want, got = truth.state_at(t), predicted.state_at(t)
        if want == got:
            current = None
        elif current and (current.truth_state, current.predicted_state) == (want, got):
            current.end_sec = t + step_sec
        else:
            current = FailureCase(t, t + step_sec, want, got, predicted.segment_at(t).reason)
            failures.append(current)
    return sorted(failures, key=lambda f: f.duration, reverse=True)[:max_cases]
