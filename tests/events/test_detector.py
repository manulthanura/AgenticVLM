import pytest

from app.events.domain.bed_event import BedEventKind
from app.events.domain.detector import BedEventDetector
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from tests.state.factories import timeline

LYING, SIT, STAND, WALK, UNK = S.LYING_IN_BED, S.SITTING_ON_BED, S.STANDING, S.WALKING, S.UNKNOWN


def detect(*runs: tuple[S, float]):
    """`(state, seconds)` pairs -> events."""
    return BedEventDetector(Settings()).detect(timeline(*runs))  # confirm windows: 5 s


def kinds(events):
    return [e.kind for e in events]


def test_bed_exit_fields():
    (exit_,) = detect((LYING, 20), (SIT, 5), (STAND, 3), (WALK, 10))
    assert exit_.kind == BedEventKind.BED_EXIT
    assert (exit_.start_time, exit_.confirmed_time) == (25, 30)  # start = when standing began
    assert (exit_.previous_state, exit_.current_state) == (SIT, WALK)
    assert exit_.confidence == pytest.approx(0.9)


def test_standing_then_sitting_back_is_not_an_exit():
    assert detect((LYING, 20), (SIT, 5), (STAND, 3), (SIT, 10)) == []
    assert detect((LYING, 20), (SIT, 5), (STAND, 10), (SIT, 10)) == []  # long, but never left


def test_exit_needs_the_confirmation_window():
    assert detect((LYING, 20), (STAND, 1), (WALK, 3)) == []  # 4 s < 5 s
    assert kinds(detect((LYING, 20), (STAND, 1), (WALK, 4))) == [BedEventKind.BED_EXIT]


def test_occlusion_inside_bed_is_not_an_event():
    assert detect((LYING, 20), (UNK, 4), (LYING, 20)) == []


def test_missed_standing_still_gives_exit_starting_after_unknown():
    (exit_,) = detect((LYING, 20), (UNK, 4), (WALK, 10))
    assert (exit_.start_time, exit_.previous_state) == (24, LYING)


def test_return_to_bed():
    events = detect((LYING, 20), (STAND, 2), (WALK, 10), (SIT, 3), (LYING, 10))
    assert kinds(events) == [BedEventKind.BED_EXIT, BedEventKind.RETURN_TO_BED]
    back = events[1]
    assert (back.start_time, back.confirmed_time) == (32, 37)
    assert (back.previous_state, back.current_state) == (WALK, SIT)


def test_touching_the_bed_briefly_is_not_a_return():
    events = detect((LYING, 20), (STAND, 2), (WALK, 10), (SIT, 2))
    assert kinds(events) == [BedEventKind.BED_EXIT]


def test_video_starting_out_of_bed_has_only_a_return():
    assert kinds(detect((WALK, 10), (SIT, 8))) == [BedEventKind.RETURN_TO_BED]


def test_standing_far_from_the_bed_counts_as_an_exit_even_without_walking():
    from app.shared.domain.time_range import TimeRange
    from app.state.domain.models import AWAY_FROM_BED, StateSegment, Timeline

    def seg(start, end, state, reason=""):
        return StateSegment(TimeRange(start, end), state, 0.9, reason)

    away = Timeline((seg(0, 20, LYING), seg(20, 40, STAND, AWAY_FROM_BED), seg(40, 55, SIT)))
    assert kinds(BedEventDetector(Settings()).detect(away)) == [
        BedEventKind.BED_EXIT,
        BedEventKind.RETURN_TO_BED,
    ]
    beside = Timeline((seg(0, 20, LYING), seg(20, 40, STAND, "bed_edge"), seg(40, 55, SIT)))
    assert BedEventDetector(Settings()).detect(beside) == []
