import pytest

from app.extract.domain.bed import BedZone
from app.extract.domain.observation import FrameFeatures
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.state.application.build_timeline import build_timeline
from app.state.domain.models import StateCandidate
from app.state.domain.state_machine import StateMachine
from tests.extract.factories import person

DT = 0.2  # 5 fps
SETTINGS = Settings()  # dwell 2 s


def run(*runs: tuple[S, int], conf=0.9):
    """`(state, n_frames)` pairs -> Timeline ending after the last frame."""
    machine = StateMachine(SETTINGS)
    t = 0.0
    for state, n in runs:
        for _ in range(n):
            machine.update(t, StateCandidate(state, conf))
            t += DT
    return machine.finish(t)


def test_short_flicker_is_ignored():
    timeline = run((S.LYING_IN_BED, 20), (S.SITTING_ON_BED, 3), (S.LYING_IN_BED, 20))
    assert [s.state for s in timeline.segments] == [S.LYING_IN_BED]


def test_change_accepted_after_dwell_and_starts_when_it_began():
    timeline = run((S.LYING_IN_BED, 20), (S.SITTING_ON_BED, 20))
    assert [s.state for s in timeline.segments] == [S.LYING_IN_BED, S.SITTING_ON_BED]
    assert timeline.segments[1].time_range.start == pytest.approx(20 * DT)


def test_invalid_transition_goes_through_unknown():
    timeline = run((S.LYING_IN_BED, 20), (S.WALKING, 40))
    assert [s.state for s in timeline.segments] == [S.LYING_IN_BED, S.UNKNOWN, S.WALKING]


def test_low_confidence_counts_as_unknown():
    machine = StateMachine(SETTINGS)
    for i in range(20):
        machine.update(i * DT, StateCandidate(S.LYING_IN_BED, 0.2))
    segment = machine.finish(4.0).segments[0]
    assert (segment.state, segment.reason) == (S.UNKNOWN, "low_confidence")


def test_durations_sum_to_analysed_duration():
    timeline = run(
        (S.LYING_IN_BED, 30),
        (S.SITTING_ON_BED, 15),
        (S.STANDING, 15),
        (S.WALKING, 25),
        (S.OUT_OF_BED, 20),
    )
    assert sum(timeline.seconds_by_state().values()) == pytest.approx(timeline.duration)
    assert timeline.duration == pytest.approx(105 * DT)
    assert timeline.state_at(0.0) == S.LYING_IN_BED
    assert timeline.state_at(timeline.duration) == S.OUT_OF_BED


def test_build_timeline_end_to_end_with_features():
    def lying(t):
        return FrameFeatures(t, person(1), 0, 1.0, 85, None, (0, 0), BedZone.INSIDE, 1.0, 0.0)

    timeline = build_timeline([lying(i * DT) for i in range(25)], end_time=5.0, settings=SETTINGS)
    assert timeline.seconds_by_state() == {S.LYING_IN_BED: pytest.approx(5.0)}


def test_flicker_inside_a_long_run_does_not_restart_its_clock():
    # the s3_seq1 bug: two WALKING frames cancelled a STANDING candidate that had held for 2 s
    timeline = run((S.SITTING_ON_BED, 20), (S.STANDING, 9), (S.WALKING, 2), (S.STANDING, 20))
    assert [s.state for s in timeline.segments] == [S.SITTING_ON_BED, S.STANDING]
    assert timeline.segments[1].time_range.start == pytest.approx(20 * DT)


def test_alternating_states_do_not_flap():
    timeline = run((S.STANDING, 20), *[(S.WALKING, 3), (S.STANDING, 3)] * 5)
    assert [s.state for s in timeline.segments] == [S.STANDING]


def test_weak_first_frame_does_not_start_unknown():
    machine = StateMachine(SETTINGS)
    machine.update(0.0, StateCandidate(S.LYING_IN_BED, 0.4))
    for i in range(1, 25):
        machine.update(i * DT, StateCandidate(S.LYING_IN_BED, 0.9))
    timeline = machine.finish(5.0)
    assert [s.state for s in timeline.segments] == [S.LYING_IN_BED]
    assert timeline.segments[0].time_range.start == 0.0


def test_video_shorter_than_one_dwell_of_weak_frames_is_unknown():
    machine = StateMachine(SETTINGS)
    for i in range(5):
        machine.update(i * DT, StateCandidate(S.LYING_IN_BED, 0.2))
    assert [s.state for s in machine.finish(1.0).segments] == [S.UNKNOWN]
