import pytest

from app.events.domain.detector import BedEventDetector
from app.extract.domain.observation import FrameFeatures
from app.reasoning.domain.cases import find_cases
from app.reasoning.domain.models import CaseKind
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.state.domain.models import LYING_OUTSIDE_BED, NO_PERSON_DETECTED
from tests.extract.factories import person
from tests.state.factories import timeline

SETTINGS = Settings()  # escalate 10 s, exit confirm 5 s, multiple persons 3 s


def cases(*runs, features=()):
    tl = timeline(*runs)
    return find_cases(tl, BedEventDetector(SETTINGS).detect(tl), list(features), SETTINGS)


def kinds(found):
    return [c.kind for c in found]


def test_lying_outside_bed_is_a_case_even_when_short():
    (case,) = cases((S.STANDING, 10), (S.UNKNOWN, 2, LYING_OUTSIDE_BED))
    assert case.kind == CaseKind.LYING_OUTSIDE_BED
    assert (case.time_range.start, case.time_range.end) == (10, 12)


@pytest.mark.parametrize(
    ("reason", "seconds", "expected"),
    [
        ("keypoints_not_visible", 5, []),  # brief occlusion: the state machine already coped
        ("keypoints_not_visible", 15, [CaseKind.LOW_CONFIDENCE]),
        (NO_PERSON_DETECTED, 15, [CaseKind.TRACK_LOST]),
    ],
)
def test_unknown_escalates_only_when_long(reason, seconds, expected):
    assert kinds(cases((S.LYING_IN_BED, 20), (S.UNKNOWN, seconds, reason))) == expected


def test_out_of_view_is_track_lost():
    assert kinds(cases((S.WALKING, 10), (S.OUT_OF_BED, 20))) == [CaseKind.TRACK_LOST]


def test_short_exit_is_a_case_but_a_clear_one_is_not():
    (short,) = cases((S.LYING_IN_BED, 20), (S.STANDING, 2), (S.WALKING, 5))
    assert short.kind == CaseKind.BED_EXIT
    assert (short.time_range.start, short.time_range.end) == (20, 27)
    assert cases((S.LYING_IN_BED, 20), (S.STANDING, 2), (S.WALKING, 30)) == []


def test_second_person_needs_to_stay_a_few_seconds():
    def frames(seconds):
        return [FrameFeatures(i * 0.2, person(1), 1) for i in range(int(seconds / 0.2))]

    assert kinds(cases((S.LYING_IN_BED, 10), features=frames(4))) == [CaseKind.MULTIPLE_PERSONS]
    assert cases((S.LYING_IN_BED, 10), features=frames(1)) == []


def test_fall_question_comes_before_others():
    found = cases((S.WALKING, 10), (S.OUT_OF_BED, 20), (S.UNKNOWN, 3, LYING_OUTSIDE_BED))
    assert kinds(found) == [CaseKind.LYING_OUTSIDE_BED, CaseKind.TRACK_LOST]
