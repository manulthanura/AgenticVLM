import pytest

from app.events.domain.detector import BedEventDetector
from app.reasoning.application.apply_findings import apply_findings, remove_rejected_exits
from app.reasoning.domain.models import RESOLVED_BY_AGENT, AgentFinding, AmbiguityCase, CaseKind
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.shared.domain.time_range import TimeRange
from tests.state.factories import timeline

MIN = 0.6


def finding(kind, start, end, state=None, confidence=0.9, exit_confirmed=None):
    case = AmbiguityCase(kind, TimeRange(start, end))
    return AgentFinding(case, state, exit_confirmed, confidence, "because")


def test_confident_finding_resolves_unknown_and_merges_neighbours():
    tl = timeline(
        (S.LYING_IN_BED, 20), (S.UNKNOWN, 15, "keypoints_not_visible"), (S.LYING_IN_BED, 20)
    )
    revised = apply_findings(tl, [finding(CaseKind.LOW_CONFIDENCE, 20, 35, S.LYING_IN_BED)], MIN)
    assert [s.state for s in revised.segments] == [S.LYING_IN_BED]
    assert revised.duration == pytest.approx(tl.duration) == pytest.approx(55)


def test_resolved_segment_is_marked_when_it_stays_separate():
    tl = timeline((S.STANDING, 10), (S.UNKNOWN, 15, "no_person_detected"))
    revised = apply_findings(tl, [finding(CaseKind.TRACK_LOST, 10, 25, S.WALKING)], MIN)
    assert [(s.state, s.reason) for s in revised.segments][1] == (S.WALKING, RESOLVED_BY_AGENT)


@pytest.mark.parametrize(
    "weak",
    [
        finding(CaseKind.LOW_CONFIDENCE, 10, 25, S.LYING_IN_BED, confidence=0.4),  # not sure enough
        finding(CaseKind.LOW_CONFIDENCE, 10, 25, S.UNKNOWN),  # nothing learned
    ],
)
def test_weak_findings_change_nothing(weak):
    tl = timeline((S.STANDING, 10), (S.UNKNOWN, 15, "keypoints_not_visible"))
    assert apply_findings(tl, [weak], MIN) == tl


def test_confident_states_are_never_overridden():
    tl = timeline((S.WALKING, 10), (S.OUT_OF_BED, 15))
    assert apply_findings(tl, [finding(CaseKind.TRACK_LOST, 10, 25, S.LYING_IN_BED)], MIN) == tl


def test_rejected_exit_is_removed_only_when_confident():
    events = BedEventDetector(Settings()).detect(
        timeline((S.LYING_IN_BED, 20), (S.STANDING, 2), (S.WALKING, 5))
    )
    assert len(events) == 1
    rejected = finding(CaseKind.BED_EXIT, 20, 27, exit_confirmed=False)
    unsure = finding(CaseKind.BED_EXIT, 20, 27, exit_confirmed=False, confidence=0.4)
    confirmed = finding(CaseKind.BED_EXIT, 20, 27, exit_confirmed=True)
    assert remove_rejected_exits(events, [rejected], MIN) == []
    assert remove_rejected_exits(events, [unsure], MIN) == events
    assert remove_rejected_exits(events, [confirmed], MIN) == events
