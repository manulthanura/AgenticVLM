import pytest

from app.alerting.domain.decision import overall_decision
from app.alerting.domain.rules import evaluate_alerts
from app.events.domain.detector import BedEventDetector
from app.reasoning.domain.models import AgentFinding, AmbiguityCase, CaseKind
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.shared.domain.states import Decision
from app.shared.domain.time_range import TimeRange
from app.state.domain.models import BED_EDGE, LYING_OUTSIDE_BED
from tests.state.factories import timeline

SETTINGS = Settings()  # edge 180, unknown 30, out-of-bed 600, out-of-view 300


def alerts(*runs: tuple[S, float] | tuple[S, float, str]):
    """`(state, seconds[, reason])` -> decisions from the whole pipeline after the state step."""
    tl = timeline(*runs)
    return evaluate_alerts(tl, BedEventDetector(SETTINGS).detect(tl), SETTINGS)


def rules(decisions):
    return {d.rule: d for d in decisions}


def test_calm_night_is_normal():
    decisions = alerts((S.LYING_IN_BED, 600))
    assert decisions == [] and overall_decision(decisions) == Decision.NORMAL


def test_bed_exit_is_monitor_and_return_is_normal():
    decisions = alerts(
        (S.LYING_IN_BED, 60), (S.STANDING, 3), (S.WALKING, 30), (S.SITTING_ON_BED, 20)
    )
    fired = rules(decisions)
    assert fired["bed_exit"].decision == Decision.MONITOR
    assert fired["return_to_bed"].decision == Decision.NORMAL
    assert overall_decision(decisions) == Decision.MONITOR


def test_lying_outside_bed_alerts_immediately():
    decisions = alerts((S.STANDING, 10), (S.UNKNOWN, 5, LYING_OUTSIDE_BED))
    fired = rules(decisions)["lying_outside_bed"]
    assert fired.decision == Decision.ALERT and fired.time_sec == 10
    assert "unknown_too_long" not in rules(decisions)  # not double-reported
    assert overall_decision(decisions) == Decision.ALERT


def test_out_of_bed_over_limit_alerts_when_limit_is_crossed():
    decisions = alerts((S.LYING_IN_BED, 60), (S.STANDING, 3), (S.WALKING, 700))
    fired = rules(decisions)["out_of_bed_too_long"]
    assert fired.decision == Decision.ALERT
    assert fired.time_sec == 60 + 600  # fires when the limit is crossed, counted from the run start
    assert fired.evidence["duration_sec"] == 703


def test_out_of_view_over_limit_alerts():
    fired = rules(alerts((S.WALKING, 10), (S.OUT_OF_BED, 301)))
    assert fired["out_of_view_too_long"].decision == Decision.ALERT
    assert "out_of_bed_too_long" not in fired  # 311 s < 600 s


def test_edge_sitting_only_counts_on_the_edge():
    assert "edge_sitting_too_long" in rules(alerts((S.SITTING_ON_BED, 181, BED_EDGE)))
    assert alerts((S.SITTING_ON_BED, 181)) == []


def test_long_unknown_is_monitor_and_pure_unknown_is_not_out_of_bed():
    decisions = alerts((S.LYING_IN_BED, 20), (S.UNKNOWN, 700, "keypoints_not_visible"))
    fired = rules(decisions)
    assert fired["unknown_too_long"].decision == Decision.MONITOR
    assert "out_of_bed_too_long" not in fired
    assert overall_decision(decisions) == Decision.MONITOR


def test_evidence_records_times_states_and_confidence():
    evidence = rules(alerts((S.SITTING_ON_BED, 200, BED_EDGE)))["edge_sitting_too_long"].evidence
    assert evidence["states"] == ["sitting_on_bed"]
    assert (evidence["start_sec"], evidence["end_sec"], evidence["limit_sec"]) == (0.0, 200.0, 180)
    assert evidence["confidence"] == pytest.approx(0.9)


def test_low_confidence_agent_finding_is_monitor_but_a_confident_one_is_silent():
    tl = timeline((S.LYING_IN_BED, 20))
    case = AmbiguityCase(CaseKind.TRACK_LOST, TimeRange(5, 9))
    unsure = AgentFinding(case, S.UNKNOWN, None, 0.3, "cannot tell")
    sure = AgentFinding(case, S.LYING_IN_BED, None, 0.9, "on the bed")
    fired = rules(evaluate_alerts(tl, [], SETTINGS, [unsure]))["agent_low_confidence"]
    assert (fired.decision, fired.time_sec) == (Decision.MONITOR, 5)
    assert fired.evidence["rationale"] == "cannot tell"
    assert evaluate_alerts(tl, [], SETTINGS, [sure]) == []
