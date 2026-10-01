import pytest

from app.extract.domain.observation import FrameFeatures
from app.reasoning.application.agent import ReasoningAgent
from app.reasoning.application.tools import AgentTools
from app.reasoning.domain.models import AmbiguityCase, CaseKind
from app.reasoning.domain.ports import VlmBudgetExceeded
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.shared.domain.time_range import TimeRange
from tests.extract.factories import person
from tests.reasoning.fakes import FakeVideo, ScriptedVlm, answer
from tests.state.factories import timeline

TIMELINE = timeline(
    (S.LYING_IN_BED, 20), (S.UNKNOWN, 15, "keypoints_not_visible"), (S.LYING_IN_BED, 20)
)
FEATURES = [FrameFeatures(i * 0.2, person(7), other_people=0) for i in range(275)]
CASE = AmbiguityCase(CaseKind.LOW_CONFIDENCE, TimeRange(20, 35))


def investigate(*script, case=CASE, **settings):
    vlm = ScriptedVlm(*script)
    agent = ReasoningAgent(vlm, AgentTools(TIMELINE, FEATURES, FakeVideo()), Settings(**settings))
    return agent.investigate(case), vlm


def test_tools_return_context_around_the_time():
    tools = AgentTools(TIMELINE, FEATURES, FakeVideo())
    assert [s.state for s in tools.get_state_history(18, 22)] == [S.LYING_IN_BED, S.UNKNOWN]
    assert tools.get_track_info(10.03).time_sec == pytest.approx(10.0)
    assert len(tools.get_keyframes(25, 4, 6)) == 4


def test_confident_first_answer_stops_after_three_steps():
    trace, vlm = investigate(answer(S.LYING_IN_BED, 0.9))
    assert [s.tool for s in trace.steps] == ["get_state_history", "get_track_info", "ask_vlm"]
    assert (trace.finding.state, trace.finding.confidence) == (S.LYING_IN_BED, 0.9)
    assert (trace.llm_calls, trace.tokens) == (1, 100)
    assert [strong for _, strong in vlm.calls] == [False]


def test_unsure_answer_escalates_once_to_the_strong_model():
    trace, vlm = investigate(answer(S.UNKNOWN, 0.3), answer(S.LYING_IN_BED, 0.9))
    assert len(trace.steps) == 4
    assert [strong for _, strong in vlm.calls] == [False, True]
    assert trace.finding.state == S.LYING_IN_BED
    assert (trace.llm_calls, trace.tokens) == (2, 200)


def test_still_unsure_after_strong_model_reports_low_confidence():
    trace, _ = investigate(answer(S.UNKNOWN, 0.3), answer(S.UNKNOWN, 0.4))
    assert trace.finding.confidence == pytest.approx(0.4)


def test_cache_hits_are_not_counted_as_calls():
    trace, _ = investigate(answer(S.LYING_IN_BED, 0.9, cached=True, tokens=0))
    assert (trace.llm_calls, trace.tokens) == (0, 0)


def test_step_limit_is_respected():
    trace, vlm = investigate(answer(), agent_max_steps=2)
    assert len(trace.steps) == 2 and vlm.calls == []
    assert (trace.finding.state, trace.finding.confidence, trace.finding.rationale) == (
        S.UNKNOWN,
        0.0,
        "no_evidence",
    )


def test_exhausted_budget_gives_a_zero_confidence_finding():
    trace, _ = investigate(VlmBudgetExceeded("cap"))
    assert trace.finding.confidence == 0.0
    assert trace.finding.rationale == "budget_exhausted"
    assert trace.steps[-1].observation == "call budget exhausted"


def test_vlm_errors_do_not_escape():
    trace, _ = investigate(RuntimeError("network down"))
    assert trace.finding.rationale == "vlm_error: network down"


def test_bed_exit_finding_carries_the_verdict_not_a_state():
    case = AmbiguityCase(CaseKind.BED_EXIT, TimeRange(20, 27))
    trace, _ = investigate(answer(S.UNKNOWN, 0.8, exit_confirmed=False), case=case)
    assert (trace.finding.state, trace.finding.exit_confirmed) == (None, False)
