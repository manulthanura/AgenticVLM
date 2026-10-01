from app.extract.domain.observation import FrameFeatures
from app.reasoning.application.tools import AgentTools
from app.reasoning.domain.models import (
    AgentFinding,
    AgentStep,
    AgentTrace,
    AmbiguityCase,
    CaseKind,
    VlmAnswer,
)
from app.reasoning.domain.ports import VisionLanguageModel, VlmBudgetExceeded
from app.shared.config import Settings
from app.shared.domain.states import ActivityState
from app.state.domain.models import StateSegment

_MIN_FRAME_WINDOW_SEC = 2.0  # frames closer together than this look identical to the model


def _describe_history(segments: list[StateSegment]) -> str:
    return "; ".join(
        f"{s.state.value} {s.time_range.start:.1f}-{s.time_range.end:.1f}s" for s in segments
    )


def _describe_track(f: FrameFeatures | None) -> str:
    if f is None or f.observation is None:
        return "nobody tracked" if f is None else f"nobody tracked, others={f.other_people}"
    return (
        f"track={f.observation.track_id} others={f.other_people} bed_zone={f.bed_zone} "
        f"bed_overlap={f.bed_overlap:.2f} visible={f.visible_ratio:.2f}"
    )


class ReasoningAgent:
    """Investigates one ambiguity case with a small, bounded loop of tool calls.

    The decision "is the context sufficient?" is a plain rule (confidence vs threshold); the VLM
    is only a perception tool. The agent never decides ALERT: it returns a finding.
    """

    def __init__(self, vlm: VisionLanguageModel, tools: AgentTools, settings: Settings) -> None:
        self._vlm, self._tools, self._settings = vlm, tools, settings

    def investigate(self, case: AmbiguityCase) -> AgentTrace:
        s, steps, answers = self._settings, [], []
        pad = s.agent_context_sec

        # Steps 1-2: cheap context first, no LLM.
        span = (case.time_range.start - pad, case.time_range.end + pad)
        history = self._tools.get_state_history(*span)
        steps.append(
            AgentStep(
                "get_state_history",
                {"t_start": span[0], "t_end": span[1]},
                _describe_history(history),
            )
        )
        info = self._tools.get_track_info(case.midpoint)
        steps.append(AgentStep("get_track_info", {"t": case.midpoint}, _describe_track(info)))

        # Step 3: look at the case with the cheap model.
        # Step 4: still unsure -> look forward in time, with a wider window, using the strong model.
        window = min(max(case.time_range.duration, _MIN_FRAME_WINDOW_SEC), pad)
        attempts = [(False, case.midpoint, window), (True, case.time_range.end, 2 * window)]
        failure = ""
        for strong, centre, width in attempts:
            if len(steps) >= s.agent_max_steps:
                break
            if answers and answers[-1].confidence >= s.agent_min_confidence:
                break  # context is sufficient
            arguments = {"t": centre, "window_sec": width, "strong": strong}
            try:
                frames = self._tools.get_keyframes(centre, s.agent_frames_per_call, width)
                if not frames:
                    failure = "no_frames"
                    steps.append(AgentStep("ask_vlm", arguments, "no frames available"))
                    break
                answer = self._vlm.ask(frames, case, strong=strong)
            except VlmBudgetExceeded:
                failure = "budget_exhausted"
                steps.append(AgentStep("ask_vlm", arguments, "call budget exhausted"))
                break
            except Exception as exc:  # noqa: BLE001 - enrichment must never fail the whole analysis
                failure = f"vlm_error: {exc}"
                steps.append(AgentStep("ask_vlm", arguments, failure))
                break
            answers.append(answer)
            steps.append(AgentStep("ask_vlm", arguments, self._describe_answer(answer)))

        return AgentTrace(
            steps=tuple(steps),
            finding=self._conclude(case, answers[-1] if answers else None, failure),
            llm_calls=sum(not a.cached for a in answers),
            tokens=sum(a.tokens for a in answers),
        )

    @staticmethod
    def _describe_answer(a: VlmAnswer) -> str:
        return f"state={a.state.value} exit_confirmed={a.exit_confirmed} conf={a.confidence:.2f}: {a.rationale}"

    @staticmethod
    def _conclude(case: AmbiguityCase, answer: VlmAnswer | None, failure: str) -> AgentFinding:
        if answer is None:  # nothing learned: say so with zero confidence, alerting will flag it
            state = None if case.kind == CaseKind.BED_EXIT else ActivityState.UNKNOWN
            return AgentFinding(case, state, None, 0.0, failure or "no_evidence")
        if case.kind == CaseKind.BED_EXIT:
            return AgentFinding(
                case, None, answer.exit_confirmed, answer.confidence, answer.rationale
            )
        return AgentFinding(case, answer.state, None, answer.confidence, answer.rationale)
