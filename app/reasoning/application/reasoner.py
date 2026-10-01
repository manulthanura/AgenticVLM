from dataclasses import dataclass

from app.events.domain.bed_event import BedEvent
from app.events.domain.detector import BedEventDetector
from app.extract.domain.observation import FrameFeatures
from app.ingestion.domain.ports import VideoSource
from app.reasoning.application.agent import ReasoningAgent
from app.reasoning.application.apply_findings import apply_findings, remove_rejected_exits
from app.reasoning.application.tools import AgentTools
from app.reasoning.domain.cases import find_cases
from app.reasoning.domain.models import AgentTrace
from app.reasoning.domain.ports import VisionLanguageModel
from app.shared.config import Settings
from app.state.domain.models import Timeline


@dataclass(frozen=True)
class ReasoningResult:
    timeline: Timeline  # revised: resolved UNKNOWN segments
    events: list[BedEvent]  # re-detected, rejected exits removed
    traces: list[AgentTrace]


class Reasoner:
    """Second pass: find ambiguous cases, investigate them, fold the findings back in."""

    def __init__(self, vlm: VisionLanguageModel, settings: Settings) -> None:
        self._vlm, self._settings = vlm, settings

    def run(
        self,
        timeline: Timeline,
        events: list[BedEvent],
        features: list[FrameFeatures],
        video: VideoSource,
    ) -> ReasoningResult:
        cases = find_cases(timeline, events, features, self._settings)
        if not cases:
            return ReasoningResult(timeline, events, [])
        agent = ReasoningAgent(self._vlm, AgentTools(timeline, features, video), self._settings)
        traces = [agent.investigate(case) for case in cases]
        findings = [t.finding for t in traces]
        min_confidence = self._settings.agent_min_confidence
        revised = apply_findings(timeline, findings, min_confidence)
        revised_events = remove_rejected_exits(
            BedEventDetector(self._settings).detect(revised), findings, min_confidence
        )
        return ReasoningResult(revised, revised_events, traces)
