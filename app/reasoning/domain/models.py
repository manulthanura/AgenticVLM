from dataclasses import dataclass
from enum import StrEnum

from app.shared.domain.states import ActivityState
from app.shared.domain.time_range import TimeRange

RESOLVED_BY_AGENT = "resolved_by_agent"  # segment reason after a finding relabelled it


class CaseKind(StrEnum):
    """Why the agent is called. Declaration order is priority: the fall question comes first."""

    LYING_OUTSIDE_BED = "lying_outside_bed"  # bed or floor?
    BED_EXIT = "bed_exit"  # really left, or stood and sat back?
    TRACK_LOST = "track_lost"  # walked out of view, or occluded?
    MULTIPLE_PERSONS = "multiple_persons"  # caregiver? which one is the patient?
    LOW_CONFIDENCE = "low_confidence"  # what is the person doing?


@dataclass(frozen=True)
class AmbiguityCase:
    kind: CaseKind
    time_range: TimeRange

    @property
    def midpoint(self) -> float:
        return (self.time_range.start + self.time_range.end) / 2


@dataclass(frozen=True)
class VlmAnswer:
    """What the vision-language model said about the frames (already validated)."""

    state: ActivityState  # UNKNOWN if it cannot tell
    exit_confirmed: bool | None  # only for BED_EXIT questions
    confidence: float
    rationale: str
    tokens: int = 0  # 0 for cache hits
    cached: bool = False


@dataclass(frozen=True)
class AgentStep:
    tool: str
    arguments: dict[str, object]
    observation: str  # short summary of what the tool returned


@dataclass(frozen=True)
class AgentFinding:
    case: AmbiguityCase
    state: ActivityState | None  # resolved state, None when the question is about an event
    exit_confirmed: bool | None
    confidence: float
    rationale: str


@dataclass(frozen=True)
class AgentTrace:
    """Evidence of the agent's reasoning for one case; saved with the job."""

    steps: tuple[AgentStep, ...]
    finding: AgentFinding
    llm_calls: int  # real calls only, cache hits excluded
    tokens: int

    @property
    def case(self) -> AmbiguityCase:
        return self.finding.case
