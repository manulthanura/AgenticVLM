from dataclasses import dataclass
from enum import StrEnum

from app.shared.domain.states import ActivityState


class BedEventKind(StrEnum):
    BED_EXIT = "bed_exit"
    RETURN_TO_BED = "return_to_bed"


@dataclass(frozen=True)
class BedEvent:
    kind: BedEventKind
    start_time: float  # when the change began (for an exit: when standing began)
    confirmed_time: float  # start + confirmation window, when we became sure
    previous_state: ActivityState
    current_state: ActivityState
    confidence: float
