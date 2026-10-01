from dataclasses import dataclass, field

from app.shared.domain.states import Decision

_SEVERITY = {Decision.NORMAL: 0, Decision.MONITOR: 1, Decision.ALERT: 2}


@dataclass(frozen=True)
class AlertDecision:
    decision: Decision
    rule: str  # which rule fired
    time_sec: float  # when it fired (for duration rules: the moment the limit was crossed)
    evidence: dict[str, object] = field(default_factory=dict)


def overall_decision(decisions: list[AlertDecision]) -> Decision:
    """The most severe decision wins; nothing fired means NORMAL."""
    return max((d.decision for d in decisions), key=_SEVERITY.__getitem__, default=Decision.NORMAL)
