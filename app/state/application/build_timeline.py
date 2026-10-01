from collections.abc import Iterable

from app.extract.domain.observation import FrameFeatures
from app.shared.config import Settings
from app.state.domain.classifier import classify
from app.state.domain.models import Timeline
from app.state.domain.state_machine import StateMachine


def build_timeline(
    features: Iterable[FrameFeatures], end_time: float, settings: Settings
) -> Timeline:
    """`end_time` is the analysed duration, so the segments always sum to it."""
    machine = StateMachine(settings)
    for f in features:
        machine.update(f.time_sec, classify(f, machine.current, settings))
    return machine.finish(end_time)
