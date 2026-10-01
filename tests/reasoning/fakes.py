import numpy as np

from app.ingestion.domain.ports import VideoSource
from app.ingestion.domain.video import SampledFrame, VideoMetadata
from app.reasoning.domain.models import AmbiguityCase, VlmAnswer
from app.reasoning.domain.ports import VisionLanguageModel
from app.shared.domain.states import ActivityState


class FakeVideo(VideoSource):
    metadata = VideoMetadata(fps=5, frame_count=500, width=64, height=48)
    sample_interval_sec = 0.2

    def frames(self):
        return iter(())

    def frame_jpeg(self, time_sec, max_side):
        return b"jpeg"

    def frames_around(self, time_sec, count, window_sec):
        return [SampledFrame(0, time_sec, np.zeros((48, 64, 3), np.uint8))] * count


def answer(
    state=ActivityState.UNKNOWN, confidence=0.9, exit_confirmed=None, cached=False, tokens=100
):
    return VlmAnswer(state, exit_confirmed, confidence, "because", tokens, cached)


class ScriptedVlm(VisionLanguageModel):
    """Returns the prepared answers in order; an Exception in the list is raised instead."""

    def __init__(self, *script):
        self._script = list(script)
        self.calls: list[tuple[AmbiguityCase, bool]] = []

    def ask(self, frames, case, strong=False):
        self.calls.append((case, strong))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
