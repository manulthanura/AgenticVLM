import numpy as np

from app.extract.domain.observation import FrameFeatures
from app.ingestion.domain.ports import VideoSource
from app.state.domain.models import StateSegment, Timeline


class AgentTools:
    """What the agent can look at. Built per job from the first-pass results and the video."""

    def __init__(
        self, timeline: Timeline, features: list[FrameFeatures], video: VideoSource
    ) -> None:
        self._timeline, self._features, self._video = timeline, features, video

    def get_state_history(self, t_start: float, t_end: float) -> list[StateSegment]:
        """Smoothed segments overlapping the window."""
        return [
            s
            for s in self._timeline.segments
            if s.time_range.end > t_start and s.time_range.start < t_end
        ]

    def get_track_info(self, t: float) -> FrameFeatures | None:
        """Features of the frame nearest to `t`: track, bed overlap, other people."""
        return min(self._features, key=lambda f: abs(f.time_sec - t), default=None)

    def get_keyframes(self, t: float, n: int, window_sec: float) -> list[np.ndarray]:
        return [f.image for f in self._video.frames_around(t, n, window_sec)]
