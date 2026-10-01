from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.ingestion.domain.video import SampledFrame, VideoMetadata


class VideoSource(ABC):
    @property
    @abstractmethod
    def metadata(self) -> VideoMetadata: ...

    @property
    @abstractmethod
    def sample_interval_sec(self) -> float:
        """Time between two sampled frames (tolerance for the duration-sum invariant)."""

    @abstractmethod
    def frames(self) -> Iterator[SampledFrame]: ...

    @abstractmethod
    def frames_around(self, time_sec: float, count: int, window_sec: float) -> list[SampledFrame]:
        """`count` frames spread over the window centred on `time_sec` (for the reasoning agent)."""

    @abstractmethod
    def frame_jpeg(self, time_sec: float, max_side: int) -> bytes:
        """The unprocessed frame at `time_sec` as JPEG (for people to look at, e.g. the UI preview)."""
