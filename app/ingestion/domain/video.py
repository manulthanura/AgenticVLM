from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class SampledFrame:
    index: int  # native frame index in the video
    time_sec: float  # seconds from video start
    image: np.ndarray  # BGR, already preprocessed


def keyframe_times(
    center: float, count: int, window_sec: float, duration_sec: float
) -> list[float]:
    """`count` evenly spaced times in the window around `center`, kept inside the video."""
    low, high = max(0.0, center - window_sec / 2), min(duration_sec, center + window_sec / 2)
    if count == 1:
        return [(low + high) / 2]
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def sample_stride(native_fps: float, target_fps: float) -> int:
    """Read every n-th native frame; never below 1 so low-fps videos are not upsampled."""
    return max(1, round(native_fps / target_fps))
