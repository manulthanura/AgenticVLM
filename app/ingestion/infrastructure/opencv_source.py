from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from app.ingestion.domain.ports import VideoSource
from app.ingestion.domain.video import (
    SampledFrame,
    VideoMetadata,
    keyframe_times,
    sample_stride,
)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Equalise local contrast on the lightness channel only, so colours stay intact (night clips)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class OpenCVVideoSource(VideoSource):
    def __init__(self, path: str | Path, sample_fps: float, clahe: bool = True) -> None:
        self._path = str(path)
        self._clahe = clahe
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self._metadata = VideoMetadata(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,  # some containers report 0
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        self._stride = sample_stride(self._metadata.fps, sample_fps)

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    @property
    def sample_interval_sec(self) -> float:
        return self._stride / self._metadata.fps

    def frames(self) -> Iterator[SampledFrame]:
        cap = cv2.VideoCapture(self._path)
        try:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    return
                if index % self._stride == 0:
                    yield self._sampled(index, image)
                index += 1
        finally:
            cap.release()

    def frames_around(self, time_sec: float, count: int, window_sec: float) -> list[SampledFrame]:
        meta = self._metadata
        cap = cv2.VideoCapture(self._path)
        frames = []
        try:
            for t in keyframe_times(time_sec, count, window_sec, meta.duration_sec):
                index = min(int(t * meta.fps), meta.frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if ok:
                    frames.append(self._sampled(index, image))
        finally:
            cap.release()
        return frames

    def frame_jpeg(self, time_sec: float, max_side: int) -> bytes:
        meta = self._metadata
        cap = cv2.VideoCapture(self._path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(time_sec * meta.fps), meta.frame_count - 1))
            ok, image = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f"No frame at {time_sec:.1f}s")
        scale = max_side / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Could not JPEG-encode frame")
        return buffer.tobytes()

    def _sampled(self, index: int, image: np.ndarray) -> SampledFrame:
        return SampledFrame(
            index, index / self._metadata.fps, apply_clahe(image) if self._clahe else image
        )
