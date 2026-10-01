from collections.abc import Sequence
from pathlib import Path

import cv2

from app.dataprep.domain.clips import StitchedClip


def _copy_clip(path: Path, writer: cv2.VideoWriter, fps: float, size: tuple[int, int]) -> int:
    """Append one clip, resized to `size` and converted to `fps` by time. Returns frames written."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open clip: {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    written, index = 0, 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                return written
            if image.shape[1::-1] != size:
                image = cv2.resize(image, size)
            # Write this source frame for every output frame time it covers (repeats or skips as needed).
            while written / fps < (index + 1) / source_fps:
                writer.write(image)
                written += 1
            index += 1
    finally:
        cap.release()


def stitch_clips(paths: Sequence[Path], out_path: Path) -> list[StitchedClip]:
    """Concatenate clips into one mp4 (size and fps from the first clip); returns where each one sits."""
    first = cv2.VideoCapture(str(paths[0]))
    fps = first.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(first.get(cv2.CAP_PROP_FRAME_WIDTH)), int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    first.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError(f"Cannot write video: {out_path}")
    clips, total_frames = [], 0
    try:
        for path in paths:
            start = total_frames / fps
            total_frames += _copy_clip(path, writer, fps, size)
            clips.append(StitchedClip("/".join(path.parts[-3:]), start, total_frames / fps))
    finally:
        writer.release()
    return clips
