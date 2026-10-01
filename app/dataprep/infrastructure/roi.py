import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

Point = tuple[int, int]


def parse_points(text: str) -> list[Point]:
    """'100,200 500,200 500,400' -> [(100, 200), (500, 200), (500, 400)]"""
    points = [tuple(int(v) for v in pair.split(",")) for pair in text.split()]
    if len(points) < 3 or any(len(p) != 2 for p in points):
        raise ValueError("Give at least 3 points as 'x,y x,y x,y'")
    return points


def read_frame(video_path: Path, time_sec: float = 0.0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"Cannot read a frame at {time_sec}s from {video_path}")
    return image


def draw_roi_preview(image: np.ndarray, polygon: Sequence[Point]) -> np.ndarray:
    """Copy of the frame with the bed polygon shaded green, for checking it by eye."""
    canvas = image.copy()
    points = np.array(polygon, np.int32)
    if len(polygon) >= 3:
        shade = canvas.copy()
        cv2.fillPoly(shade, [points], (0, 200, 0))
        canvas = cv2.addWeighted(shade, 0.3, canvas, 0.7, 0)
    if len(polygon) >= 2:
        cv2.polylines(canvas, [points], len(polygon) >= 3, (0, 255, 0), 2)
    for point in polygon:
        cv2.circle(canvas, point, 4, (0, 0, 255), -1)
    return canvas


def pick_polygon(image: np.ndarray) -> list[Point]:
    """Click the bed corners in an OpenCV window. Enter = save, u = undo, Esc = cancel (returns [])."""
    title = "Click bed corners | Enter = save | u = undo | Esc = cancel"
    points: list[Point] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while True:
            cv2.imshow(title, draw_roi_preview(image, points))
            key = cv2.waitKey(30) & 0xFF
            if key in (10, 13) and len(points) >= 3:
                return points
            if key == ord("u") and points:
                points.pop()
            if key == 27:
                return []
    finally:
        cv2.destroyAllWindows()


def save_roi(path: Path, polygon: Sequence[Point], frame_size: tuple[int, int]) -> None:
    """Write `{"bed_polygon": [[x, y], ...], "frame_size": [w, h]}`"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"bed_polygon": [list(p) for p in polygon], "frame_size": list(frame_size)}
    path.write_text(json.dumps(data), encoding="utf-8")
