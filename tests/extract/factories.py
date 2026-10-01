import numpy as np

from app.extract.domain.observation import PersonObservation


def keypoints(hip=(100.0, 200.0), lying=False, seated=False, conf=1.0) -> np.ndarray:
    """Synthetic COCO keypoints: shoulders above hips (or beside them when lying), legs straight or bent."""
    hx, hy = hip
    kp = np.zeros((17, 3))
    kp[:, 2] = conf
    kp[[5, 6], :2] = (hx + 80, hy) if lying else (hx, hy - 80)
    kp[[11, 12], :2] = hip
    kp[[13, 14], :2] = (hx + 60, hy) if seated else (hx, hy + 60)
    kp[[15, 16], :2] = (hx + 60, hy + 60) if seated else (hx, hy + 120)
    return kp


def person(track_id, hip=(100.0, 200.0), **kw) -> PersonObservation:
    hx, hy = hip
    return PersonObservation(
        track_id, (hx - 40, hy - 100, hx + 40, hy + 130), 0.9, keypoints(hip, **kw)
    )
