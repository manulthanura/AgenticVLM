from dataclasses import dataclass

import numpy as np

from app.extract.domain.bed import BedZone

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class PersonObservation:
    track_id: int | None
    bbox: BBox
    confidence: float
    keypoints: np.ndarray  # (17, 3): x, y, confidence (COCO order)


@dataclass(frozen=True)
class FrameFeatures:
    """Everything the state classifier may look at for one sampled frame.

    `observation` is the primary person; None means nobody is tracked in this frame.
    Optional numbers are None when the keypoints needed to compute them were not visible.
    """

    time_sec: float
    observation: PersonObservation | None
    other_people: int  # caregiver signal
    visible_ratio: float = 0.0  # share of keypoints above the confidence floor (occlusion signal)
    torso_angle_deg: float | None = None  # 0 = upright, 90 = horizontal
    knee_angle_deg: float | None = None  # ~180 = legs straight, ~90 = seated
    hip_center: tuple[float, float] | None = None
    bed_zone: BedZone | None = None
    bed_overlap: float = 0.0
    speed: float | None = None  # hip speed in bbox heights per second
