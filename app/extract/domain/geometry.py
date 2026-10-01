import numpy as np

# COCO keypoint indices
SHOULDERS = (5, 6)
HIPS = (11, 12)
LEG_CHAINS = ((11, 13, 15), (12, 14, 16))  # hip, knee, ankle per side


def midpoint(kp: np.ndarray, indices: tuple[int, ...], min_conf: float) -> np.ndarray | None:
    """Mean of the visible keypoints among `indices` (one visible side is enough)."""
    points = [kp[i, :2] for i in indices if kp[i, 2] >= min_conf]
    return np.mean(points, axis=0) if points else None


def visible_ratio(kp: np.ndarray, min_conf: float) -> float:
    return float((kp[:, 2] >= min_conf).mean())


def torso_angle_deg(kp: np.ndarray, min_conf: float) -> float | None:
    shoulders, hips = midpoint(kp, SHOULDERS, min_conf), midpoint(kp, HIPS, min_conf)
    if shoulders is None or hips is None:
        return None
    dx, dy = np.abs(shoulders - hips)
    return float(np.degrees(np.arctan2(dx, dy)))


def knee_angle_deg(kp: np.ndarray, min_conf: float) -> float | None:
    angles = []
    for hip, knee, ankle in LEG_CHAINS:
        if all(kp[i, 2] >= min_conf for i in (hip, knee, ankle)):
            thigh, shin = kp[hip, :2] - kp[knee, :2], kp[ankle, :2] - kp[knee, :2]
            cos = thigh @ shin / (np.linalg.norm(thigh) * np.linalg.norm(shin) + 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return float(np.mean(angles)) if angles else None
