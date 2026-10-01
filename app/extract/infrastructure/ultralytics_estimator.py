import numpy as np
from ultralytics import YOLO

from app.extract.domain.observation import PersonObservation
from app.extract.domain.ports import PoseEstimator


class UltralyticsPoseEstimator(PoseEstimator):
    """YOLO pose + built-in ByteTrack. `persist=True` keeps track IDs across successive calls."""

    def __init__(self, model_name: str, detection_conf: float) -> None:
        self._model = YOLO(model_name)  # downloads weights on first use
        self._conf = detection_conf

    def estimate(self, image: np.ndarray) -> list[PersonObservation]:
        result = self._model.track(
            image, persist=True, tracker="bytetrack.yaml", conf=self._conf, verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        ids = (
            result.boxes.id.int().tolist()
            if result.boxes.id is not None
            else [None] * len(result.boxes)
        )
        return [
            PersonObservation(
                track_id=track_id,
                bbox=tuple(box.tolist()),
                confidence=float(conf),
                keypoints=kp.cpu().numpy(),
            )
            for track_id, box, conf, kp in zip(
                ids, result.boxes.xyxy, result.boxes.conf, result.keypoints.data, strict=True
            )
        ]
