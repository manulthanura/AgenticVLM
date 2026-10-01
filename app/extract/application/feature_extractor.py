import math
from collections import deque

from app.extract.domain import geometry
from app.extract.domain.bed import BedRegion
from app.extract.domain.observation import FrameFeatures, PersonObservation
from app.extract.domain.ports import PoseEstimator
from app.ingestion.domain.video import SampledFrame
from app.shared.config import Settings


class FeatureExtractor:
    """Turns sampled frames into FrameFeatures for the primary person. Stateful: feed frames in order."""

    def __init__(self, estimator: PoseEstimator, bed: BedRegion, settings: Settings) -> None:
        self._estimator, self._bed, self._settings = estimator, bed, settings
        self._track_id: int | None = None
        self._last_position: tuple[float, float] | None = None
        self._last_height = 0.0
        self._hips: deque[tuple[float, float, float]] = deque()  # time, x, y over the speed window

    def extract(self, frame: SampledFrame) -> FrameFeatures:
        people = self._estimator.estimate(frame.image)
        person = self._pick_primary(people)
        if person is None:
            self._hips.clear()  # a speed across a gap in tracking would be meaningless
            return FrameFeatures(frame.time_sec, None, other_people=len(people))

        min_conf = self._settings.keypoint_conf_min
        kp = person.keypoints
        hip = geometry.midpoint(kp, geometry.HIPS, min_conf)
        x1, y1, x2, y2 = person.bbox
        self._last_position = (
            (float(hip[0]), float(hip[1])) if hip is not None else ((x1 + x2) / 2, (y1 + y2) / 2)
        )
        self._last_height = y2 - y1
        if hip is None:
            self._hips.clear()
        else:
            window = self._settings.speed_window_sec
            if self._hips and self._step_speed(frame.time_sec, hip) > self._settings.speed_jump_max:
                self._hips.clear()  # the track jumped: do not let the jump count as movement for a whole window
            self._hips.append((frame.time_sec, float(hip[0]), float(hip[1])))
            while len(self._hips) > 1 and frame.time_sec - self._hips[1][0] >= window:
                self._hips.popleft()
        speed = self._speed(hip)
        return FrameFeatures(
            time_sec=frame.time_sec,
            observation=person,
            other_people=len(people) - 1,
            visible_ratio=geometry.visible_ratio(kp, min_conf),
            torso_angle_deg=geometry.torso_angle_deg(kp, min_conf),
            knee_angle_deg=geometry.knee_angle_deg(kp, min_conf),
            hip_center=None if hip is None else (float(hip[0]), float(hip[1])),
            bed_zone=None if hip is None else self._bed.zone(*hip),
            bed_overlap=self._bed.overlap_ratio(person.bbox),
            speed=speed,
        )

    def _pick_primary(self, people: list[PersonObservation]) -> PersonObservation | None:
        if not people:
            return None
        if self._track_id is None:
            # First lock: the person on the bed; confidence breaks ties (e.g. nobody in bed yet).
            chosen = max(people, key=lambda p: (self._bed.overlap_ratio(p.bbox), p.confidence))
        else:
            chosen = next((p for p in people if p.track_id == self._track_id), None)
            chosen = chosen or self._nearest_to_last_position(people)
        if chosen is not None:
            self._track_id = chosen.track_id
        return chosen

    def _nearest_to_last_position(
        self, people: list[PersonObservation]
    ) -> PersonObservation | None:
        """Re-lock after a tracker ID switch, but only to someone close to where we last saw the person.

        A caregiver appearing elsewhere after the person walked out of view must not be adopted.
        """
        if self._last_position is None:
            return None
        lx, ly = self._last_position

        def distance(p: PersonObservation) -> float:
            x1, y1, x2, y2 = p.bbox
            return math.hypot((x1 + x2) / 2 - lx, (y1 + y2) / 2 - ly)

        nearest = min(people, key=distance)
        limit = self._settings.relock_max_dist_bbox_heights * self._last_height
        return nearest if distance(nearest) <= limit else None

    def _step_speed(self, time_sec: float, hip) -> float:
        t0, x0, y0 = self._hips[-1]
        dt = time_sec - t0
        if dt <= 0 or not self._last_height:
            return 0.0
        return math.hypot(hip[0] - x0, hip[1] - y0) / dt / self._last_height

    def _speed(self, hip) -> float | None:
        """Displacement over the speed window: one 0.2 s step is too noisy to separate walking from sway."""
        if hip is None or len(self._hips) < 2:
            return None
        t1, x1, y1 = self._hips[-1]
        t0, x0, y0 = self._hips[0]
        dt = t1 - t0
        # Normalising by bbox height makes speed independent of how far the person is from the camera.
        return (
            math.hypot(x1 - x0, y1 - y0) / dt / self._last_height
            if dt > 0 and self._last_height
            else None
        )
