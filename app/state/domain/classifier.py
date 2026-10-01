from app.extract.domain.bed import BedZone
from app.extract.domain.observation import FrameFeatures
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.state.domain.models import (
    AWAY_FROM_BED,
    BED_EDGE,
    LYING_OUTSIDE_BED,
    NO_PERSON_DETECTED,
    RECLINED,
    StateCandidate,
)

# Only someone who was walking can plausibly have left the view (see allowed transitions).
_MAY_HAVE_LEFT_VIEW = {S.WALKING, S.OUT_OF_BED}


def classify(f: FrameFeatures, previous: S | None, settings: Settings) -> StateCandidate:
    """Rule-based single-frame state. Prefers UNKNOWN (with a reason) over a forced label."""
    person = f.observation
    if person is None:
        if previous in _MAY_HAVE_LEFT_VIEW:
            return StateCandidate(S.OUT_OF_BED, 1.0, "person_left_view")
        return StateCandidate(S.UNKNOWN, 1.0, NO_PERSON_DETECTED)

    confidence = min(person.confidence, f.visible_ratio)
    x1, y1, x2, y2 = person.bbox
    if f.bed_zone == BedZone.OUTSIDE and (x2 - x1) >= settings.lying_bbox_aspect_min * (y2 - y1):
        # Judged by the box, not the torso angle: lying toward the camera makes the torso look upright.
        return StateCandidate(S.UNKNOWN, confidence, LYING_OUTSIDE_BED)
    if f.visible_ratio < settings.visible_ratio_min or f.torso_angle_deg is None:
        return StateCandidate(S.UNKNOWN, confidence, "keypoints_not_visible")

    on_bed = f.bed_zone in (BedZone.INSIDE, BedZone.EDGE)
    if f.torso_angle_deg >= settings.lying_torso_angle_deg:
        if on_bed:
            return StateCandidate(S.LYING_IN_BED, confidence)
        return StateCandidate(S.UNKNOWN, confidence, LYING_OUTSIDE_BED)

    # Propped up on a pillow the torso is only ~40-50 deg off vertical. On the bed that is still resting;
    # outside the bed the 60 deg rule above stays, so leaning next to the bed is not called lying.
    if f.bed_zone == BedZone.INSIDE and f.torso_angle_deg >= settings.reclined_torso_angle_deg:
        return StateCandidate(S.LYING_IN_BED, confidence, RECLINED)

    if f.knee_angle_deg is None:  # legs hidden, typically under a blanket
        if on_bed:
            return StateCandidate(S.SITTING_ON_BED, confidence * 0.5, "legs_not_visible")
        return StateCandidate(S.UNKNOWN, confidence, "legs_not_visible")
    if f.knee_angle_deg < settings.seated_knee_max_deg:
        if not on_bed:
            return StateCandidate(S.SITTING_OUTSIDE_BED, confidence)
        edge = f.bed_zone == BedZone.EDGE
        return StateCandidate(S.SITTING_ON_BED, confidence, BED_EDGE if edge else "")

    if f.speed is None:  # first frame or after a tracking gap: cannot tell walking from standing
        return StateCandidate(S.STANDING, confidence * 0.5, "speed_unknown")
    walking = f.speed >= settings.walking_speed_min
    if walking:
        return StateCandidate(S.WALKING, confidence)
    # Hips clearly off the bed: evidence of a real bed exit even if the person never walks (e.g. at a wardrobe).
    away = f.bed_zone == BedZone.OUTSIDE
    return StateCandidate(S.STANDING, confidence, AWAY_FROM_BED if away else "")
