import pytest

from app.extract.domain.bed import BedZone
from app.extract.domain.observation import FrameFeatures, PersonObservation
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.state.domain.classifier import classify
from tests.extract.factories import person

SETTINGS = Settings()


def frame(zone=BedZone.INSIDE, torso=0.0, knee=170.0, speed=0.0, visible=1.0, t=0.0):
    return FrameFeatures(t, person(1), 0, visible, torso, knee, (0, 0), zone, 1.0, speed)


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (frame(torso=85, knee=None), S.LYING_IN_BED),
        (frame(torso=10, knee=100), S.SITTING_ON_BED),
        (frame(zone=BedZone.EDGE, torso=10, knee=100), S.SITTING_ON_BED),
        (frame(zone=BedZone.OUTSIDE, torso=10, knee=100), S.SITTING_OUTSIDE_BED),
        (frame(zone=BedZone.OUTSIDE, speed=0.05), S.STANDING),
        (frame(zone=BedZone.OUTSIDE, speed=0.8), S.WALKING),
    ],
)
def test_each_state(features, expected):
    assert classify(features, None, SETTINGS).state == expected


def test_lying_outside_bed_is_unknown_with_fall_reason():
    c = classify(frame(zone=BedZone.OUTSIDE, torso=85), None, SETTINGS)
    assert (c.state, c.reason) == (S.UNKNOWN, "lying_outside_bed")


def test_occlusion_is_unknown():
    assert classify(frame(visible=0.2), None, SETTINGS).state == S.UNKNOWN


def test_legs_hidden_under_blanket_is_low_confidence_sitting():
    c = classify(frame(torso=10, knee=None), None, SETTINGS)
    assert c.state == S.SITTING_ON_BED and c.confidence < SETTINGS.state_confidence_min


def test_missing_person_depends_on_previous_state():
    gone = FrameFeatures(0.0, None, 0)
    assert classify(gone, S.WALKING, SETTINGS).state == S.OUT_OF_BED
    assert classify(gone, S.LYING_IN_BED, SETTINGS).state == S.UNKNOWN


def test_reclined_on_bed_is_lying():
    c = classify(frame(torso=45, knee=150), None, SETTINGS)
    assert (c.state, c.reason) == (S.LYING_IN_BED, "reclined")


def test_leaning_next_to_the_bed_is_not_lying():
    assert (
        classify(frame(zone=BedZone.OUTSIDE, torso=45, knee=150), None, SETTINGS).state
        != S.LYING_IN_BED
    )
    assert (
        classify(frame(zone=BedZone.EDGE, torso=45, knee=150), None, SETTINGS).state
        != S.LYING_IN_BED
    )


def test_standing_outside_the_bed_is_marked_away():
    assert (
        classify(frame(zone=BedZone.OUTSIDE, speed=0.05), None, SETTINGS).reason == "away_from_bed"
    )
    assert classify(frame(zone=BedZone.EDGE, speed=0.05), None, SETTINGS).reason == ""


def wide(zone, torso=0.0, knee=100.0):
    """Box wider than tall (300 x 250 px), e.g. someone lying toward the camera."""
    p = person(1)
    box = PersonObservation(1, (100, 400, 400, 650), p.confidence, p.keypoints)
    return FrameFeatures(0.0, box, 0, 1.0, torso, knee, (0, 0), zone, 1.0, 0.0)


def test_wide_box_outside_the_bed_is_lying_outside_bed_even_if_torso_looks_upright():
    c = classify(wide(BedZone.OUTSIDE), None, SETTINGS)
    assert (c.state, c.reason) == (S.UNKNOWN, "lying_outside_bed")


def test_wide_box_on_the_bed_is_not_a_fall():
    assert classify(wide(BedZone.INSIDE), None, SETTINGS).reason != "lying_outside_bed"
    assert classify(wide(BedZone.EDGE), None, SETTINGS).reason != "lying_outside_bed"


def test_tall_box_outside_the_bed_is_not_lying():
    assert (
        classify(frame(zone=BedZone.OUTSIDE, torso=5, knee=100), None, SETTINGS).state
        == S.SITTING_OUTSIDE_BED
    )
