import pytest

from app.extract.domain import geometry
from app.extract.domain.bed import BedRegion, BedZone
from tests.extract.factories import keypoints

MIN_CONF = 0.3


def test_torso_angle_upright_vs_lying():
    assert geometry.torso_angle_deg(keypoints(), MIN_CONF) == pytest.approx(0)
    assert geometry.torso_angle_deg(keypoints(lying=True), MIN_CONF) == pytest.approx(90)


def test_knee_angle_straight_vs_seated():
    assert geometry.knee_angle_deg(keypoints(), MIN_CONF) == pytest.approx(180)
    assert geometry.knee_angle_deg(keypoints(seated=True), MIN_CONF) == pytest.approx(90)


def test_low_confidence_keypoints_give_none():
    kp = keypoints(conf=0.1)
    assert geometry.torso_angle_deg(kp, MIN_CONF) is None
    assert geometry.knee_angle_deg(kp, MIN_CONF) is None
    assert geometry.visible_ratio(kp, MIN_CONF) == 0


def test_bed_zones_and_overlap():
    bed = BedRegion([(0, 0), (200, 0), (200, 200), (0, 200)], edge_margin_px=10)
    assert bed.zone(100, 100) == BedZone.INSIDE
    assert bed.zone(195, 100) == BedZone.EDGE
    assert bed.zone(205, 100) == BedZone.EDGE  # just outside the border still counts as edge
    assert bed.zone(400, 100) == BedZone.OUTSIDE
    assert bed.overlap_ratio((0, 0, 100, 100)) == pytest.approx(1)
    assert bed.overlap_ratio((100, 0, 300, 100)) == pytest.approx(0.5)
