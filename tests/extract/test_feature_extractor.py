import numpy as np
import pytest

from app.extract.application.feature_extractor import FeatureExtractor
from app.extract.domain.bed import BedRegion, BedZone
from app.extract.domain.ports import PoseEstimator
from app.ingestion.domain.video import SampledFrame
from app.shared.config import Settings
from tests.extract.factories import person

BED = BedRegion([(0, 0), (200, 0), (200, 300), (0, 300)], edge_margin_px=10)


class ScriptedEstimator(PoseEstimator):
    """Returns one prepared list of people per call."""

    def __init__(self, *frames):
        self._frames = list(frames)

    def estimate(self, image):
        return self._frames.pop(0)


def run(*frames):
    extractor = FeatureExtractor(ScriptedEstimator(*frames), BED, Settings())
    image = np.zeros((1, 1, 3), np.uint8)
    return [extractor.extract(SampledFrame(i, i * 0.2, image)) for i in range(len(frames))]


def test_primary_is_person_on_bed_and_others_are_counted():
    on_bed, visitor = person(1, hip=(100, 150)), person(2, hip=(500, 150))
    f = run([visitor, on_bed])[0]
    assert f.observation.track_id == 1
    assert f.other_people == 1
    assert f.bed_zone == BedZone.INSIDE
    assert f.torso_angle_deg == pytest.approx(0)


def test_no_person_gives_empty_features():
    f = run([])[0]
    assert f.observation is None and f.other_people == 0


def test_speed_is_normalised_by_bbox_height():
    f = run([person(1, hip=(100, 150))], [person(1, hip=(123, 150))])[1]
    assert f.speed == pytest.approx(23 / 0.2 / 230)  # 23 px in 0.2 s, bbox 230 px tall


def test_relock_after_id_switch_only_when_nearby():
    switched = run([person(1, hip=(100, 150))], [person(7, hip=(110, 150))])[1]
    assert switched.observation.track_id == 7

    other_end = run([person(1, hip=(100, 150))], [person(8, hip=(900, 150))])[1]
    assert other_end.observation is None and other_end.other_people == 1


def test_speed_resets_after_tracking_gap():
    f = run([person(1, hip=(100, 150))], [], [person(1, hip=(150, 150))])[2]
    assert f.speed is None


def test_speed_is_measured_over_a_window_not_one_step():
    # steady 5 px per 0.2 s step = 25 px/s, at any frame rate; the extractor keeps ~1 s of history
    hips = [(100 + 5 * i, 150) for i in range(8)]
    f = run(*[[person(1, hip=h)] for h in hips])[-1]
    assert f.speed == pytest.approx(25 / 230)


def test_hip_jitter_does_not_read_as_walking():
    # +-8 px wobble every frame: a single step looks like 80 px/s, over 1 s the net movement is ~0
    hips = [(100 + (8 if i % 2 else -8), 150) for i in range(8)]
    f = run(*[[person(1, hip=h)] for h in hips])[-1]
    assert f.speed < Settings().walking_speed_min


def test_track_jump_does_not_count_as_walking_for_a_whole_window():
    # hips teleport 700 px (stitch cut / ID switch), then the person stands still
    still = [(800, 150)] * 4 + [(100, 150)] * 6
    frames = run(*[[person(1, hip=h)] for h in still])
    assert frames[4].speed is None  # history restarted at the jump
    assert all(f.speed < Settings().walking_speed_min for f in frames[5:])
