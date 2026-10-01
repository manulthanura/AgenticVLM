import numpy as np
import pytest

from app.dataprep.domain.clips import StitchedClip, ground_truth_template
from app.evaluation.domain.ground_truth import parse_ground_truth
from app.shared.domain.states import ActivityState as S

cv2 = pytest.importorskip("cv2")

from app.dataprep.infrastructure.roi import (
    draw_roi_preview,
    parse_points,
    read_frame,
    save_roi,
)
from app.dataprep.infrastructure.stitcher import stitch_clips
from app.extract.infrastructure.roi_loader import load_bed_region


def make_clip(path, fps, size, seconds, value):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(int(fps * seconds)):
        writer.write(np.full((size[1], size[0], 3), value, np.uint8))
    writer.release()
    return path


def duration(path):
    cap = cv2.VideoCapture(str(path))
    seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return seconds, size


def test_stitch_joins_clips_of_different_size_and_fps(tmp_path):
    a = make_clip(tmp_path / "a.mp4", 10, (64, 48), 2, 50)
    b = make_clip(tmp_path / "b.mp4", 5, (32, 24), 3, 200)  # other size and fps
    clips = stitch_clips([a, b], tmp_path / "out" / "seq.mp4")
    assert [(round(c.start_sec, 1), round(c.end_sec, 1)) for c in clips] == [(0, 2), (2, 5)]
    seconds, size = duration(tmp_path / "out" / "seq.mp4")
    assert seconds == pytest.approx(5, abs=0.2) and size == (64, 48)
    # the second clip is in the second part of the stitched video, resized to the first clip's size
    assert (
        read_frame(tmp_path / "out" / "seq.mp4", 4.0).mean()
        > read_frame(tmp_path / "out" / "seq.mp4", 1.0).mean()
    )


def test_stitch_reports_missing_clips(tmp_path):
    a = make_clip(tmp_path / "a.mp4", 10, (64, 48), 1, 50)
    with pytest.raises(FileNotFoundError):
        stitch_clips([a, tmp_path / "missing.mp4"], tmp_path / "seq.mp4")


def test_ground_truth_template_is_valid_only_after_filling_it_in():
    clips = [StitchedClip("s3/ADL/03.mp4", 0.0, 9.9), StitchedClip("s3/ADL/18.mp4", 9.9, 21.4)]
    template = ground_truth_template(clips)
    assert "# s3/ADL/03.mp4" in template
    with pytest.raises(ValueError, match="state is empty"):
        parse_ground_truth(template)
    filled = template.replace("0.00,9.90,", "0.00,9.90,lying_in_bed").replace(
        "9.90,21.40,", "9.90,21.40,walking"
    )
    assert [s.state for s in parse_ground_truth(filled).segments] == [S.LYING_IN_BED, S.WALKING]


def test_parse_points():
    assert parse_points("1,2 3,4 5,6") == [(1, 2), (3, 4), (5, 6)]
    for bad in ("1,2 3,4", "1,2,3 4,5 6,7", "a,b c,d e,f"):
        with pytest.raises(ValueError):
            parse_points(bad)


def test_saved_roi_loads_as_a_bed_region_and_preview_marks_the_polygon(tmp_path):
    polygon = [(10, 10), (90, 10), (90, 60), (10, 60)]
    save_roi(tmp_path / "rois" / "clip.json", polygon, (100, 80))
    bed = load_bed_region(tmp_path / "rois" / "clip.json", edge_margin_px=5)
    assert bed.zone(50, 35).value == "inside" and bed.zone(300, 300).value == "outside"

    frame = np.zeros((80, 100, 3), np.uint8)
    preview = draw_roi_preview(frame, polygon)
    assert preview.shape == frame.shape and preview[35, 50, 1] > 0  # green shading inside the bed
    assert preview[75, 5].sum() == 0  # untouched outside
    assert frame.sum() == 0  # the original is not modified
