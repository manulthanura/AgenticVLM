import numpy as np
import pytest

from app.ingestion.domain.video import keyframe_times, sample_stride

cv2 = pytest.importorskip("cv2")

from app.ingestion.infrastructure.opencv_source import OpenCVVideoSource


def test_stride_never_below_one():
    assert sample_stride(30, 5) == 6
    assert sample_stride(3, 5) == 1


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(30):  # 3 s at 10 fps
        writer.write(np.full((48, 64, 3), 100, np.uint8))
    writer.release()
    return path


def test_samples_at_target_fps(video):
    source = OpenCVVideoSource(video, sample_fps=5)
    frames = list(source.frames())
    assert source.metadata.duration_sec == pytest.approx(3.0)
    assert source.sample_interval_sec == pytest.approx(0.2)
    assert len(frames) == 15
    assert frames[1].time_sec == pytest.approx(0.2)


def test_keyframe_times_are_evenly_spaced_and_stay_inside_the_video():
    assert keyframe_times(5, 3, 2, 10) == [4, 5, 6]
    assert keyframe_times(0.5, 3, 2, 10) == [0, 0.75, 1.5]  # window clipped at the start
    assert keyframe_times(5, 1, 2, 10) == [5]


def test_frames_around_returns_requested_number_near_the_time(video):
    frames = OpenCVVideoSource(video, sample_fps=5).frames_around(1.5, count=3, window_sec=1.0)
    assert [round(f.time_sec, 1) for f in frames] == [1.0, 1.5, 2.0]
    assert frames[0].image.shape == (48, 64, 3)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        OpenCVVideoSource(tmp_path / "nope.mp4", sample_fps=5)
