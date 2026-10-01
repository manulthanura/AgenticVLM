import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.analysis.application.run_analysis import RunAnalysis
from app.analysis.domain.job import JobStatus
from app.analysis.infrastructure.job_store import FileJobStore
from app.extract.domain.bed import BedRegion
from app.extract.domain.ports import PoseEstimator
from app.ingestion.domain.ports import VideoSource
from app.ingestion.domain.video import SampledFrame, VideoMetadata
from app.main import create_app
from app.reporting.infrastructure.file_writer import write_files
from app.shared.config import Settings
from tests.extract.factories import person

FPS, SECONDS = 5, 10


class FakeSource(VideoSource):
    metadata = VideoMetadata(fps=FPS, frame_count=FPS * SECONDS, width=640, height=480)
    sample_interval_sec = 1 / FPS

    def frames(self):
        for i in range(self.metadata.frame_count):
            yield SampledFrame(i, i / FPS, np.zeros((1, 1, 3), np.uint8))

    def frames_around(self, time_sec, count, window_sec):
        return [SampledFrame(0, time_sec, np.zeros((1, 1, 3), np.uint8))] * count

    def frame_jpeg(self, time_sec, max_side):
        if time_sec > SECONDS:
            raise ValueError("past the end")
        return b"jpeg"


class LyingOnBed(PoseEstimator):
    def estimate(self, image):
        return [person(1, hip=(100.0, 150.0), lying=True)]


BED = BedRegion([(0, 0), (300, 0), (300, 300), (0, 300)], edge_margin_px=10)


def make_run(estimator=LyingOnBed, settings=None):
    return RunAnalysis(
        settings or Settings(),
        open_video=lambda path: FakeSource(),
        load_bed=lambda path: BED,
        make_estimator=estimator,
        write_report=write_files,
    )


def test_pipeline_end_to_end_with_fakes(tmp_path):
    job = FileJobStore(tmp_path).new_job()
    make_run().run(job)
    assert job.status == JobStatus.DONE and job.progress == 1.0
    summary = json.loads((job.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_state"] == "lying_in_bed"
    assert summary["activity_duration_sec"]["lying_in_bed"] == pytest.approx(SECONDS)
    # The en dash needs UTF-8; Windows' default encoding would garble it.
    assert (
        (job.output_dir / "timeline.txt")
        .read_text(encoding="utf-8")
        .startswith("00:00:00 – 00:00:10 LYING_IN_BED")
    )
    alerts = json.loads((job.output_dir / "alerts.json").read_text(encoding="utf-8"))
    assert alerts["overall_decision"] == "NORMAL"


def test_failure_is_recorded_on_the_job(tmp_path):
    def broken():
        raise RuntimeError("model missing")

    job = FileJobStore(tmp_path).new_job()
    make_run(estimator=broken).run(job)
    assert (job.status, job.error) == (JobStatus.FAILED, "model missing")


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), outputs_dir=str(tmp_path / "outputs"))
    (tmp_path / "data" / "videos").mkdir(parents=True)
    (tmp_path / "data" / "rois").mkdir(parents=True)
    (tmp_path / "data" / "videos" / "clip.mp4").write_bytes(b"x")
    (tmp_path / "data" / "rois" / "clip.json").write_text("{}")
    app = create_app(settings, make_run(settings=settings), open_video=lambda path: FakeSource())
    return TestClient(app)


def test_health(client):
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_analyze_by_path_then_read_results(client):
    job_id = client.post("/api/v1/analysis", data={"video_path": "clip.mp4"}).json()["job_id"]
    status = client.get(f"/api/v1/analysis/{job_id}").json()
    assert (status["status"], status["progress"]) == ("done", 1.0)
    assert client.get(f"/api/v1/analysis/{job_id}/summary").json()["final_state"] == "lying_in_bed"
    assert "LYING_IN_BED" in client.get(f"/api/v1/analysis/{job_id}/timeline").text
    assert client.get(f"/api/v1/analysis/{job_id}/events").json() == []
    assert client.get(f"/api/v1/analysis/{job_id}/alerts").json()["overall_decision"] == "NORMAL"
    assert client.get(f"/api/v1/analysis/{job_id}/traces").json() == {
        "llm_calls": 0,
        "total_tokens": 0,
        "traces": [],
    }


def test_analyze_by_upload(client):
    files = {"video": ("a.mp4", b"x"), "roi": ("roi.json", b"{}")}
    assert client.post("/api/v1/analysis", files=files).status_code == 202


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"data": {}}, 400),  # nothing given
        ({"data": {"video_path": "missing.mp4"}}, 400),
        ({"data": {"video_path": "../../etc/passwd"}}, 400),
        ({"files": {"video": ("a.mp4", b"x")}}, 400),  # upload without polygon
    ],
)
def test_bad_requests(client, kwargs, code):
    assert client.post("/api/v1/analysis", **kwargs).status_code == code


def test_unknown_job_is_404(client):
    assert client.get("/api/v1/analysis/nope").status_code == 404
    assert client.get("/api/v1/analysis/nope/summary").status_code == 404


def test_ui_is_served_at_root_and_hidden_from_swagger(client):
    page = client.get("/")
    assert page.status_code == 200 and "cdn.tailwindcss.com" in page.text
    assert "/" not in client.get("/openapi.json").json()["paths"]
    assert client.get("/docs").status_code == 200


def test_old_unversioned_paths_are_gone(client):
    assert client.get("/health").status_code == 404
    assert client.post("/analysis", data={"video_path": "clip.mp4"}).status_code == 404


def test_sources_lists_server_files(client):
    assert client.get("/api/v1/analysis/sources").json() == {
        "videos": ["clip.mp4"],
        "rois": ["clip.json"],
    }


def test_ui_endpoints_result_and_frame(client):
    job_id = client.post("/api/v1/analysis", data={"video_path": "clip.mp4"}).json()["job_id"]
    result = client.get(f"/api/v1/analysis/{job_id}/result").json()
    assert result["segments"][0]["state"] == "lying_in_bed"
    frame = client.get(f"/api/v1/analysis/{job_id}/frame?t=3")
    assert (frame.content, frame.headers["content-type"]) == (b"jpeg", "image/jpeg")
    assert client.get(f"/api/v1/analysis/{job_id}/frame?t=99").status_code == 404
    assert client.get(f"/api/v1/analysis/{job_id}/frame?t=-1").status_code == 422
    assert client.get("/api/v1/analysis/nope/frame").status_code == 404
