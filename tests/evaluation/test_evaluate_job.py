import json

import pytest
from fastapi.testclient import TestClient

from app.evaluation.application.evaluate import evaluate_texts, render_failure_cases_md
from app.evaluation.infrastructure.job_evaluator import evaluate_job_dir
from app.main import create_app
from app.reporting.application.raw_result import dump_raw_result, load_raw_result
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from tests.analysis.test_analysis import make_run
from tests.reasoning.fakes import FakeVideo
from tests.state.factories import timeline

pytest.importorskip("cv2")

SETTINGS = Settings()
PREDICTED = timeline((S.LYING_IN_BED, 10), (S.UNKNOWN, 10, "keypoints_not_visible"))
GROUND_TRUTH = "start_sec,end_sec,state\n0,10,lying_in_bed\n10,20,sitting_on_bed\n"


def test_raw_result_round_trips():
    loaded, events = load_raw_result(dump_raw_result(PREDICTED, []))
    assert loaded == PREDICTED and events == []


def test_evaluate_texts_reads_usage_from_the_traces_file():
    traces = json.dumps({"llm_calls": 3, "total_tokens": 900, "traces": []})
    report = evaluate_texts(GROUND_TRUTH, dump_raw_result(PREDICTED, []), traces, SETTINGS)
    assert report.to_dict()["llm"] == {"calls": 3, "total_tokens": 900}
    assert report.to_dict()["frame_accuracy"] == 0.5


def test_failure_case_markdown_has_times_reason_and_a_todo():
    report = evaluate_texts(GROUND_TRUTH, dump_raw_result(PREDICTED, []), "", SETTINGS)
    text = render_failure_cases_md(report, {1: "failure_frames/case_1.jpg"})
    assert "00:00:10 – 00:00:20" in text and "keypoints_not_visible" in text
    assert "![case 1](failure_frames/case_1.jpg)" in text and "_TODO_" in text


def test_evaluate_job_dir_writes_report_markdown_and_a_frame(tmp_path):
    (tmp_path / "raw_result.json").write_text(dump_raw_result(PREDICTED, []), encoding="utf-8")
    result = evaluate_job_dir(tmp_path, GROUND_TRUTH, FakeVideo(), SETTINGS)
    assert result["frame_accuracy"] == 0.5
    assert json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8")) == result
    assert "case_1.jpg" in (tmp_path / "failure_cases.md").read_text(encoding="utf-8")
    assert (tmp_path / "failure_frames" / "case_1.jpg").exists()


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), outputs_dir=str(tmp_path / "outputs"))
    for folder in ("videos", "rois", "ground_truth"):
        (tmp_path / "data" / folder).mkdir(parents=True)
    (tmp_path / "data" / "videos" / "clip.mp4").write_bytes(
        b"x"
    )  # not a real video: no frames saved
    (tmp_path / "data" / "rois" / "clip.json").write_text("{}")
    return TestClient(create_app(settings, make_run(settings=settings)))


def finished_job(client):
    return client.post("/api/v1/analysis", data={"video_path": "clip.mp4"}).json()[
        "job_id"
    ]  # 10 s lying


def upload(csv_text):
    return {"ground_truth": ("gt.csv", csv_text.encode())}


def test_api_evaluates_against_an_uploaded_csv(client):
    job_id = finished_job(client)
    ok = client.post(f"/api/v1/evaluation/{job_id}", files=upload("0,10,lying_in_bed"))
    assert ok.status_code == 200 and ok.json()["frame_accuracy"] == 1.0
    wrong = client.post(f"/api/v1/evaluation/{job_id}", files=upload("0,10,walking")).json()
    assert (
        wrong["frame_accuracy"] == 0.0 and wrong["failure_cases"][0]["predicted"] == "lying_in_bed"
    )


def test_api_uses_default_ground_truth_file_when_nothing_is_uploaded(client, tmp_path):
    job_id = finished_job(client)
    assert client.post(f"/api/v1/evaluation/{job_id}").status_code == 400
    (tmp_path / "data" / "ground_truth" / "clip.csv").write_text("0,10,lying_in_bed")
    assert client.post(f"/api/v1/evaluation/{job_id}").json()["frame_accuracy"] == 1.0


def test_api_errors(client):
    assert client.post("/api/v1/evaluation/nope", files=upload("0,10,walking")).status_code == 404
    job_id = finished_job(client)
    assert client.post(f"/api/v1/evaluation/{job_id}", files=upload("0,10,")).status_code == 422
