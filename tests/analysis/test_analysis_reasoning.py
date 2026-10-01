import json

import pytest

from app.analysis.application.run_analysis import RunAnalysis
from app.analysis.domain.job import JobStatus
from app.analysis.infrastructure.job_store import FileJobStore
from app.extract.domain.bed import BedRegion
from app.reasoning.application.reasoner import Reasoner
from app.reporting.infrastructure.file_writer import write_files
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from tests.analysis.test_analysis import FakeSource, LyingOnBed
from tests.reasoning.fakes import ScriptedVlm, answer

# The person lies at x~100 but the bed is drawn far to the right: "lying outside the bed".
BED_ELSEWHERE = BedRegion([(500, 0), (800, 0), (800, 300), (500, 300)], edge_margin_px=10)
SETTINGS = Settings()


def run(tmp_path, vlm=None):
    analysis = RunAnalysis(
        SETTINGS,
        open_video=lambda path: FakeSource(),
        load_bed=lambda path: BED_ELSEWHERE,
        make_estimator=LyingOnBed,
        write_report=write_files,
        make_reasoner=(lambda: Reasoner(vlm, SETTINGS)) if vlm else None,
    )
    job = FileJobStore(tmp_path).new_job()
    analysis.run(job)
    assert job.status == JobStatus.DONE, job.error

    def read(name):
        return json.loads((job.output_dir / name).read_text(encoding="utf-8"))

    return read


def test_without_the_agent_lying_outside_the_bed_is_an_alert(tmp_path):
    read = run(tmp_path)
    alerts = read("alerts.json")
    assert alerts["overall_decision"] == "ALERT"
    assert alerts["decisions"][0]["rule"] == "lying_outside_bed"
    assert read("agent_traces.json")["traces"] == []


def test_agent_that_sees_the_bed_resolves_the_case_and_leaves_a_trace(tmp_path):
    vlm = ScriptedVlm(answer(S.LYING_IN_BED, 0.9))
    read = run(tmp_path, vlm)
    assert read("summary.json")["activity_duration_sec"]["lying_in_bed"] == pytest.approx(10)
    assert read("alerts.json")["overall_decision"] == "NORMAL"
    traces = read("agent_traces.json")
    assert (traces["llm_calls"], traces["total_tokens"]) == (1, 100)
    assert traces["traces"][0]["case"]["kind"] == "lying_outside_bed"
    assert traces["traces"][0]["finding"]["state"] == "lying_in_bed"


def test_agent_that_cannot_tell_keeps_the_alert_and_adds_a_monitor(tmp_path):
    vlm = ScriptedVlm(answer(S.UNKNOWN, 0.3), answer(S.UNKNOWN, 0.4))
    read = run(tmp_path, vlm)
    rules = {d["rule"] for d in read("alerts.json")["decisions"]}
    assert rules == {"lying_outside_bed", "agent_low_confidence"}
    assert read("alerts.json")["overall_decision"] == "ALERT"
