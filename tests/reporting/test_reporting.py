import json

import pytest

from app.alerting.domain.rules import evaluate_alerts
from app.events.domain.detector import BedEventDetector
from app.reasoning.domain.models import AgentFinding, AgentStep, AgentTrace, AmbiguityCase, CaseKind
from app.reporting.application.render_report import render_report
from app.reporting.domain.formatting import clock, human
from app.reporting.domain.summary import build_summary
from app.reporting.infrastructure.file_writer import write_files
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.shared.domain.time_range import TimeRange
from tests.state.factories import timeline

SETTINGS = Settings()
# lying 5 min, sits up, stands, walks 40 s, comes back, lies down again
STORY = timeline(
    (S.LYING_IN_BED, 300),
    (S.SITTING_ON_BED, 8),
    (S.STANDING, 4),
    (S.WALKING, 40),
    (S.SITTING_ON_BED, 6),
    (S.LYING_IN_BED, 42),
)


def run(tl):
    events = BedEventDetector(SETTINGS).detect(tl)
    return events, evaluate_alerts(tl, events, SETTINGS)


def test_time_formatting():
    assert clock(308) == "00:05:08"
    assert human(702) == "11m 42s"
    assert human(59) == "59s"
    assert human(3723) == "1h 2m 3s"


def test_summary_sums_to_observation_duration():
    events, _ = run(STORY)
    summary = build_summary(STORY, events)
    assert sum(summary.activity_duration_sec.values()) == pytest.approx(
        summary.observation_duration_sec
    )
    assert summary.total_in_bed_sec + summary.total_out_of_bed_sec == pytest.approx(400)
    assert summary.total_in_bed_sec == pytest.approx(356)
    assert summary.longest_out_of_bed_period_sec == pytest.approx(44)
    assert (summary.bed_exit_count, summary.bed_return_count) == (1, 1)
    assert summary.final_state == S.LYING_IN_BED


def test_rendered_files_follow_the_brief():
    files = render_report(STORY, *run(STORY))
    assert files["timeline.txt"].splitlines()[0] == "00:00:00 – 00:05:00 LYING_IN_BED"
    exit_row = json.loads(files["events.json"])[0]
    assert exit_row == {
        "event": "bed_exit",
        "start_time": "00:05:08",
        "confirmed_time": "00:05:13",
        "previous_state": "sitting_on_bed",
        "current_state": "walking",
        "confidence": 0.9,
        "decision": "MONITOR",
    }
    alerts = json.loads(files["alerts.json"])
    assert alerts["overall_decision"] == "MONITOR"
    assert {d["rule"] for d in alerts["decisions"]} == {"bed_exit", "return_to_bed"}
    summary = json.loads(files["summary.json"])
    assert summary["final_state"] == "lying_in_bed"
    assert summary["longest_out_of_bed_period_human"] == "44s"


def test_empty_timeline_renders():
    files = render_report(timeline(), [], [])
    assert files["timeline.txt"] == "\n"
    assert json.loads(files["summary.json"])["final_state"] is None


def test_write_files(tmp_path):
    write_files(tmp_path / "job1", {"a.txt": "hello"})
    assert (tmp_path / "job1" / "a.txt").read_text() == "hello"


def test_agent_traces_are_rendered_with_usage_totals():
    case = AmbiguityCase(CaseKind.LOW_CONFIDENCE, TimeRange(20, 35))
    finding = AgentFinding(case, S.LYING_IN_BED, None, 0.9, "on the bed")
    step = AgentStep("ask_vlm", {"t": 27.5}, "state=lying_in_bed conf=0.90")
    trace = AgentTrace((step,), finding, llm_calls=1, tokens=420)
    rendered = json.loads(render_report(STORY, *run(STORY), traces=[trace])["agent_traces.json"])
    assert (rendered["llm_calls"], rendered["total_tokens"]) == (1, 420)
    (row,) = rendered["traces"]
    assert row["case"] == {
        "kind": "low_confidence",
        "start_time": "00:00:20",
        "end_time": "00:00:35",
    }
    assert row["steps"][0]["tool"] == "ask_vlm"
    assert row["finding"]["state"] == "lying_in_bed"


def test_no_agent_gives_empty_traces_file():
    rendered = json.loads(render_report(STORY, *run(STORY))["agent_traces.json"])
    assert rendered == {"llm_calls": 0, "total_tokens": 0, "traces": []}
