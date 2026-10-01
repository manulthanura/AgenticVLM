import json

from app.evaluation.domain.ground_truth import parse_ground_truth
from app.evaluation.domain.metrics import (
    duration_errors,
    evaluated_duration,
    find_failures,
    frame_metrics,
    match_events,
)
from app.evaluation.domain.report import EvaluationReport
from app.events.domain.bed_event import BedEvent
from app.events.domain.detector import BedEventDetector
from app.reporting.application.raw_result import load_raw_result
from app.reporting.domain.formatting import clock, human
from app.shared.config import Settings
from app.state.domain.models import Timeline


def evaluate(
    truth: Timeline,
    predicted: Timeline,
    predicted_events: list[BedEvent],
    settings: Settings,
    llm_calls: int = 0,
    total_tokens: int = 0,
) -> EvaluationReport:
    # Ground-truth events are derived with the same detector, so both sides follow the same event rules.
    truth_events = BedEventDetector(settings).detect(truth)
    step = settings.evaluation_sample_sec
    return EvaluationReport(
        evaluated_sec=evaluated_duration(truth, predicted),
        frames=frame_metrics(truth, predicted, step),
        events=match_events(predicted_events, truth_events, settings.event_match_tolerance_sec),
        event_tolerance_sec=settings.event_match_tolerance_sec,
        durations=duration_errors(truth, predicted),
        failures=find_failures(truth, predicted, step, settings.failure_cases_max),
        llm_calls=llm_calls,
        total_tokens=total_tokens,
    )


def evaluate_texts(
    ground_truth_csv: str, raw_result_json: str, traces_json: str, settings: Settings
) -> EvaluationReport:
    """Same as `evaluate`, from the ground-truth CSV and the job's raw_result.json / agent_traces.json."""
    predicted, events = load_raw_result(raw_result_json)
    usage = json.loads(traces_json) if traces_json else {}
    return evaluate(
        parse_ground_truth(ground_truth_csv),
        predicted,
        events,
        settings,
        usage.get("llm_calls", 0),
        usage.get("total_tokens", 0),
    )


def render_failure_cases_md(report: EvaluationReport, frame_files: dict[int, str]) -> str:
    """Draft of docs/failure-cases.md: facts filled in, the 'why' left for the author to write."""
    lines = [
        "# Failure cases",
        "",
        "Extracted automatically from an evaluation run. Write the analysis under each case.",
        "",
    ]
    for i, f in enumerate(report.failures, start=1):
        lines += [
            f"## {i}. {clock(f.start_sec)} – {clock(f.end_sec)} ({human(f.duration)})",
            f"- Ground truth: `{f.truth_state.value}`",
            f"- Predicted: `{f.predicted_state.value}`",
            f"- System's own reason: {f.reason or 'none recorded'}",
        ]
        if i in frame_files:
            lines.append(f"- Frame: ![case {i}]({frame_files[i]})")
        lines += ["- Why it failed: _TODO_", ""]
    if not report.failures:
        lines.append("No mismatching seconds in the evaluated span.")
    return "\n".join(lines) + "\n"
