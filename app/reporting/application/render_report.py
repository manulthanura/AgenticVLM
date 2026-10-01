import json
from collections.abc import Sequence

from app.alerting.domain.decision import AlertDecision, overall_decision
from app.events.domain.bed_event import BedEvent
from app.reasoning.domain.models import AgentTrace
from app.reporting.application.raw_result import dump_raw_result
from app.reporting.domain.formatting import clock
from app.reporting.domain.summary import build_summary
from app.state.domain.models import Timeline


def _dump(data: object) -> str:
    return json.dumps(data, indent=2)


def _decision_for(event: BedEvent, decisions: list[AlertDecision]) -> AlertDecision:
    """The alerting rule for an event is named after its kind and fires at its confirmation time."""
    return next(
        d for d in decisions if d.rule == event.kind.value and d.time_sec == event.confirmed_time
    )


def _trace_row(trace: AgentTrace) -> dict[str, object]:
    finding = trace.finding
    return {
        "case": {
            "kind": trace.case.kind.value,
            "start_time": clock(trace.case.time_range.start),
            "end_time": clock(trace.case.time_range.end),
        },
        "steps": [
            {"tool": s.tool, "arguments": s.arguments, "observation": s.observation}
            for s in trace.steps
        ],
        "finding": {
            "state": finding.state.value if finding.state else None,
            "exit_confirmed": finding.exit_confirmed,
            "confidence": round(finding.confidence, 2),
            "rationale": finding.rationale,
        },
        "llm_calls": trace.llm_calls,
        "tokens": trace.tokens,
    }


def render_report(
    timeline: Timeline,
    events: list[BedEvent],
    decisions: list[AlertDecision],
    traces: Sequence[AgentTrace] = (),
) -> dict[str, str]:
    """File name -> file content. Pure, so it is testable without touching the disk."""
    lines = [
        f"{clock(s.time_range.start)} – {clock(s.time_range.end)} {s.state.name}"
        + (f"  [{s.reason}]" if s.reason else "")
        for s in timeline.segments
    ]
    event_rows = [
        {
            "event": e.kind.value,
            "start_time": clock(e.start_time),
            "confirmed_time": clock(e.confirmed_time),
            "previous_state": e.previous_state.value,
            "current_state": e.current_state.value,
            "confidence": round(e.confidence, 2),
            "decision": _decision_for(e, decisions).decision.value,
        }
        for e in events
    ]
    alert_rows = [
        {
            "time": clock(d.time_sec),
            "decision": d.decision.value,
            "rule": d.rule,
            "evidence": d.evidence,
        }
        for d in decisions
    ]
    return {
        "timeline.txt": "\n".join(lines) + "\n",
        "summary.json": _dump(build_summary(timeline, events).to_dict()),
        "events.json": _dump(event_rows),
        "alerts.json": _dump(
            {"overall_decision": overall_decision(decisions).value, "decisions": alert_rows}
        ),
        "raw_result.json": dump_raw_result(timeline, events),
        "agent_traces.json": _dump(
            {
                "llm_calls": sum(t.llm_calls for t in traces),  # real calls, cache hits excluded
                "total_tokens": sum(t.tokens for t in traces),
                "traces": [_trace_row(t) for t in traces],
            }
        ),
    }
