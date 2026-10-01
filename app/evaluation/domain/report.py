from dataclasses import dataclass

from app.evaluation.domain.metrics import (
    DurationError,
    EventScore,
    FailureCase,
    FrameMetrics,
)
from app.events.domain.bed_event import BedEventKind
from app.reporting.domain.formatting import clock
from app.shared.domain.states import ActivityState


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


@dataclass(frozen=True)
class EvaluationReport:
    evaluated_sec: float
    frames: FrameMetrics
    events: dict[BedEventKind, EventScore]
    event_tolerance_sec: float
    durations: dict[ActivityState, DurationError]
    failures: list[FailureCase]
    llm_calls: int
    total_tokens: int

    def to_dict(self) -> dict[str, object]:
        exits = self.events[BedEventKind.BED_EXIT]
        return {
            "evaluated_duration_sec": round(self.evaluated_sec, 2),
            "frames_evaluated": self.frames.n_frames,
            "frame_accuracy": _round(self.frames.accuracy),
            "per_class": {
                s.value: {
                    "precision": _round(c.precision),
                    "recall": _round(c.recall),
                    "support": c.support,
                }
                for s, c in self.frames.per_class.items()
            },
            "confusion_matrix": {
                t.value: {p.value: n for p, n in row.items()}
                for t, row in self.frames.confusion.items()
            },
            "events": {
                **{
                    kind.value: {
                        "precision": _round(score.precision),
                        "recall": _round(score.recall),
                        "matched": score.matched,
                        "predicted": score.predicted,
                        "ground_truth": score.truth,
                    }
                    for kind, score in self.events.items()
                },
                "false_bed_exits": exits.predicted - exits.matched,
                "tolerance_sec": self.event_tolerance_sec,
            },
            "durations": {
                s.value: {
                    "predicted_sec": round(d.predicted_sec, 2),
                    "ground_truth_sec": round(d.truth_sec, 2),
                    "abs_error_sec": round(d.abs_error_sec, 2),
                }
                for s, d in self.durations.items()
            },
            "total_abs_duration_error_sec": round(
                sum(d.abs_error_sec for d in self.durations.values()), 2
            ),
            "failure_cases": [
                {
                    "start_sec": f.start_sec,
                    "end_sec": f.end_sec,
                    "start_time": clock(f.start_sec),
                    "end_time": clock(f.end_sec),
                    "ground_truth": f.truth_state.value,
                    "predicted": f.predicted_state.value,
                    "reason": f.reason,
                }
                for f in self.failures
            ],
            "llm": {"calls": self.llm_calls, "total_tokens": self.total_tokens},
        }
