import csv

from app.shared.domain.states import ActivityState
from app.shared.domain.time_range import TimeRange
from app.state.domain.models import StateSegment, Timeline

_TOLERANCE_SEC = 0.01  # rounding slack when checking that rows touch


def parse_ground_truth(text: str) -> Timeline:
    """`start_sec,end_sec,state` rows -> a Timeline with confidence 1.0.

    Must start at 0 and be contiguous (no gaps or overlaps), so every second has a label.
    Blank lines, `#` comments and a header row are ignored.
    """
    rows = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    segments: list[StateSegment] = []
    expected_start = 0.0
    for n, row in enumerate(csv.reader(rows), start=1):
        if n == 1 and row[0].strip().lower() == "start_sec":
            continue  # header
        if len(row) < 3 or not row[2].strip():
            raise ValueError(f"row {n}: state is empty (fill in the template)")
        try:
            start, end = float(row[0]), float(row[1])
            state = ActivityState(row[2].strip().lower())
        except ValueError as exc:
            raise ValueError(f"row {n}: expected start_sec,end_sec,state ({exc})") from exc
        if end <= start:
            raise ValueError(f"row {n}: end_sec must be greater than start_sec")
        if abs(start - expected_start) > _TOLERANCE_SEC:
            raise ValueError(
                f"row {n}: starts at {start} but the previous row ended at {expected_start}"
                " (ground truth must be contiguous from 0)"
            )
        segments.append(StateSegment(TimeRange(start, end), state, 1.0))
        expected_start = end
    if not segments:
        raise ValueError("ground truth has no rows")
    return Timeline(tuple(segments))
