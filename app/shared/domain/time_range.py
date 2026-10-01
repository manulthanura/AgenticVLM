from dataclasses import dataclass


@dataclass(frozen=True)
class TimeRange:
    """Half-open interval [start, end) in float seconds from video start."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("end must be >= start")

    @property
    def duration(self) -> float:
        return self.end - self.start
