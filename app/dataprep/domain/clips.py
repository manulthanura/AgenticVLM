from dataclasses import dataclass


@dataclass(frozen=True)
class StitchedClip:
    name: str
    start_sec: float  # where this clip starts in the stitched video
    end_sec: float


def ground_truth_template(clips: list[StitchedClip]) -> str:
    """CSV skeleton with one row per clip and the state left empty.

    The ground-truth parser rejects empty states, so the template cannot be used by accident.
    Split a row when a clip contains several states.
    """
    lines = ["start_sec,end_sec,state"]
    for clip in clips:
        lines.append(f"# {clip.name}: one row per state, then fill in the state")
        lines.append(f"{clip.start_sec:.2f},{clip.end_sec:.2f},")
    return "\n".join(lines) + "\n"
