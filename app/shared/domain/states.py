from enum import StrEnum


class ActivityState(StrEnum):
    LYING_IN_BED = "lying_in_bed"
    SITTING_ON_BED = "sitting_on_bed"
    SITTING_OUTSIDE_BED = "sitting_outside_bed"
    STANDING = "standing"
    WALKING = "walking"
    OUT_OF_BED = "out_of_bed"  # not visible after having left the bed
    UNKNOWN = "unknown"  # prefer this over forcing a classification


class Decision(StrEnum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    ALERT = "ALERT"


# Bed accounting: everything not listed here counts as out of bed (including UNKNOWN).
IN_BED_STATES = frozenset({ActivityState.LYING_IN_BED, ActivityState.SITTING_ON_BED})

_S = ActivityState
# Undirected pairs plus the one-way WALKING -> OUT_OF_BED / OUT_OF_BED -> ...
ALLOWED_TRANSITIONS: frozenset[tuple[ActivityState, ActivityState]] = frozenset(
    {
        (_S.LYING_IN_BED, _S.SITTING_ON_BED),
        (_S.SITTING_ON_BED, _S.LYING_IN_BED),
        (_S.SITTING_ON_BED, _S.STANDING),
        (_S.STANDING, _S.SITTING_ON_BED),
        (_S.SITTING_OUTSIDE_BED, _S.STANDING),
        (_S.STANDING, _S.SITTING_OUTSIDE_BED),
        (_S.STANDING, _S.WALKING),
        (_S.WALKING, _S.STANDING),
        (_S.WALKING, _S.OUT_OF_BED),
        (_S.OUT_OF_BED, _S.WALKING),
        (_S.OUT_OF_BED, _S.STANDING),
    }
)


def is_transition_allowed(src: ActivityState, dst: ActivityState) -> bool:
    if src == dst:
        return True
    if _S.UNKNOWN in (src, dst):  # any <-> UNKNOWN
        return True
    return (src, dst) in ALLOWED_TRANSITIONS
