from app.shared.domain.states import ActivityState as S
from app.shared.domain.states import is_transition_allowed


def test_lying_to_walking_is_invalid():
    assert not is_transition_allowed(S.LYING_IN_BED, S.WALKING)


def test_sitting_to_standing_is_allowed_both_ways():
    assert is_transition_allowed(S.SITTING_ON_BED, S.STANDING)
    assert is_transition_allowed(S.STANDING, S.SITTING_ON_BED)


def test_any_state_can_go_through_unknown():
    assert is_transition_allowed(S.LYING_IN_BED, S.UNKNOWN)
    assert is_transition_allowed(S.UNKNOWN, S.WALKING)
