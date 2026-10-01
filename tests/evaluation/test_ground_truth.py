import pytest

from app.evaluation.domain.ground_truth import parse_ground_truth
from app.shared.domain.states import ActivityState as S

GOOD = """start_sec,end_sec,state
# the person wakes up
0,20,LYING_IN_BED
20,25.5,sitting_on_bed

25.5,60,walking
"""


def test_parses_header_comments_blank_lines_and_state_case():
    timeline = parse_ground_truth(GOOD)
    assert [s.state for s in timeline.segments] == [S.LYING_IN_BED, S.SITTING_ON_BED, S.WALKING]
    assert timeline.duration == 60
    assert timeline.state_at(21) == S.SITTING_ON_BED


def test_header_is_optional():
    assert parse_ground_truth("0,5,standing").duration == 5


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no rows"),
        ("0,5,", "state is empty"),  # an unfilled template row
        ("0,5,flying", "row 1"),
        ("0,x,walking", "row 1"),
        ("5,10,walking", "contiguous from 0"),  # does not start at 0
        ("0,5,walking\n6,10,standing", "contiguous"),  # gap
        ("0,5,walking\n4,10,standing", "contiguous"),  # overlap
        ("0,0,walking", "greater than start_sec"),
    ],
)
def test_bad_ground_truth_is_rejected_with_a_helpful_message(text, message):
    with pytest.raises(ValueError, match=message):
        parse_ground_truth(text)
