import pytest

from app.evaluation.application.evaluate import evaluate
from app.evaluation.domain.metrics import (
    duration_errors,
    find_failures,
    frame_metrics,
    match_events,
)
from app.events.domain.bed_event import BedEventKind
from app.events.domain.detector import BedEventDetector
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from tests.state.factories import timeline

SETTINGS = Settings()  # events matched within 5 s, sampled every 1 s
TRUTH = timeline((S.LYING_IN_BED, 10), (S.SITTING_ON_BED, 10))


def events(*runs):
    return BedEventDetector(SETTINGS).detect(timeline(*runs))


def test_perfect_prediction():
    report = evaluate(TRUTH, TRUTH, [], SETTINGS).to_dict()
    assert report["frame_accuracy"] == 1.0
    assert report["failure_cases"] == []
    assert report["total_abs_duration_error_sec"] == 0


def test_confusion_matrix_and_per_class_scores():
    predicted = timeline((S.LYING_IN_BED, 10), (S.UNKNOWN, 10))
    metrics = frame_metrics(TRUTH, predicted, 1.0)
    assert metrics.n_frames == 20 and metrics.accuracy == 0.5
    assert metrics.confusion[S.SITTING_ON_BED] == {S.UNKNOWN: 10}
    assert metrics.per_class[S.LYING_IN_BED].precision == 1.0
    assert metrics.per_class[S.SITTING_ON_BED].recall == 0.0
    assert metrics.per_class[S.UNKNOWN].precision == 0.0  # predicted 10 times, never right
    assert metrics.per_class[S.SITTING_ON_BED].support == 10


def test_only_the_span_both_timelines_cover_is_compared():
    short = timeline((S.LYING_IN_BED, 10))
    assert frame_metrics(TRUTH, short, 1.0).n_frames == 10
    errors = duration_errors(TRUTH, short)
    assert errors[S.LYING_IN_BED].abs_error_sec == 0
    assert S.SITTING_ON_BED not in errors  # beyond the covered span


def test_durations_report_absolute_error():
    predicted = timeline((S.LYING_IN_BED, 8), (S.SITTING_ON_BED, 12))
    errors = duration_errors(TRUTH, predicted)
    assert (errors[S.LYING_IN_BED].predicted_sec, errors[S.LYING_IN_BED].truth_sec) == (8, 10)
    assert errors[S.SITTING_ON_BED].abs_error_sec == pytest.approx(2)


def test_events_match_within_tolerance_and_extra_exits_are_false_alarms():
    truth = events((S.LYING_IN_BED, 20), (S.STANDING, 3), (S.WALKING, 20))  # exit at 20 s
    near = events((S.LYING_IN_BED, 24), (S.STANDING, 3), (S.WALKING, 20))  # 4 s late: matches
    far = events((S.LYING_IN_BED, 30), (S.STANDING, 3), (S.WALKING, 20))  # 10 s late: does not
    exit_kind = BedEventKind.BED_EXIT
    good = match_events(near, truth, 5)[exit_kind]
    assert (good.matched, good.precision, good.recall) == (1, 1.0, 1.0)
    bad = match_events(far, truth, 5)[exit_kind]
    assert (bad.matched, bad.precision, bad.recall) == (0, 0.0, 0.0)


def test_one_event_cannot_match_twice_and_empty_sides_give_none():
    truth = events((S.LYING_IN_BED, 20), (S.STANDING, 3), (S.WALKING, 20))
    twice = truth + truth  # two predictions for one true exit
    score = match_events(twice, truth, 5)[BedEventKind.BED_EXIT]
    assert (score.matched, score.predicted) == (1, 2) and score.precision == 0.5
    nothing = match_events([], [], 5)[BedEventKind.BED_EXIT]
    assert nothing.precision is None and nothing.recall is None


def test_false_bed_exits_are_reported():
    truth = timeline((S.LYING_IN_BED, 60))
    predicted = timeline(
        (S.LYING_IN_BED, 20), (S.STANDING, 3), (S.WALKING, 20), (S.LYING_IN_BED, 17)
    )
    report = evaluate(truth, predicted, BedEventDetector(SETTINGS).detect(predicted), SETTINGS)
    assert report.to_dict()["events"]["false_bed_exits"] == 1


def test_failures_are_grouped_and_longest_first():
    truth = timeline((S.SITTING_ON_BED, 20))
    predicted = timeline(
        (S.UNKNOWN, 5, "keypoints_not_visible"), (S.SITTING_ON_BED, 5), (S.LYING_IN_BED, 10)
    )
    failures = find_failures(truth, predicted, 1.0, max_cases=5)
    assert [(f.start_sec, f.end_sec, f.predicted_state) for f in failures] == [
        (10, 20, S.LYING_IN_BED),
        (0, 5, S.UNKNOWN),
    ]
    assert failures[1].reason == "keypoints_not_visible"
    assert len(find_failures(truth, predicted, 1.0, max_cases=1)) == 1
