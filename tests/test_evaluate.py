import numpy as np
import pandas as pd

from nad.evaluate import false_alarm_rate, recall_by_category, recall_on_seen_and_unseen


def test_false_alarm_rate_is_share_of_normal_rows_flagged():
    y_true = pd.Series([0, 0, 0, 0, 1, 1])
    predictions = np.array([1, 0, 0, 0, 1, 0])
    assert false_alarm_rate(y_true, predictions) == 0.25


def test_recall_on_seen_and_unseen_splits_by_attack_type():
    train_classes = pd.Series(["normal", "neptune"])
    test_classes = pd.Series(["normal", "neptune", "neptune", "mailbomb", "mailbomb"])
    predictions = np.array([1, 1, 0, 1, 1])
    result = recall_on_seen_and_unseen(train_classes, test_classes, predictions)
    assert result == {"recall_seen": 0.5, "recall_unseen": 1.0}


def test_recall_by_category_ignores_normal_rows():
    test_classes = pd.Series(["normal", "smurf", "satan", "guess_passwd", "rootkit"])
    predictions = np.array([1, 1, 0, 1, 0])
    result = recall_by_category(test_classes, predictions)
    assert result == {"recall_dos": 1.0, "recall_probe": 0.0, "recall_r2l": 1.0, "recall_u2r": 0.0}
