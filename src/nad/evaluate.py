def recall_on_seen_and_unseen(train_classes, test_classes, predictions):
    is_attack = test_classes != "normal"
    is_seen = test_classes.isin(train_classes.unique())
    return {
        "recall_seen": predictions[is_attack & is_seen].mean(),
        "recall_unseen": predictions[is_attack & ~is_seen].mean(),
    }


def false_alarm_rate(y_true, predictions):
    return predictions[y_true == 0].mean()
