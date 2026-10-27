from nad.data import load_nsl_kdd, split_features_labels
from nad.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, build_preprocessor
from nad.train import DATA_DIR


def load_sample():
    X, _ = split_features_labels(load_nsl_kdd(DATA_DIR / "KDDTest+.txt"))
    return X.head(100)


def test_every_feature_is_used_except_the_constant_column():
    X = load_sample()
    unused = set(X.columns) - set(CATEGORICAL_COLUMNS) - set(NUMERIC_COLUMNS)
    assert unused == {"num_outbound_cmds"}


def test_unknown_category_does_not_crash_or_change_width():
    X = load_sample()
    preprocessor = build_preprocessor().fit(X)
    new_row = X.head(1).copy()
    new_row["service"] = "service_never_seen_before"
    assert preprocessor.transform(new_row).shape[1] == preprocessor.transform(X).shape[1]
