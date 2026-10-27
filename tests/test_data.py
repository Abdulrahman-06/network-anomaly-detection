from nad.data import CATEGORY_OF, load_nsl_kdd, split_features_labels
from nad.train import DATA_DIR


def test_loader_drops_difficulty_and_adds_label():
    df = load_nsl_kdd(DATA_DIR / "KDDTest+.txt")
    assert "difficulty_level" not in df.columns
    assert (df["is_attack"] == (df["class"] != "normal")).all()


def test_split_removes_label_columns_from_features():
    X, y = split_features_labels(load_nsl_kdd(DATA_DIR / "KDDTest+.txt"))
    assert "class" not in X.columns
    assert "is_attack" not in X.columns
    assert y.name == "is_attack"


def test_every_attack_name_has_a_category():
    train = load_nsl_kdd(DATA_DIR / "KDDTrain+.txt")
    test = load_nsl_kdd(DATA_DIR / "KDDTest+.txt")
    attack_names = (set(train["class"]) | set(test["class"])) - {"normal"}
    missing = attack_names - set(CATEGORY_OF)
    assert missing == set()
