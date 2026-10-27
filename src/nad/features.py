from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import COLUMN_NAMES

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# num_outbound_cmds is 0 in every training row, so it carries no information
EXCLUDED_COLUMNS = CATEGORICAL_COLUMNS + ["class", "difficulty_level", "num_outbound_cmds"]

NUMERIC_COLUMNS = [name for name in COLUMN_NAMES if name not in EXCLUDED_COLUMNS]


def build_preprocessor():
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
    ])
