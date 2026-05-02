import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLUMNS = [
    "champion",
    "role",
    "lane",
    "time_bucket",
]

ENEMY_COLUMNS = [
    "enemy_1",
    "enemy_2",
    "enemy_3",
    "enemy_4",
    "enemy_5",
]

NUMERIC_COLUMNS = [
    "gold",
    "gold_per_min",
    "win",
    "kills",
    "deaths",
    "assists",
    "kda_ratio",
    "cs",
    "cs_per_min",
    "vision",
    "vision_per_min",
    "enemy_ad_count",
    "enemy_ap_count",
    "enemy_crit_count",
]


def build_encoders(df, save_path):
    encoders = {}

    for col in CATEGORICAL_COLUMNS:
        encoders[f"{col}_le"] = LabelEncoder().fit(df[col].fillna("unknown").astype(str))

    encoders["item_le"] = LabelEncoder().fit(df["label_item"].astype(str))

    all_champs = set()
    for col in ENEMY_COLUMNS:
        for champ in df[col].fillna("None").astype(str).unique():
            all_champs.add(champ)

    encoders["enemy_vocab"] = {
        champ: i for i, champ in enumerate(sorted(all_champs))
    }

    encoders["categorical_columns"] = CATEGORICAL_COLUMNS
    encoders["enemy_columns"] = ENEMY_COLUMNS
    encoders["numeric_columns"] = NUMERIC_COLUMNS

    joblib.dump(encoders, save_path)
    return encoders


def load_encoders(path):
    return joblib.load(path)


def safe_label_encode(value, label_encoder):
    value = str(value)

    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]

    return -1


def parse_old_kda(row):
    """
    Backward compatibility for API inputs that still send kda as '2/1/2'.
    """
    kda = str(row.get("kda", "0/0/0"))

    try:
        kills, deaths, assists = [float(x) for x in kda.split("/")]
    except ValueError:
        kills, deaths, assists = 0.0, 0.0, 0.0

    kda_ratio = (kills + assists) / max(1.0, deaths)

    return kills, deaths, assists, kda_ratio


def encode_row(row, enc):
    x = []

    # Categorical features
    for col in enc.get("categorical_columns", CATEGORICAL_COLUMNS):
        value = row.get(col, "unknown")
        label_encoder = enc[f"{col}_le"]
        x.append(safe_label_encode(value, label_encoder))

    # Backward compatibility if API does not send new KDA columns yet
    parsed_kills, parsed_deaths, parsed_assists, parsed_kda_ratio = parse_old_kda(row)

    numeric_defaults = {
        "kills": parsed_kills,
        "deaths": parsed_deaths,
        "assists": parsed_assists,
        "kda_ratio": parsed_kda_ratio,
        "gold": 0.0,
        "gold_per_min": 0.0,
        "win": 0.0,
        "cs": 0.0,
        "cs_per_min": 0.0,
        "vision": 0.0,
        "vision_per_min": 0.0,
        "enemy_ad_count": 0.0,
        "enemy_ap_count": 0.0,
        "enemy_crit_count": 0.0,
    }

    # Numeric features
    for col in enc.get("numeric_columns", NUMERIC_COLUMNS):
        value = row.get(col, numeric_defaults.get(col, 0.0))

        try:
            value = float(value)
        except ValueError:
            value = numeric_defaults.get(col, 0.0)

        x.append(value)

    # Enemy champion multi-hot vector
    enemy_vec = [0] * len(enc["enemy_vocab"])

    for col in enc.get("enemy_columns", ENEMY_COLUMNS):
        champ = str(row.get(col, "None"))

        if champ in enc["enemy_vocab"]:
            enemy_vec[enc["enemy_vocab"][champ]] = 1

    x.extend(enemy_vec)

    return np.array(x, dtype=float)