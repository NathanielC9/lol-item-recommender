import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_encoders(df, save_path):
    encoders = {}
    encoders["champion_le"] = LabelEncoder().fit(df["champion"].astype(str))
    encoders["lane_le"] = LabelEncoder().fit(df["lane"].astype(str))
    encoders["time_le"] = LabelEncoder().fit(df["time_bucket"].astype(str))
    encoders["item_le"] = LabelEncoder().fit(df["label_item"].astype(str))

    all_champs = set()
    for col in ["enemy_1", "enemy_2", "enemy_3", "enemy_4", "enemy_5"]:
        for c in df[col].fillna("None").astype(str).unique():
            all_champs.add(c)
    encoders["enemy_vocab"] = {c: i for i, c in enumerate(sorted(all_champs))}

    joblib.dump(encoders, save_path)
    return encoders

def load_encoders(path):
    return joblib.load(path)

def encode_row(row, enc):
    champ = str(row.get("champion", "None"))
    lane = str(row.get("lane", "NONE"))
    timeb = str(row.get("time_bucket", "mid"))
    gold = float(row.get("gold", 0))
    win = float(row.get("win", 0))
    k, d, a = [float(x) for x in str(row.get("kda", "0/0/0")).split("/")]
    cs = float(row.get("cs", 0))
    vision = float(row.get("vision", 0))

    x = [
        enc["champion_le"].transform([champ])[0] if champ in enc["champion_le"].classes_ else -1,
        enc["lane_le"].transform([lane])[0] if lane in enc["lane_le"].classes_ else -1,
        enc["time_le"].transform([timeb])[0] if timeb in enc["time_le"].classes_ else -1,
        gold, win, k, d, a, cs, vision
    ]

    enemy_vec = [0] * len(enc["enemy_vocab"])
    for col in ["enemy_1", "enemy_2", "enemy_3", "enemy_4", "enemy_5"]:
        c = str(row.get(col, "None"))
        if c in enc["enemy_vocab"]:
            enemy_vec[enc["enemy_vocab"][c]] = 1
    x.extend(enemy_vec)

    return np.array(x, dtype=float)