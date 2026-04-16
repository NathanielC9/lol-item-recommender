import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_encoders(df, save_path):
    encoders = {}
    encoders["champion_le"] = LabelEncoder().fit(df["champion"].astype(str))
    encoders["lane_le"] = LabelEncoder().fit(df["lane"].astype(str))
    encoders["time_le"] = LabelEncoder().fit(df["time_bucket"].astype(str))
    encoders["item_le"] = LabelEncoder().fit(df["label_item"].astype(str))
    joblib.dump(encoders, save_path)
    return encoders

def load_encoders(path):
    return joblib.load(path)

def encode_row(row, enc):
    champ = str(row["champion"])
    lane = str(row.get("lane", "NONE"))
    timeb = str(row["time_bucket"])
    gold = float(row.get("gold_diff", 0))
    win = float(row.get("win", 0))
    k, d, a = [float(x) for x in str(row.get("kda", "0/0/0")).split("/")]

    x = [
        enc["champion_le"].transform([champ])[0] if champ in enc["champion_le"].classes_ else -1,
        enc["lane_le"].transform([lane])[0] if lane in enc["lane_le"].classes_ else -1,
        enc["time_le"].transform([timeb])[0] if timeb in enc["time_le"].classes_ else -1,
        gold, win, k, d, a
    ]
    return np.array(x, dtype=float)