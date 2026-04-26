import os
import joblib
import numpy as np
import torch

from preprocessing.encode import load_encoders, encode_row
from utils.item_names import get_item_name
from utils.game_logic import (
    is_valid_item_for_role,
    item_bonus,
    explain_item
)
from models.mlp_model import MLP

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def load_pipeline():
    encoders = load_encoders(os.path.join(MODEL_PATH, "encoders.joblib"))

    scaler_path = os.path.join(MODEL_PATH, "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    for model_name in ["mlp_model.pt", "rf_model.joblib"]:
        model_path = os.path.join(MODEL_PATH, model_name)

        if os.path.exists(model_path):
            return {
                "encoders": encoders,
                "model_path": model_path,
                "scaler": scaler
            }

    raise FileNotFoundError("No trained model found in saved_models/")


def _load_model_predictions(pipeline, X):
    model_path = pipeline["model_path"]

    if model_path.endswith(".joblib"):
        model = joblib.load(model_path)

        probs = model.predict_proba([X])[0]
        classes = np.array(model.classes_)

        return classes, probs

    meta = joblib.load(os.path.join(os.path.dirname(model_path), "mlp_meta.joblib"))

    model = MLP(
        in_dim=meta["in_dim"],
        out_dim=meta["out_dim"]
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    classes = np.array(meta["classes"])

    return classes, probs


def _rerank_items(classes, probs, raw_row, top_n_pool=40):
    champion = raw_row.get("champion", "")

    ranked_idx = np.argsort(probs)[::-1]
    candidates = []

    for idx in ranked_idx[:top_n_pool]:
        item_id = classes[idx]
        item_name = get_item_name(item_id)

        if not is_valid_item_for_role(item_name, champion):
            continue

        model_prob = float(probs[idx])
        bonus = item_bonus(item_name, raw_row)
        adjusted_score = model_prob + bonus

        candidates.append({
            "item_id": item_id,
            "item": item_name,
            "model_prob": model_prob,
            "rule_bonus": bonus,
            "adjusted_score": adjusted_score,
            "reason": explain_item(item_name, raw_row)
        })

    if not candidates:
        for idx in ranked_idx[:3]:
            item_id = classes[idx]
            item_name = get_item_name(item_id)

            candidates.append({
                "item_id": item_id,
                "item": item_name,
                "model_prob": float(probs[idx]),
                "rule_bonus": 0.0,
                "adjusted_score": float(probs[idx]),
                "reason": explain_item(item_name, raw_row)
            })

    candidates = sorted(
        candidates,
        key=lambda x: x["adjusted_score"],
        reverse=True
    )

    top3 = []

    for c in candidates[:3]:
        top3.append({
            "item": c["item"],
            "prob": round(c["model_prob"], 4),
            "adjusted_score": round(c["adjusted_score"], 4),
            "rule_bonus": round(c["rule_bonus"], 4),
            "reason": c["reason"]
        })

    return top3


def predict_next_item(pipeline, raw_row):
    encoders = pipeline["encoders"]

    X = encode_row(raw_row, encoders)

    if pipeline.get("scaler") is not None:
        X = pipeline["scaler"].transform([X])[0]

    classes, probs = _load_model_predictions(pipeline, X)

    top3 = _rerank_items(classes, probs, raw_row)

    prediction = top3[0]["item"]

    return prediction, top3