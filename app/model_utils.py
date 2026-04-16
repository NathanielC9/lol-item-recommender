import os
import joblib
import numpy as np
from preprocessing.encode import load_encoders, encode_row
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models")

def load_pipeline():
    encoders = load_encoders(os.path.join(MODEL_PATH, "encoders.joblib"))
    for m in ["mlp_model.pt", "rf_model.joblib"]:
        p = os.path.join(MODEL_PATH, m)
        if os.path.exists(p):
            return {"encoders": encoders, "model_path": p}
    raise FileNotFoundError("No trained model found in saved_models/")

def predict_next_item(pipeline, raw_row):
    enc = pipeline["encoders"]
    X = encode_row(raw_row, enc)
    model_path = pipeline["model_path"]

    if model_path.endswith(".joblib"):
        model = joblib.load(model_path)
        probs = model.predict_proba([X])[0]
        classes = model.classes_
    else:
        import torch
        meta = joblib.load(os.path.join(os.path.dirname(model_path), "mlp_meta.joblib"))
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        classes = meta["classes"]

    top3_idx = np.argsort(probs)[-3:][::-1]
    top1 = classes[top3_idx[0]]
    top3 = [{"item": classes[i], "prob": round(float(probs[i]), 4)} for i in top3_idx]
    return top1, top3

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()