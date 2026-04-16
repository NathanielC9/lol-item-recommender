from flask import Flask, request, jsonify, render_template_string
import os
from app.model_utils import load_pipeline, predict_next_item
from dotenv import load_dotenv
load_dotenv()

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

app = Flask(__name__)
pipeline = load_pipeline()

with open(os.path.join(os.path.dirname(__file__), "ui.html"), "r") as f:
    UI_HTML = f.read()

@app.route("/")
def index():
    return render_template_string(UI_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        pred, top3 = predict_next_item(pipeline, data)
        return jsonify({"prediction": pred, "top3": top3})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)