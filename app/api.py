from flask import Flask, request, jsonify, render_template_string
import os
import sys
from model_utils import load_pipeline, predict_next_item
from dotenv import load_dotenv
load_dotenv()

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