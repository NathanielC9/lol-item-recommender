from flask import Flask, request, jsonify, render_template_string
import os
from dotenv import load_dotenv
from app.model_utils import load_pipeline, predict_next_item

load_dotenv()

app = Flask(__name__)
pipeline = load_pipeline()

def get_champion_options():
    champs = pipeline["encoders"]["champion_le"].classes_
    options = "\n".join([f'<option value="{c}">{c}</option>' for c in champs])
    return options

with open(os.path.join(os.path.dirname(__file__), "ui.html"), "r") as f:
    UI_HTML = f.read()

@app.route("/")
def index():
    html = UI_HTML.replace("{{CHAMPION_OPTIONS}}", get_champion_options())
    return render_template_string(html)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        pred, top3 = predict_next_item(pipeline, data)
        return jsonify({
            "prediction": pred,
            "top3": top3,
            "input_state": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)