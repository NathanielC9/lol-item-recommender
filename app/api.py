from flask import Flask, request, jsonify, render_template_string
import os
from dotenv import load_dotenv
from app.model_utils import load_pipeline, predict_next_item
from utils.game_logic import get_champion_role

load_dotenv()

app = Flask(__name__)
pipeline = load_pipeline()


def get_champion_options():
    champs = pipeline["encoders"]["champion_le"].classes_
    options = "\n".join([f'<option value="{c}">{c}</option>' for c in champs])
    return options


def validate_input(data, pipeline):
    """Validate and clean API input before sending it to the model."""
    errors = []

    if not isinstance(data, dict):
        return None, ["Request body must be valid JSON."]

    clean_data = data.copy()
    encoders = pipeline["encoders"]

    champion_classes = set(encoders["champion_le"].classes_)
    lane_classes = set(encoders["lane_le"].classes_)
    time_classes = set(encoders["time_bucket_le"].classes_)
    role_classes = set(encoders["role_le"].classes_)

    champion = clean_data.get("champion")
    lane = clean_data.get("lane")
    time_bucket = clean_data.get("time_bucket")

    # Required categorical fields from dropdowns
    if not champion:
        errors.append("Champion is required.")
    elif champion not in champion_classes:
        errors.append(f"Invalid champion: {champion}")

    if not lane:
        errors.append("Lane is required.")
    elif lane not in lane_classes:
        errors.append(f"Invalid lane: {lane}")

    if not time_bucket:
        errors.append("Time bucket is required.")
    elif time_bucket not in time_classes:
        errors.append(f"Invalid time_bucket: {time_bucket}")

    # The frontend does not send role. Derive it from champion so the model
    # receives a real role value instead of always defaulting to "unknown".
    role = get_champion_role(champion) if champion in champion_classes else "unknown"
    if role not in role_classes:
        role = "unknown"
    clean_data["role"] = role

    # Enemy team validation
    enemy_fields = ["enemy_1", "enemy_2", "enemy_3", "enemy_4", "enemy_5"]
    enemy_champs = []

    for field in enemy_fields:
        enemy = clean_data.get(field)

        if not enemy or enemy == "None":
            errors.append(f"{field} is required.")
        elif enemy not in champion_classes:
            errors.append(f"Invalid enemy champion for {field}: {enemy}")
        else:
            enemy_champs.append(enemy)

    if len(enemy_champs) != len(set(enemy_champs)):
        errors.append("Enemy team cannot contain duplicate champions.")

    if champion in enemy_champs:
        errors.append("Player champion cannot also be on the enemy team.")

    # KDA validation. The UI sends KDA as a string like "2/1/2".
    kda = str(clean_data.get("kda", "0/0/0")).strip() or "0/0/0"
    parts = kda.split("/")

    if len(parts) != 3:
        errors.append("KDA must be in kills/deaths/assists format, such as 2/1/2.")
    else:
        try:
            kills, deaths, assists = [float(part) for part in parts]

            if kills < 0 or deaths < 0 or assists < 0:
                errors.append("KDA values cannot be negative.")
            else:
                clean_data["kills"] = kills
                clean_data["deaths"] = deaths
                clean_data["assists"] = assists
                clean_data["kda_ratio"] = (kills + assists) / max(1.0, deaths)
        except ValueError:
            errors.append("KDA must contain numbers, such as 2/1/2.")

    # Validate simple numeric fields used by the encoder.
    def validate_number(field, default=0.0, min_value=None, max_value=None):
        raw_value = clean_data.get(field, default)

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"Invalid number for {field}: {raw_value}")
            value = default

        if min_value is not None and value < min_value:
            errors.append(f"{field} cannot be less than {min_value}.")
            value = min_value

        if max_value is not None and value > max_value:
            errors.append(f"{field} cannot be greater than {max_value}.")
            value = max_value

        clean_data[field] = value

    validate_number("gold", default=0.0, min_value=-3000.0, max_value=3000.0)
    validate_number("win", default=0.0, min_value=0.0, max_value=1.0)
    validate_number("cs", default=0.0, min_value=0.0, max_value=1000.0)
    validate_number("vision", default=0.0, min_value=0.0, max_value=500.0)

    return clean_data, errors


with open(os.path.join(os.path.dirname(__file__), "ui.html"), "r") as f:
    UI_HTML = f.read()


@app.route("/")
def index():
    html = UI_HTML.replace("{{CHAMPION_OPTIONS}}", get_champion_options())
    return render_template_string(html)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    clean_data, errors = validate_input(data, pipeline)

    if errors:
        return jsonify({
            "error": "Invalid input",
            "details": errors
        }), 400

    try:
        pred, top3 = predict_next_item(pipeline, clean_data)

        if len(top3) != 3:
            return jsonify({
                "error": "Prediction failed",
                "details": ["Expected exactly 3 recommendations."]
            }), 500

        return jsonify({
            "prediction": pred,
            "top3": top3,
            "input_state": clean_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
