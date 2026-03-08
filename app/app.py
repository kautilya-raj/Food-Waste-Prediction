"""
app.py
------
Flask application for the Food Waste Prediction system.
Serves a modern web UI and a JSON REST API.

Routes
------
GET  /                     → Landing / venue selection page
GET  /predict/<venue_type> → Input form for a specific venue
POST /predict/<venue_type> → Submit form, return results page
GET  /api/predict          → JSON API (POST body: {venue_type, ...features})
GET  /api/model-info       → JSON model metadata for all venues
GET  /train                → Trigger re-training (admin endpoint)
"""

import os
import sys
import logging

from flask import Flask, render_template, request, jsonify, redirect, url_for

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import setup_logging, validate_input, FIELD_SPECS
from src.inference import predict, load_model_bundle
from src.training import train_all_models

# ──────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "food-waste-dev-key-change-in-prod")

VENUE_TYPES = ["hostel", "hotel", "wedding", "household"]

VENUE_META = {
    "hostel":    {"icon": "🏫", "color": "#4CAF50", "label": "Hostel / Mess",    "image": "hostel.png"},
    "hotel":     {"icon": "🏨", "color": "#2196F3", "label": "Hotel / Restaurant","image": "hotel.png"},
    "wedding":   {"icon": "💍", "color": "#E91E63", "label": "Event",             "image": "event.png"},
    "household": {"icon": "🏠", "color": "#FF9800", "label": "Household",         "image": "home.png"},
}

# ──────────────────────────────────────────────
# Helper: get model info if available
# ──────────────────────────────────────────────
def _get_model_info(venue_type: str) -> dict:
    try:
        b = load_model_bundle(venue_type)
        return {"loaded": True, "model_name": b.get("model_name"), "metrics": b.get("metrics")}
    except FileNotFoundError:
        return {"loaded": False}


# ══════════════════════════════════════════════
# Page routes
# ══════════════════════════════════════════════

@app.route("/")
def landing():
    """Full-screen landing page."""
    return render_template("landing.html")



@app.route("/app")
def index():
    """Landing page — show all four venue cards."""
    model_status = {vt: _get_model_info(vt) for vt in VENUE_TYPES}
    return render_template("index.html",
                           venue_types=VENUE_TYPES,
                           venue_meta=VENUE_META,
                           model_status=model_status)


@app.route("/predict/<venue_type>", methods=["GET"])
def predict_form(venue_type: str):
    """Render the prediction input form for a specific venue type."""
    if venue_type not in VENUE_TYPES:
        return render_template("error.html", message=f"Unknown venue type: {venue_type}"), 404

    model_info = _get_model_info(venue_type)
    field_specs = FIELD_SPECS.get(venue_type, [])
    return render_template("predict_form.html",
                           venue_type=venue_type,
                           meta=VENUE_META[venue_type],
                           field_specs=field_specs,
                           model_info=model_info)


@app.route("/predict/<venue_type>", methods=["POST"])
def predict_submit(venue_type: str):
    """Process the prediction form and render results."""
    if venue_type not in VENUE_TYPES:
        return render_template("error.html", message=f"Unknown venue type: {venue_type}"), 404

    try:
        # Validate + clean inputs
        cleaned = validate_input(venue_type, request.form)

        # Run prediction
        result = predict(venue_type, cleaned)

        return render_template("result.html",
                               venue_type=venue_type,
                               meta=VENUE_META[venue_type],
                               inputs=cleaned,
                               result=result)

    except FileNotFoundError as e:
        return render_template("error.html",
                               message=str(e),
                               action_url=url_for("train_models"),
                               action_label="Train Models Now"), 503

    except ValueError as e:
        field_specs = FIELD_SPECS.get(venue_type, [])
        return render_template("predict_form.html",
                               venue_type=venue_type,
                               meta=VENUE_META[venue_type],
                               field_specs=field_specs,
                               model_info=_get_model_info(venue_type),
                               error=str(e),
                               form_data=request.form), 400

    except Exception as e:
        logger.exception("Unexpected error during prediction")
        return render_template("error.html", message=f"Unexpected error: {e}"), 500


@app.route("/train")
def train_models():
    """Admin endpoint: re-train all models."""
    logger.info("Training all models triggered via web UI")
    try:
        summary = train_all_models(n_samples=2000)
        return render_template("train_result.html", summary=summary, venue_meta=VENUE_META)
    except Exception as e:
        logger.exception("Training failed")
        return render_template("error.html", message=f"Training error: {e}"), 500



@app.route("/about")
def about():
    """About page — ML models, tech stack, project structure."""
    model_status = {}
    for vt in VENUE_TYPES:
        info = _get_model_info(vt)
        if info["loaded"]:
            info["model_name"] = info.get("model_name", "?")
        model_status[vt] = info
    return render_template("about.html", venue_meta=VENUE_META, model_status=model_status)

# ══════════════════════════════════════════════
# JSON API routes
# ══════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON REST endpoint for predictions.

    Request body (JSON):
    {
        "venue_type": "hotel",
        "num_guests": 120,
        "meal_type": "dinner",
        ...
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    venue_type = data.pop("venue_type", None)
    if not venue_type:
        return jsonify({"error": "Missing field: venue_type"}), 400

    try:
        cleaned = validate_input(venue_type, data)
        result  = predict(venue_type, cleaned)
        return jsonify({"venue_type": venue_type, "inputs": cleaned, **result})
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "hint": "POST /api/train to train models first"}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("API prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Return metadata for all trained models."""
    info = {}
    for vt in VENUE_TYPES:
        info[vt] = _get_model_info(vt)
    return jsonify(info)


@app.route("/api/train", methods=["POST"])
def api_train():
    """Trigger model (re-)training via API."""
    try:
        summary = train_all_models(n_samples=2000)
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        logger.exception("API training error")
        return jsonify({"status": "error", "message": str(e)}), 500



# ══════════════════════════════════════════════
# CSV Upload Training
# ══════════════════════════════════════════════

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}

def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/train/upload")
def train_upload_page():
    """Show the CSV upload training page."""
    from src.utils import FIELD_SPECS
    model_status = {vt: _get_model_info(vt) for vt in VENUE_TYPES}
    return render_template("train_upload.html",
                           venue_types=VENUE_TYPES,
                           venue_meta=VENUE_META,
                           model_status=model_status,
                           field_specs=FIELD_SPECS)


@app.route("/train/upload", methods=["POST"])
def train_upload_submit():
    """Accept a CSV file, validate it, train the model, return results."""
    import pandas as pd
    from src.training import train_model_from_df

    venue_type = request.form.get("venue_type", "").lower()
    if venue_type not in VENUE_TYPES:
        return render_template("train_upload.html",
                               venue_types=VENUE_TYPES,
                               venue_meta=VENUE_META,
                               model_status={vt: _get_model_info(vt) for vt in VENUE_TYPES},
                               field_specs=__import__("src.utils", fromlist=["FIELD_SPECS"]).FIELD_SPECS,
                               error="Please select a valid venue type."), 400

    if "csv_file" not in request.files or request.files["csv_file"].filename == "":
        return render_template("train_upload.html",
                               venue_types=VENUE_TYPES,
                               venue_meta=VENUE_META,
                               model_status={vt: _get_model_info(vt) for vt in VENUE_TYPES},
                               field_specs=__import__("src.utils", fromlist=["FIELD_SPECS"]).FIELD_SPECS,
                               error="No file selected. Please upload a CSV."), 400

    file = request.files["csv_file"]
    if not _allowed_file(file.filename):
        return render_template("train_upload.html",
                               venue_types=VENUE_TYPES,
                               venue_meta=VENUE_META,
                               model_status={vt: _get_model_info(vt) for vt in VENUE_TYPES},
                               field_specs=__import__("src.utils", fromlist=["FIELD_SPECS"]).FIELD_SPECS,
                               error="Only CSV files are accepted."), 400

    try:
        df = pd.read_csv(file)
        logger.info("Uploaded CSV: %d rows, columns: %s", len(df), list(df.columns))

        result = train_model_from_df(venue_type, df)

        return render_template("train_upload_result.html",
                               venue_type=venue_type,
                               meta=VENUE_META[venue_type],
                               result=result,
                               n_rows=len(df))

    except ValueError as e:
        from src.utils import FIELD_SPECS
        return render_template("train_upload.html",
                               venue_types=VENUE_TYPES,
                               venue_meta=VENUE_META,
                               model_status={vt: _get_model_info(vt) for vt in VENUE_TYPES},
                               field_specs=FIELD_SPECS,
                               error=str(e)), 400
    except Exception as e:
        logger.exception("CSV training error")
        return render_template("error.html", message=f"Training failed: {e}"), 500


@app.route("/train/sample-csv/<venue_type>")
def download_sample_csv(venue_type):
    """Generate and serve a sample CSV for the given venue type."""
    import io
    import pandas as pd
    from flask import send_file
    from src.preprocessing import generate_synthetic_data

    if venue_type not in VENUE_TYPES:
        return render_template("error.html", message=f"Unknown venue type: {venue_type}"), 404

    df = generate_synthetic_data(venue_type, n_samples=50)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"sample_{venue_type}.csv"
    )

# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
