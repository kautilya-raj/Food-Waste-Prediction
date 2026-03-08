"""inference.py — load model and predict food waste."""
import os, pickle, logging, pandas as pd
logger     = logging.getLogger(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_model_bundle(venue_type):
    path = os.path.join(MODELS_DIR, f"{venue_type.lower()}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Run: python main.py train")
    with open(path, "rb") as f: return pickle.load(f)

def predict(venue_type, input_data):
    bundle = load_model_bundle(venue_type)
    df     = pd.DataFrame([input_data])
    X      = bundle["preprocessor"].transform(df)
    kg     = max(0.0, round(float(bundle["model"].predict(X)[0]), 2))
    return {"prediction_kg": kg, "tips": _tips(venue_type, input_data, kg),
            "model_name": bundle.get("model_name","?"), "r2_score": bundle.get("metrics",{}).get("R2")}

def _tips(venue_type, data, kg):
    tips = []
    if kg > 50:   tips.append("⚠️ High waste predicted — reduce quantities by 20%.")
    elif kg > 20: tips.append("📊 Moderate waste — consider reducing portions slightly.")
    else:         tips.append("✅ Waste levels are manageable. Keep monitoring.")
    if venue_type == "hostel":
        if data.get("special_occasion") == 1: tips.append("🎉 Special occasions increase waste — prepare 10-15% less.")
        if float(data.get("leftover_yesterday", 0)) > 10: tips.append("♻️ High leftovers yesterday — incorporate them today.")
        tips.append("📋 Post daily menus in advance so students can plan attendance.")
    elif venue_type == "hotel":
        if data.get("buffet_style") == 1: tips.append("🍽️ Buffets waste more — use smaller trays replenished frequently.")
        tips.append("📉 Track real-time consumption to adjust replenishment dynamically.")
    elif venue_type == "wedding":
        if int(data.get("num_dishes", 0)) > 20: tips.append("🍲 Too many dishes — cutting to 15 reduces waste by ~25%.")
        if data.get("outdoor_event") == 1: tips.append("🌿 Outdoor events see more spoilage — use covered serving stations.")
        tips.append("📬 Confirm final guest count 48 hours before the event.")
    elif venue_type == "household":
        tips.append("🛒 Plan meals for the week before grocery shopping.")
        tips.append("🥡 Schedule a leftover night each week to clear the fridge.")
    return tips
