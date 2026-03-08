"""main.py — FoodWise entry point.  Usage: python main.py [train|serve|predict]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import setup_logging
setup_logging()
import logging; logger = logging.getLogger(__name__)

def cmd_train():
    from src.training import train_all_models
    print("\n🤖  Training all models...\n")
    s = train_all_models(n_samples=2000)
    print(f"\n{'Venue':<15} {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-"*65)
    for vt, info in s.items():
        if "error" in info:
            print(f"{vt:<15} ERROR: {info['error']}")
        else:
            m = info["metrics"]
            print(f"{vt:<15} {info['best_model_name']:<22} {m['MAE']:>8} {m['RMSE']:>8} {m['R2']:>8}")
    print("\n✅  Models saved to ./models/\n")

def cmd_serve():
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG","false").lower() == "true"
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    pkls = [f for f in os.listdir(models_dir) if f.endswith("_model.pkl")]
    if not pkls:
        print("No models found — training now...\n")
        from src.training import train_all_models
        train_all_models()
    print(f"\n🌿  FoodWise running at http://localhost:{port}\n")
    from app.app import app
    app.run(host="0.0.0.0", port=port, debug=debug)

def cmd_predict():
    from src.inference import predict
    from src.utils import validate_input, FIELD_SPECS
    print("\n🍽️  FoodWise CLI Prediction\n")
    vts = list(FIELD_SPECS.keys())
    print("Venue types:", ", ".join(vts))
    vt = input("Enter venue type: ").strip().lower()
    if vt not in vts:
        print(f"Unknown: {vt}"); return
    raw = {}
    for (field, dtype, choices, lo, hi) in FIELD_SPECS[vt]:
        hint = f" [{'/'.join(choices)}]" if choices else (f" ({lo}-{hi})" if lo is not None else "")
        raw[field] = input(f"  {field.replace('_',' ').title()}{hint}: ").strip()
    try:
        r = predict(vt, validate_input(vt, raw))
        print(f"\n✅  Predicted waste: {r['prediction_kg']} kg  (model={r['model_name']}, R2={r['r2_score']})\n")
        for tip in r["tips"]:
            print(f"   {tip}")
    except Exception as e:
        print(f"\n❌  {e}")

CMDS = {"train": cmd_train, "serve": cmd_serve, "predict": cmd_predict}
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd not in CMDS:
        print(f"Usage: python main.py [train|serve|predict]")
        sys.exit(1)
    CMDS[cmd]()
