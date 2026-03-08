"""training.py — train and persist models for each venue type."""
import os, pickle, logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.preprocessing import FoodWastePreprocessor, generate_synthetic_data

logger     = logging.getLogger(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def _candidates():
    return {
        "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        "Ridge":            Ridge(alpha=1.0),
    }

def train_model(venue_type, n_samples=2000):
    df = generate_synthetic_data(venue_type, n_samples=n_samples)
    pre = FoodWastePreprocessor()
    X_tr, X_te, y_tr, y_te = pre.fit_transform(df, venue_type)
    results = {}
    for name, mdl in _candidates().items():
        mdl.fit(X_tr, y_tr); yp = mdl.predict(X_te)
        results[name] = (mdl, {"MAE": round(float(mean_absolute_error(y_te,yp)),4),
                                "RMSE": round(float(mean_squared_error(y_te,yp)**0.5),4),
                                "R2":   round(float(r2_score(y_te,yp)),4)})
    best = max(results, key=lambda k: results[k][1]["R2"])
    bm, bmet = results[best]
    ens = VotingRegressor([(n,m) for n,(m,_) in results.items()], n_jobs=-1)
    ens.fit(X_tr, y_tr); yp = ens.predict(X_te)
    emet = {"MAE": round(float(mean_absolute_error(y_te,yp)),4),
            "RMSE": round(float(mean_squared_error(y_te,yp)**0.5),4),
            "R2":   round(float(r2_score(y_te,yp)),4)}
    if emet["R2"] > bmet["R2"]: best, bm, bmet = "Ensemble", ens, emet
    bundle = {"model": bm, "preprocessor": pre, "venue_type": venue_type,
              "feature_names": pre.feature_names, "metrics": bmet, "model_name": best}
    with open(os.path.join(MODELS_DIR, f"{venue_type}_model.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Saved %s model (best=%s R2=%.4f)", venue_type, best, bmet["R2"])
    return {"venue_type": venue_type, "best_model_name": best, "metrics": bmet,
            "feature_names": pre.feature_names, "all_results": {k: v[1] for k,v in results.items()}}

def train_all_models(n_samples=2000):
    summary = {}
    for vt in ["hostel","hotel","wedding","household"]:
        try:    summary[vt] = train_model(vt, n_samples=n_samples)
        except Exception as e: logger.error("Failed %s: %s", vt, e); summary[vt] = {"error": str(e)}
    return summary


def train_model_from_df(venue_type: str, df) -> dict:
    """
    Train a model from a user-supplied DataFrame (uploaded CSV).
    Validates that required columns are present, then trains and saves the model.

    Parameters
    ----------
    venue_type : One of hostel / hotel / wedding / household
    df         : pandas DataFrame loaded from uploaded CSV

    Returns
    -------
    dict with best_model_name, metrics, n_rows, feature_names, all_results
    """
    import pandas as pd
    from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

    vt = venue_type.lower()
    required_cols = FEATURE_COLUMNS[vt] + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Required: {required_cols}"
        )

    # Drop rows with nulls in required columns
    df = df[required_cols].dropna()
    if len(df) < 20:
        raise ValueError(f"After removing nulls, only {len(df)} rows remain. Need at least 20.")

    logger.info("Training '%s' from uploaded CSV (%d rows)", vt, len(df))

    pre = FoodWastePreprocessor()
    X_tr, X_te, y_tr, y_te = pre.fit_transform(df, vt)

    from sklearn.metrics import r2_score
    results = {}
    for name, mdl in _candidates().items():
        mdl.fit(X_tr, y_tr); yp = mdl.predict(X_te)
        results[name] = (mdl, {
            "MAE":  round(float(mean_absolute_error(y_te, yp)), 4),
            "RMSE": round(float(mean_squared_error(y_te, yp) ** 0.5), 4),
            "R2":   round(float(r2_score(y_te, yp)), 4),
        })

    best = max(results, key=lambda k: results[k][1]["R2"])
    bm, bmet = results[best]

    ens = VotingRegressor([(n, m) for n, (m, _) in results.items()], n_jobs=-1)
    ens.fit(X_tr, y_tr); yp = ens.predict(X_te)
    emet = {"MAE": round(float(mean_absolute_error(y_te,yp)),4),
            "RMSE": round(float(mean_squared_error(y_te,yp)**0.5),4),
            "R2":   round(float(r2_score(y_te,yp)),4)}
    if emet["R2"] > bmet["R2"]: best, bm, bmet = "Ensemble", ens, emet

    bundle = {"model": bm, "preprocessor": pre, "venue_type": vt,
              "feature_names": pre.feature_names, "metrics": bmet, "model_name": best}
    with open(os.path.join(MODELS_DIR, f"{vt}_model.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    logger.info("Saved CSV-trained %s model (best=%s R2=%.4f)", vt, best, bmet["R2"])
    return {
        "venue_type": vt, "best_model_name": best, "metrics": bmet,
        "feature_names": pre.feature_names, "n_rows": len(df),
        "all_results": {k: v[1] for k, v in results.items()},
    }
