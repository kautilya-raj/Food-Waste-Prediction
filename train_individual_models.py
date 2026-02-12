"""
Train Individual Models - Modular Approach
==========================================
Train each model type separately for flexibility
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== CONFIGURATION ================== #
DATA_PATH = "data/hotel_food_waste.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ================== LOAD AND PREPARE DATA ================== #

def load_and_prepare_data():
    """Load and prepare data for training"""
    print("📂 Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Separate features and target
    X = df.drop("Wastage Food Amount", axis=1)
    y = df["Wastage Food Amount"]
    
    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test, label_encoders

# ================== TRAIN RANDOM FOREST ================== #

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    print("\n1️⃣  Training Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²: {r2:.4f}")
    
    # Save
    joblib.dump(model, MODEL_DIR / "hotel_model_random_forest.pkl")
    print("   ✅ Saved: hotel_model_random_forest.pkl")
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ================== TRAIN GRADIENT BOOSTING ================== #

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting model"""
    print("\n2️⃣  Training Gradient Boosting...")
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=RANDOM_STATE
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²: {r2:.4f}")
    
    # Save
    joblib.dump(model, MODEL_DIR / "hotel_model_gradient_boosting.pkl")
    print("   ✅ Saved: hotel_model_gradient_boosting.pkl")
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ================== TRAIN XGBOOST ================== #

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model"""
    try:
        from xgboost import XGBRegressor
        
        print("\n3️⃣  Training XGBoost...")
        
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        
        # Save
        joblib.dump(model, MODEL_DIR / "hotel_model_xgboost.pkl")
        print("   ✅ Saved: hotel_model_xgboost.pkl")
        
        return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    except ImportError:
        print("   ⚠️  XGBoost not installed. Skipping...")
        return None, None

# ================== TRAIN LIGHTGBM ================== #

def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train LightGBM model"""
    try:
        from lightgbm import LGBMRegressor
        
        print("\n4️⃣  Training LightGBM...")
        
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        
        # Save
        joblib.dump(model, MODEL_DIR / "hotel_model_lightgbm.pkl")
        print("   ✅ Saved: hotel_model_lightgbm.pkl")
        
        return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    except ImportError:
        print("   ⚠️  LightGBM not installed. Skipping...")
        return None, None

# ================== TRAIN CATBOOST ================== #

def train_catboost(X_train, X_test, y_train, y_test):
    """Train CatBoost model"""
    try:
        from catboost import CatBoostRegressor
        
        print("\n5️⃣  Training CatBoost...")
        
        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        
        # Save
        joblib.dump(model, MODEL_DIR / "hotel_model_catboost.pkl")
        print("   ✅ Saved: hotel_model_catboost.pkl")
        
        return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    except ImportError:
        print("   ⚠️  CatBoost not installed. Skipping...")
        return None, None

# ================== MAIN ================== #

def main():
    """Train all models"""
    print("\n" + "="*70)
    print("  🚀 TRAINING INDIVIDUAL MODELS")
    print("="*70 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test, encoders = load_and_prepare_data()
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Save encoders
    joblib.dump(encoders, MODEL_DIR / "hotel_encoders.pkl")
    print("   ✅ Saved: hotel_encoders.pkl")
    
    # Train all models
    models = {}
    metrics = {}
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    if rf_model:
        models['random_forest'] = rf_model
        metrics['random_forest'] = rf_metrics
    
    # Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting(X_train, X_test, y_train, y_test)
    if gb_model:
        models['gradient_boosting'] = gb_model
        metrics['gradient_boosting'] = gb_metrics
    
    # XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    if xgb_model:
        models['xgboost'] = xgb_model
        metrics['xgboost'] = xgb_metrics
    
    # LightGBM
    lgbm_model, lgbm_metrics = train_lightgbm(X_train, X_test, y_train, y_test)
    if lgbm_model:
        models['lightgbm'] = lgbm_model
        metrics['lightgbm'] = lgbm_metrics
    
    # CatBoost
    cat_model, cat_metrics = train_catboost(X_train, X_test, y_train, y_test)
    if cat_model:
        models['catboost'] = cat_model
        metrics['catboost'] = cat_metrics
    
    # Find best model
    if metrics:
        best_model_name = min(metrics.items(), key=lambda x: x[1]['MAE'])[0]
        best_model = models[best_model_name]
        
        print("\n" + "="*70)
        print(f"  🏆 BEST MODEL: {best_model_name.upper()}")
        print(f"     MAE: {metrics[best_model_name]['MAE']:.4f}")
        print(f"     RMSE: {metrics[best_model_name]['RMSE']:.4f}")
        print(f"     R²: {metrics[best_model_name]['R2']:.4f}")
        print("="*70)
        
        # Save best model as default
        joblib.dump(best_model, MODEL_DIR / "hotel_model.pkl")
        print(f"\n✅ Best model saved as: hotel_model.pkl")
    
    print("\n✅ All models trained successfully!\n")

if __name__ == "__main__":
    main()
