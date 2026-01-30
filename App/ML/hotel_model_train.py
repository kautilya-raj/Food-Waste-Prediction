"""
Enhanced Hotel Food Wastage Prediction Model
============================================
Improvements:
- Multiple model comparison (RF, XGBoost, LightGBM, CatBoost)
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- Feature importance analysis
- Comprehensive error metrics
- Model ensemble for better predictions
- Proper data validation and error handling
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)

# Advanced models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Install with: pip install catboost")

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ================== #
DATA_PATH = "data/hotel_food_waste.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ================== HELPER FUNCTIONS ================== #

def load_and_validate_data(filepath):
    """Load data with validation"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️  Missing values detected:")
            print(missing[missing > 0])
            print("\n🔧 Handling missing values...")
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation"""
    # Train predictions
    y_train_pred = model.predict(X_train)
    # Test predictions
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_test_pred) * 100
    }
    
    # Cross-validation score
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=CV_FOLDS, 
        scoring='neg_mean_absolute_error'
    )
    metrics['CV MAE (mean)'] = -cv_scores.mean()
    metrics['CV MAE (std)'] = cv_scores.std()
    
    return metrics


def print_metrics(metrics_dict):
    """Print metrics in a formatted way"""
    print(f"\n{'='*70}")
    print(f"  {metrics_dict['Model']} - Performance Metrics")
    print(f"{'='*70}")
    print(f"  Train MAE:      {metrics_dict['Train MAE']:.4f}")
    print(f"  Test MAE:       {metrics_dict['Test MAE']:.4f}")
    print(f"  Train RMSE:     {metrics_dict['Train RMSE']:.4f}")
    print(f"  Test RMSE:      {metrics_dict['Test RMSE']:.4f}")
    print(f"  Train R²:       {metrics_dict['Train R²']:.4f}")
    print(f"  Test R²:        {metrics_dict['Test R²']:.4f}")
    print(f"  Test MAPE:      {metrics_dict['Test MAPE']:.2f}%")
    print(f"  CV MAE:         {metrics_dict['CV MAE (mean)']:.4f} ± {metrics_dict['CV MAE (std)']:.4f}")
    print(f"{'='*70}\n")


def get_feature_importance(model, feature_names, top_n=10):
    """Extract and display feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n🔍 Top {top_n} Most Important Features:")
        print("=" * 50)
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"  {row['Feature']:30s} {row['Importance']:.4f}")
        print("=" * 50)
        
        return importance_df
    else:
        print("⚠️  Model does not support feature importance")
        return None


# ================== MAIN TRAINING PIPELINE ================== #

def main():
    print("\n" + "="*70)
    print("  🚀 ENHANCED HOTEL FOOD WASTAGE PREDICTION MODEL")
    print("="*70 + "\n")
    
    # ---------------- LOAD DATA ---------------- #
    print("📂 Loading data...")
    df = load_and_validate_data(DATA_PATH)
    
    # Check if target column exists
    if "Wastage Food Amount" not in df.columns:
        print(f"❌ Error: 'Wastage Food Amount' column not found")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # ---------------- FEATURES & TARGET ---------------- #
    X = df.drop("Wastage Food Amount", axis=1)
    y = df["Wastage Food Amount"]
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Target mean: {y.mean():.2f} ± {y.std():.2f}")
    
    # ---------------- ENCODE CATEGORICAL FEATURES ---------------- #
    print("\n🔄 Encoding categorical features...")
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    if categorical_cols:
        print(f"  Categorical columns: {categorical_cols}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        print(f"  ✅ Encoded {len(categorical_cols)} categorical features")
    else:
        print("  ℹ️  No categorical features found")
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # ---------------- TRAIN / TEST SPLIT ---------------- #
    print(f"\n✂️  Splitting data (test size: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # ---------------- FEATURE SCALING (Optional) ---------------- #
    # Uncomment if you want to scale features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # ---------------- MODEL TRAINING ---------------- #
    print("\n🤖 Training models...")
    
    all_metrics = []
    trained_models = {}
    
    # 1. Enhanced Random Forest with GridSearch
    print("\n1️⃣  Training Random Forest with hyperparameter tuning...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid = GridSearchCV(
        rf_base, rf_params, cv=3, scoring='neg_mean_absolute_error', 
        n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    
    print(f"  ✅ Best RF params: {rf_grid.best_params_}")
    rf_metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest (Tuned)")
    all_metrics.append(rf_metrics)
    print_metrics(rf_metrics)
    trained_models['random_forest'] = rf_model
    
    # 2. Gradient Boosting
    print("\n2️⃣  Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=RANDOM_STATE
    )
    gb_model.fit(X_train, y_train)
    gb_metrics = evaluate_model(gb_model, X_train, X_test, y_train, y_test, "Gradient Boosting")
    all_metrics.append(gb_metrics)
    print_metrics(gb_metrics)
    trained_models['gradient_boosting'] = gb_model
    
    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n3️⃣  Training XGBoost...")
        xgb_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")
        all_metrics.append(xgb_metrics)
        print_metrics(xgb_metrics)
        trained_models['xgboost'] = xgb_model
    
    # 4. LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        print("\n4️⃣  Training LightGBM...")
        lgbm_model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        lgbm_model.fit(X_train, y_train)
        lgbm_metrics = evaluate_model(lgbm_model, X_train, X_test, y_train, y_test, "LightGBM")
        all_metrics.append(lgbm_metrics)
        print_metrics(lgbm_metrics)
        trained_models['lightgbm'] = lgbm_model
    
    # 5. CatBoost (if available)
    if CATBOOST_AVAILABLE:
        print("\n5️⃣  Training CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=0
        )
        cat_model.fit(X_train, y_train)
        cat_metrics = evaluate_model(cat_model, X_train, X_test, y_train, y_test, "CatBoost")
        all_metrics.append(cat_metrics)
        print_metrics(cat_metrics)
        trained_models['catboost'] = cat_model
    
    # ---------------- MODEL COMPARISON ---------------- #
    print("\n📊 MODEL COMPARISON SUMMARY")
    print("="*90)
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.to_string(index=False))
    print("="*90)
    
    # Find best model
    best_model_name = metrics_df.loc[metrics_df['Test MAE'].idxmin(), 'Model']
    best_test_mae = metrics_df['Test MAE'].min()
    print(f"\n🏆 Best Model: {best_model_name} (Test MAE: {best_test_mae:.4f})")
    
    # ---------------- ENSEMBLE MODEL ---------------- #
    print("\n🔗 Creating Ensemble Model...")
    
    # Use top 3 models for ensemble
    top_models = []
    for model_key, model in trained_models.items():
        top_models.append((model_key, model))
    
    if len(top_models) >= 2:
        ensemble = VotingRegressor(estimators=top_models)
        ensemble.fit(X_train, y_train)
        ensemble_metrics = evaluate_model(ensemble, X_train, X_test, y_train, y_test, "Ensemble (Voting)")
        print_metrics(ensemble_metrics)
        trained_models['ensemble'] = ensemble
    
    # ---------------- FEATURE IMPORTANCE ---------------- #
    # Use the best single model for feature importance
    best_model_key = list(trained_models.keys())[0]  # Random Forest
    best_model = trained_models[best_model_key]
    importance_df = get_feature_importance(best_model, feature_names)
    
    # ---------------- SAVE BEST MODEL & ARTIFACTS ---------------- #
    print("\n💾 Saving models and artifacts...")
    
    # Save the best performing model
    best_idx = metrics_df['Test MAE'].idxmin()
    best_model_type = list(trained_models.keys())[best_idx] if best_idx < len(trained_models) else 'random_forest'
    best_model_to_save = trained_models[best_model_type]
    
    joblib.dump(best_model_to_save, MODEL_DIR / "hotel_model.pkl")
    joblib.dump(label_encoders, MODEL_DIR / "hotel_encoders.pkl")
    
    # Save ensemble if available
    if 'ensemble' in trained_models:
        joblib.dump(trained_models['ensemble'], MODEL_DIR / "hotel_model_ensemble.pkl")
    
    # Save all models for comparison
    for model_name, model in trained_models.items():
        joblib.dump(model, MODEL_DIR / f"hotel_model_{model_name}.pkl")
    
    # Save metrics
    metrics_df.to_csv(MODEL_DIR / "model_comparison_metrics.csv", index=False)
    
    # Save feature importance
    if importance_df is not None:
        importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    
    print(f"  ✅ Best model saved: hotel_model.pkl ({best_model_type})")
    print(f"  ✅ Encoders saved: hotel_encoders.pkl")
    print(f"  ✅ All models saved in: {MODEL_DIR}")
    print(f"  ✅ Metrics saved: model_comparison_metrics.csv")
    
    # ---------------- FINAL SUMMARY ---------------- #
    print("\n" + "="*70)
    print("  ✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"  Best Model: {best_model_name}")
    print(f"  Test MAE: {best_test_mae:.4f}")
    print(f"  Test R²: {metrics_df.loc[best_idx, 'Test R²']:.4f}")
    print(f"  Test MAPE: {metrics_df.loc[best_idx, 'Test MAPE']:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
