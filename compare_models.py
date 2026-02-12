"""
Model Evaluation and Comparison Tool
====================================
Load and compare all trained models
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# ================== CONFIGURATION ================== #
DATA_PATH = "data/hotel_food_waste.csv"
MODEL_DIR = Path("models")

# ================== LOAD TEST DATA ================== #

def load_test_data():
    """Load and prepare test data"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Wastage Food Amount", axis=1)
    y = df["Wastage Food Amount"]
    
    # Encode
    encoders = joblib.load(MODEL_DIR / "hotel_encoders.pkl")
    for col, encoder in encoders.items():
        if col in X.columns:
            X[col] = encoder.transform(X[col].astype(str))
    
    # Split (use same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_test, y_test

# ================== EVALUATE MODEL ================== #

def evaluate_single_model(model_path, X_test, y_test):
    """Evaluate a single model"""
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        metrics = {
            'Model': model_path.stem.replace('hotel_model_', '').replace('hotel_model', 'default'),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        return metrics
    except Exception as e:
        print(f"Error evaluating {model_path.name}: {str(e)}")
        return None

# ================== COMPARE ALL MODELS ================== #

def compare_all_models():
    """Compare all models in the models directory"""
    
    print("\n" + "="*70)
    print("  📊 MODEL COMPARISON")
    print("="*70 + "\n")
    
    # Load test data
    print("Loading test data...")
    X_test, y_test = load_test_data()
    print(f"Test samples: {len(X_test)}\n")
    
    # Find all model files
    model_files = list(MODEL_DIR.glob("hotel_model*.pkl"))
    model_files = [f for f in model_files if 'encoders' not in f.name]
    
    if not model_files:
        print("❌ No model files found in models/ directory")
        return
    
    print(f"Found {len(model_files)} models:\n")
    
    # Evaluate each model
    all_metrics = []
    for model_file in model_files:
        print(f"Evaluating: {model_file.name}...")
        metrics = evaluate_single_model(model_file, X_test, y_test)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("❌ No models could be evaluated")
        return
    
    # Create comparison DataFrame
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.sort_values('MAE')
    
    # Display results
    print("\n" + "="*90)
    print("  RESULTS")
    print("="*90 + "\n")
    print(df_metrics.to_string(index=False))
    print("\n" + "="*90)
    
    # Highlight best model
    best_idx = df_metrics['MAE'].idxmin()
    best_model = df_metrics.loc[best_idx]
    
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   MAE:  {best_model['MAE']:.4f}")
    print(f"   RMSE: {best_model['RMSE']:.4f}")
    print(f"   R²:   {best_model['R2']:.4f}")
    print(f"   MAPE: {best_model['MAPE']:.2f}%")
    print("="*90 + "\n")
    
    # Save comparison
    output_file = MODEL_DIR / "model_comparison_metrics.csv"
    df_metrics.to_csv(output_file, index=False)
    print(f"✅ Comparison saved to: {output_file}\n")
    
    return df_metrics

# ================== MAIN ================== #

if __name__ == "__main__":
    compare_all_models()
