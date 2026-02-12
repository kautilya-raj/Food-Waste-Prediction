# Complete File Structure and Code Reference
# ==========================================

## 📁 Directory Structure

```
hotel-food-wastage-prediction/
│
├── 📂 data/
│   └── hotel_food_waste.csv              # Your training data (CSV format)
│
├── 📂 models/
│   ├── hotel_model.pkl                   # Best model (auto-selected)
│   ├── hotel_encoders.pkl                # Label encoders for categorical features
│   ├── hotel_model_random_forest.pkl     # Random Forest model
│   ├── hotel_model_gradient_boosting.pkl # Gradient Boosting model
│   ├── hotel_model_xgboost.pkl           # XGBoost model (if installed)
│   ├── hotel_model_lightgbm.pkl          # LightGBM model (if installed)
│   ├── hotel_model_catboost.pkl          # CatBoost model (if installed)
│   ├── hotel_model_ensemble.pkl          # Ensemble/Voting model
│   ├── model_comparison_metrics.csv      # Performance comparison CSV
│   └── feature_importance.csv            # Feature importance rankings
│
├── 📂 visualizations/
│   ├── model_comparison.png              # 4-panel model comparison chart
│   ├── train_vs_test.png                 # Overfitting analysis
│   ├── cv_scores.png                     # Cross-validation results
│   ├── feature_importance.png            # Feature importance bar chart
│   └── summary_table.png                 # Performance summary table
│
├── 📂 predictions/
│   └── (prediction output files)         # Batch prediction results
│
├── 🐍 Core Training Scripts
│   ├── train_hotel_model_enhanced.py     # Main training script (6 models + tuning)
│   ├── train_individual_models.py        # Train models separately
│   └── data_preprocessing.py             # Data cleaning utilities
│
├── 🎯 Prediction Scripts
│   ├── predict_hotel_wastage.py          # Single & batch prediction
│   └── compare_models.py                 # Load and compare all models
│
├── 📊 Visualization Scripts
│   └── visualize_models.py               # Generate all charts
│
├── 🌐 API & Deployment
│   ├── api_server.py                     # Flask REST API server
│   └── api_examples.py                   # API usage examples
│
├── 🔧 Setup & Utilities
│   ├── setup.py                          # Automated setup script
│   ├── run.py                            # Master pipeline runner
│   └── requirements.txt                  # Package dependencies
│
└── 📚 Documentation
    ├── README_HOTEL_MODEL.md             # Complete documentation
    ├── UPGRADE_SUMMARY.md                # What's new/improved
    └── FILE_STRUCTURE.md                 # This file
```

## 📝 File Descriptions

### 1. train_hotel_model_enhanced.py
**Purpose:** Main training script with advanced ML models

**Features:**
- Trains 6 different models (RF, GB, XGB, LGBM, CatBoost, Ensemble)
- Automated hyperparameter tuning via GridSearchCV
- 5-fold cross-validation
- Comprehensive metrics (MAE, RMSE, R², MAPE)
- Feature importance analysis
- Auto-selects best model

**Usage:**
```bash
python train_hotel_model_enhanced.py
```

**Outputs:**
- models/hotel_model.pkl (best model)
- models/hotel_encoders.pkl
- models/model_comparison_metrics.csv
- models/feature_importance.csv

---

### 2. predict_hotel_wastage.py
**Purpose:** Make predictions with trained models

**Features:**
- Single prediction
- Batch prediction from CSV
- Interactive mode
- Error handling for unseen categories
- Non-negative prediction enforcement

**Usage:**
```python
from predict_hotel_wastage import load_model_and_encoders, predict_wastage

model, encoders = load_model_and_encoders()

input_data = {
    'Type of Food': 'Rice',
    'Number of Guests': 100,
    # ... other features
}

prediction = predict_wastage(input_data, model, encoders)
print(f"Predicted Wastage: {prediction:.2f} kg")
```

---

### 3. visualize_models.py
**Purpose:** Generate professional visualizations

**Features:**
- Model comparison charts (4-panel)
- Train vs test performance
- Cross-validation scores with error bars
- Feature importance plot
- Summary performance table

**Usage:**
```bash
python visualize_models.py
```

**Outputs:**
- visualizations/model_comparison.png
- visualizations/train_vs_test.png
- visualizations/cv_scores.png
- visualizations/feature_importance.png
- visualizations/summary_table.png

---

### 4. api_server.py
**Purpose:** REST API for web integration

**Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `GET /features` - Required features list
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

**Usage:**
```bash
python api_server.py
# Server runs on http://localhost:5000
```

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Type of Food": "Rice",
    "Number of Guests": 100,
    "Event Type": "Wedding",
    "Quantity of Food": 50.0,
    ...
  }'
```

---

### 5. api_examples.py
**Purpose:** Example API usage scripts

**Features:**
- Health check example
- Single prediction example
- Batch prediction example
- Error handling examples
- CURL command examples

**Usage:**
```bash
python api_examples.py
```

---

### 6. train_individual_models.py
**Purpose:** Train models separately (modular approach)

**Features:**
- Train each model type individually
- Simpler than full pipeline
- Good for testing specific models
- Auto-saves best as hotel_model.pkl

**Usage:**
```bash
python train_individual_models.py
```

---

### 7. data_preprocessing.py
**Purpose:** Data cleaning and preparation utilities

**Features:**
- Missing value handling
- Outlier removal (IQR/Z-score)
- Feature encoding
- Derived feature creation
- Data validation
- Summary statistics

**Usage:**
```python
from data_preprocessing import handle_missing_values, remove_outliers

df_clean = handle_missing_values(df)
df_clean = remove_outliers(df_clean, 'Wastage Food Amount')
```

---

### 8. compare_models.py
**Purpose:** Load and compare all trained models

**Features:**
- Evaluates all models in models/ directory
- Side-by-side metric comparison
- Identifies best model
- Exports comparison CSV

**Usage:**
```bash
python compare_models.py
```

---

### 9. setup.py
**Purpose:** Automated installation and setup

**Features:**
- Creates directory structure
- Installs dependencies
- Verifies installation
- Creates sample data (optional)
- Minimal or full installation

**Usage:**
```bash
python setup.py
```

---

### 10. run.py
**Purpose:** Master pipeline runner with menu

**Features:**
- Interactive menu system
- Run full pipeline
- Train models only
- Quick prediction mode
- Start API server
- Generate visualizations

**Usage:**
```bash
python run.py
```

**Menu Options:**
1. Run Full Pipeline
2. Train Models Only
3. Quick Prediction
4. Start API Server
5. Generate Visualizations
6. Compare Models
7. Setup/Install
0. Exit

---

### 11. requirements.txt
**Purpose:** Python package dependencies

**Contents:**
```
# Core (Required)
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0

# Advanced Models (Optional but Recommended)
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0

# Visualization (Optional)
matplotlib>=3.7.0
seaborn>=0.12.0

# API (Optional)
flask>=2.3.0
flask-cors>=4.0.0
```

---

## 🚀 Quick Start Guide

### Step 1: Setup
```bash
# Create directory and navigate
mkdir hotel-food-wastage-prediction
cd hotel-food-wastage-prediction

# Copy all scripts to this directory

# Run setup
python setup.py
```

### Step 2: Prepare Data
```bash
# Add your data file
# Place CSV at: data/hotel_food_waste.csv

# Or create sample data (for testing)
# This is done automatically in setup.py
```

### Step 3: Train Models
```bash
# Option A: Use master runner
python run.py
# Select option 1 (Full Pipeline)

# Option B: Train directly
python train_hotel_model_enhanced.py
```

### Step 4: Make Predictions
```bash
# Option A: Interactive
python run.py
# Select option 3 (Quick Prediction)

# Option B: Programmatic
python predict_hotel_wastage.py
```

### Step 5: Deploy (Optional)
```bash
# Start API server
python api_server.py

# Access at http://localhost:5000
```

---

## 📊 Data Format

### Required CSV Columns

Your `data/hotel_food_waste.csv` should have these columns:

1. **Wastage Food Amount** (float) - Target variable
2. **Type of Food** (string) - e.g., Rice, Biryani, Curry
3. **Number of Guests** (int) - Number of people
4. **Event Type** (string) - Wedding, Corporate, etc.
5. **Quantity of Food** (float) - Food prepared in kg
6. **Storage Conditions** (string) - Refrigerated, Room Temperature
7. **Purchase History** (string) - Bulk, Moderate, Just-in-Time
8. **Seasonality** (string) - Summer, Winter, Monsoon
9. **Preparation Method** (string) - Freshly Cooked, Pre-cooked
10. **Geographical Location** (string) - Urban, Rural
11. **Pricing** (float) - Cost per plate in rupees

### Example CSV Format:
```csv
Type of Food,Number of Guests,Event Type,Quantity of Food,Storage Conditions,Purchase History,Seasonality,Preparation Method,Geographical Location,Pricing,Wastage Food Amount
Rice,100,Wedding,50.0,Refrigerated,Bulk Purchase,Summer,Freshly Cooked,Urban,500.0,7.5
Biryani,150,Corporate Event,75.0,Room Temperature,Moderate Purchase,Winter,Pre-cooked,Urban,800.0,11.2
```

---

## 🔧 Troubleshooting

### Issue: Import errors
```bash
# Install missing packages
pip install package_name --break-system-packages
```

### Issue: Model not found
```bash
# Train the model first
python train_hotel_model_enhanced.py
```

### Issue: Data file not found
```bash
# Check file location
ls data/hotel_food_waste.csv

# Create sample data
python setup.py
# Choose option to create sample data
```

### Issue: Poor performance
- Need 100+ training samples minimum
- Check data quality
- Remove outliers
- Ensure features are relevant

---

## 📈 Performance Metrics Explained

### MAE (Mean Absolute Error)
- Average prediction error
- In kg of wastage
- Lower is better
- Example: MAE=2.5 means average error is 2.5 kg

### RMSE (Root Mean Squared Error)
- Penalizes large errors more
- In kg of wastage
- Lower is better
- More sensitive to outliers than MAE

### R² (R-squared)
- Variance explained by model
- Range: 0 to 1
- Higher is better
- Example: R²=0.92 means model explains 92% of variance

### MAPE (Mean Absolute Percentage Error)
- Error as percentage
- Easy to interpret
- Lower is better
- Example: MAPE=10% means average 10% error

---

## 🎯 Best Practices

1. **Data Collection**
   - Collect 100+ samples minimum
   - Ensure data quality
   - Include diverse scenarios
   - Regular updates

2. **Model Training**
   - Retrain monthly with new data
   - Monitor performance metrics
   - Keep old models for comparison
   - Document changes

3. **Production Deployment**
   - Use ensemble model for stability
   - Implement logging
   - Monitor predictions
   - Set up alerts

4. **Maintenance**
   - Track prediction errors
   - Collect feedback
   - Update features as needed
   - Regular retraining

---

## 📞 Getting Help

1. Check error messages carefully
2. Verify file locations
3. Ensure all packages installed
4. Review documentation files
5. Test with sample data first

---

**Last Updated:** January 2026
**Version:** 2.0
**Author:** Enhanced ML Pipeline
