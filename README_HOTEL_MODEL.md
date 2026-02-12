# 🍽️ Enhanced Hotel Food Wastage Prediction System

## Overview
This enhanced machine learning system predicts hotel food wastage with high accuracy using multiple state-of-the-art algorithms including Random Forest, XGBoost, LightGBM, CatBoost, and ensemble methods.

## 🚀 Key Improvements Over Original Version

### 1. **Multiple Advanced Models**
- ✅ Random Forest with hyperparameter tuning (GridSearchCV)
- ✅ Gradient Boosting Regressor
- ✅ XGBoost (if installed)
- ✅ LightGBM (if installed)
- ✅ CatBoost (if installed)
- ✅ Ensemble model combining best performers

### 2. **Robust Evaluation**
- Cross-validation (5-fold CV by default)
- Multiple metrics: MAE, RMSE, R², MAPE
- Train/test performance comparison
- Comprehensive model comparison

### 3. **Better Error Handling**
- Data validation and missing value handling
- Unseen category handling in predictions
- Comprehensive error messages
- Input validation

### 4. **Feature Analysis**
- Feature importance ranking
- Detailed feature contribution analysis
- CSV export of importance scores

### 5. **Production Ready**
- Batch prediction support
- Interactive prediction mode
- Multiple model formats saved
- Comprehensive logging

## 📋 Requirements

### Minimum Requirements
```bash
pip install pandas numpy scikit-learn joblib
```

### Recommended (for best performance)
```bash
pip install -r requirements.txt
```

This includes:
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0 (optional)
- lightgbm >= 4.0.0 (optional)
- catboost >= 1.2.0 (optional)

## 📁 Project Structure

```
project/
├── data/
│   └── hotel_food_waste.csv          # Training data
├── models/
│   ├── hotel_model.pkl                # Best model (auto-selected)
│   ├── hotel_encoders.pkl             # Label encoders
│   ├── hotel_model_ensemble.pkl       # Ensemble model
│   ├── hotel_model_*.pkl              # Individual models
│   ├── model_comparison_metrics.csv   # Performance comparison
│   └── feature_importance.csv         # Feature rankings
├── train_hotel_model_enhanced.py      # Training script
├── predict_hotel_wastage.py           # Prediction script
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## 🎓 Usage

### 1. Training the Model

```bash
python train_hotel_model_enhanced.py
```

**What happens:**
1. Loads and validates data from `data/hotel_food_waste.csv`
2. Handles missing values automatically
3. Encodes categorical features
4. Trains multiple models with cross-validation
5. Performs hyperparameter tuning
6. Compares all models
7. Saves best model and all artifacts

**Output:**
```
✅ TRAINING COMPLETED SUCCESSFULLY!
Best Model: XGBoost (Test MAE: 2.1234)
Test R²: 0.9456
Test MAPE: 8.32%
```

### 2. Making Predictions

#### Single Prediction
```python
from predict_hotel_wastage import load_model_and_encoders, predict_wastage

# Load model
model, encoders = load_model_and_encoders()

# Prepare input
input_data = {
    'Type of Food': 'Rice',
    'Number of Guests': 100,
    'Event Type': 'Wedding',
    'Quantity of Food': 50.0,
    'Storage Conditions': 'Refrigerated',
    'Purchase History': 'Bulk Purchase',
    'Seasonality': 'Summer',
    'Preparation Method': 'Freshly Cooked',
    'Geographical Location': 'Urban',
    'Pricing': 500.0
}

# Predict
wastage = predict_wastage(input_data, model, encoders)
print(f"Predicted Wastage: {wastage:.2f} kg")
```

#### Batch Prediction
```python
from predict_hotel_wastage import load_model_and_encoders, predict_batch

model, encoders = load_model_and_encoders()

predict_batch(
    input_file='data/hotel_test_data.csv',
    output_file='predictions/results.csv',
    model=model,
    encoders=encoders
)
```

#### Interactive Mode
```bash
python predict_hotel_wastage.py
# Follow the prompts to enter values
```

## 📊 Model Performance Metrics

The system provides comprehensive metrics:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error - average prediction error |
| **RMSE** | Root Mean Squared Error - penalizes large errors |
| **R²** | R-squared score - variance explained (0-1) |
| **MAPE** | Mean Absolute Percentage Error |
| **CV Score** | Cross-validation score with std deviation |

### Expected Performance
With proper data, you should see:
- Test R² > 0.85
- Test MAE < 5% of mean wastage
- Test MAPE < 15%

## 🔧 Troubleshooting

### Issue: Import Error for XGBoost/LightGBM/CatBoost
**Solution:** These are optional. The system will work with just Random Forest and Gradient Boosting.
```bash
# Install individually if needed
pip install xgboost
pip install lightgbm
pip install catboost
```

### Issue: "File not found" error
**Solution:** Ensure your data is at `data/hotel_food_waste.csv`
```bash
mkdir -p data models predictions
# Place your CSV in the data folder
```

### Issue: Poor model performance
**Possible causes:**
1. Insufficient training data (need 100+ samples)
2. High correlation between features
3. Target variable has extreme outliers
4. Categorical features with too many unique values

**Solutions:**
- Collect more data
- Check for data quality issues
- Remove outliers or use robust scaling
- Consider feature engineering

### Issue: Prediction errors for new data
**Solution:** The system handles unseen categories automatically by using the most common category.

## 🎯 Best Practices

### Data Preparation
1. **Minimum samples:** 100+ rows for reliable training
2. **Feature quality:** Ensure features are relevant to wastage
3. **Target distribution:** Check for outliers in wastage amounts
4. **Missing values:** Will be handled automatically, but better to clean beforehand

### Model Selection
1. Start with Random Forest (most robust)
2. Try XGBoost for better performance
3. Use ensemble for production (combines strengths)

### Hyperparameter Tuning
Modify `rf_params` in training script for different settings:
```python
rf_params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

## 📈 Feature Importance

After training, check `models/feature_importance.csv` to see which features matter most:

```csv
Feature,Importance
Quantity of Food,0.3521
Number of Guests,0.2134
Event Type,0.1876
Pricing,0.1234
...
```

Use this to:
- Focus data collection on important features
- Remove irrelevant features
- Understand wastage drivers

## 🔄 Model Updates

To retrain with new data:
1. Add new records to `data/hotel_food_waste.csv`
2. Run training script again
3. Compare metrics with previous version
4. Deploy if performance improves

## 💡 Tips for Best Results

1. **Regular Updates:** Retrain monthly with new data
2. **Feature Engineering:** Create interaction features (e.g., wastage_per_guest)
3. **Seasonality:** Consider adding temporal features
4. **Validation:** Test on recent data before deployment
5. **Monitoring:** Track prediction errors in production

## 📞 Support

For issues or questions:
1. Check the error message carefully
2. Verify data format matches training data
3. Ensure all required packages are installed
4. Check model files exist in `models/` directory

## 🎉 What's Next?

Consider these enhancements:
- [ ] Add time-series forecasting for future wastage
- [ ] Implement SHAP values for explainability
- [ ] Create web API using Flask/FastAPI
- [ ] Add confidence intervals to predictions
- [ ] Implement automated retraining pipeline
- [ ] Add data drift detection

## 📝 License

This is an educational/research project. Modify as needed for your use case.

---

**Happy Predicting! 🚀**
