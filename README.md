# 🍽️ Food Waste Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org)

## 📋 Overview

Machine Learning-based system to predict food wastage for hotels, hostels, weddings, and households using advanced ML algorithms (Random Forest, XGBoost, LightGBM, CatBoost).

## ✨ Features

- 🤖 **6 ML Models** - Auto-selects best performer
- 📊 **Advanced Analytics** - Feature importance & visualizations
- 🌐 **REST API** - Flask-based web API
- 📈 **Interactive UI** - Web interface for predictions
- 🎯 **High Accuracy** - 90%+ R² score on test data
- 📉 **Batch Predictions** - Process multiple inputs at once

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/kautilya-raj/Food-Waste-Prediction.git
cd Food-Waste-Prediction

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py
```

### Usage
```bash
# Train models
python train_hotel_model_enhanced.py

# Make predictions
python predict_hotel_wastage.py

# Start web app
python main_app.py

# Or use menu system
python run.py
```

## 📁 Project Structure
```
Food-Waste-Prediction/
├── App/                    # Flask web application
├── data/                   # Training data directory
├── models/                 # Trained ML models
├── visualizations/         # Generated charts
├── predictions/            # Prediction outputs
├── config/                 # Configuration files
├── train_*.py              # Training scripts
├── predict_*.py            # Prediction scripts
└── api_server.py           # REST API server
```

## 📊 Models Used

1. **Random Forest** - With hyperparameter tuning
2. **Gradient Boosting** - Sequential ensemble
3. **XGBoost** - Extreme Gradient Boosting
4. **LightGBM** - Fast gradient boosting
5. **CatBoost** - Categorical boosting
6. **Ensemble** - Voting regressor combining top models

## 🎯 Performance

- **Test R²:** 0.90-0.95
- **MAE:** 20-40% better than baseline
- **Cross-validation:** 5-fold CV validated

## 📚 Documentation

- [Complete Guide](README_HOTEL_MODEL.md)
- [Upgrade Summary](UPGRADE_SUMMARY.md)
- [File Structure](FILE_STRUCTURE.md)

## 🔗 API Endpoints
```bash
GET  /              # API info
GET  /health        # Health check
POST /predict       # Single prediction
POST /predict/batch # Batch predictions
```

## 🌟 Demo

[Add screenshot or demo link here]

## 📝 Data Format

Required CSV columns:
- Type of Food
- Number of Guests
- Event Type
- Quantity of Food
- Storage Conditions
- Purchase History
- Seasonality
- Preparation Method
- Geographical Location
- Pricing
- Wastage Food Amount (target)

## 👥 Contributors

- [@kautilya-raj](https://github.com/kautilya-raj)

## 📄 License

[Specify your license]

## 🙏 Acknowledgments

Built with enhanced ML pipeline featuring state-of-the-art algorithms and comprehensive evaluation metrics.

---

**⭐ Star this repo if you find it useful!**
