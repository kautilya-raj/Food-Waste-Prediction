# 🌿 FoodWise — Food Waste Prediction System

> An ML-powered web application that predicts food waste for hostels, hotels, weddings,
> and households — and gives actionable tips to reduce it.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Description

FoodWise uses machine learning regression models to predict **how many kilograms of food
will be wasted** at a given event or setting. Users select their venue type, fill in
a simple form, and instantly receive a prediction along with personalised reduction tips.

The system supports four venue categories:

| Venue | Description |
|-------|-------------|
| 🏫 Hostel / Mess | Predicts daily mess waste based on student count, meal type, and special events |
| 🏨 Hotel / Restaurant | Predicts buffet and banquet waste based on guest count and event type |
| 💍 Wedding / Event | Predicts large-scale catering waste for weddings and gatherings |
| 🏠 Household | Predicts daily household food waste based on family size and habits |

---

## ✨ Features

- **4 ML models** (one per venue type) — automatically selects the best from Random Forest,
  Gradient Boosting, Ridge Regression, and an Ensemble
- **REST API** — JSON endpoint for external integrations
- **Modern responsive UI** — clean, mobile-friendly Flask + HTML/CSS frontend
- **Actionable tips** — contextual waste-reduction advice per prediction
- **One-click training** — `/train` route re-trains all models from the browser
- **Auto-training on startup** — server trains models if none are found
- **Input validation** — comprehensive server-side and client-side checks
- **Rotating log files** — structured logging to `logs/app.log`

---

## 📁 Folder Structure

```
food-waste-prediction/
│
├── data/
│   └── sample/              # Place your CSV datasets here (optional)
│
├── models/                  # Saved .pkl model bundles (auto-generated after training)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Feature engineering, encoding, scaling, data generation
│   ├── training.py          # Model training pipeline + evaluation
│   ├── inference.py         # Prediction + tip generation
│   └── utils.py             # Logging, input validation, field specs
│
├── app/
│   ├── __init__.py
│   ├── app.py               # Flask routes (web UI + JSON API)
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── predict_form.html
│   │   ├── result.html
│   │   ├── train_result.html
│   │   └── error.html
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── notebooks/               # Jupyter notebooks for EDA (optional)
├── logs/                    # Rotating log files (auto-created)
│
├── main.py                  # CLI entrypoint
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/food-waste-prediction.git
cd food-waste-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — Web Server (recommended)
```bash
python main.py runserver
```
Then open **http://localhost:5000** in your browser.

> ℹ️ Models are auto-trained on first startup if not already present.

### Option B — Train models first, then run
```bash
python main.py train
python main.py runserver
```

### Option C — CLI interactive prediction
```bash
python main.py predict
```

### Option D — Docker
```bash
docker build -t foodwise .
docker run -p 5000:5000 foodwise
```

---

## 🔌 REST API

### Predict (POST)
```http
POST /api/predict
Content-Type: application/json

{
  "venue_type": "hotel",
  "num_guests": 120,
  "meal_type": "dinner",
  "day_of_week": "Sat",
  "season": "winter",
  "event_type": "gala",
  "buffet_style": 1,
  "avg_rating": 4.2
}
```

**Response:**
```json
{
  "venue_type": "hotel",
  "prediction_kg": 28.4,
  "model_name": "Ensemble",
  "r2_score": 0.912,
  "tips": [
    "🟠 High waste predicted...",
    "🍽️ Buffet setups typically waste more..."
  ]
}
```

### Get model info (GET)
```http
GET /api/model-info
```

### Trigger training (POST)
```http
POST /api/train
```

---

## 📊 Model Performance

After training on 2,000 synthetic samples per venue type, typical metrics:

| Venue     | Model            | R²    | MAE  |
|-----------|------------------|-------|------|
| Hostel    | Gradient Boosting| ~0.93 | ~2.1 |
| Hotel     | Ensemble         | ~0.91 | ~3.4 |
| Wedding   | Random Forest    | ~0.94 | ~6.2 |
| Household | Ridge Regression | ~0.89 | ~0.3 |

> Metrics are approximate and vary with random seed.

---

## 📦 Requirements

- Python 3.10+
- Flask 3.0
- scikit-learn 1.5
- numpy, pandas
- gunicorn (production)

See `requirements.txt` for pinned versions.

---

## 🌱 Using Your Own Data

Replace synthetic data with real CSVs by modifying `src/training.py`:

```python
# Instead of:
df = generate_synthetic_data(venue_type, ...)

# Load your CSV:
df = pd.read_csv(f"data/{venue_type}.csv")
```

Ensure your CSV has the feature columns listed in `src/preprocessing.py → FEATURE_COLUMNS`
plus a `food_waste_kg` target column.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
