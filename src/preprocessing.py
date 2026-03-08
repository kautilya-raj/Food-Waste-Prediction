"""preprocessing.py — feature encoding, scaling, synthetic data."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = {
    "hostel":    ["num_students","meal_type","day_of_week","season","special_occasion","leftover_yesterday"],
    "hotel":     ["num_guests","meal_type","day_of_week","season","event_type","buffet_style","avg_rating"],
    "wedding":   ["num_guests","cuisine_type","num_dishes","catering_experience_years","outdoor_event","season"],
    "household": ["family_size","meal_type","day_of_week","shopping_frequency","income_level","season"],
}
TARGET_COLUMN = "food_waste_kg"

def generate_synthetic_data(venue_type, n_samples=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    vt  = venue_type.lower()
    if vt == "hostel":
        df = pd.DataFrame({
            "num_students":       rng.integers(50,500,n_samples),
            "meal_type":          rng.choice(["breakfast","lunch","dinner"],n_samples),
            "day_of_week":        rng.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],n_samples),
            "season":             rng.choice(["summer","winter","monsoon","spring"],n_samples),
            "special_occasion":   rng.integers(0,2,n_samples),
            "leftover_yesterday": rng.uniform(0,20,n_samples).round(2),
        })
        df[TARGET_COLUMN] = (df["num_students"]*rng.uniform(0.04,0.12,n_samples)
            + df["leftover_yesterday"]*0.3 + df["special_occasion"]*rng.uniform(5,15,n_samples)
            + rng.normal(0,2,n_samples)).clip(0).round(2)
    elif vt == "hotel":
        df = pd.DataFrame({
            "num_guests":   rng.integers(20,300,n_samples),
            "meal_type":    rng.choice(["breakfast","lunch","dinner","brunch"],n_samples),
            "day_of_week":  rng.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],n_samples),
            "season":       rng.choice(["summer","winter","monsoon","spring"],n_samples),
            "event_type":   rng.choice(["conference","wedding","regular","gala"],n_samples),
            "buffet_style": rng.integers(0,2,n_samples),
            "avg_rating":   rng.uniform(2.5,5.0,n_samples).round(1),
        })
        df[TARGET_COLUMN] = (df["num_guests"]*rng.uniform(0.1,0.3,n_samples)
            + df["buffet_style"]*rng.uniform(10,30,n_samples)
            + rng.normal(0,3,n_samples)).clip(0).round(2)
    elif vt == "wedding":
        df = pd.DataFrame({
            "num_guests":                rng.integers(50,1000,n_samples),
            "cuisine_type":              rng.choice(["indian","continental","chinese","mixed"],n_samples),
            "num_dishes":                rng.integers(5,40,n_samples),
            "catering_experience_years": rng.integers(1,30,n_samples),
            "outdoor_event":             rng.integers(0,2,n_samples),
            "season":                    rng.choice(["summer","winter","monsoon","spring"],n_samples),
        })
        df[TARGET_COLUMN] = (df["num_guests"]*rng.uniform(0.15,0.4,n_samples)
            + df["num_dishes"]*rng.uniform(1,3,n_samples)
            + df["outdoor_event"]*rng.uniform(5,20,n_samples)
            + rng.normal(0,5,n_samples)).clip(0).round(2)
    elif vt == "household":
        df = pd.DataFrame({
            "family_size":        rng.integers(1,10,n_samples),
            "meal_type":          rng.choice(["breakfast","lunch","dinner"],n_samples),
            "day_of_week":        rng.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],n_samples),
            "shopping_frequency": rng.choice(["daily","weekly","biweekly","monthly"],n_samples),
            "income_level":       rng.choice(["low","medium","high"],n_samples),
            "season":             rng.choice(["summer","winter","monsoon","spring"],n_samples),
        })
        df[TARGET_COLUMN] = (df["family_size"]*rng.uniform(0.05,0.2,n_samples)
            + rng.normal(0,0.5,n_samples)).clip(0).round(2)
    else:
        raise ValueError(f"Unknown venue_type: {venue_type!r}")
    return df

class FoodWastePreprocessor:
    def __init__(self):
        self.encoders = {}; self.scaler = StandardScaler()
        self.feature_names = []; self.is_fitted = False
    def fit_transform(self, df, venue_type):
        self.feature_names = FEATURE_COLUMNS[venue_type.lower()]
        X = self._encode(df[self.feature_names].copy(), fit=True)
        y = df[TARGET_COLUMN].values
        X = self.scaler.fit_transform(X)
        self.is_fitted = True
        return train_test_split(X, y, test_size=0.2, random_state=42)
    def transform(self, df):
        if not self.is_fitted: raise RuntimeError("Preprocessor not fitted.")
        return self.scaler.transform(self._encode(df[self.feature_names].copy(), fit=False))
    def _encode(self, X, fit):
        X = X.copy()
        for col in X.select_dtypes(include=["object","category"]).columns:
            if fit:
                le = LabelEncoder(); X[col] = le.fit_transform(X[col].astype(str)); self.encoders[col] = le
            else:
                le = self.encoders[col]; known = set(le.classes_)
                X[col] = X[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
                X[col] = le.transform(X[col])
        return X
