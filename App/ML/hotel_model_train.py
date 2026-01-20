import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("data/hotel_food_waste.csv")

# ---------------- FEATURES & TARGET ---------------- #
X = df.drop("Wastage Food Amount", axis=1)
y = df["Wastage Food Amount"]

# ---------------- ENCODE CATEGORICAL FEATURES ---------------- #
label_encoders = {}

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# ---------------- TRAIN / TEST SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ---------------- #
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- SAVE MODEL & ENCODERS ---------------- #
joblib.dump(model, "models/hotel_model.pkl")
joblib.dump(label_encoders, "models/hotel_encoders.pkl")

print("✅ Hotel regression model trained and saved successfully")

