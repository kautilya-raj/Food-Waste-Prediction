import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("data/wedding_food_waste.csv")

# ---------------- FEATURES & TARGET ---------------- #
X = df[
    [
        "Event_Type",
        "Guests_Count",
        "Menu_Type",
        "Meal_Type",
        "Season",
        "Food_Prepared_kg"
    ]
]

y = df["Food_Wasted_kg"]

# ---------------- COLUMN TYPES ---------------- #
categorical_features = [
    "Event_Type",
    "Menu_Type",
    "Meal_Type",
    "Season"
]

numeric_features = [
    "Guests_Count",
    "Food_Prepared_kg"
]

# ---------------- PREPROCESSING ---------------- #
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# ---------------- MODEL ---------------- #
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# ---------------- PIPELINE ---------------- #
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ---------------- TRAIN / TEST SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ---------------- #
pipeline.fit(X_train, y_train)

# ---------------- EVALUATE ---------------- #
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Wedding Model MAE: {mae:.2f} kg")

# ---------------- SAVE MODEL ---------------- #
joblib.dump(pipeline, "models/wedding_model.pkl")
print("Wedding model saved at models/wedding_model.pkl")

