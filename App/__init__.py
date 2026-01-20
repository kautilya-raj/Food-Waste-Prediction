import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS

from App.Controllers import (
    app_info_blueprint,
    data_collector_blueprint,
    preprocessing_blueprint,
    predictor_blueprint
)


def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    CORS(app)

    # ---------------- REGISTER API BLUEPRINTS ---------------- #
    app.register_blueprint(app_info_blueprint)
    app.register_blueprint(data_collector_blueprint)
    app.register_blueprint(preprocessing_blueprint)
    app.register_blueprint(predictor_blueprint)
    # --------------------------------------------------------- #

    # ==================== UI ROUTES ==================== #

    @app.route("/ui")
    def ui_home():
        return render_template("menu.html")

    @app.route("/ui/predict/<place>", methods=["GET", "POST"])
    def ui_predict(place):
        result = None
        place = place.lower()

        # ==================== HOTEL (UNCHANGED) ==================== #
        if request.method == "POST" and place == "hotel":

            hotel_data = {
                "type_of_food": request.form.get("type_of_food"),
                "num_guests": int(request.form.get("num_guests")),
                "event_type": request.form.get("event_type"),
                "quantity_food": float(request.form.get("quantity_food")),
                "storage_condition": request.form.get("storage_condition"),
                "purchase_history": request.form.get("purchase_history"),
                "seasonality": request.form.get("seasonality"),
                "prep_method": request.form.get("prep_method"),
                "location": request.form.get("location"),
                "pricing": float(request.form.get("pricing"))
            }

            food_per_guest = hotel_data["quantity_food"] / hotel_data["num_guests"]

            if food_per_guest > 0.9:
                result = "HIGH Food Wastage Expected"
            elif food_per_guest > 0.6:
                result = "MEDIUM Food Wastage Expected"
            else:
                result = "LOW Food Wastage Expected"

            return render_template("index.html", place="Hotel", result=result)

        # ==================== WEDDING (ML ONLY) ==================== #
        if request.method == "POST" and place == "wedding":

            model = joblib.load("models/wedding_model.pkl")

            input_data = pd.DataFrame([{
                "Event_Type": request.form.get("event_type"),
                "Guests_Count": int(request.form.get("guests_count")),
                "Menu_Type": request.form.get("menu_type"),
                "Meal_Type": request.form.get("meal_type"),
                "Season": request.form.get("season"),
                "Food_Prepared_kg": float(request.form.get("food_prepared_kg"))
            }])

            predicted_waste = model.predict(input_data)[0]
            waste_percentage = (predicted_waste / input_data["Food_Prepared_kg"][0]) * 100

            result = (
                f"Estimated Food Waste: {predicted_waste:.2f} kg "
                f"({waste_percentage:.1f}%)"
            )

            return render_template("wedding.html", result=result)

        # ==================== GET REQUESTS ==================== #
        if request.method == "GET":

            if place == "wedding":
                return render_template("wedding.html")

            return render_template("index.html", place=place.title())

        return render_template("index.html", place=place.title(), result=result)

    return app

