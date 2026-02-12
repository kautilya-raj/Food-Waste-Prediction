"""
Flask API for Hotel Food Wastage Prediction
==========================================
Deploy the model as a REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ================== CONFIGURATION ================== #
app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/hotel_model.pkl"
ENCODERS_PATH = "models/hotel_encoders.pkl"

# ================== LOAD MODEL ON STARTUP ================== #
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    logger.info("✅ Model and encoders loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    model = None
    encoders = None

# ================== HELPER FUNCTIONS ================== #

def validate_input(data):
    """Validate input data"""
    required_fields = list(encoders.keys())
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    return True, "Valid"


def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical features
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    logger.warning(f"Unknown category in '{col}': {df[col].values[0]}")
                    df[col] = encoder.transform([encoder.classes_[0]])
        
        return df, None
    
    except Exception as e:
        return None, str(e)


# ================== API ENDPOINTS ================== #

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Hotel Food Wastage Prediction API',
        'version': '2.0',
        'status': 'active' if model is not None else 'error',
        'endpoints': {
            '/predict': 'POST - Make a single prediction',
            '/predict/batch': 'POST - Make batch predictions',
            '/health': 'GET - Check API health',
            '/features': 'GET - Get required features'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'encoders_loaded': encoders is not None
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Get list of required features"""
    if encoders is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    features = {}
    for col, encoder in encoders.items():
        features[col] = {
            'type': 'categorical',
            'options': encoder.classes_.tolist()
        }
    
    return jsonify({
        'required_features': list(encoders.keys()),
        'feature_details': features
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    
    Expected JSON format:
    {
        "Type of Food": "Rice",
        "Number of Guests": 100,
        "Event Type": "Wedding",
        ...
    }
    """
    if model is None or encoders is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Preprocess
        df, error = preprocess_input(data)
        if error:
            return jsonify({'error': f'Preprocessing error: {error}'}), 400
        
        # Make prediction
        prediction = model.predict(df)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Calculate wastage percentage if quantity is provided
        wastage_percentage = None
        if 'Quantity of Food' in data:
            quantity = float(data['Quantity of Food'])
            if quantity > 0:
                wastage_percentage = (prediction / quantity) * 100
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'wastage_amount': round(float(prediction), 2),
                'unit': 'kg',
                'wastage_percentage': round(wastage_percentage, 2) if wastage_percentage else None
            },
            'input': data
        }
        
        logger.info(f"Prediction made: {prediction:.2f} kg")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON format:
    {
        "data": [
            {"Type of Food": "Rice", ...},
            {"Type of Food": "Biryani", ...},
            ...
        ]
    }
    """
    if model is None or encoders is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'No data provided. Expected {"data": [...]}'}), 400
        
        batch_data = request_data['data']
        
        if not isinstance(batch_data, list):
            return jsonify({'error': 'Data must be a list of objects'}), 400
        
        predictions = []
        errors = []
        
        # Process each item
        for idx, item in enumerate(batch_data):
            try:
                # Validate
                is_valid, message = validate_input(item)
                if not is_valid:
                    errors.append({'index': idx, 'error': message})
                    continue
                
                # Preprocess
                df, error = preprocess_input(item)
                if error:
                    errors.append({'index': idx, 'error': f'Preprocessing error: {error}'})
                    continue
                
                # Predict
                prediction = model.predict(df)[0]
                prediction = max(0, prediction)
                
                # Calculate percentage
                wastage_percentage = None
                if 'Quantity of Food' in item:
                    quantity = float(item['Quantity of Food'])
                    if quantity > 0:
                        wastage_percentage = (prediction / quantity) * 100
                
                predictions.append({
                    'index': idx,
                    'wastage_amount': round(float(prediction), 2),
                    'wastage_percentage': round(wastage_percentage, 2) if wastage_percentage else None
                })
            
            except Exception as e:
                errors.append({'index': idx, 'error': str(e)})
        
        response = {
            'success': True,
            'total_items': len(batch_data),
            'successful_predictions': len(predictions),
            'failed_predictions': len(errors),
            'predictions': predictions,
            'errors': errors if errors else None
        }
        
        logger.info(f"Batch prediction: {len(predictions)} successful, {len(errors)} failed")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


# ================== ERROR HANDLERS ================== #

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ================== MAIN ================== #

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🚀 STARTING HOTEL FOOD WASTAGE PREDICTION API")
    print("="*70 + "\n")
    print("  API will be available at: http://localhost:5000")
    print("\n  Endpoints:")
    print("    - GET  /              : API information")
    print("    - GET  /health        : Health check")
    print("    - GET  /features      : Required features")
    print("    - POST /predict       : Single prediction")
    print("    - POST /predict/batch : Batch predictions")
    print("\n" + "="*70 + "\n")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
