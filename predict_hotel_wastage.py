"""
Enhanced Hotel Food Wastage Prediction Script
============================================
Load the trained model and make predictions with proper validation
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ================== #
MODEL_PATH = "models/hotel_model.pkl"
ENCODERS_PATH = "models/hotel_encoders.pkl"

# ================== LOAD MODEL ================== #

def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        print("✅ Model and encoders loaded successfully")
        return model, encoders
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please train the model first.")
        raise
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

# ================== PREDICTION FUNCTION ================== #

def predict_wastage(input_data, model, encoders):
    """
    Predict food wastage for given input data
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing input features
    model : trained model
        The trained ML model
    encoders : dict
        Dictionary of label encoders
        
    Returns:
    --------
    float : Predicted wastage amount
    """
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError as e:
                    # Handle unseen categories
                    print(f"⚠️  Warning: Unknown category in '{col}': {df[col].values[0]}")
                    print(f"   Known categories: {list(encoder.classes_)}")
                    # Use the most frequent category as fallback
                    df[col] = encoder.transform([encoder.classes_[0]])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        return prediction
        
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        raise

# ================== BATCH PREDICTION ================== #

def predict_batch(input_file, output_file, model, encoders):
    """
    Make predictions for a batch of inputs from CSV file
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to save predictions
    model : trained model
        The trained ML model
    encoders : dict
        Dictionary of label encoders
    """
    try:
        # Load input data
        df = pd.read_csv(input_file)
        print(f"📂 Loaded {len(df)} records from {input_file}")
        
        # Encode categorical features
        for col, encoder in encoders.items():
            if col in df.columns:
                # Handle unseen categories
                def safe_transform(x):
                    try:
                        return encoder.transform([str(x)])[0]
                    except ValueError:
                        return encoder.transform([encoder.classes_[0]])[0]
                
                df[col] = df[col].apply(safe_transform)
        
        # Make predictions
        predictions = model.predict(df)
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        
        # Add predictions to dataframe
        df['Predicted_Wastage'] = predictions
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"✅ Predictions saved to {output_file}")
        
        # Print summary statistics
        print(f"\n📊 Prediction Summary:")
        print(f"  Mean wastage: {predictions.mean():.2f}")
        print(f"  Std deviation: {predictions.std():.2f}")
        print(f"  Min wastage: {predictions.min():.2f}")
        print(f"  Max wastage: {predictions.max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {str(e)}")
        raise

# ================== EXAMPLE USAGE ================== #

def main():
    """Example usage of the prediction functions"""
    
    # Load model and encoders
    model, encoders = load_model_and_encoders()
    
    # Example 1: Single prediction
    print("\n" + "="*70)
    print("  EXAMPLE 1: Single Prediction")
    print("="*70)
    
    # Replace with your actual feature names and values
    sample_input = {
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
        # Add all your features here
    }
    
    try:
        prediction = predict_wastage(sample_input, model, encoders)
        print(f"\n🎯 Predicted Wastage: {prediction:.2f} kg")
        print(f"   Wastage Percentage: {(prediction / sample_input['Quantity of Food']) * 100:.2f}%")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
    
    # Example 2: Batch prediction (uncomment to use)
    """
    print("\n" + "="*70)
    print("  EXAMPLE 2: Batch Prediction")
    print("="*70)
    
    input_csv = "data/hotel_test_data.csv"
    output_csv = "predictions/hotel_predictions.csv"
    
    try:
        results = predict_batch(input_csv, output_csv, model, encoders)
        print(f"\n✅ Batch prediction completed successfully!")
    except Exception as e:
        print(f"Batch prediction failed: {str(e)}")
    """

# ================== INTERACTIVE PREDICTION ================== #

def interactive_prediction():
    """Interactive prediction mode"""
    print("\n" + "="*70)
    print("  🎯 INTERACTIVE HOTEL FOOD WASTAGE PREDICTION")
    print("="*70 + "\n")
    
    # Load model
    model, encoders = load_model_and_encoders()
    
    # Get input from user
    input_data = {}
    
    print("Please enter the following information:\n")
    
    # You'll need to customize this based on your actual features
    # Here's a template:
    
    features_to_input = {
        'Type of Food': str,
        'Number of Guests': int,
        'Event Type': str,
        'Quantity of Food': float,
        'Storage Conditions': str,
        'Purchase History': str,
        'Seasonality': str,
        'Preparation Method': str,
        'Geographical Location': str,
        'Pricing': float
    }
    
    for feature, dtype in features_to_input.items():
        while True:
            try:
                value = input(f"  {feature}: ")
                input_data[feature] = dtype(value)
                break
            except ValueError:
                print(f"  ⚠️  Invalid input. Please enter a valid {dtype.__name__}")
    
    # Make prediction
    try:
        prediction = predict_wastage(input_data, model, encoders)
        print("\n" + "="*70)
        print(f"  🎯 PREDICTED WASTAGE: {prediction:.2f} kg")
        if 'Quantity of Food' in input_data:
            wastage_pct = (prediction / input_data['Quantity of Food']) * 100
            print(f"  📊 WASTAGE PERCENTAGE: {wastage_pct:.2f}%")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run interactive mode
    # interactive_prediction()
