#!/usr/bin/env python3
"""
Master Run Script - Complete ML Pipeline
========================================
Run the entire ML pipeline from start to finish
"""

import subprocess
import sys
from pathlib import Path
import time

def print_header(text, char="="):
    """Print formatted header"""
    print("\n" + char*70)
    print(f"  {text}")
    print(char*70 + "\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, command],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - COMPLETED\n")
            return True
        else:
            print(f"\n❌ {description} - FAILED\n")
            return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        return False

def check_prerequisites():
    """Check if prerequisites are met"""
    print_header("🔍 CHECKING PREREQUISITES")
    
    checks = []
    
    # Check data file
    data_file = Path("data/hotel_food_waste.csv")
    if data_file.exists():
        print("✅ Data file found")
        checks.append(True)
    else:
        print("❌ Data file not found: data/hotel_food_waste.csv")
        checks.append(False)
    
    # Check required packages
    required = ['pandas', 'numpy', 'sklearn', 'joblib']
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package} installed")
            checks.append(True)
        except ImportError:
            print(f"❌ {package} not installed")
            checks.append(False)
    
    return all(checks)

def run_full_pipeline():
    """Run the complete ML pipeline"""
    
    print_header("🚀 HOTEL FOOD WASTAGE ML PIPELINE", "=")
    print("This will run the complete machine learning pipeline:\n")
    print("  1. Train models")
    print("  2. Compare performance")
    print("  3. Generate visualizations")
    print("  4. Test predictions\n")
    
    input("Press Enter to start...")
    
    start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please run setup.py first.\n")
        return
    
    # Step 1: Train models
    success = run_command(
        "train_hotel_model_enhanced.py",
        "STEP 1: Training Models"
    )
    
    if not success:
        print("❌ Training failed. Aborting pipeline.")
        return
    
    # Step 2: Compare models (optional - if file exists)
    if Path("compare_models.py").exists():
        run_command(
            "compare_models.py",
            "STEP 2: Comparing Models"
        )
    
    # Step 3: Visualize
    if Path("visualize_models.py").exists():
        run_command(
            "visualize_models.py",
            "STEP 3: Generating Visualizations"
        )
    
    # Step 4: Test prediction
    print_header("STEP 4: Testing Predictions")
    print("Running a test prediction...\n")
    
    try:
        from predict_hotel_wastage import load_model_and_encoders, predict_wastage
        
        model, encoders = load_model_and_encoders()
        
        # Sample input
        test_input = {
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
        
        prediction = predict_wastage(test_input, model, encoders)
        
        print("Test Input:")
        for key, value in test_input.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎯 Predicted Wastage: {prediction:.2f} kg")
        print(f"   Wastage %: {(prediction/test_input['Quantity of Food'])*100:.2f}%")
        
        print("\n✅ Prediction test - COMPLETED\n")
        
    except Exception as e:
        print(f"❌ Prediction test failed: {str(e)}\n")
    
    # Summary
    elapsed = time.time() - start_time
    
    print_header("📊 PIPELINE SUMMARY", "=")
    print(f"⏱️  Total Time: {elapsed:.2f} seconds\n")
    print("Generated Files:")
    
    files_to_check = [
        "models/hotel_model.pkl",
        "models/hotel_encoders.pkl",
        "models/model_comparison_metrics.csv",
        "visualizations/model_comparison.png",
        "visualizations/feature_importance.png"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"  ✅ {file_path} ({size:.1f} KB)")
        else:
            print(f"  ⚠️  {file_path} (not found)")
    
    print("\n" + "="*70)
    print("  ✅ PIPELINE COMPLETED!")
    print("="*70 + "\n")
    
    print("Next Steps:")
    print("  - Check visualizations/ for charts")
    print("  - Review models/model_comparison_metrics.csv")
    print("  - Use predict_hotel_wastage.py for predictions")
    print("  - Run api_server.py to deploy as API\n")

def run_quick_prediction():
    """Quick prediction mode"""
    print_header("🎯 QUICK PREDICTION MODE")
    
    try:
        from predict_hotel_wastage import load_model_and_encoders, predict_wastage
        
        model, encoders = load_model_and_encoders()
        print("✅ Model loaded\n")
        
        print("Enter prediction details:\n")
        
        input_data = {}
        
        # Get user input
        fields = {
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
        
        for field, dtype in fields.items():
            while True:
                try:
                    value = input(f"  {field}: ")
                    input_data[field] = dtype(value)
                    break
                except ValueError:
                    print(f"    ⚠️  Invalid input. Please enter a {dtype.__name__}")
        
        # Predict
        prediction = predict_wastage(input_data, model, encoders)
        
        print("\n" + "="*70)
        print(f"  🎯 PREDICTED WASTAGE: {prediction:.2f} kg")
        if input_data['Quantity of Food'] > 0:
            pct = (prediction / input_data['Quantity of Food']) * 100
            print(f"  📊 WASTAGE PERCENTAGE: {pct:.2f}%")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("❌ Model not found. Please train the model first.\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")

def main():
    """Main menu"""
    
    while True:
        print("\n" + "="*70)
        print("  🍽️  HOTEL FOOD WASTAGE PREDICTION SYSTEM")
        print("="*70 + "\n")
        
        print("Options:")
        print("  1. Run Full Pipeline (Train → Visualize → Test)")
        print("  2. Train Models Only")
        print("  3. Quick Prediction (Interactive)")
        print("  4. Start API Server")
        print("  5. Generate Visualizations")
        print("  6. Compare Models")
        print("  7. Setup/Install")
        print("  0. Exit\n")
        
        choice = input("Enter choice: ")
        
        if choice == '1':
            run_full_pipeline()
        
        elif choice == '2':
            run_command("train_hotel_model_enhanced.py", "Training Models")
        
        elif choice == '3':
            run_quick_prediction()
        
        elif choice == '4':
            print("\n🚀 Starting API Server...")
            print("Press Ctrl+C to stop\n")
            try:
                subprocess.run([sys.executable, "api_server.py"])
            except KeyboardInterrupt:
                print("\n\n✅ API Server stopped\n")
        
        elif choice == '5':
            run_command("visualize_models.py", "Generating Visualizations")
        
        elif choice == '6':
            if Path("compare_models.py").exists():
                run_command("compare_models.py", "Comparing Models")
            else:
                print("❌ compare_models.py not found")
        
        elif choice == '7':
            run_command("setup.py", "Running Setup")
        
        elif choice == '0':
            print("\nGoodbye! 👋\n")
            break
        
        else:
            print("\n❌ Invalid choice\n")

if __name__ == "__main__":
    main()
