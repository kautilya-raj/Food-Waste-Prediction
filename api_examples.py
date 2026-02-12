"""
API Usage Examples
==================
Examples of how to use the Hotel Food Wastage Prediction API
"""

import requests
import json

# ================== CONFIGURATION ================== #
API_URL = "http://localhost:5000"

# ================== EXAMPLE 1: HEALTH CHECK ================== #

def check_health():
    """Check if API is running"""
    response = requests.get(f"{API_URL}/health")
    print("\n" + "="*70)
    print("  HEALTH CHECK")
    print("="*70)
    print(json.dumps(response.json(), indent=2))
    return response.json()


# ================== EXAMPLE 2: GET FEATURES ================== #

def get_features():
    """Get list of required features"""
    response = requests.get(f"{API_URL}/features")
    print("\n" + "="*70)
    print("  REQUIRED FEATURES")
    print("="*70)
    print(json.dumps(response.json(), indent=2))
    return response.json()


# ================== EXAMPLE 3: SINGLE PREDICTION ================== #

def single_prediction():
    """Make a single prediction"""
    
    # Example input data
    data = {
        "Type of Food": "Rice",
        "Number of Guests": 100,
        "Event Type": "Wedding",
        "Quantity of Food": 50.0,
        "Storage Conditions": "Refrigerated",
        "Purchase History": "Bulk Purchase",
        "Seasonality": "Summer",
        "Preparation Method": "Freshly Cooked",
        "Geographical Location": "Urban",
        "Pricing": 500.0
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print("\n" + "="*70)
    print("  SINGLE PREDICTION")
    print("="*70)
    print("\nInput:")
    print(json.dumps(data, indent=2))
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))
    
    return response.json()


# ================== EXAMPLE 4: BATCH PREDICTION ================== #

def batch_prediction():
    """Make batch predictions"""
    
    # Example batch data
    batch_data = {
        "data": [
            {
                "Type of Food": "Rice",
                "Number of Guests": 100,
                "Event Type": "Wedding",
                "Quantity of Food": 50.0,
                "Storage Conditions": "Refrigerated",
                "Purchase History": "Bulk Purchase",
                "Seasonality": "Summer",
                "Preparation Method": "Freshly Cooked",
                "Geographical Location": "Urban",
                "Pricing": 500.0
            },
            {
                "Type of Food": "Biryani",
                "Number of Guests": 150,
                "Event Type": "Corporate Event",
                "Quantity of Food": 75.0,
                "Storage Conditions": "Room Temperature",
                "Purchase History": "Moderate Purchase",
                "Seasonality": "Winter",
                "Preparation Method": "Pre-cooked",
                "Geographical Location": "Urban",
                "Pricing": 800.0
            },
            {
                "Type of Food": "Curry",
                "Number of Guests": 200,
                "Event Type": "Wedding",
                "Quantity of Food": 100.0,
                "Storage Conditions": "Refrigerated",
                "Purchase History": "Bulk Purchase",
                "Seasonality": "Monsoon",
                "Preparation Method": "Live Counter",
                "Geographical Location": "Rural",
                "Pricing": 600.0
            }
        ]
    }
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=batch_data,
        headers={"Content-Type": "application/json"}
    )
    
    print("\n" + "="*70)
    print("  BATCH PREDICTION")
    print("="*70)
    print(f"\nRequesting predictions for {len(batch_data['data'])} items...")
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))
    
    return response.json()


# ================== EXAMPLE 5: ERROR HANDLING ================== #

def error_handling_example():
    """Example of handling errors"""
    
    # Missing required field
    incomplete_data = {
        "Type of Food": "Rice",
        "Number of Guests": 100
        # Missing other required fields
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=incomplete_data,
        headers={"Content-Type": "application/json"}
    )
    
    print("\n" + "="*70)
    print("  ERROR HANDLING EXAMPLE")
    print("="*70)
    print("\nIncomplete input (missing fields):")
    print(json.dumps(incomplete_data, indent=2))
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))


# ================== MAIN FUNCTION ================== #

def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("  🚀 API USAGE EXAMPLES")
    print("="*70)
    print("\nMake sure the API server is running first!")
    print("Run: python api_server.py")
    print("\nPress Enter to continue...")
    input()
    
    try:
        # Example 1: Health Check
        check_health()
        
        # Example 2: Get Features
        get_features()
        
        # Example 3: Single Prediction
        single_prediction()
        
        # Example 4: Batch Prediction
        batch_prediction()
        
        # Example 5: Error Handling
        error_handling_example()
        
        print("\n" + "="*70)
        print("  ✅ ALL EXAMPLES COMPLETED")
        print("="*70 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server")
        print("Make sure the server is running: python api_server.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


# ================== CURL EXAMPLES ================== #

def print_curl_examples():
    """Print curl command examples"""
    
    print("\n" + "="*70)
    print("  📝 CURL COMMAND EXAMPLES")
    print("="*70)
    
    print("\n1. Health Check:")
    print("curl http://localhost:5000/health")
    
    print("\n2. Get Features:")
    print("curl http://localhost:5000/features")
    
    print("\n3. Single Prediction:")
    print("""curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "Type of Food": "Rice",
    "Number of Guests": 100,
    "Event Type": "Wedding",
    "Quantity of Food": 50.0,
    "Storage Conditions": "Refrigerated",
    "Purchase History": "Bulk Purchase",
    "Seasonality": "Summer",
    "Preparation Method": "Freshly Cooked",
    "Geographical Location": "Urban",
    "Pricing": 500.0
  }'""")
    
    print("\n4. Batch Prediction:")
    print("""curl -X POST http://localhost:5000/predict/batch \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [
      {
        "Type of Food": "Rice",
        "Number of Guests": 100,
        ...
      }
    ]
  }'""")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Uncomment the function you want to run
    main()
    # print_curl_examples()
