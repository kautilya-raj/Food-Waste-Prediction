"""
Setup Script for Hotel Food Wastage Prediction System
=====================================================
Automated setup and installation
"""

import subprocess
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def create_directories():
    """Create necessary directories"""
    print_header("📁 CREATING DIRECTORIES")
    
    directories = ['data', 'models', 'predictions', 'visualizations']
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"✅ Created: {directory}/")
        else:
            print(f"ℹ️  Already exists: {directory}/")

def install_requirements(minimal=False):
    """Install required packages"""
    print_header("📦 INSTALLING PACKAGES")
    
    if minimal:
        packages = [
            'pandas>=1.5.0',
            'numpy>=1.24.0',
            'scikit-learn>=1.3.0',
            'joblib>=1.3.0'
        ]
        print("Installing minimal requirements...\n")
    else:
        packages = [
            'pandas>=1.5.0',
            'numpy>=1.24.0',
            'scikit-learn>=1.3.0',
            'joblib>=1.3.0',
            'xgboost>=2.0.0',
            'lightgbm>=4.0.0',
            'catboost>=1.2.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'flask>=2.3.0',
            'flask-cors>=4.0.0'
        ]
        print("Installing full requirements...\n")
    
    for package in packages:
        try:
            print(f"Installing {package.split('>=')[0]}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                package, '--break-system-packages', '-q'
            ])
            print(f"  ✅ {package.split('>=')[0]} installed")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Failed to install {package.split('>=')[0]}")

def verify_installation():
    """Verify that all packages are installed"""
    print_header("✅ VERIFYING INSTALLATION")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    optional_packages = {
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'catboost': 'CatBoost',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'flask': 'Flask'
    }
    
    print("Required packages:")
    all_required_ok = True
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            all_required_ok = False
    
    print("\nOptional packages:")
    for module, name in optional_packages.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - Not installed (optional)")
    
    return all_required_ok

def check_data_file():
    """Check if data file exists"""
    print_header("📊 CHECKING DATA FILE")
    
    data_file = Path("data/hotel_food_waste.csv")
    
    if data_file.exists():
        print(f"✅ Data file found: {data_file}")
        
        # Try to load and show info
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            print(f"\n📈 Data Info:")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Columns: {', '.join(df.columns.tolist()[:5])}...")
        except Exception as e:
            print(f"⚠️  Could not read data file: {str(e)}")
    else:
        print(f"⚠️  Data file not found: {data_file}")
        print(f"\nPlease add your data file to: data/hotel_food_waste.csv")
        print(f"Required columns:")
        print(f"  - Wastage Food Amount (target)")
        print(f"  - Type of Food")
        print(f"  - Number of Guests")
        print(f"  - Event Type")
        print(f"  - Quantity of Food")
        print(f"  - Storage Conditions")
        print(f"  - Purchase History")
        print(f"  - Seasonality")
        print(f"  - Preparation Method")
        print(f"  - Geographical Location")
        print(f"  - Pricing")

def create_sample_data():
    """Create a sample data file"""
    print_header("📝 CREATING SAMPLE DATA")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'Type of Food': np.random.choice(['Rice', 'Biryani', 'Curry', 'Roti', 'Salad'], n_samples),
            'Number of Guests': np.random.randint(50, 300, n_samples),
            'Event Type': np.random.choice(['Wedding', 'Corporate Event', 'Birthday', 'Reception'], n_samples),
            'Quantity of Food': np.random.uniform(30, 150, n_samples),
            'Storage Conditions': np.random.choice(['Refrigerated', 'Room Temperature', 'No Storage'], n_samples),
            'Purchase History': np.random.choice(['Bulk Purchase', 'Moderate Purchase', 'Just-in-Time'], n_samples),
            'Seasonality': np.random.choice(['Summer', 'Winter', 'Monsoon', 'Festival Season'], n_samples),
            'Preparation Method': np.random.choice(['Freshly Cooked', 'Pre-cooked', 'Frozen & Reheated', 'Live Counter'], n_samples),
            'Geographical Location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
            'Pricing': np.random.uniform(300, 1000, n_samples),
        }
        
        # Create wastage (with some logic)
        wastage = []
        for i in range(n_samples):
            base_wastage = data['Quantity of Food'][i] * 0.15
            if data['Storage Conditions'][i] == 'No Storage':
                base_wastage *= 1.5
            if data['Event Type'][i] == 'Wedding':
                base_wastage *= 1.3
            wastage.append(base_wastage + np.random.normal(0, 2))
        
        data['Wastage Food Amount'] = wastage
        
        df = pd.DataFrame(data)
        
        # Save
        output_file = Path("data/hotel_food_waste.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Sample data created: {output_file}")
        print(f"   Samples: {len(df)}")
        print(f"   This is SAMPLE data for testing only!")
        print(f"   Replace with your real data for production use.")
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {str(e)}")

def print_next_steps():
    """Print next steps for the user"""
    print_header("🚀 NEXT STEPS")
    
    print("1. Add your data:")
    print("   - Place your CSV file at: data/hotel_food_waste.csv")
    print("   - Or use sample data created above for testing\n")
    
    print("2. Train the model:")
    print("   python train_hotel_model_enhanced.py\n")
    
    print("3. Make predictions:")
    print("   python predict_hotel_wastage.py\n")
    
    print("4. Generate visualizations:")
    print("   python visualize_models.py\n")
    
    print("5. Start API server (optional):")
    print("   python api_server.py\n")
    
    print("For more info, see: README_HOTEL_MODEL.md\n")

def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("  🚀 HOTEL FOOD WASTAGE PREDICTION - SETUP")
    print("="*70)
    
    print("\nThis script will:")
    print("  1. Create necessary directories")
    print("  2. Install required packages")
    print("  3. Verify installation")
    print("  4. Check for data file")
    
    response = input("\nProceed with setup? (y/n): ")
    
    if response.lower() != 'y':
        print("\nSetup cancelled.")
        return
    
    # Installation mode
    print("\nChoose installation mode:")
    print("  1. Minimal (scikit-learn only)")
    print("  2. Full (includes XGBoost, LightGBM, CatBoost)")
    
    mode = input("Enter choice (1/2): ")
    minimal = mode == '1'
    
    # Create directories
    create_directories()
    
    # Install packages
    install_requirements(minimal=minimal)
    
    # Verify
    all_ok = verify_installation()
    
    # Check data
    check_data_file()
    
    # Offer to create sample data
    if not Path("data/hotel_food_waste.csv").exists():
        response = input("\nCreate sample data for testing? (y/n): ")
        if response.lower() == 'y':
            create_sample_data()
    
    # Next steps
    if all_ok:
        print_next_steps()
        print("="*70)
        print("  ✅ SETUP COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
    else:
        print("\n⚠️  Setup completed with warnings.")
        print("Some required packages may be missing.\n")

if __name__ == "__main__":
    main()
