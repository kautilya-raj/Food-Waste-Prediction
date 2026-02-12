"""
Model Comparison and Visualization Script
=========================================
Compare different models and visualize their performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ================== CONFIGURATION ================== #
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# ================== LOAD METRICS ================== #

def load_metrics():
    """Load model comparison metrics"""
    metrics_file = MODEL_DIR / "model_comparison_metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    else:
        print("⚠️  Metrics file not found. Train the model first.")
        return None

# ================== VISUALIZATION FUNCTIONS ================== #

def plot_model_comparison(metrics_df):
    """Create comparison charts for all models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hotel Food Wastage Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. Test MAE Comparison
    ax1 = axes[0, 0]
    metrics_df.plot(x='Model', y='Test MAE', kind='bar', ax=ax1, color='skyblue', legend=False)
    ax1.set_title('Test Mean Absolute Error (Lower is Better)', fontweight='bold')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(metrics_df['Test MAE']):
        ax1.text(i, v + 0.1, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Test R² Comparison
    ax2 = axes[0, 1]
    metrics_df.plot(x='Model', y='Test R²', kind='bar', ax=ax2, color='lightcoral', legend=False)
    ax2.set_title('Test R² Score (Higher is Better)', fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add value labels
    for i, v in enumerate(metrics_df['Test R²']):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Test RMSE Comparison
    ax3 = axes[1, 0]
    metrics_df.plot(x='Model', y='Test RMSE', kind='bar', ax=ax3, color='lightgreen', legend=False)
    ax3.set_title('Test RMSE (Lower is Better)', fontweight='bold')
    ax3.set_ylabel('RMSE')
    ax3.set_xlabel('')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(metrics_df['Test RMSE']):
        ax3.text(i, v + 0.1, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Test MAPE Comparison
    ax4 = axes[1, 1]
    metrics_df.plot(x='Model', y='Test MAPE', kind='bar', ax=ax4, color='plum', legend=False)
    ax4.set_title('Test MAPE - Mean Absolute Percentage Error', fontweight='bold')
    ax4.set_ylabel('MAPE (%)')
    ax4.set_xlabel('')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(metrics_df['Test MAPE']):
        ax4.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'model_comparison.png'}")
    plt.close()


def plot_train_vs_test(metrics_df):
    """Plot train vs test performance to check overfitting"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Train vs Test Performance (Overfitting Check)', fontsize=16, fontweight='bold')
    
    # MAE comparison
    ax1 = axes[0]
    x = np.arange(len(metrics_df))
    width = 0.35
    
    ax1.bar(x - width/2, metrics_df['Train MAE'], width, label='Train MAE', color='lightblue')
    ax1.bar(x + width/2, metrics_df['Test MAE'], width, label='Test MAE', color='darkblue')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('MAE')
    ax1.set_title('MAE: Train vs Test')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # R² comparison
    ax2 = axes[1]
    ax2.bar(x - width/2, metrics_df['Train R²'], width, label='Train R²', color='lightcoral')
    ax2.bar(x + width/2, metrics_df['Test R²'], width, label='Test R²', color='darkred')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R²: Train vs Test')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'train_vs_test.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'train_vs_test.png'}")
    plt.close()


def plot_feature_importance():
    """Plot feature importance"""
    
    importance_file = MODEL_DIR / "feature_importance.csv"
    if not importance_file.exists():
        print("⚠️  Feature importance file not found.")
        return
    
    importance_df = pd.read_csv(importance_file)
    
    # Plot top 15 features
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), top_features['Importance'], color='teal')
    plt.yticks(range(top_n), top_features['Feature'])
    plt.xlabel('Importance Score', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(top_features['Importance']):
        plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'feature_importance.png'}")
    plt.close()


def plot_cv_scores(metrics_df):
    """Plot cross-validation scores with error bars"""
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(metrics_df))
    plt.errorbar(x, metrics_df['CV MAE (mean)'], yerr=metrics_df['CV MAE (std)'], 
                 fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
    
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Cross-Validation MAE', fontweight='bold')
    plt.title('Cross-Validation Performance (5-Fold CV)', fontsize=16, fontweight='bold')
    plt.xticks(x, metrics_df['Model'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cv_scores.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'cv_scores.png'}")
    plt.close()


def create_summary_table(metrics_df):
    """Create a formatted summary table"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Select key metrics
    table_data = metrics_df[['Model', 'Test MAE', 'Test RMSE', 'Test R²', 'Test MAPE']].copy()
    
    # Highlight best values
    best_mae_idx = table_data['Test MAE'].idxmin()
    best_r2_idx = table_data['Test R²'].idxmax()
    
    # Format numbers
    table_data['Test MAE'] = table_data['Test MAE'].apply(lambda x: f'{x:.4f}')
    table_data['Test RMSE'] = table_data['Test RMSE'].apply(lambda x: f'{x:.4f}')
    table_data['Test R²'] = table_data['Test R²'].apply(lambda x: f'{x:.4f}')
    table_data['Test MAPE'] = table_data['Test MAPE'].apply(lambda x: f'{x:.2f}%')
    
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best performers
    for i in range(len(table_data.columns)):
        if i > 0:  # Skip model name column
            table[(best_mae_idx + 1, i)].set_facecolor('#90EE90')
    
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'summary_table.png'}")
    plt.close()


# ================== MAIN ================== #

def main():
    """Generate all visualizations"""
    
    print("\n" + "="*70)
    print("  📊 GENERATING MODEL COMPARISON VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Load metrics
    metrics_df = load_metrics()
    if metrics_df is None:
        return
    
    print(f"📈 Loaded metrics for {len(metrics_df)} models\n")
    
    # Generate plots
    print("Creating visualizations...")
    
    try:
        plot_model_comparison(metrics_df)
        plot_train_vs_test(metrics_df)
        plot_cv_scores(metrics_df)
        plot_feature_importance()
        create_summary_table(metrics_df)
        
        print("\n" + "="*70)
        print("  ✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY")
        print("="*70)
        print(f"\n📁 Saved to: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        print("  - model_comparison.png (4 metric comparisons)")
        print("  - train_vs_test.png (overfitting analysis)")
        print("  - cv_scores.png (cross-validation results)")
        print("  - feature_importance.png (top features)")
        print("  - summary_table.png (performance summary)")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {str(e)}")


if __name__ == "__main__":
    main()
