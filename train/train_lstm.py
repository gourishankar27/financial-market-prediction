import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_lstm import LSTMStockPredictor
from src.utils_lstm import (
    create_comprehensive_visualization,
    compute_strategy_returns,
    compute_sharpe_ratio,
)
from src.data_processing import create_financial_dataset


warnings.filterwarnings('ignore')


TRAIN_ON_LATEST_DATA = False

def main():

    if TRAIN_ON_LATEST_DATA:
        create_financial_dataset()

    try:
        df = pd.read_csv(
            'data/core_proposal_daily_fixed_train.csv',
            index_col=0,
            parse_dates=True
        )


        # After loading df
        drop_cols = [c for c in df.columns if c == 'CPI']
        if drop_cols:
            df = df.drop(columns=drop_cols)
            print(f"Dropped CPI columns from modeling dataset: {drop_cols}")

        print(f"Data loaded successfully: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print("Target column: SP500_Returns\n")
    except FileNotFoundError:
        print("Please ensure your data file exists in the correct location.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Display data info
    print("Data Overview:")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Features: {len(df.columns) - 1}\n")  # Excluding target

    # Initialize predictor
    print("Initializing LSTM predictor with corrected implementation...")
    predictor = LSTMStockPredictor(
        sequence_length=30,    # Look back 30 days
        hidden_size=64,
        num_layers=3,
        learning_rate=0.001,
        batch_size=32,
        epochs=1000            # Max epochs (early stopping will likely trigger)
    )
    print()

    # Prepare data
    print("Preparing data (no leakage: scalers fit on train only, next-day target)...")
    try:
        input_size = predictor.prepare_data(
            df,
            target_col='SP500_Returns',
            test_size=0.15,
            val_size=0.15
        )
        print("Data preparation completed!")
        print(f"Input features: {input_size}\n")
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return

    # Build model
    print("Building LSTM model...")
    try:
        predictor.build_model(input_size)
        print("Model built successfully!\n")
    except Exception as e:
        print(f"Error building model: {e}")
        return

    # Train model
    print("Starting training...")
    print("Note: Training will use early stopping to prevent overfitting\n")

    best_weight_path = 'weights/best_lstm_model.pth'
    os.makedirs(os.path.dirname(best_weight_path), exist_ok=True)

    try:
        train_losses, val_losses = predictor.train_model(best_weight_path)
        print("Training completed!\n")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Evaluate model
    print("Evaluating model on test set...")
    try:
        metrics = predictor.evaluate_model()

        # NEW: compute Sharpe ratio for LSTM strategy (test set)
        lstm_strategy_returns = compute_strategy_returns(
            metrics['Predictions'], metrics['Targets']
        )
        lstm_sharpe = compute_sharpe_ratio(lstm_strategy_returns)
        metrics['Sharpe_Ratio'] = lstm_sharpe

        print(f"LSTM Sharpe Ratio (test set): {lstm_sharpe:.4f}\n")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return



    # Create visualizations
    print("Creating result visualizations...")
    try:
        create_comprehensive_visualization(
            predictor,
            metrics,
            train_losses=train_losses,
            val_losses=val_losses,
        )
        print("Visualizations saved!\n")
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
        print("This is not critical - your model training was successful.\n")

    # Save final results
    print("Saving results...")
    try:
        # Strategy returns: long/short based on LSTM predictions
        strategy_returns = compute_strategy_returns(
            metrics['Predictions'],
            metrics['Targets']
        )

        results_df = pd.DataFrame({
            'Actual_Returns': metrics['Targets'],
            'Predicted_Returns': metrics['Predictions'],
            'Strategy_Returns': strategy_returns,
            'Actual_Direction': np.sign(metrics['Targets']),
            'Predicted_Direction': np.sign(metrics['Predictions']),
        })
        results_df.to_csv('reports/results/lstm_predictions_corrected.csv', index=False)

        metrics_to_save = {
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'Directional_Accuracy': metrics['Directional_Accuracy'],
            'Correlation': metrics['Correlation'],
            'Sharpe_Ratio': metrics.get('Sharpe_Ratio', np.nan),
        }

        with open('reports/results/model_metrics_corrected.txt', 'w') as f:
            f.write("LSTM MODEL METRICS\n")
            f.write("=" * 30 + "\n")
            for key, value in metrics_to_save.items():
                if key == 'Directional_Accuracy':
                    f.write(f"{key}: {value:.1%}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")

        print("Results saved:")
        print("  • lstm_predictions_corrected.csv")
        print("  • model_metrics_corrected.txt")
        print("  • best_lstm_model.pth\n")

    except Exception as e:
        print(f"Warning: Could not save results: {e}")



    # Comapre with ARIMAX results
    #compare with ARIMAX baseline if results file exists
    try:
        print("Comparing LSTM against ARIMAX baseline (if available)...")
        predictor.compare_with_baseline()
    except Exception as e:
        print(f"[WARN] Could not run LSTM vs ARIMAX comparison: {e}")


if __name__ == "__main__":
    main()
