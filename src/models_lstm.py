import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.utils_lstm import compute_strategy_returns, compute_sharpe_ratio


class FinancialLSTM(nn.Module):
    """
    LSTM Model for Financial Time Series Forecasting

    Architecture:
    - Input: Sequence of financial features
    - LSTM layers with dropout for regularization
    - Dense layers for final prediction
    - Output: Next-day return prediction
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1):
        super(FinancialLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the last output in the sequence
        out = out[:, -1, :]

        # Apply dropout
        out = self.dropout(out)

        # Dense layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class FinancialDataset(Dataset):
    """
    For each index i, returns:
      X_i = features[i : i + sequence_length]
      y_i = target at time (i + sequence_length)  # next‑day
    """
    def __init__(self, features, targets, sequence_length):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        # For next-step prediction we lose `sequence_length` observations
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        feature_seq = self.features[idx:idx + self.sequence_length]   # up to time t
        target = self.targets[idx + self.sequence_length]             # time t+1
        return torch.FloatTensor(feature_seq), torch.FloatTensor([target])



class LSTMStockPredictor:
    """
    Complete LSTM-based stock market prediction system.
    """

    def __init__(self, sequence_length=30, hidden_size=64, num_layers=2,
                 learning_rate=0.001, batch_size=32, epochs=100):

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Scalers for normalization
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def prepare_data(self, df, target_col='SP500_Returns',
                    test_size=0.1, val_size=0.1):
        print("Preparing data for LSTM.")

        # Separate features and target
        # We explicitly drop CPI so it is never used as an input feature
        ignore_cols = {target_col, 'CPI'}
        feature_cols = [col for col in df.columns if col not in ignore_cols]
        
        features = df[feature_cols].values.astype(np.float32)
        targets  = df[target_col].values.astype(np.float32)

        total_len = len(features)
        train_len = int(total_len * (1 - test_size - val_size))
        val_len   = int(total_len * val_size)

        # 1) temporal splits (raw)
        train_features_raw = features[:train_len]
        val_features_raw   = features[train_len:train_len + val_len]
        test_features_raw  = features[train_len + val_len:]

        train_targets_raw  = targets[:train_len]
        val_targets_raw    = targets[train_len:train_len + val_len]
        test_targets_raw   = targets[train_len + val_len:]

        # 2) fit scalers on TRAIN ONLY
        self.feature_scaler.fit(train_features_raw)
        train_features = self.feature_scaler.transform(train_features_raw)
        val_features   = self.feature_scaler.transform(val_features_raw)
        test_features  = self.feature_scaler.transform(test_features_raw)

        self.target_scaler.fit(train_targets_raw.reshape(-1, 1))
        train_targets = self.target_scaler.transform(train_targets_raw.reshape(-1, 1)).ravel()
        val_targets   = self.target_scaler.transform(val_targets_raw.reshape(-1, 1)).ravel()
        test_targets  = self.target_scaler.transform(test_targets_raw.reshape(-1, 1)).ravel()

        # 3) datasets & loaders (next‑day via FinancialDataset above)
        train_dataset = FinancialDataset(train_features, train_targets, self.sequence_length)
        val_dataset   = FinancialDataset(val_features,   val_targets,   self.sequence_length)
        test_dataset  = FinancialDataset(test_features,  test_targets,  self.sequence_length)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=False)
        self.test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False)

        print("Data prepared:")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Validation sequences: {len(val_dataset)}")
        print(f"  Test sequences: {len(test_dataset)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Sequence length: {self.sequence_length}")

        return len(feature_cols)


    def build_model(self, input_size):
        """Build and initialize the LSTM model."""
        self.model = FinancialLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        ).to(self.device)

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        print("Model built:")
        print(f"  Architecture: LSTM({input_size} -> {self.hidden_size})")
        print(f"  Layers: {self.num_layers}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_model(self, best_weight_path):
        """Train the LSTM model with early stopping."""
        print("\nStarting training...")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20  # can be reduced to 20 if desired

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_features, batch_targets in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}'):
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_features, batch_targets in self.val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                    val_loss += loss.item()

            # Average losses
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"Resetting counter: {patience_counter}")
                torch.save(self.model.state_dict(), best_weight_path)
            else:
                patience_counter += 1
                print(f"Updating counter: {patience_counter}")

            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}")
                print(f"  Train Loss: {avg_train_loss:.6f}")
                print(f"  Val Loss: {avg_val_loss:.6f}")
                print(f"  Best Val Loss: {best_val_loss:.6f}")

        # Load the best model
        self.model.load_state_dict(torch.load(best_weight_path))
        print("\nTraining completed!")

        return train_losses, val_losses

    def evaluate_model(self):
        """Evaluate the trained model on the test set."""
        print("\nEvaluating model...")

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_features, batch_targets in self.test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                outputs = self.model(batch_features)
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())

        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()

        # Inverse transform to actual scale
        predictions_actual = self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        targets_actual = self.target_scaler.inverse_transform(
            targets.reshape(-1, 1)
        ).flatten()

        
        # Metrics
        mse = mean_squared_error(targets_actual, predictions_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets_actual, predictions_actual)

        actual_directions = np.sign(targets_actual)
        predicted_directions = np.sign(predictions_actual)
        directional_accuracy = np.mean(actual_directions == predicted_directions)

        correlation = np.corrcoef(targets_actual, predictions_actual)[0, 1]

        # --- NEW: strategy returns + Sharpe ratio (daily data → 252) ---
        lstm_strategy_returns = compute_strategy_returns(
            predictions_actual,
            targets_actual,
        )
        lstm_sharpe = compute_sharpe_ratio(
            lstm_strategy_returns,  # daily returns
            periods_per_year=252    # daily trading days
        )

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Directional_Accuracy': directional_accuracy,
            'Correlation': correlation,
            'Predictions': predictions_actual,
            'Targets': targets_actual,
            'Sharpe_Ratio': lstm_sharpe,
        }

        print("\nLSTM Model Performance:")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  MAE: {mae:.4f}%")
        print(f"  Directional Accuracy: {directional_accuracy:.1%}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Sharpe Ratio: {lstm_sharpe:.4f}")

        return metrics


    # Baseline comparison (optional)
    def compare_with_baseline(
        self,
        baseline_path: str = "reports/results/arimax_baseline_results.csv",
        fallback_arimax_rmse: float = 1.210,
        fallback_arimax_dir_acc: float = 0.488,
        fallback_arimax_corr: float = 0.0005,
    ):
        """
        Compare LSTM performance with ARIMAX baseline.

        Priority:
        1. Try to load ARIMAX metrics from baseline_path.
        2. If that fails, fall back to the provided defaults.
        """
        import os
        import pandas as pd

        # First get LSTM metrics
        metrics = self.evaluate_model()

        # Defaults (in case we can't read the file)
        arimax_rmse = fallback_arimax_rmse
        arimax_dir_acc = fallback_arimax_dir_acc
        arimax_corr = fallback_arimax_corr

        # Try loading ARIMAX results from CSV
        try:
            if os.path.exists(baseline_path):
                df = pd.read_csv(baseline_path)

                arimax_rmse = float(df["Out_Sample_RMSE"].iloc[0])
                arimax_dir_acc = float(df["Directional_Accuracy"].iloc[0])
                arimax_corr = float(df["Correlation"].iloc[0])

                print("\nLoaded ARIMAX baseline metrics from file:")
                print(f"  File: {baseline_path}")
                print(f"  RMSE: {arimax_rmse:.4f}%")
                print(f"  Directional Accuracy: {arimax_dir_acc:.1%}")
                print(f"  Correlation: {arimax_corr:.4f}")
            else:
                print(
                    f"\n[WARN] Baseline file not found: {baseline_path}\n"
                    "Using fallback ARIMAX metrics instead."
                )
        except Exception as e:
            print(
                f"\n[WARN] Error reading ARIMAX baseline from {baseline_path}: {e}\n"
                "Using fallback ARIMAX metrics instead."
            )

        print("\nLSTM vs ARIMAX COMPARISON:")
        print("=" * 45)

        rmse_improvement = ((arimax_rmse - metrics["RMSE"]) / arimax_rmse) * 100
        dir_improvement = (
            (metrics["Directional_Accuracy"] - arimax_dir_acc) / arimax_dir_acc
        ) * 100
        corr_improvement = metrics["Correlation"] - arimax_corr

        print("RMSE:")
        print(f"  ARIMAX: {arimax_rmse:.4f}%")
        print(f"  LSTM:   {metrics['RMSE']:.4f}%")
        print(f"  Improvement: {rmse_improvement:+.1f}%")

        print("\nDirectional Accuracy:")
        print(f"  ARIMAX: {arimax_dir_acc:.1%}")
        print(f"  LSTM:   {metrics['Directional_Accuracy']:.1%}")
        print(f"  Improvement: {dir_improvement:+.1f}%")

        print("\nCorrelation:")
        print(f"  ARIMAX: {arimax_corr:.4f}")
        print(f"  LSTM:   {metrics['Correlation']:.4f}")
        print(f"  Improvement: {corr_improvement:+.4f}")

        print("\nSUCCESS METRICS:")
        if metrics["Directional_Accuracy"] > 0.55:
            print("Above-random directional accuracy achieved!")
        else:
            print("Still below 55% directional accuracy target")

        if metrics["RMSE"] < arimax_rmse:
            print("RMSE improvement over ARIMAX!")
        else:
            print("RMSE not improved over ARIMAX")

        if metrics["Correlation"] > 0.1:
            print("Meaningful correlation achieved!")
        else:
            print("Correlation still weak")

        return metrics
