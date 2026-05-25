"""
Evaluate trained Temporal Fusion Transformer on test set.

Comprehensive evaluation including statistical metrics, financial metrics,
and diagnostic outputs with full experiment logging.

Usage:
    # Basic evaluation (automatically uses best checkpoint)
    python evaluate_tft.py --experiment-name exp004
    python evaluate_tft.py --experiment-name 00_baseline_exploration/exp004
    
    # Specific checkpoint
    python evaluate_tft.py \\
        --experiment-name 00_baseline_exploration/exp004 \\
        --checkpoint experiments/00_baseline_exploration/exp004/checkpoints/tft-epoch=00-val_loss=0.1191.ckpt
    
    # Custom test split (for fixed vs vintage comparison)
    python evaluate_tft.py \\
        --experiment-name exp004 \\
        --test-split data/splits/fixed/core_proposal_daily_fixed_test.csv
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore',message='lightning_fabric')

import os
import sys
import logging
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


# ============================================================================
# FREQUENCY-AWARE CONSTANTS
# ============================================================================

FREQUENCY_CONFIG = {
    'daily': {
        'annualization_factor': 252,      # Trading days per year
        'rolling_window': 60,             # ~3 months of daily data
        'range_threshold': 0.02,          # 2% range for daily returns
        'change_threshold': 0.001,        # 0.1% daily change threshold
        'periods_per_year': 252,
        'period_label': 'days',
    },
    'weekly': {
        'annualization_factor': 52,       # Weeks per year
        'rolling_window': 13,             # ~3 months of weekly data
        'range_threshold': 0.05,          # 5% range for weekly returns (higher volatility)
        'change_threshold': 0.005,        # 0.5% weekly change threshold
        'periods_per_year': 52,
        'period_label': 'weeks',
    },
    'monthly': {
        'annualization_factor': 12,       # Months per year
        'rolling_window': 6,              # 6 months (need more for stability)
        'range_threshold': 0.10,          # 10% range for monthly returns
        'change_threshold': 0.01,         # 1% monthly change threshold
        'periods_per_year': 12,
        'period_label': 'months',
    },
}


def infer_frequency(dates):
    """
    Infer data frequency from date spacing.
    
    Parameters:
    -----------
    dates : array-like
        Array of dates (datetime or string)
        
    Returns:
    --------
    str
        'daily', 'weekly', or 'monthly'
    """
    dates_series = pd.to_datetime(pd.Series(dates))
    date_diffs = dates_series.diff().dropna()
    
    # Use median to be robust to holidays/gaps
    median_diff_days = date_diffs.median().days
    
    if median_diff_days <= 3:
        return 'daily'
    elif median_diff_days <= 10:
        return 'weekly'
    else:
        return 'monthly'


def get_frequency_config(frequency):
    """Get configuration dict for a frequency, with fallback to daily."""
    if frequency not in FREQUENCY_CONFIG:
        print(f"WARNING: Unknown frequency '{frequency}', defaulting to 'daily'")
        frequency = 'daily'
    return FREQUENCY_CONFIG[frequency]


# ============================================================================
# LOGGING SETUP
# ============================================================================

class TeeLogger:
    """
    Tee print statements to both console and log file.
    
    Usage:
        logger = TeeLogger(log_path)
        # All print() statements now go to both console and file
        logger.close()  # When done
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', buffering=1)  # Line buffered
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        sys.stdout = self.terminal
        self.log.close()


def setup_logging(output_dir):
    """Setup logging to both console and file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(output_dir, f'evaluation_{timestamp}.log')
    
    # Redirect stdout to tee logger
    logger = TeeLogger(log_path)
    sys.stdout = logger
    
    print(f"Logging to: {log_path}")
    print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return logger


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained TFT model on test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Name of experiment (for loading config)')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (if None, uses best from training)')
    parser.add_argument('--checkpoint-type', type=str, default='best_val_loss_path',
                        choices=['best_val_loss_path', 'best_pred_std_path', 'best_unique_path'],
                        help='Checkpoint selection metric (ignored if --checkpoint is provided)')
    parser.add_argument('--test-split', type=str, default=None,
                        help='Path to test CSV (if None, infers from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (if None, uses experiment dir)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference')
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(experiment_name):
    """Load experiment configuration from training run."""
    # Try multiple possible paths for phase subdirectories
    possible_paths = [
        f'experiments/{experiment_name}/config.json',                    # Direct path
        f'experiments/00_baseline_exploration/{experiment_name}/config.json',
        f'experiments/01_staleness_features/{experiment_name}/config.json',
        f'experiments/01_staleness_features_fixed/{experiment_name}/config.json',
    ]
    
    # Also check if user provided full path
    if '/' in experiment_name:
        possible_paths.insert(0, f'experiments/{experiment_name}/config.json')
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if config_path is None:
        # Print helpful error
        print(f"\nERROR: Could not find config.json for experiment: {experiment_name}")
        print(f"Tried the following paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print(f"\nTip: Specify full path like: 00_baseline_exploration/exp_name")
        raise FileNotFoundError(f"Config not found for experiment: {experiment_name}")
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(config, test_split_path=None):
    """Load test split and prepare for evaluation."""
    # Get splits directory from config
    base_splits_dir = config.get('data', {}).get('splits_dir', 'data/splits')
    split_prefix = f"{config['feature_set']}_{config['frequency']}"
    
    # Check if config specifies release_date_mode (fixed vs vintage)
    release_mode = config.get('data', {}).get('release_date_mode', 'fixed')
    
    # Construct split directory path
    # Try multiple patterns to handle different directory structures
    possible_splits_dirs = [
        f"{base_splits_dir}/{release_mode}",  # data/splits/fixed/
        base_splits_dir,                       # data/splits/
        f"data/splits/{release_mode}",         # absolute path
        "data/splits",                         # fallback
    ]
    
    # Determine test split path
    if test_split_path is None:
        # Try to find the test file in possible locations
        for splits_dir in possible_splits_dirs:
            # Try with release_mode suffix
            test_path_with_mode = f"{splits_dir}/{split_prefix}_{release_mode}_test.csv"
            test_path_without_mode = f"{splits_dir}/{split_prefix}_test.csv"
            
            if os.path.exists(test_path_with_mode):
                test_split_path = test_path_with_mode
                train_path = f"{splits_dir}/{split_prefix}_{release_mode}_train.csv"
                break
            elif os.path.exists(test_path_without_mode):
                test_split_path = test_path_without_mode
                train_path = f"{splits_dir}/{split_prefix}_train.csv"
                break
        
        if test_split_path is None or not os.path.exists(test_split_path):
            # Print diagnostic info
            print(f"\nERROR: Could not find test split file!")
            print(f"Tried the following locations:")
            for splits_dir in possible_splits_dirs:
                print(f"  - {splits_dir}/{split_prefix}_{release_mode}_test.csv")
                print(f"  - {splits_dir}/{split_prefix}_test.csv")
            raise FileNotFoundError(
                f"Could not find test split. Config specifies:\n"
                f"  base_splits_dir: {base_splits_dir}\n"
                f"  split_prefix: {split_prefix}\n"
                f"  release_mode: {release_mode}\n"
                f"Please check your config.json and data directory structure."
            )
    else:
        # User specified path - derive train path from it
        train_path = test_split_path.replace('_test.csv', '_train.csv')
    
    print(f"Loading data from:")
    print(f"  Test:  {test_split_path}")
    print(f"  Train: {train_path}")
    
    # Load test data
    test_df = pd.read_csv(test_split_path, index_col='Date', parse_dates=True)
    
    # Load training data
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training split not found: {train_path}")
    train_df = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    
    # Check if staleness features are expected based on config
    features_list = config['features']['all']
    has_staleness = any('days_since' in f or 'is_fresh' in f for f in features_list)
    
    if has_staleness:
        from src.data_utils import add_staleness_features
        
        print("Detected staleness features in config, adding to data...")
        train_df = add_staleness_features(train_df, verbose=False)
        test_df = add_staleness_features(test_df, verbose=False)
        
        # Apply same log normalization as training
        staleness_cols = [c for c in train_df.columns if 'days_since' in c]
        if staleness_cols:
            # Calculate max from actual data (same as training)
            max_train = train_df[staleness_cols].max().max()
            max_test = test_df[staleness_cols].max().max()
            MAX_DAYS_STALE = np.ceil(max(max_train, max_test)) + 5
            
            print(f"Normalizing staleness features with max={MAX_DAYS_STALE:.0f} days")
            
            for col in staleness_cols:
                train_df[col] = np.log1p(train_df[col]) / np.log1p(MAX_DAYS_STALE)
                test_df[col] = np.log1p(test_df[col]) / np.log1p(MAX_DAYS_STALE)
    
    return train_df, test_df


def prepare_test_dataset(train_df, test_df, config):
    """Prepare test dataset in TimeSeriesDataSet format."""
    # Reset index and add required columns
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    
    # Combine train and test for continuous time index
    # This ensures test samples have enough history for encoder
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df['time_idx'] = range(len(combined_df))
    combined_df['group'] = 'SP500'
    
    # Get feature list from config
    features = config['features']['all']
    
    # Create training dataset for normalization parameters (train only)
    train_df['time_idx'] = range(len(train_df))
    train_df['group'] = 'SP500'
    
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="SP500_Returns",
        group_ids=["group"],
        max_encoder_length=config['architecture']['max_encoder_length'],
        max_prediction_length=1,
        time_varying_known_reals=[],
        time_varying_unknown_reals=features,
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    
    # Create test dataset from combined data (uses train normalization)
    # Only samples from test period will be predicted, but with train history available
    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        combined_df,
        predict=False,  # false for rolling predictions
        stop_randomization=True
    )
    
    # Filter dataset to only test period indices
    # Indices in combined_df: 0 to len(train_df)-1 are training, rest are test
    test_start_idx = len(train_df)
    test_indices = list(range(test_start_idx, len(combined_df)))

    # DEBUG
    #print(f"test_dataset.index columns: {test_dataset.index.columns.tolist()}")
    #print(f"test_dataset.index shape: {test_dataset.index.shape}")
    #print(f"First few rows:\n{test_dataset.index.head()}")

    # Manually filter the dataset's index
    test_dataset.index = test_dataset.index[test_dataset.index['time'] >= test_start_idx]

    # DEBUG
    #print(f"Filtered test_dataset size: {len(test_dataset.index)}")
    
    return test_dataset, test_df, test_start_idx


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path, config):
    """
    Load trained TFT model from checkpoint.
    
    Automatically detects model type (custom vs baseline) by inspecting
    checkpoint hyperparameters and loads from appropriate module.
    
    Detection logic:
    - If 'num_encoder_features' in hparams → Custom TFT (models/tft_model.py)
    - Otherwise → Baseline TFT (pytorch_forecasting)
    
    Both models have identical prediction interfaces, so downstream evaluation
    code works unchanged.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Check if this model used regime output
    regime_config = config.get('regime_output', {})
    use_regime_output = regime_config.get('enabled', False)
    
    # Check if this model used distribution loss
    training_config = config.get('training', {})
    mean_weight = training_config.get('dist_loss_mean_weight', 0.0)
    std_weight = training_config.get('dist_loss_std_weight', 0.0)
    uses_dist_loss = mean_weight > 0 or std_weight > 0

    # Check if this model used classification head
    classification_config = config.get('classification', {})
    use_classification = classification_config.get('enabled', False)

    # Case 0: Classification model
    if use_classification:
        from src.classification_tft import ClassificationTFT
        model = ClassificationTFT.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model
    
    # Case 1: Baseline model (no regime output, no dist loss)
    if not uses_dist_loss and not use_regime_output:
        import torch
        
        # Check if regime attention is used
        regime_attn_config = config.get('regime_attention', {})
        use_regime_attention = regime_attn_config.get('enabled', False)
        
        if use_regime_attention:
            print(f"  Detected regime attention checkpoint, applying architecture modification...")
            
            # Load checkpoint dict first
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            hparams = checkpoint['hyper_parameters']
            
            # Create baseline model
            model = TemporalFusionTransformer(**hparams)
            
            # Apply regime attention modification
            from src.regime_attention import replace_attention_module
            
            model = replace_attention_module(
                model,
                regime_mode=regime_attn_config.get('regime_mode', 'vix_threshold'),
                vix_threshold=regime_attn_config.get('vix_threshold', 25.0),
                num_regimes=regime_attn_config.get('num_regimes', 2)
            )
            
            # Load weights
            model.load_state_dict(checkpoint['state_dict'])
            print(f"  Successfully loaded regime attention state_dict")
        else:
            # Standard baseline loading
            model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        
        model.eval()
        return model
    
    # Case 2: Regime output without dist loss (clean loading path)
    if use_regime_output and not uses_dist_loss:
        import torch
        
        print(f"  Detected regime output checkpoint, applying architecture modification...")
        
        # Load checkpoint dict first (don't instantiate model yet)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract hparams from checkpoint
        hparams = checkpoint['hyper_parameters']
        
        # Create model with baseline architecture
        model = TemporalFusionTransformer(**hparams)
        
        # Apply regime output architecture before loading weights
        from src.regime_output import replace_output_layer
        
        num_regimes = regime_config.get('num_regimes', 2)
        routing_mode = regime_config.get('routing_mode', 'learned')
        routing_strategy = regime_config.get('routing_strategy', 'learned')
        vix_threshold = regime_config.get('vix_threshold', 25.0)
        vix_threshold_low = regime_config.get('vix_threshold_low', None)
        vix_threshold_high = regime_config.get('vix_threshold_high', None)
        load_balance_weight = regime_config.get('load_balance_weight', 0.5)
        expert_hidden_size = regime_config.get('expert_hidden_size', 0)
        hard_routing_train = False  # Always soft routing during evaluation
        
        model = replace_output_layer(
            model,
            num_regimes=num_regimes,
            routing_mode=routing_mode,
            routing_strategy=routing_strategy,
            vix_threshold=vix_threshold,
            vix_threshold_low=vix_threshold_low,
            vix_threshold_high=vix_threshold_high,
            load_balance_weight=load_balance_weight,
            expert_hidden_size=expert_hidden_size,
            hard_routing_train=hard_routing_train
        )
        
        # Check if regime attention is also used
        regime_attn_config = config.get('regime_attention', {})
        use_regime_attention = regime_attn_config.get('enabled', False)
        
        if use_regime_attention:
            print(f"  Detected regime attention, applying attention modification...")
            from regime_attention import replace_attention_module
            
            model = replace_attention_module(
                model,
                regime_mode=regime_attn_config.get('regime_mode', 'vix_threshold'),
                vix_threshold=regime_attn_config.get('vix_threshold', 25.0),
                num_regimes=regime_attn_config.get('num_regimes', 2)
            )
        
        # NOW load the MoE weights
        model.load_state_dict(checkpoint['state_dict'])
        print(f"  Successfully loaded regime output state_dict")
        
        model.eval()
        return model
    
    # Case 3: Distribution loss (with or without regime output)
    print(f"  Loading distribution loss checkpoint (bypassing corrupted loss)...")
    
    import torch
    import pickle
    import io
    import zipfile
    import tempfile
    
    class FixedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'QuantileLoss' and 'pytorch_forecasting' in module:
                from pytorch_forecasting.metrics import QuantileLoss
                return QuantileLoss
            return super().find_class(module, name)
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = FixedUnpickler(f).load()
    except (pickle.UnpicklingError, zipfile.BadZipFile, Exception) as e:
        print(f"  Direct unpickling failed, checkpoint may be unfixable: {e}")
        raise RuntimeError(
            f"Cannot load checkpoint with monkey-patched loss. "
            f"The checkpoint is corrupted beyond repair. "
            f"Recommendation: Retrain this experiment without distribution loss, "
            f"or skip distribution loss analysis for Phase 3."
        )
    
    if 'hyper_parameters' in checkpoint and 'loss' in checkpoint['hyper_parameters']:
        from pytorch_forecasting.metrics import QuantileLoss
        checkpoint['hyper_parameters']['loss'] = QuantileLoss()
    
    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        if use_regime_output:
            print(f"  Detected regime output checkpoint, applying architecture modification...")
            
            hparams = checkpoint['hyper_parameters']
            model = TemporalFusionTransformer(**hparams)
            
            from src.regime_output import replace_output_layer
            
            num_regimes = regime_config.get('num_regimes', 2)
            routing_mode = regime_config.get('routing_mode', 'learned')
            routing_strategy = regime_config.get('routing_strategy', 'learned')
            vix_threshold = regime_config.get('vix_threshold', 25.0)
            vix_threshold_low = regime_config.get('vix_threshold_low', None)
            vix_threshold_high = regime_config.get('vix_threshold_high', None)
            load_balance_weight = regime_config.get('load_balance_weight', 0.5)
            expert_hidden_size = regime_config.get('expert_hidden_size', 0)
            hard_routing_train = False
            
            model = replace_output_layer(
                model,
                num_regimes=num_regimes,
                routing_mode=routing_mode,
                routing_strategy=routing_strategy,
                vix_threshold=vix_threshold,
                vix_threshold_low=vix_threshold_low,
                vix_threshold_high=vix_threshold_high,
                load_balance_weight=load_balance_weight,
                expert_hidden_size=expert_hidden_size,
                hard_routing_train=hard_routing_train
            )
            
            # Check if regime attention is also used
            regime_attn_config = config.get('regime_attention', {})
            use_regime_attention = regime_attn_config.get('enabled', False)
            
            if use_regime_attention:
                print(f"  Detected regime attention, applying attention modification...")
                from regime_attention import replace_attention_module
                
                model = replace_attention_module(
                    model,
                    regime_mode=regime_attn_config.get('regime_mode', 'vix_threshold'),
                    vix_threshold=regime_attn_config.get('vix_threshold', 25.0),
                    num_regimes=regime_attn_config.get('num_regimes', 2)
                )
            
            model.load_state_dict(checkpoint['state_dict'])
            print(f"  Successfully loaded regime output state_dict")
        else:
            # Check if regime attention is used without regime output
            regime_attn_config = config.get('regime_attention', {})
            use_regime_attention = regime_attn_config.get('enabled', False)
            
            if use_regime_attention:
                print(f"  Detected regime attention, applying attention modification...")
                hparams = checkpoint['hyper_parameters']
                model = TemporalFusionTransformer(**hparams)
                
                from regime_attention import replace_attention_module
                
                model = replace_attention_module(
                    model,
                    regime_mode=regime_attn_config.get('regime_mode', 'vix_threshold'),
                    vix_threshold=regime_attn_config.get('vix_threshold', 25.0),
                    num_regimes=regime_attn_config.get('num_regimes', 2)
                )
                
                model.load_state_dict(checkpoint['state_dict'])
                print(f"  Successfully loaded regime attention state_dict")
            else:
                # Standard loading
                torch.save(checkpoint, tmp_path)
                model = TemporalFusionTransformer.load_from_checkpoint(tmp_path)
        
    finally:
        os.unlink(tmp_path)
    
    model.eval()
    
    from loss_wrapper import add_distribution_penalties
    model = add_distribution_penalties(
        model,
        mean_weight=mean_weight,
        std_weight=std_weight,
        target_mean=training_config.get('dist_loss_target_mean', 0.0003),
        target_std=training_config.get('dist_loss_target_std', 0.01)
    )
    print(f"  Re-applied distribution penalties: mean_weight={mean_weight}, std_weight={std_weight}")
    
    return model

# ============================================================================
# PREDICTION
# ============================================================================

def generate_predictions(model, test_dataset, batch_size=128):
    """Generate predictions on test set."""
    # Create dataloader
    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True
    )
    
    # DEBUG
    #print(f"Test dataloader batches: {len(test_dataloader)}")
    #print(f"Test dataset size: {len(test_dataset)}")
    
    # Generate predictions
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # Unpack batch - dataloader returns (x_dict, y_tuple)
            x, y = batch
            
            # Get predictions
            pred = model(x)
            
            # DEBUG check what pred actually is
            #print(f"pred type: {type(pred)}")
            #print(f"pred: {pred if not isinstance(pred, torch.Tensor) else pred.shape}")
            
            # Extract point prediction (median quantile)
            # TFT returns a named tuple with 'prediction' attribute
            # Shape: [batch_size, prediction_length, num_quantiles]
            if hasattr(pred, 'prediction'):
                pred_tensor = pred.prediction
            elif isinstance(pred, tuple):
                pred_tensor = pred[0]
            elif isinstance(pred, dict) and 'prediction' in pred:
                pred_tensor = pred['prediction']
            else:
                pred_tensor = pred
            
            # For prediction_length=1, quantiles=[0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
            # Index 3 is the median (0.50 quantile)
            point_pred = pred_tensor[:, 0, 3]
            
            # Extract actual values - y is also a tuple (target, weight) or just target
            if isinstance(y, tuple):
                y_actual = y[0]
            else:
                y_actual = y
            
            # DEBUG
            #print(f"Batch {i}: predictions shape={point_pred.shape}, actuals shape={y_actual.shape}")
            
            predictions.append(point_pred.cpu().numpy())
            actuals.append(y_actual[:, 0].cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    print(f"Total predictions: {len(predictions)}, Total actuals: {len(actuals)}")
    
    return predictions, actuals


# ============================================================================
# STATISTICAL METRICS
# ============================================================================

def compute_statistical_metrics(predictions, actuals):
    """Compute standard statistical forecast metrics."""
    # Basic error metrics
    errors = actuals - predictions
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    metrics = {
        'mse': float(np.mean(squared_errors)),
        'rmse': float(np.sqrt(np.mean(squared_errors))),
        'mae': float(np.mean(abs_errors)),
        'mape': float(np.mean(np.abs(errors / (actuals + 1e-10))) * 100),
    }
    
    # R-squared (out-of-sample)
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    metrics['r2'] = float(1 - (ss_res / ss_tot))
    
    # Mean and std of errors
    metrics['mean_error'] = float(np.mean(errors))
    metrics['std_error'] = float(np.std(errors))
    
    return metrics


# ============================================================================
# FINANCIAL METRICS
# ============================================================================

def compute_financial_metrics(predictions, actuals, frequency='daily'):
    """
    Compute financial performance metrics.
    
    Parameters:
    -----------
    predictions : np.array
        Model predictions
    actuals : np.array
        Actual returns
    frequency : str
        Data frequency for annualization ('daily', 'weekly', 'monthly')
    """
    freq_config = get_frequency_config(frequency)
    annualization_factor = freq_config['annualization_factor']
    
    # Directional accuracy
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    directional_accuracy = float(np.mean(pred_direction == actual_direction))
    
    # Simple long-only strategy (trade when predicted return > 0)
    strategy_returns = np.where(predictions > 0, actuals, 0)
    cumulative_returns = np.cumprod(1 + strategy_returns / 100) - 1
    
    # Sharpe ratio (annualized using frequency-appropriate factor)
    if len(strategy_returns) > 1:
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe = float((mean_return / std_return) * np.sqrt(annualization_factor) if std_return > 0 else 0)
    else:
        sharpe = 0.0
    
    # Maximum drawdown
    cumulative_wealth = np.cumprod(1 + strategy_returns / 100)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = float(np.min(drawdown))
    
    # Hit rate (percentage of profitable trades)
    profitable_trades = strategy_returns > 0
    hit_rate = float(np.mean(profitable_trades)) if len(strategy_returns) > 0 else 0.0
    
    metrics = {
        'directional_accuracy': directional_accuracy,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': float(cumulative_returns[-1]) if len(cumulative_returns) > 0 else 0.0,
        'hit_rate': hit_rate,
        'num_trades': int(np.sum(predictions > 0)),
    }

    # Binary predictions (0 = down, 1 = up)
    pred_binary = (predictions > 0).astype(int)
    actual_binary = (actuals > 0).astype(int)

    # Precision, recall, F1
    precision = float(precision_score(actual_binary, pred_binary, zero_division=0))
    recall = float(recall_score(actual_binary, pred_binary, zero_division=0))
    f1 = float(f1_score(actual_binary, pred_binary, zero_division=0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    conf_matrix = confusion_matrix(actual_binary, pred_binary).tolist()

    # AUC-ROC (using continuous predictions as probability scores)
    # Normalize predictions to [0,1] range for AUC calculation
    pred_normalized = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-10)
    try:
        auc_roc = float(roc_auc_score(actual_binary, pred_normalized))
    except:
        auc_roc = 0.5  # Default to random if calculation fails

    # Alpha (excess return over buy-and-hold)
    buy_hold_returns = actuals  # Buy and hold = just hold through all periods
    buy_hold_cumulative = np.cumprod(1 + buy_hold_returns / 100) - 1
    strategy_cumulative = np.cumprod(1 + strategy_returns / 100) - 1

    alpha = float(strategy_cumulative[-1] - buy_hold_cumulative[-1]) if len(strategy_cumulative) > 0 else 0.0

    # Add to metrics dict (before the return statement)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    metrics['confusion_matrix'] = conf_matrix
    metrics['auc_roc'] = auc_roc
    metrics['alpha'] = alpha
    
    # DEBUG
    print(f"Predictions - min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}")
    print(f"Negative predictions: {np.sum(predictions < 0)}/{len(predictions)}")
    
    return metrics, strategy_returns


# ============================================================================
# DIAGNOSTIC OUTPUTS
# ============================================================================

def compute_residual_diagnostics(predictions, actuals):
    """Compute residual diagnostics."""
    errors = actuals - predictions
    
    # Normality test
    _, normality_p = stats.normaltest(errors)
    
    # Autocorrelation at lag 1
    if len(errors) > 1:
        acf_lag1 = float(np.corrcoef(errors[:-1], errors[1:])[0, 1])
    else:
        acf_lag1 = 0.0
    
    diagnostics = {
        'normality_p_value': float(normality_p),
        'acf_lag1': acf_lag1,
        'skewness': float(stats.skew(errors)),
        'kurtosis': float(stats.kurtosis(errors)),
    }
    
    return diagnostics


def detect_model_collapse(predictions, actuals, dates, window=None, frequency='daily'):
    """
    Four-mode financial collapse detection with prediction quality analysis.
    
    Structural methods (3):
    1. Variance-based: Low rolling variance vs baseline (20% threshold)
    2. Range-based: Tight min-max bounds over windows (frequency-adjusted threshold)
    3. Consecutive-similarity: Minimal step-to-step changes (frequency-adjusted threshold)
    
    Quality metrics (2):
    4. Correlation: Predictions vs actuals correlation
    5. Directional accuracy: Sign agreement with actuals
    
    Four modes (priority order):
    - STRONG_COLLAPSE: 3/3 structural methods flag (predictions not varying)
    - WEAK_COLLAPSE: 2/3 structural methods flag (structural issues)
    - DEGRADED: Predictions vary but poor quality (low correlation OR low directional acc)
    - HEALTHY: Predictions vary and accurate
    
    Parameters:
    -----------
    predictions : np.array
        Model predictions
    actuals : np.array
        Actual returns
    dates : array-like
        Dates for temporal summary
    window : int, optional
        Rolling window size. If None, uses frequency-appropriate default.
    frequency : str
        Data frequency ('daily', 'weekly', 'monthly')
    
    Returns comprehensive assessment with temporal breakdown.
    """
    # Get frequency-specific configuration
    freq_config = get_frequency_config(frequency)
    
    if window is None:
        window = freq_config['rolling_window']
    
    range_threshold = freq_config['range_threshold']
    change_threshold = freq_config['change_threshold']
    period_label = freq_config['period_label']
    
    # ========================================================================
    # STRUCTURAL METHOD 1: VARIANCE-BASED
    # ========================================================================
    rolling_std = pd.Series(predictions).rolling(window=window, min_periods=window//2).std()
    baseline_start = window
    baseline_end = min(baseline_start + window, len(predictions))
    initial_variance = np.var(predictions[baseline_start:baseline_end])
    collapse_threshold_var = initial_variance * 0.2
    variance_collapsed = rolling_std.values**2 < collapse_threshold_var
    
    # ========================================================================
    # STRUCTURAL METHOD 2: RANGE-BASED (frequency-adjusted threshold)
    # ========================================================================
    rolling_max = pd.Series(predictions).rolling(window=window, min_periods=window//2).max()
    rolling_min = pd.Series(predictions).rolling(window=window, min_periods=window//2).min()
    rolling_range = rolling_max - rolling_min
    range_collapsed = rolling_range.values < range_threshold
    
    # ========================================================================
    # STRUCTURAL METHOD 3: CONSECUTIVE-SIMILARITY (frequency-adjusted threshold)
    # ========================================================================
    pred_changes = np.abs(np.diff(predictions))
    pred_changes = np.concatenate([[np.nan], pred_changes])
    rolling_mean_change = pd.Series(pred_changes).rolling(window=window, min_periods=window//2).mean()
    consecutive_collapsed = rolling_mean_change.values < change_threshold
    
    # ========================================================================
    # QUALITY METRIC 1: CORRELATION
    # ========================================================================
    rolling_corr = []
    for i in range(len(predictions)):
        if i < window // 2:
            rolling_corr.append(np.nan)
        else:
            start_idx = max(0, i - window + 1)
            window_preds = predictions[start_idx:i+1]
            window_actuals = actuals[start_idx:i+1]
            
            # Correlation (handle edge case of zero variance)
            if np.std(window_preds) < 1e-10 or np.std(window_actuals) < 1e-10:
                corr = 0.0
            else:
                corr = np.corrcoef(window_preds, window_actuals)[0, 1]
            rolling_corr.append(corr)
    
    rolling_corr = np.array(rolling_corr)
    
    # ========================================================================
    # QUALITY METRIC 2: DIRECTIONAL ACCURACY
    # ========================================================================
    rolling_dir_acc = []
    for i in range(len(predictions)):
        if i < window // 2:
            rolling_dir_acc.append(np.nan)
        else:
            start_idx = max(0, i - window + 1)
            window_preds = predictions[start_idx:i+1]
            window_actuals = actuals[start_idx:i+1]
            
            # Directional accuracy
            dir_acc = np.mean(np.sign(window_preds) == np.sign(window_actuals))
            rolling_dir_acc.append(dir_acc)
    
    rolling_dir_acc = np.array(rolling_dir_acc)
    
    # ========================================================================
    # COMBINE STRUCTURAL METHODS INTO CONSENSUS
    # ========================================================================
    collapse_matrix = np.stack([
        variance_collapsed,
        range_collapsed,
        consecutive_collapsed,
    ], axis=0)
    
    collapse_matrix = np.nan_to_num(collapse_matrix, nan=False)
    structural_consensus = np.sum(collapse_matrix, axis=0)
    
    # ========================================================================
    # QUALITY DEGRADATION: COMBINED THRESHOLD
    # Flag as degraded only if BOTH correlation AND directional accuracy are poor
    # This avoids flagging normal financial prediction challenges
    # ========================================================================
    low_correlation = rolling_corr < 0.00   # Negative correlation
    low_dir_acc = rolling_dir_acc < 0.52    # Barely better than coin flip
    
    # Degraded = BOTH metrics are poor (predictions provide no useful information)
    quality_degraded = np.nan_to_num(low_correlation & low_dir_acc, nan=False)
    
    # ========================================================================
    # UNIDIRECTIONAL PREDICTION CHECK
    # Catch models that only predict one direction (always positive or always negative)
    # Even if they pass structural and quality checks, this is fundamentally broken
    # ========================================================================
    rolling_pct_positive = []
    for i in range(len(predictions)):
        if i < window // 2:
            rolling_pct_positive.append(np.nan)
        else:
            start_idx = max(0, i - window + 1)
            window_preds = predictions[start_idx:i+1]
            pct_pos = np.mean(window_preds > 0)
            rolling_pct_positive.append(pct_pos)
    
    rolling_pct_positive = np.array(rolling_pct_positive)
    # Flag if >98% positive or >98% negative (essentially unidirectional)
    unidirectional = np.nan_to_num((rolling_pct_positive > 0.98) | (rolling_pct_positive < 0.02), nan=False)
    
    # Determine mode for each timestep (priority order: worst to best)
    modes = []
    for i in range(len(predictions)):
        if structural_consensus[i] >= 3:
            # Worst: Predictions frozen (constant or near-constant)
            modes.append('STRONG_COLLAPSE')
        elif structural_consensus[i] >= 2:
            # Predictions barely varying (reduced variation)
            modes.append('WEAK_COLLAPSE')
        elif unidirectional[i]:
            # Predictions vary but stuck in one direction (structural blindness to market state)
            modes.append('UNIDIRECTIONAL')
        elif quality_degraded[i] or structural_consensus[i] >= 1:
            # Predictions vary in both directions but poor quality (wrong magnitude/timing)
            # OR borderline structural issues (1/3 checks failed)
            modes.append('DEGRADED')
        else:
            # Best: Predictions vary appropriately with good quality
            modes.append('HEALTHY')
    
    modes = np.array(modes)
    
    # ========================================================================
    # GENERATE TEMPORAL SUMMARY
    # ========================================================================
    temporal_summary = []
    method_details = []
    current_mode = None
    state_start_idx = 0
    
    for i, mode in enumerate(modes):
        if mode != current_mode:
            # State transition
            if current_mode is not None and i > state_start_idx + 5:
                start_date = pd.to_datetime(dates[state_start_idx]).strftime('%Y-%m-%d')
                end_date = pd.to_datetime(dates[i-1]).strftime('%Y-%m-%d')
                duration_days = i - state_start_idx
                
                # Get period stats
                period_preds = predictions[state_start_idx:i]
                period_actuals = actuals[state_start_idx:i]
                period_structural = structural_consensus[state_start_idx:i]
                period_corr = rolling_corr[state_start_idx:i]
                period_dir_acc = rolling_dir_acc[state_start_idx:i]
                
                period_std = np.std(period_preds)
                period_range = np.max(period_preds) - np.min(period_preds)
                period_mean_change = np.mean(np.abs(np.diff(period_preds)))
                avg_structural_methods = int(np.nanmean(period_structural))
                
                # Handle empty slices (all NaN) gracefully
                avg_corr = np.nanmean(period_corr) if not np.all(np.isnan(period_corr)) else 0.0
                avg_dir_acc = np.nanmean(period_dir_acc) if not np.all(np.isnan(period_dir_acc)) else 0.5
                
                # Symbol based on mode
                if current_mode == 'HEALTHY':
                    symbol = "[ OK ]"
                elif current_mode == 'DEGRADED':
                    symbol = "[DEGD]"
                elif current_mode == 'UNIDIRECTIONAL':
                    symbol = "[UNI ]"
                elif current_mode == 'WEAK_COLLAPSE':
                    symbol = "[WEAK]"
                else:  # STRONG_COLLAPSE
                    symbol = "[FAIL]"
                
                temporal_summary.append(
                    f"{symbol} {start_date} to {end_date} ({duration_days:4d} {period_label}): "
                    f"{current_mode:20s} - std={period_std:.4f}, corr={avg_corr:+.3f}, "
                    f"dir_acc={avg_dir_acc:.3f}"
                )
                
                method_details.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_periods': duration_days,
                    'period_label': period_label,
                    'mode': current_mode,
                    'structural_methods_flagged': avg_structural_methods,
                    'std': float(period_std),
                    'range': float(period_range),
                    'avg_change': float(period_mean_change),
                    'correlation': float(avg_corr),
                    'directional_accuracy': float(avg_dir_acc),
                })
            
            current_mode = mode
            state_start_idx = i
    
    # Add final period
    if current_mode is not None and len(predictions) > state_start_idx + 5:
        start_date = pd.to_datetime(dates[state_start_idx]).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')
        duration_days = len(predictions) - state_start_idx
        
        period_preds = predictions[state_start_idx:]
        period_actuals = actuals[state_start_idx:]
        period_structural = structural_consensus[state_start_idx:]
        period_corr = rolling_corr[state_start_idx:]
        period_dir_acc = rolling_dir_acc[state_start_idx:]
        
        period_std = np.std(period_preds)
        period_range = np.max(period_preds) - np.min(period_preds)
        period_mean_change = np.mean(np.abs(np.diff(period_preds)))
        avg_structural_methods = int(np.nanmean(period_structural))
        
        # Handle empty slices (all NaN) gracefully
        avg_corr = np.nanmean(period_corr) if not np.all(np.isnan(period_corr)) else 0.0
        avg_dir_acc = np.nanmean(period_dir_acc) if not np.all(np.isnan(period_dir_acc)) else 0.5
        
        if current_mode == 'HEALTHY':
            symbol = "[ OK ]"
        elif current_mode == 'DEGRADED':
            symbol = "[DEGD]"
        elif current_mode == 'UNIDIRECTIONAL':
            symbol = "[UNI ]"
        elif current_mode == 'WEAK_COLLAPSE':
            symbol = "[WEAK]"
        else:
            symbol = "[FAIL]"
        
        temporal_summary.append(
            f"{symbol} {start_date} to {end_date} ({duration_days:4d} {period_label}): "
            f"{current_mode:20s} - std={period_std:.4f}, corr={avg_corr:+.3f}, "
            f"dir_acc={avg_dir_acc:.3f}"
        )
        
        method_details.append({
            'start_date': start_date,
            'end_date': end_date,
            'duration_periods': duration_days,
            'period_label': period_label,
            'mode': current_mode,
            'structural_methods_flagged': avg_structural_methods,
            'std': float(period_std),
            'range': float(period_range),
            'avg_change': float(period_mean_change),
            'correlation': float(avg_corr),
            'directional_accuracy': float(avg_dir_acc),
        })
    
    # Calculate summary statistics by mode
    healthy_periods = int(np.sum(modes == 'HEALTHY'))
    degraded_periods = int(np.sum(modes == 'DEGRADED'))
    unidirectional_periods = int(np.sum(modes == 'UNIDIRECTIONAL'))
    weak_collapse_periods = int(np.sum(modes == 'WEAK_COLLAPSE'))
    strong_collapse_periods = int(np.sum(modes == 'STRONG_COLLAPSE'))
    total_periods = len(modes)
    
    # Individual method statistics
    method_stats = {
        'variance': int(np.sum(variance_collapsed)),
        'range': int(np.sum(range_collapsed)),
        'consecutive': int(np.sum(consecutive_collapsed)),
        'low_correlation': int(np.sum(low_correlation)),
        'low_directional_acc': int(np.sum(low_dir_acc)),
        'quality_degraded': int(np.sum(quality_degraded)),
        'unidirectional': int(np.sum(unidirectional)),
    }
    
    # Determine overall status
    collapse_detected = strong_collapse_periods > 0 or weak_collapse_periods > 0
    degradation_detected = degraded_periods > 0 or unidirectional_periods > 0
    
    return {
        'collapse_detected': collapse_detected,
        'degradation_detected': degradation_detected,
        'temporal_summary': temporal_summary,
        'method_details': method_details,
        'frequency': frequency,
        'period_label': period_label,
        'mode_stats': {
            'healthy_periods': healthy_periods,
            'degraded_periods': degraded_periods,
            'unidirectional_periods': unidirectional_periods,
            'weak_collapse_periods': weak_collapse_periods,
            'strong_collapse_periods': strong_collapse_periods,
            'total_periods': total_periods,
            'healthy_pct': healthy_periods / total_periods * 100,
            'degraded_pct': degraded_periods / total_periods * 100,
            'unidirectional_pct': unidirectional_periods / total_periods * 100,
            'weak_collapse_pct': weak_collapse_periods / total_periods * 100,
            'strong_collapse_pct': strong_collapse_periods / total_periods * 100,
        },
        'method_stats': method_stats,
        'global_stats': {
            'std': float(np.std(predictions)),
            'mean': float(np.mean(predictions)),
            'unique_count': int(len(np.unique(predictions))),
        },
    }

def create_diagnostic_plots(predictions, actuals, dates, output_dir, frequency='daily'):
    """
    Create comprehensive diagnostic plots showing prediction quality over time.
    
    Creates a 3-panel figure:
    1. Rolling correlation (frequency-appropriate window)
    2. Rolling directional accuracy (frequency-appropriate window)
    3. Prediction magnitude over time
    """
    freq_config = get_frequency_config(frequency)
    window = freq_config['rolling_window']
    
    dates_dt = pd.to_datetime(dates)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # ========================================================================
    # 1. ROLLING CORRELATION
    # ========================================================================
    rolling_corr = []
    for i in range(len(predictions)):
        if i < window // 2:
            rolling_corr.append(np.nan)
        else:
            start_idx = max(0, i - window + 1)
            window_preds = predictions[start_idx:i+1]
            window_actuals = actuals[start_idx:i+1]
            
            if np.std(window_preds) < 1e-10 or np.std(window_actuals) < 1e-10:
                corr = 0.0
            else:
                corr = np.corrcoef(window_preds, window_actuals)[0, 1]
            rolling_corr.append(corr)
    
    axes[0].plot(dates_dt, rolling_corr, linewidth=1.5, color='darkblue')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Zero correlation')
    axes[0].axhline(y=0.05, color='orange', linestyle=':', alpha=0.5, label='Positive threshold (0.05)')
    axes[0].axhline(y=-0.05, color='orange', linestyle=':', alpha=0.5, label='Negative threshold (-0.05)')
    axes[0].fill_between(dates_dt, 0, rolling_corr, where=np.array(rolling_corr) < 0, 
                         alpha=0.3, color='red', label='Negative correlation')
    axes[0].fill_between(dates_dt, 0, rolling_corr, where=np.array(rolling_corr) > 0, 
                         alpha=0.3, color='green', label='Positive correlation')
    axes[0].set_ylabel(f'Correlation ({window}-day)')
    axes[0].set_title('Rolling Correlation: Predictions vs Actuals')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # ========================================================================
    # 2. ROLLING DIRECTIONAL ACCURACY
    # ========================================================================
    rolling_dir_acc = []
    for i in range(len(predictions)):
        if i < window // 2:
            rolling_dir_acc.append(np.nan)
        else:
            start_idx = max(0, i - window + 1)
            window_preds = predictions[start_idx:i+1]
            window_actuals = actuals[start_idx:i+1]
            
            dir_acc = np.mean(np.sign(window_preds) == np.sign(window_actuals))
            rolling_dir_acc.append(dir_acc)
    
    axes[1].plot(dates_dt, rolling_dir_acc, linewidth=1.5, color='darkgreen')
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Random (50%)')
    axes[1].axhline(y=0.52, color='orange', linestyle=':', alpha=0.5, label='Threshold (52%)')
    axes[1].fill_between(dates_dt, 0.5, rolling_dir_acc, 
                         where=np.array(rolling_dir_acc) > 0.5, 
                         alpha=0.3, color='green', label='Above random')
    axes[1].fill_between(dates_dt, 0.5, rolling_dir_acc, 
                         where=np.array(rolling_dir_acc) <= 0.5, 
                         alpha=0.3, color='red', label='Below random')
    axes[1].set_ylabel(f'Directional Accuracy ({window}-day)')
    axes[1].set_title('Rolling Directional Accuracy')
    axes[1].set_ylim([0.35, 0.70])
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # ========================================================================
    # 3. PREDICTION MAGNITUDE OVER TIME
    # ========================================================================
    abs_predictions = np.abs(predictions)
    rolling_mean_mag = pd.Series(abs_predictions).rolling(window=window, min_periods=window//2).mean()
    rolling_std_mag = pd.Series(abs_predictions).rolling(window=window, min_periods=window//2).std()
    
    axes[2].plot(dates_dt, abs_predictions, alpha=0.3, linewidth=0.5, color='lightblue', label='Daily |prediction|')
    axes[2].plot(dates_dt, rolling_mean_mag, linewidth=2, color='darkblue', label=f'Rolling mean ({window}-day)')
    axes[2].fill_between(dates_dt, 
                         rolling_mean_mag - rolling_std_mag, 
                         rolling_mean_mag + rolling_std_mag,
                         alpha=0.2, color='blue', label='Ã‚Â±1 std')
    axes[2].axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, label='1% threshold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('|Prediction| (%)')
    axes[2].set_title('Prediction Magnitude Over Time')
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    # Format x-axis for all subplots
    import matplotlib.dates as mdates
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Quality diagnostic plots saved to: {output_dir}/quality_diagnostics.png")


def create_regression_diagnostic_plots(predictions, actuals, dates, output_dir):
    """
    Create classic regression diagnostic plots (4-panel layout).
    
    Standard ML regression diagnostics:
    1. Actual vs Predicted scatter
    2. Residuals over time
    3. Residual distribution histogram
    4. Q-Q plot for normality
    """
    errors = actuals - predictions
    dates_dt = pd.to_datetime(dates)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========================================================================
    # 1. ACTUAL VS PREDICTED
    # ========================================================================
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual Returns (%)')
    axes[0, 0].set_ylabel('Predicted Returns (%)')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ========================================================================
    # 2. RESIDUALS OVER TIME
    # ========================================================================
    axes[0, 1].plot(dates_dt, errors, alpha=0.6, linewidth=0.8, color='steelblue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Prediction Error (%)')
    axes[0, 1].set_title('Residuals Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Format x-axis
    import matplotlib.dates as mdates
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========================================================================
    # 3. RESIDUAL DISTRIBUTION
    # ========================================================================
    axes[1, 0].hist(errors, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    # Overlay normal distribution
    mu, sigma = errors.mean(), errors.std()
    x = np.linspace(errors.min(), errors.max(), 100)
    axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal(mu={mu:.3f}, sigma={sigma:.3f})')
    axes[1, 0].set_xlabel('Prediction Error (%)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ========================================================================
    # 4. Q-Q PLOT
    # ========================================================================
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    # Enhance Q-Q plot styling
    axes[1, 1].get_lines()[0].set_markerfacecolor('steelblue')
    axes[1, 1].get_lines()[0].set_markeredgecolor('black')
    axes[1, 1].get_lines()[0].set_markersize(4)
    axes[1, 1].get_lines()[1].set_color('red')
    axes[1, 1].get_lines()[1].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Regression diagnostic plots saved to: {output_dir}/diagnostic_plots.png")


def create_diagnostic_plots_OLD_BACKUP(predictions, actuals, dates, output_dir):
    """Create diagnostic plots."""
    errors = actuals - predictions
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=10)
    axes[0, 0].plot([actuals.min(), actuals.max()], 
                     [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Returns (%)')
    axes[0, 0].set_ylabel('Predicted Returns (%)')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals over time
    axes[0, 1].plot(dates, errors, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Prediction Error (%)')
    axes[0, 1].set_title('Residuals Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Predictions over time (to visualize collapse)
    axes[0, 2].plot(dates, predictions, alpha=0.6, label='Predictions')
    axes[0, 2].axhline(y=np.mean(predictions), color='r', linestyle='--', 
                      alpha=0.5, label=f'Mean: {np.mean(predictions):.4f}')
    axes[0, 2].set_xlabel('Date')
    axes[0, 2].set_ylabel('Predicted Returns (%)')
    axes[0, 2].set_title('Predictions Over Time')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Residual distribution
    axes[1, 0].hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error (%)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Q-Q plot
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Rolling variance of predictions (60-day window)
    window = min(60, len(predictions) // 4)
    if window > 5:
        rolling_var = pd.Series(predictions).rolling(window=window).std()
        axes[1, 2].plot(dates, rolling_var, linewidth=2, label='Rolling Std')
        axes[1, 2].axhline(y=np.std(predictions), color='r', linestyle='--', 
                          alpha=0.5, label=f'Overall Std: {np.std(predictions):.4f}')
        axes[1, 2].set_xlabel('Date')
        axes[1, 2].set_ylabel('Prediction Std Dev')
        axes[1, 2].set_title(f'Rolling Prediction Variance ({window}-day)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_plots.png'), dpi=150)
    plt.close()


def create_performance_plots(actuals, strategy_returns, dates, output_dir, collapse_diagnostics=None):
    """Create performance plots with optional collapse markers."""
    # Cumulative returns
    cumulative_strategy = np.cumprod(1 + strategy_returns / 100)
    cumulative_buy_hold = np.cumprod(1 + actuals / 100)
    
    # Convert dates to datetime for proper plotting
    dates_dt = pd.to_datetime(dates)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)  # sharex=True for aligned ticks
    
    # 1. Cumulative returns comparison
    axes[0].plot(dates_dt, cumulative_buy_hold, label='Buy & Hold', linewidth=2)
    axes[0].plot(dates_dt, cumulative_strategy, label='TFT Strategy', linewidth=2)
    
    # Add mode background shading
    if collapse_diagnostics and 'method_details' in collapse_diagnostics:
        # Track if we've added each label already (for legend)
        added_labels = set()
        
        for period in collapse_diagnostics['method_details']:
            start = pd.to_datetime(period['start_date'])
            end = pd.to_datetime(period['end_date'])
            
            if period['mode'] == 'HEALTHY':
                color, alpha = 'green', 0.08
                label = 'Healthy' if 'Healthy' not in added_labels else ''
                added_labels.add('Healthy')
            elif period['mode'] == 'DEGRADED':
                color, alpha = 'yellow', 0.12
                label = 'Degraded' if 'Degraded' not in added_labels else ''
                added_labels.add('Degraded')
            elif period['mode'] == 'UNIDIRECTIONAL':
                color, alpha = 'gold', 0.15
                label = 'Unidirectional' if 'Unidirectional' not in added_labels else ''
                added_labels.add('Unidirectional')
            elif period['mode'] == 'WEAK_COLLAPSE':
                color, alpha = 'orange', 0.18
                label = 'Weak collapse' if 'Weak collapse' not in added_labels else ''
                added_labels.add('Weak collapse')
            else:  # STRONG_COLLAPSE
                color, alpha = 'red', 0.22
                label = 'Strong collapse' if 'Strong collapse' not in added_labels else ''
                added_labels.add('Strong collapse')
            
            axes[0].axvspan(start, end, alpha=alpha, color=color, label=label if label else None)
    
    # Add mode transition markers
    if collapse_diagnostics and 'method_details' in collapse_diagnostics:
        # Find first occurrence of each problematic mode
        first_degraded = None
        first_weak = None
        first_strong = None
        
        for period in collapse_diagnostics['method_details']:
            if period['mode'] == 'DEGRADED' and first_degraded is None:
                first_degraded = pd.to_datetime(period['start_date'])
            elif period['mode'] == 'WEAK_COLLAPSE' and first_weak is None:
                first_weak = pd.to_datetime(period['start_date'])
            elif period['mode'] == 'STRONG_COLLAPSE' and first_strong is None:
                first_strong = pd.to_datetime(period['start_date'])
        
        # Plot markers (reverse order so most severe is on top)
        if first_degraded:
            axes[0].axvline(x=first_degraded, color='gold', linewidth=2, linestyle='--',
                           label=f'Ã¢â€ â€™ Degraded: {first_degraded.strftime("%Y-%m")}', alpha=0.7)
        if first_weak:
            axes[0].axvline(x=first_weak, color='orange', linewidth=2, linestyle='--',
                           label=f'Ã¢â€ â€™ Weak collapse: {first_weak.strftime("%Y-%m")}', alpha=0.7)
        if first_strong:
            axes[0].axvline(x=first_strong, color='red', linewidth=2, linestyle='--',
                           label=f'Ã¢â€ â€™ Strong collapse: {first_strong.strftime("%Y-%m")}', alpha=0.7)
    
    axes[0].set_ylabel('Cumulative Return (Growth of $1)')
    axes[0].set_title('Strategy Performance (shaded by quality mode)')
    axes[0].legend(loc='best', fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    # Add month markers (every 3 months for readability)
    for i, date in enumerate(dates_dt):
        if date.day == 1 and date.month % 3 == 1:  # Jan, Apr, Jul, Oct
            axes[0].axvline(x=date, color='gray', alpha=0.2, linewidth=0.5, linestyle='--')
    
    # 2. Rolling Sharpe ratio (60-day window)
    window = min(60, len(strategy_returns) // 4)
    if window > 5:
        rolling_mean = pd.Series(strategy_returns).rolling(window).mean()
        rolling_std = pd.Series(strategy_returns).rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        # Use dates array directly - matplotlib will handle NaN values correctly
        axes[1].plot(dates_dt, rolling_sharpe, linewidth=2)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Add mode transition markers to rolling Sharpe too
        if collapse_diagnostics and 'method_details' in collapse_diagnostics:
            first_degraded = None
            first_weak = None
            first_strong = None
            
            for period in collapse_diagnostics['method_details']:
                if period['mode'] == 'DEGRADED' and first_degraded is None:
                    first_degraded = pd.to_datetime(period['start_date'])
                elif period['mode'] == 'WEAK_COLLAPSE' and first_weak is None:
                    first_weak = pd.to_datetime(period['start_date'])
                elif period['mode'] == 'STRONG_COLLAPSE' and first_strong is None:
                    first_strong = pd.to_datetime(period['start_date'])
            
            if first_degraded:
                axes[1].axvline(x=first_degraded, color='gold', linewidth=2, linestyle='--', alpha=0.7)
            if first_weak:
                axes[1].axvline(x=first_weak, color='orange', linewidth=2, linestyle='--', alpha=0.7)
            if first_strong:
                axes[1].axvline(x=first_strong, color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel(f'Rolling Sharpe ({window}-day)')
        axes[1].set_title('Rolling Sharpe Ratio')
        axes[1].grid(True, alpha=0.3)
        
        # Add month markers to rolling sharpe too
        for i, date in enumerate(dates_dt):
            if date.day == 1 and date.month % 3 == 1:
                axes[1].axvline(x=date, color='gray', alpha=0.2, linewidth=0.5, linestyle='--')
    
    # Format x-axis for BOTH subplots (sharex=True ensures they're synchronized)
    import matplotlib.dates as mdates
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Rotate labels
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_plots.png'), dpi=150)
    plt.close()


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(predictions, actuals, dates, metrics_stat, metrics_fin, 
                 diagnostics, collapse_diagnostics, output_dir):
    """Save all evaluation results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for NaN or infinite values
    nan_count = np.isnan(predictions).sum()
    inf_count = np.isinf(predictions).sum()
    
    if nan_count > 0 or inf_count > 0:
        print(f"\nWARNING: Found {nan_count} NaN and {inf_count} Inf predictions!")
    
    # Save predictions CSV with more detail
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Predicted': predictions,
        'Error': actuals - predictions,
        'Abs_Error': np.abs(actuals - predictions),
    })
    
    # Add debug columns
    results_df['Prediction_Change'] = results_df['Predicted'].diff().abs()
    results_df['Is_Constant'] = results_df['Prediction_Change'] < 1e-6
    
    # Flag suspicious constant regions
    constant_run_length = 0
    constant_runs = []
    for i, is_const in enumerate(results_df['Is_Constant']):
        if is_const:
            constant_run_length += 1
        else:
            if constant_run_length > 10:  # Flag runs of 10+ identical predictions
                constant_runs.append((i - constant_run_length, i, constant_run_length))
            constant_run_length = 0
    
    if constant_runs:
        print(f"\nWARNING: Found {len(constant_runs)} suspicious constant prediction regions:")
        for start, end, length in constant_runs[:5]:  # Show first 5
            print(f"  {dates[start]} to {dates[end-1]}: {length} identical predictions = {predictions[start]:.6f}")
    
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Compute prediction statistics
    prediction_stats = {
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'range': float(np.ptp(predictions)),
        'num_unique': int(len(np.unique(np.round(predictions, 6)))),
        'pct_positive': float((predictions > 0).mean() * 100),
        'pct_negative': float((predictions < 0).mean() * 100),
        'num_predictions': int(len(predictions)),
    }
    
    # Combine all metrics
    all_metrics = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'prediction_stats': prediction_stats,
        'statistical_metrics': metrics_stat,
        'financial_metrics': metrics_fin,
        'residual_diagnostics': diagnostics,
        'mode_stats': collapse_diagnostics['mode_stats'],
        'collapse_detected': collapse_diagnostics['collapse_detected'],
        'degradation_detected': collapse_diagnostics['degradation_detected'],
    }
    
    # Save metrics JSON
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics


def print_summary(metrics_stat, metrics_fin):
    """Print evaluation summary to console."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\nSTATISTICAL METRICS:")
    print(f"  MSE:              {metrics_stat['mse']:.6f}")
    print(f"  RMSE:             {metrics_stat['rmse']:.6f}")
    print(f"  MAE:              {metrics_stat['mae']:.6f}")
    print(f"  Out-of-sample RÂ²: {metrics_stat['r2']:.4f}")
    
    print("\nFINANCIAL METRICS:")
    print(f"  Directional Acc:  {metrics_fin['directional_accuracy']:.2%}")
    print(f"  Sharpe Ratio:     {metrics_fin['sharpe_ratio']:.4f}")
    print(f"  Total Return:     {metrics_fin['total_return']:.2%}")
    print(f"  Max Drawdown:     {metrics_fin['max_drawdown']:.2%}")
    print(f"  Hit Rate:         {metrics_fin['hit_rate']:.2%}")
    print(f"  Number of Trades: {metrics_fin['num_trades']}")

    print("\nCLASSIFICATION METRICS (Binary Up/Down):")
    print(f"  Precision:        {metrics_fin['precision']:.4f}")
    print(f"  Recall:           {metrics_fin['recall']:.4f}")
    print(f"  F1 Score:         {metrics_fin['f1_score']:.4f}")
    print(f"  AUC-ROC:          {metrics_fin['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    conf = metrics_fin['confusion_matrix']
    print(f"                   Predicted Down  Predicted Up")
    print(f"  Actual Down:     {conf[0][0]:6d}          {conf[0][1]:6d}")
    print(f"  Actual Up:       {conf[1][0]:6d}          {conf[1][1]:6d}")
    print(f"\nALPHA (vs Buy-and-Hold):")
    print(f"  Excess Return:    {metrics_fin['alpha']:.2%}")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate():
    """Main evaluation pipeline."""
    args = parse_args()
    
    print("="*70)
    print(f"Evaluating TFT: {args.experiment_name}")
    print("="*70)
    
    # Determine checkpoint path
    if args.checkpoint is None:
        possible_metrics_paths = [
            f'experiments/{args.experiment_name}/final_metrics.json',
            f'experiments/00_baseline_exploration/{args.experiment_name}/final_metrics.json',
            f'experiments/01_staleness_features_fixed/{args.experiment_name}/final_metrics.json',
        ]
        
        if '/' in args.experiment_name:
            possible_metrics_paths.insert(0, f'experiments/{args.experiment_name}/final_metrics.json')
        
        metrics_path = None
        for path in possible_metrics_paths:
            if os.path.exists(path):
                metrics_path = path
                exp_base_path = os.path.dirname(path)  # Get experiment directory
                break
        
        if metrics_path is None:
            print(f"\nERROR: Could not find final_metrics.json")
            print(f"Tried:")
            for path in possible_metrics_paths:
                print(f"  - {path}")
            raise FileNotFoundError(
                f"No final_metrics.json found.\n"
                f"Either specify --checkpoint manually or ensure training completed successfully."
            )
        
        ckpt_type = args.checkpoint_type
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        if ckpt_type not in metrics:
            raise KeyError(
                f"Checkpoint type '{ckpt_type}' not found in final_metrics.json.\n"
                f"Available: {list(metrics.keys())}\n"
                f"This might be an old experiment without multi-metric checkpointing."
            )
        
        checkpoint_path = metrics[ckpt_type]
        
        # Checkpoint path in final_metrics.json might not include phase directory
        # If it doesn't exist, prepend the phase directory from metrics_path
        if not os.path.exists(checkpoint_path):
            # Extract phase directory from metrics_path
            # e.g., experiments/00_baseline_exploration/exp/final_metrics.json
            #   -> experiments/00_baseline_exploration
            phase_dir = '/'.join(metrics_path.split('/')[:-2])  # Remove exp_name/final_metrics.json
            
            # Extract just the experiment part from checkpoint path
            # e.g., experiments/exp/checkpoints/best.ckpt -> exp/checkpoints/best.ckpt
            if checkpoint_path.startswith('experiments/'):
                exp_relative_path = '/'.join(checkpoint_path.split('/')[1:])  # Remove 'experiments/'
                corrected_checkpoint_path = f"{phase_dir}/{exp_relative_path}"
                
                if os.path.exists(corrected_checkpoint_path):
                    print(f"Note: Corrected checkpoint path to include phase directory")
                    checkpoint_path = corrected_checkpoint_path
                else:
                    print(f"WARNING: Could not find checkpoint at:")
                    print(f"  Original: {metrics[ckpt_type]}")
                    print(f"  Corrected: {corrected_checkpoint_path}")
            else:
                print(f"WARNING: Checkpoint path doesn't start with 'experiments/': {checkpoint_path}")
        
        print(f"\nUsing best checkpoint from training:")
        print(f"  {checkpoint_path}")
        print(f"  Validation loss: {metrics['best_val_loss']:.6f}")
        
        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"The path in final_metrics.json may be incorrect."
            )
    else:
        checkpoint_path = args.checkpoint
        # Derive experiment base path from checkpoint
        exp_base_path = os.path.dirname(os.path.dirname(checkpoint_path))
        print(f"\nUsing specified checkpoint:")
        print(f"  {checkpoint_path}")
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config(args.experiment_name)
    
    # Extract frequency from config, or infer from data later
    data_frequency = config.get('frequency', None)
    if data_frequency:
        print(f"Data frequency (from config): {data_frequency}")
    
    # Set output directory
    if args.output_dir is None:
        # Use the same base path we found for metrics/checkpoint
        if args.checkpoint is None and 'exp_base_path' in locals():
            output_dir = os.path.join(exp_base_path, 'evaluation')
        else:
            # Try to find experiment directory
            for base in ['experiments/', 'experiments/00_baseline_exploration/', 
                        'experiments/01_staleness_features/', 'experiments/01_staleness_features_fixed/']:
                exp_path = f"{base}{args.experiment_name}".rstrip('/')
                if os.path.exists(exp_path):
                    output_dir = os.path.join(exp_path, 'evaluation')
                    break
            else:
                # Fallback
                output_dir = os.path.join('experiments', args.experiment_name, 'evaluation')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Setup logging to capture all output
    logger = setup_logging(output_dir)
    
    # Load test data
    print("Loading test data...")
    train_df, test_df = load_test_data(config, args.test_split)
    print(f"Test samples: {len(test_df)}")
    
    # Warn if using vintage data with old fixed-trained experiments
    if args.test_split and 'vintage' in args.test_split.lower():
        # Check if experiment is in old phase directories (trained with fixed data)
        if any(phase in args.experiment_name or phase in str(config_path) 
               for phase in ['00_baseline_exploration', '01_staleness_features']):
            print("\n" + "!"*70)
            print("WARNING: POTENTIAL DATA MISMATCH")
            print("!"*70)
            print("You are using VINTAGE test data with an experiment from phase 00 or 01.")
            print("These experiments were trained with FIXED release dates.")
            print("Evaluation results may not be meaningful for cross-comparison.")
            print("!"*70 + "\n")
    
    # Prepare test dataset
    print("Preparing test dataset...")
    test_dataset, test_df_indexed, test_start_idx = prepare_test_dataset(train_df, test_df, config)
    
    # Count actual test predictions
    # Need to account for encoder length requirement
    max_encoder_length = config['architecture']['max_encoder_length']
    expected_predictions = len(test_df) - max_encoder_length + 1
    print(f"Expected test predictions: {expected_predictions} (test samples - encoder length + 1)")
    
    # Load model
    print(f"Loading model from checkpoint...")
    model = load_model(checkpoint_path, config)
    
    # Generate predictions
    print("Generating predictions...")
    predictions, actuals = generate_predictions(model, test_dataset, args.batch_size)
    
    print(f"\nPrediction Generation Summary:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Test dataset size: {len(test_dataset)}")
    print(f"  Test dataloader batches: {len(test_dataset.to_dataloader(train=False, batch_size=args.batch_size))}")
    print(f"  Prediction statistics:")
    print(f"    Min: {predictions.min():.6f}")
    print(f"    Max: {predictions.max():.6f}")
    print(f"    Mean: {predictions.mean():.6f}")
    print(f"    Std: {predictions.std():.6f}")
    print(f"    Unique values: {len(np.unique(predictions))}")
    
    # Get dates for plotting - need to align with actual predictions made
    # The test_dataset has been filtered to test period, but predictions may be shorter
    # due to encoder length requirements on the first samples
    # Use the test_dataset.index to get the correct time indices for predictions
    if len(predictions) < len(test_df_indexed):
        # Predictions are shorter than test_df - take the last N dates
        # This happens because first max_encoder_length samples can't be predicted
        dates = test_df_indexed['Date'].values[-len(predictions):]
        print(f"\nWARNING: Predictions ({len(predictions)}) < Test samples ({len(test_df_indexed)})")
        print(f"Date range: {dates[0]} to {dates[-1]}")
    else:
        # Full coverage
        dates = test_df_indexed['Date'].values
    
    # Verify alignment
    assert len(dates) == len(predictions) == len(actuals), \
        f"Length mismatch: dates={len(dates)}, predictions={len(predictions)}, actuals={len(actuals)}"
    
    print(f"\nPrediction period: {dates[0]} to {dates[-1]} ({len(predictions)} samples)")
    
    # Infer or confirm frequency
    if data_frequency is None:
        data_frequency = infer_frequency(dates)
        print(f"Data frequency (inferred): {data_frequency}")
    else:
        inferred = infer_frequency(dates)
        if inferred != data_frequency:
            print(f"WARNING: Config frequency ({data_frequency}) differs from inferred ({inferred})")
    
    freq_config = get_frequency_config(data_frequency)
    period_label = freq_config['period_label']
    
    # Compute metrics
    print("Computing metrics...")
    metrics_stat = compute_statistical_metrics(predictions, actuals)
    metrics_fin, strategy_returns = compute_financial_metrics(predictions, actuals, frequency=data_frequency)
    diagnostics = compute_residual_diagnostics(predictions, actuals)
    
    # Detect model collapse
    print("Analyzing prediction behavior for collapse...")
    collapse_diagnostics = detect_model_collapse(predictions, actuals, dates, frequency=data_frequency)
    
    # Print collapse diagnostics
    print("\n" + "="*70)
    print("TEMPORAL QUALITY ANALYSIS (4-Mode Detection)")
    print("="*70)
    
    # Print method-specific statistics
    print("\nStructural Detection Methods:")
    print("-" * 70)
    structural_methods = {
        'variance': '1. Variance-based',
        'range': '2. Range-based',
        'consecutive': '3. Consecutive-similarity',
    }
    
    for method_key, method_name in structural_methods.items():
        periods_flagged = collapse_diagnostics['method_stats'][method_key]
        total_periods = collapse_diagnostics['mode_stats']['total_periods']
        pct = periods_flagged / total_periods * 100 if total_periods > 0 else 0
        symbol = "[WARN]" if pct > 10 else "[ OK ]"
        print(f"{method_name:30s} {symbol} {periods_flagged:4d} {period_label} ({pct:5.1f}%)")
    
    print("\nQuality Metrics:")
    print("-" * 70)
    
    # Show individual components
    corr_flagged = collapse_diagnostics['method_stats']['low_correlation']
    dir_flagged = collapse_diagnostics['method_stats']['low_directional_acc']
    combined_flagged = collapse_diagnostics['method_stats']['quality_degraded']
    unidirectional_flagged = collapse_diagnostics['method_stats']['unidirectional']
    total_periods = collapse_diagnostics['mode_stats']['total_periods']
    
    corr_pct = corr_flagged / total_periods * 100 if total_periods > 0 else 0
    dir_pct = dir_flagged / total_periods * 100 if total_periods > 0 else 0
    combined_pct = combined_flagged / total_periods * 100 if total_periods > 0 else 0
    unidirectional_pct = unidirectional_flagged / total_periods * 100 if total_periods > 0 else 0
    
    print(f"{'  Correlation < 0.0':30s}        {corr_flagged:4d} {period_label} ({corr_pct:5.1f}%)")
    print(f"{'  Directional acc < 52%':30s}        {dir_flagged:4d} {period_label} ({dir_pct:5.1f}%)")
    print(f"{'  Combined (BOTH poor)':30s}        {combined_flagged:4d} {period_label} ({combined_pct:5.1f}%)")
    print(f"{'  Unidirectional (>98%)':30s} {('[WARN]' if unidirectional_pct > 10 else '[ OK ]'):7s} {unidirectional_flagged:4d} {period_label} ({unidirectional_pct:5.1f}%)")
    
    # Print temporal summary
    print("\n" + "-" * 70)
    print("Temporal Summary:")
    print("-" * 70)
    for line in collapse_diagnostics['temporal_summary']:
        print(line)
    print("-" * 70)
    
    # Print mode statistics
    ms = collapse_diagnostics['mode_stats']
    print("\nMode Distribution:")
    print(f"  HEALTHY:          {ms['healthy_periods']:4d} {period_label} ({ms['healthy_pct']:5.1f}%)")
    print(f"  DEGRADED:         {ms['degraded_periods']:4d} {period_label} ({ms['degraded_pct']:5.1f}%)")
    print(f"  UNIDIRECTIONAL:   {ms['unidirectional_periods']:4d} {period_label} ({ms['unidirectional_pct']:5.1f}%)")
    print(f"  WEAK_COLLAPSE:    {ms['weak_collapse_periods']:4d} {period_label} ({ms['weak_collapse_pct']:5.1f}%)")
    print(f"  STRONG_COLLAPSE:  {ms['strong_collapse_periods']:4d} {period_label} ({ms['strong_collapse_pct']:5.1f}%)")
    
    problematic_periods = ms['degraded_periods'] + ms['unidirectional_periods'] + ms['weak_collapse_periods'] + ms['strong_collapse_periods']
    problematic_pct = ms['degraded_pct'] + ms['unidirectional_pct'] + ms['weak_collapse_pct'] + ms['strong_collapse_pct']
    
    print(f"\n  Total problematic: {problematic_periods:4d} {period_label} ({problematic_pct:.1f}%)")
    
    if collapse_diagnostics['collapse_detected']:
        print(f"\n[WARN] STRUCTURAL COLLAPSE DETECTED")
    if collapse_diagnostics['degradation_detected']:
        print(f"[WARN] PREDICTION QUALITY DEGRADATION DETECTED")
    if not collapse_diagnostics['collapse_detected'] and not collapse_diagnostics['degradation_detected']:
        print(f"\n[ OK ] NO SIGNIFICANT ISSUES DETECTED")
    
    print("="*70 + "\n")
    
    # Create plots (pass collapse diagnostics for visual markers)
    print("Creating diagnostic plots...")
    create_regression_diagnostic_plots(predictions, actuals, dates, output_dir)  # Classic 4-panel regression diagnostics
    create_diagnostic_plots(predictions, actuals, dates, output_dir, frequency=data_frequency)  # Financial quality diagnostics
    create_performance_plots(actuals, strategy_returns, dates, output_dir, 
                           collapse_diagnostics=collapse_diagnostics)
    
    # Save results
    print("Saving results...")
    all_metrics = save_results(predictions, actuals, dates, metrics_stat, 
                                metrics_fin, diagnostics, collapse_diagnostics, output_dir)
    
    # Add checkpoint and frequency info to saved metrics
    all_metrics['checkpoint_used'] = checkpoint_path
    all_metrics['frequency'] = data_frequency
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    print_summary(metrics_stat, metrics_fin)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - predictions.csv")
    print(f"  - evaluation_metrics.json")
    print(f"  - diagnostic_plots.png")
    print(f"  - performance_plots.png")
    print(f"  - evaluation_<timestamp>.log")
    
    # Close logger
    print(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    logger.close()
    
    return all_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    evaluate()
