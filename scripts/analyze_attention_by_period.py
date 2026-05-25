"""
Analyze TFT attention patterns across temporal regimes.

Extracts attention weights from trained TFT models and compares how attention
distribution changes across different time periods to understand regime shifts.

Usage:
    # Default: year-by-year analysis
    python scripts/analyze_attention_by_period.py \\
        --experiment 00_baseline_exploration/sweep2_h16_drop_0.25
    
    # Custom periods
    python scripts/analyze_attention_by_period.py \\
        --experiment 00_baseline_exploration/sweep2_h16_drop_0.25 \\
        --periods "2020-01-01:2021-12-31" "2022-01-01:2023-12-31" \\
        --period-labels "Pre-inflation" "Inflation-shock"
    
    # Specific checkpoint
    python scripts/analyze_attention_by_period.py \\
        --experiment 00_baseline_exploration/sweep2_h16_drop_0.25 \\
        --checkpoint experiments/.../checkpoints/best.ckpt
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


# ============================================================================
# LOGGING SETUP
# ============================================================================

class TeeLogger:
    """Tee print statements to both console and log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', buffering=1)
        
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
    log_path = os.path.join(output_dir, f'attention_analysis_{timestamp}.log')
    
    logger = TeeLogger(log_path)
    sys.stdout = logger
    
    print(f"Logging to: {log_path}")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return logger


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze TFT attention patterns across temporal periods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., 00_baseline_exploration/sweep2_h16_drop_0.25)')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (if None, uses best from training)')
    parser.add_argument('--test-split', type=str, default=None,
                        help='Path to test CSV (if None, infers from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (if None, uses experiment/attention_analysis/)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference')
    
    # Period definition
    parser.add_argument('--periods', nargs='+', default=None,
                        help='Custom period ranges (format: "YYYY-MM-DD:YYYY-MM-DD")')
    parser.add_argument('--period-labels', nargs='+', default=None,
                        help='Labels for custom periods (must match --periods length)')
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION LOADING (reuse from evaluate_tft.py)
# ============================================================================

def load_config(experiment_name):
    """Load experiment configuration from training run."""
    possible_paths = [
        f'experiments/{experiment_name}/config.json',
        f'experiments/00_baseline_exploration/{experiment_name}/config.json',
        f'experiments/01_staleness_features/{experiment_name}/config.json',
        f'experiments/01_staleness_features_fixed/{experiment_name}/config.json',
    ]
    
    if '/' in experiment_name:
        possible_paths.insert(0, f'experiments/{experiment_name}/config.json')
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if config_path is None:
        print(f"\nERROR: Could not find config.json for experiment: {experiment_name}")
        print(f"Tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        raise FileNotFoundError(f"Config not found for experiment: {experiment_name}")
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def find_checkpoint(experiment_name, checkpoint_path=None):
    """Find checkpoint file for experiment."""
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Using specified checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    # Try to find best checkpoint
    possible_dirs = [
        f'experiments/{experiment_name}/checkpoints',
        f'experiments/00_baseline_exploration/{experiment_name}/checkpoints',
        f'experiments/01_staleness_features/{experiment_name}/checkpoints',
        f'experiments/01_staleness_features_fixed/{experiment_name}/checkpoints',
    ]
    
    for ckpt_dir in possible_dirs:
        if os.path.exists(ckpt_dir):
            # Find checkpoint with lowest val_loss
            ckpts = list(Path(ckpt_dir).glob('*.ckpt'))
            if ckpts:
                # Try to parse val_loss from filename
                def get_val_loss(p):
                    try:
                        return float(p.stem.split('val_loss=')[1].split('-')[0])
                    except:
                        return float('inf')
                
                best_ckpt = min(ckpts, key=get_val_loss)
                print(f"Using best checkpoint: {best_ckpt}")
                return str(best_ckpt)
    
    raise FileNotFoundError(f"No checkpoint found for experiment: {experiment_name}")


# ============================================================================
# DATA LOADING (reuse from evaluate_tft.py)
# ============================================================================

def load_test_data(config, test_split_path=None):
    """Load test split and prepare for evaluation."""
    base_splits_dir = config.get('data', {}).get('splits_dir', 'data/splits')
    split_prefix = f"{config['feature_set']}_{config['frequency']}"
    release_mode = config.get('data', {}).get('release_date_mode', 'fixed')
    
    possible_splits_dirs = [
        f"{base_splits_dir}/{release_mode}",
        base_splits_dir,
        f"data/splits/{release_mode}",
        "data/splits",
    ]
    
    if test_split_path is None:
        for splits_dir in possible_splits_dirs:
            test_path_with_mode = f"{splits_dir}/{split_prefix}_{release_mode}_test.csv"
            test_path_without_mode = f"{splits_dir}/{split_prefix}_test.csv"
            
            if os.path.exists(test_path_with_mode):
                test_split_path = test_path_with_mode
                break
            elif os.path.exists(test_path_without_mode):
                test_split_path = test_path_without_mode
                break
        
        if test_split_path is None:
            raise FileNotFoundError(
                f"Could not find test split. Tried:\n" +
                "\n".join([f"  - {d}/{split_prefix}_{{release_mode}}_test.csv" 
                          for d in possible_splits_dirs])
            )
    
    print(f"Loading test data from: {test_split_path}")
    
    # Also need train data for TimeSeriesDataSet creation
    train_split_path = test_split_path.replace('_test.csv', '_train.csv')
    if not os.path.exists(train_split_path):
        raise FileNotFoundError(f"Train split not found: {train_split_path}")
    
    train_df = pd.read_csv(train_split_path, index_col='Date', parse_dates=True)
    test_df = pd.read_csv(test_split_path, index_col='Date', parse_dates=True)
    
    # Check if staleness features are expected based on config
    features_list = config.get('features', {}).get('all', [])
    has_staleness = any('days_since' in f or 'is_fresh' in f for f in features_list)
    
    if has_staleness:
        try:
            from data_utils import add_staleness_features
        except ImportError:
            try:
                from src.data_utils import add_staleness_features
            except ImportError:
                print("Warning: Could not import add_staleness_features, skipping staleness feature generation")
                return train_df, test_df
        
        print("Detected staleness features in config, adding to data...")
        train_df = add_staleness_features(train_df, verbose=False)
        test_df = add_staleness_features(test_df, verbose=False)
        
        # Apply same normalization as training
        staleness_cols = [c for c in train_df.columns if 'days_since' in c]
        for col in staleness_cols:
            train_df[col] = train_df[col] / 30.0
            test_df[col] = test_df[col] / 30.0
    
    return train_df, test_df


def prepare_test_dataset(train_df, test_df, config):
    """Prepare TimeSeriesDataSet for test data."""
    # Get features from config
    features_config = config.get('features', {})
    feature_list = features_config.get('all', [])
    
    # Reset index and add required columns
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    
    # Combine train and test for continuous time index
    # This ensures test samples have enough history for encoder
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df['time_idx'] = range(len(combined_df))
    combined_df['group'] = 'SP500'
    
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
        time_varying_unknown_reals=feature_list,
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    
    # Create test dataset from combined data (uses train normalization)
    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        combined_df,
        predict=False,  # False for rolling predictions
        stop_randomization=True
    )
    
    # Filter dataset to only test period indices
    test_start_idx = len(train_df)
    test_dataset.index = test_dataset.index[test_dataset.index['time'] >= test_start_idx]
    
    print(f"Test dataset size after filtering: {len(test_dataset.index)}")
    
    # Keep indexed version for date alignment
    test_df_indexed = test_df.copy()
    
    return test_dataset, test_df_indexed, test_start_idx


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path, config):
    """Load trained TFT model from checkpoint (supports regime output)."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Check if this model used regime output
    regime_config = config.get('regime_output', {})
    use_regime_output = regime_config.get('enabled', False)
    
    # Check if this model used distribution loss
    training_config = config.get('training', {})
    mean_weight = training_config.get('dist_loss_mean_weight', 0.0)
    std_weight = training_config.get('dist_loss_std_weight', 0.0)
    uses_dist_loss = mean_weight > 0 or std_weight > 0
    
    # Case 1: Baseline model (no regime output, no dist loss)
    if not uses_dist_loss and not use_regime_output:
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"  Loaded baseline model, moved to device: {device}")
        return model
    
    # Case 2: Regime output without dist loss
    if use_regime_output and not uses_dist_loss:
        print(f"  Detected regime output checkpoint, applying architecture modification...")
        
        # Load checkpoint dict first
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract hparams from checkpoint
        hparams = checkpoint['hyper_parameters']
        
        # Create model with baseline architecture (no weights loaded)
        model = TemporalFusionTransformer(**hparams)
        
        # Apply regime output architecture before loading weights
        from regime_output import replace_output_layer
        
        num_regimes = regime_config.get('num_regimes', 2)
        routing_mode = regime_config.get('routing_mode', 'learned')
        routing_strategy = regime_config.get('routing_strategy', 'learned')
        vix_threshold = regime_config.get('vix_threshold', 25.0)
        
        model = replace_output_layer(
            model,
            num_regimes=num_regimes,
            routing_mode=routing_mode,
            routing_strategy=routing_strategy,
            vix_threshold=vix_threshold
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"  Model moved to device: {device}")
        return model
    
    # Case 3: Distribution loss (with or without regime output)
    print(f"  Loading distribution loss checkpoint (bypassing corrupted loss)...")
    
    import pickle
    import tempfile
    
    # Custom unpickler for dist loss
    class FixedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'QuantileLoss' and 'pytorch_forecasting' in module:
                from pytorch_forecasting.metrics import QuantileLoss
                return QuantileLoss
            return super().find_class(module, name)
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = FixedUnpickler(f).load()
    except Exception as e:
        print(f"  Direct unpickling failed: {e}")
        raise RuntimeError(
            f"Cannot load checkpoint with monkey-patched loss. "
            f"Recommendation: Retrain without distribution loss."
        )
    
    # Fix hyperparameters if loss object exists
    if 'hyper_parameters' in checkpoint and 'loss' in checkpoint['hyper_parameters']:
        from pytorch_forecasting.metrics import QuantileLoss
        checkpoint['hyper_parameters']['loss'] = QuantileLoss()
    
    # Save fixed checkpoint temporarily
    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        if use_regime_output:
            print(f"  Detected regime output checkpoint, applying architecture modification...")
            
            # Create model from hyperparameters (no weights loaded)
            hparams = checkpoint['hyper_parameters']
            model = TemporalFusionTransformer(**hparams)
            
            # Apply regime output architecture before loading weights
            from regime_output import replace_output_layer
            
            num_regimes = regime_config.get('num_regimes', 2)
            routing_mode = regime_config.get('routing_mode', 'learned')
            routing_strategy = regime_config.get('routing_strategy', 'learned')
            vix_threshold = regime_config.get('vix_threshold', 25.0)
            
            model = replace_output_layer(
                model,
                num_regimes=num_regimes,
                routing_mode=routing_mode,
                routing_strategy=routing_strategy,
                vix_threshold=vix_threshold
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
            
            # Now load the state dict with MoE architecture
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
        import os
        os.unlink(tmp_path)
    
    model.eval()
    
    # Re-apply distribution penalties
    from loss_wrapper import add_distribution_penalties
    model = add_distribution_penalties(
        model,
        mean_weight=mean_weight,
        std_weight=std_weight,
        target_mean=training_config.get('dist_loss_target_mean', 0.0003),
        target_std=training_config.get('dist_loss_target_std', 0.01)
    )
    print(f"  Re-applied distribution penalties: mean_weight={mean_weight}, std_weight={std_weight}")
    
    # Check if this is a custom TFT model
    is_custom = False
    if 'state_dict' in checkpoint:
        # Look for custom TFT signatures in state dict
        state_dict_keys = checkpoint['state_dict'].keys()
        if any('static_transform' in k or 'variable_selection' in k for k in state_dict_keys):
            # Check if it has pytorch-forecasting specific keys
            has_pf_keys = any('loss.quantiles' in k for k in state_dict_keys)
            is_custom = not has_pf_keys
   
    """
    if is_custom:
        print("  Detected custom TFT model")
        # Import custom TFT
        import sys
        from pathlib import Path
        
        # Add project root to path (script is in scripts/, need parent)
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent  # Go up from scripts/ to project root
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from models.tft_model import TemporalFusionTransformer as CustomTFT
        
        # Load hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Create model instance
        model = CustomTFT(**hparams)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    else:
        print("  Detected pytorch-forecasting TFT model")
        # Use pytorch-forecasting loader
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        model.eval()
    """
    
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Model moved to device: {device}")
    
    return model


# ============================================================================
# PERIOD SPLITTING
# ============================================================================

def create_periods(dates, custom_periods=None, custom_labels=None):
    """
    Split dates into analysis periods.
    
    Parameters:
    -----------
    dates : array-like
        Array of dates
    custom_periods : list of str, optional
        Custom period ranges in format ["YYYY-MM-DD:YYYY-MM-DD", ...]
    custom_labels : list of str, optional
        Labels for custom periods
        
    Returns:
    --------
    dict mapping period_label -> boolean mask
    """
    dates = pd.to_datetime(dates)
    
    if custom_periods is not None:
        # Parse custom periods
        periods = {}
        labels = custom_labels if custom_labels else [f"Period_{i+1}" for i in range(len(custom_periods))]
        
        if len(labels) != len(custom_periods):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of periods ({len(custom_periods)})")
        
        for label, period_str in zip(labels, custom_periods):
            start_str, end_str = period_str.split(':')
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
            mask = (dates >= start) & (dates <= end)
            periods[label] = mask
            print(f"  {label}: {start_str} to {end_str} ({mask.sum()} samples)")
    else:
        # Default: year-by-year
        years = sorted(dates.year.unique())
        periods = {}
        
        for year in years:
            mask = dates.year == year
            periods[str(year)] = mask
            print(f"  {year}: {mask.sum()} samples")
    
    return periods


# ============================================================================
# ATTENTION EXTRACTION
# ============================================================================

def extract_attention_patterns(model, test_dataset, batch_size=128):
    """
    Extract attention weights for all test samples.
    
    Returns:
    --------
    dict with:
        'attention': array of shape [n_samples, encoder_length]
        'encoder_variables': array of shape [n_samples, encoder_length, n_features]
        'predictions': array of shape [n_samples]
        'actuals': array of shape [n_samples]
    """
    model.eval()
    dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    all_attention = []
    all_encoder_vars = []
    all_predictions = []
    all_actuals = []
    
    print("Extracting attention patterns...")
    print(f"Total batches to process: {len(dataloader)}")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            # Move to device
            if torch.cuda.is_available():
                x = {k: v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
            
            # Forward pass
            output = model(x)
            
            # Extract predictions (same logic as evaluate_tft.py)
            if hasattr(output, 'prediction'):
                pred_tensor = output.prediction
            elif isinstance(output, tuple):
                pred_tensor = output[0]
            elif isinstance(output, dict) and 'prediction' in output:
                pred_tensor = output['prediction']
            else:
                pred_tensor = output
            
            # Median quantile (index 3)
            preds = pred_tensor[:, 0, 3]
            
            # Extract actuals
            if isinstance(y, tuple):
                y_actual = y[0]
            else:
                y_actual = y
            
            all_predictions.append(preds.cpu().numpy())
            all_actuals.append(y_actual[:, 0].cpu().numpy())
            
            # Extract attention using interpret_output or getter method
            # Note: interpret_output can be expensive, so we do it after storing preds/actuals
            attn_weights = None
            
            # Method 1: Custom TFT model - use getter method
            if hasattr(model, 'get_attention_weights'):
                output_again = model(x)  # Need fresh forward pass for getter
                attn_weights = model.get_attention_weights()
                
                if attn_weights is not None:
                    # Shape: [batch, num_heads, max_prediction_length, encoder_length + max_prediction_length]
                    # Average across heads to match pytorch-forecasting format
                    attn_weights = attn_weights.mean(dim=1)  # [batch, max_prediction_length, seq_len]
                    
                    # Extract first prediction timestep
                    if attn_weights.size(1) == 1:
                        attn_weights = attn_weights[:, 0, :]  # [batch, seq_len]
                    else:
                        attn_weights = attn_weights[:, 0, :]  # [batch, seq_len]
                    
                    all_attention.append(attn_weights.cpu().numpy())
                    
            # Method 2: pytorch-forecasting TFT model - use interpret_output
            elif hasattr(model, 'interpret_output'):
                try:
                    interpretation = model.interpret_output(
                        output,
                        reduction='none',
                        attention_prediction_horizon=0  # First prediction step
                    )
                    
                    if 'attention' in interpretation:
                        attn = interpretation['attention'].cpu().numpy()
                        all_attention.append(attn)
                    
                    if 'encoder_variables' in interpretation:
                        enc_vars = interpretation['encoder_variables'].cpu().numpy()
                        all_encoder_vars.append(enc_vars)
                        
                except Exception as e:
                    print(f"\nWarning: interpret_output failed on batch {batch_idx}: {e}")
                    print("Continuing without attention extraction for this batch...")
                    # Still keep the predictions/actuals
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)
    
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Total actuals: {len(actuals)}")
    
    results = {
        'predictions': predictions,
        'actuals': actuals,
    }
    
    if all_attention:
        results['attention'] = np.concatenate(all_attention)
        print(f"Attention shape: {results['attention'].shape}")
    else:
        print("WARNING: No attention data extracted!")
    
    if all_encoder_vars:
        results['encoder_variables'] = np.concatenate(all_encoder_vars)
        print(f"Encoder variables shape: {results['encoder_variables'].shape}")
    
    return results


# ============================================================================
# ATTENTION ANALYSIS
# ============================================================================

def compute_attention_entropy(attention_weights):
    """
    Compute entropy of attention distribution.
    
    H = -sum(p_i * log(p_i))
    
    Higher entropy = more diffuse attention
    Lower entropy = concentrated attention
    """
    # Normalize attention to sum to 1
    attn_norm = attention_weights / (attention_weights.sum(axis=-1, keepdims=True) + 1e-10)
    
    # Compute entropy
    entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10), axis=-1)
    
    return entropy


def analyze_attention_by_period(attention_data, periods, feature_names=None):
    """
    Analyze attention patterns for each period.
    
    Returns:
    --------
    dict mapping period_label -> statistics dict
    """
    attention = attention_data['attention']
    n_features = attention.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    results = {}
    
    for period_name, mask in periods.items():
        period_attention = attention[mask]
        
        if len(period_attention) == 0:
            print(f"Warning: No samples in period {period_name}")
            continue
        
        # Compute statistics
        mean_attention = period_attention.mean(axis=0)
        std_attention = period_attention.std(axis=0)
        
        # Compute entropy
        entropy = compute_attention_entropy(period_attention)
        
        # Feature importance ranking
        sorted_indices = np.argsort(mean_attention)[::-1]
        
        results[period_name] = {
            'n_samples': int(mask.sum()),
            'mean_attention': mean_attention,
            'std_attention': std_attention,
            'entropy_mean': float(entropy.mean()),
            'entropy_std': float(entropy.std()),
            'top_features': [(feature_names[i], float(mean_attention[i])) 
                            for i in sorted_indices[:5]],
            'attention_concentration': float((mean_attention ** 2).sum()),  # Herfindahl index
        }
    
    return results


def compare_attention_patterns(period_stats):
    """
    Compare attention patterns across periods to identify shifts.
    
    Returns:
    --------
    dict with comparison metrics
    """
    period_names = list(period_stats.keys())
    
    if len(period_names) < 2:
        return {}
    
    # Compute pairwise differences
    comparisons = {}
    
    for i, period1 in enumerate(period_names[:-1]):
        for period2 in period_names[i+1:]:
            stats1 = period_stats[period1]
            stats2 = period_stats[period2]
            
            # Cosine similarity of attention vectors
            attn1 = stats1['mean_attention']
            attn2 = stats2['mean_attention']
            cos_sim = np.dot(attn1, attn2) / (np.linalg.norm(attn1) * np.linalg.norm(attn2) + 1e-10)
            
            # L2 distance
            l2_dist = np.linalg.norm(attn1 - attn2)
            
            # Entropy difference
            entropy_diff = stats2['entropy_mean'] - stats1['entropy_mean']
            
            # Concentration difference
            conc_diff = stats2['attention_concentration'] - stats1['attention_concentration']
            
            comparisons[f"{period1}_vs_{period2}"] = {
                'cosine_similarity': float(cos_sim),
                'l2_distance': float(l2_dist),
                'entropy_change': float(entropy_diff),
                'concentration_change': float(conc_diff),
            }
    
    return comparisons


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_attention_heatmap(period_stats, feature_names, output_path):
    """Create heatmap showing attention weights across periods and features."""
    period_names = list(period_stats.keys())
    n_features = len(feature_names)
    
    # Build matrix: periods” features
    attention_matrix = np.zeros((len(period_names), n_features))
    for i, period in enumerate(period_names):
        attention_matrix[i] = period_stats[period]['mean_attention']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(period_names) * 0.5)))
    
    sns.heatmap(
        attention_matrix,
        xticklabels=feature_names,
        yticklabels=period_names,
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Mean Attention Weight'},
        ax=ax
    )
    
    ax.set_title('Attention Weights by Period and Feature', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Period', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention heatmap: {output_path}")


def plot_attention_trends(period_stats, output_path):
    """Plot entropy and concentration trends across periods."""
    period_names = list(period_stats.keys())
    
    entropy_means = [period_stats[p]['entropy_mean'] for p in period_names]
    entropy_stds = [period_stats[p]['entropy_std'] for p in period_names]
    concentrations = [period_stats[p]['attention_concentration'] for p in period_names]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Entropy plot
    ax = axes[0]
    x = range(len(period_names))
    ax.errorbar(x, entropy_means, yerr=entropy_stds, marker='o', linewidth=2, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(period_names, rotation=45, ha='right')
    ax.set_ylabel('Attention Entropy', fontsize=12)
    ax.set_title('Attention Entropy by Period (Higher = More Diffuse)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Concentration plot
    ax = axes[1]
    ax.plot(x, concentrations, marker='s', linewidth=2, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(period_names, rotation=45, ha='right')
    ax.set_ylabel('Attention Concentration', fontsize=12)
    ax.set_title('Attention Concentration by Period (Higher = More Concentrated)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention trends: {output_path}")


def plot_feature_importance_comparison(period_stats, feature_names, output_path):
    """Plot top features by period in bar chart."""
    period_names = list(period_stats.keys())
    n_periods = len(period_names)
    
    # Limit to top 10 features overall
    all_attention = np.zeros(len(feature_names))
    for stats in period_stats.values():
        all_attention += stats['mean_attention']
    top_10_indices = np.argsort(all_attention)[-10:][::-1]
    top_10_features = [feature_names[i] for i in top_10_indices]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top_10_features))
    width = 0.8 / n_periods
    
    for i, period in enumerate(period_names):
        attention = period_stats[period]['mean_attention']
        values = [attention[idx] for idx in top_10_indices]
        offset = (i - n_periods/2) * width + width/2
        ax.bar(x + offset, values, width, label=period)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Feature Importance by Period (Top 10 Features)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_10_features, rotation=45, ha='right')
    ax.legend(title='Period', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature importance comparison: {output_path}")


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(period_stats, comparisons, output_dir):
    """Save analysis results to JSON."""
    results = {
        'period_statistics': {},
        'period_comparisons': comparisons,
        'analysis_timestamp': datetime.now().isoformat(),
    }
    
    # Convert numpy arrays to lists for JSON serialization
    for period, stats in period_stats.items():
        results['period_statistics'][period] = {
            'n_samples': stats['n_samples'],
            'entropy_mean': stats['entropy_mean'],
            'entropy_std': stats['entropy_std'],
            'attention_concentration': stats['attention_concentration'],
            'top_features': stats['top_features'],
            'mean_attention': stats['mean_attention'].tolist(),
            'std_attention': stats['std_attention'].tolist(),
        }
    
    output_path = os.path.join(output_dir, 'attention_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results: {output_path}")
    
    return results


def print_summary(period_stats, comparisons):
    """Print human-readable summary of results."""
    print("\n" + "="*70)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*70)
    
    # Per-period summary
    print("\nPer-Period Statistics:")
    print("-" * 70)
    for period, stats in period_stats.items():
        print(f"\n{period}:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Entropy: {stats['entropy_mean']:.4f} +/- {stats['entropy_std']:.4f}")
        print(f"  Concentration: {stats['attention_concentration']:.4f}")
        print(f"  Top 5 features:")
        for feat, weight in stats['top_features']:
            print(f"    {feat:30s} {weight:.4f}")
    
    # Comparisons
    if comparisons:
        print("\n" + "-" * 70)
        print("Period Comparisons:")
        print("-" * 70)
        for comp_name, metrics in comparisons.items():
            print(f"\n{comp_name}:")
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
            print(f"  L2 distance: {metrics['l2_distance']:.4f}")
            print(f"  Entropy change: {metrics['entropy_change']:.4f}")
            print(f"  Concentration change: {metrics['concentration_change']:.4f}")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.experiment)
    
    # Setup output directory
    if args.output_dir is None:
        exp_dir = f"experiments/{args.experiment}"
        args.output_dir = os.path.join(exp_dir, 'attention_analysis')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    print(f"Experiment: {args.experiment}")
    print(f"Output directory: {args.output_dir}")
    
    # Find checkpoint
    checkpoint_path = find_checkpoint(args.experiment, args.checkpoint)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_test_data(config, args.test_split)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Prepare test dataset
    print("\nPreparing test dataset...")
    test_dataset, test_df_indexed, test_start_idx = prepare_test_dataset(train_df, test_df, config)
    
    # Load model
    print("\nLoading model...")
    model = load_model(checkpoint_path, config)
    
    # Extract attention patterns
    print("\nExtracting attention patterns...")
    attention_data = extract_attention_patterns(model, test_dataset, args.batch_size)
    
    # Check if we got attention data
    if 'attention' not in attention_data:
        print("\n" + "!"*70)
        print("ERROR: Failed to extract attention patterns from model")
        print("!"*70)
        print("\nPossible causes:")
        print("1. Model's interpret_output() method is not working properly")
        print("2. PyTorch Forecasting version incompatibility")
        print("3. Model architecture doesn't support attention extraction")
        print("\nSaving predictions/actuals but cannot analyze attention.")
        print("!"*70 + "\n")
        
        # Save what we have
        output_path = os.path.join(args.output_dir, 'predictions_only.json')
        with open(output_path, 'w') as f:
            json.dump({
                'predictions': attention_data['predictions'].tolist(),
                'actuals': attention_data['actuals'].tolist(),
                'error': 'Failed to extract attention patterns'
            }, f, indent=2)
        print(f"Saved predictions to: {output_path}")
        logger.close()
        return
    
    # Get dates aligned with predictions
    max_encoder_length = config['architecture']['max_encoder_length']
    dates = test_df_indexed['Date'].values[-len(attention_data['predictions']):]
    
    print(f"\nPrediction period: {dates[0]} to {dates[-1]}")
    print(f"Total predictions: {len(dates)}")
    
    # Create periods
    print("\nCreating temporal periods:")
    periods = create_periods(dates, args.periods, args.period_labels)
    
    # Get feature names
    feature_list = config.get('features', {}).get('all', [])
    # Attention is over encoder timesteps, not features
    # We need to understand what TFT's attention represents
    
    # TFT attention is over encoder timesteps, not features directly
    # We'll analyze temporal attention patterns
    encoder_length = attention_data['attention'].shape[1]
    timestep_labels = [f"t-{encoder_length-i}" for i in range(encoder_length)]
    
    print(f"\nAttention over {encoder_length} encoder timesteps")
    
    # Analyze attention by period
    print("\nAnalyzing attention patterns by period...")
    period_stats = analyze_attention_by_period(attention_data, periods, timestep_labels)
    
    # Compare periods
    print("\nComparing attention patterns across periods...")
    comparisons = compare_attention_patterns(period_stats)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_attention_heatmap(
        period_stats, 
        timestep_labels,
        os.path.join(args.output_dir, 'attention_heatmap.png')
    )
    
    plot_attention_trends(
        period_stats,
        os.path.join(args.output_dir, 'attention_trends.png')
    )
    
    plot_feature_importance_comparison(
        period_stats,
        timestep_labels,
        os.path.join(args.output_dir, 'temporal_attention_comparison.png')
    )
    
    # Save results
    print("\nSaving results...")
    save_results(period_stats, comparisons, args.output_dir)
    
    # Print summary
    print_summary(period_stats, comparisons)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"  - attention_analysis_results.json")
    print(f"  - attention_heatmap.png")
    print(f"  - attention_trends.png")
    print(f"  - temporal_attention_comparison.png")
    
    logger.close()


if __name__ == "__main__":
    main()