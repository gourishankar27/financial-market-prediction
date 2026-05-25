"""
Analyze TFT Variable Selection Network (VSN) weights across market regimes.

Extracts VSN weights showing which input FEATURES (VIX, CPI, yields, etc.) the model
selects, complementing temporal attention analysis (which shows which TIMESTEPS matter).

IMPORTANT NOTE ON TARGET VARIABLE:
----------------------------------
pytorch-forecasting's TFT implementation separates the target variable 
(SP500_Returns) into a privileged 'encoder_target' pathway that bypasses 
the Variable Selection Network entirely. This differs from the original 
TFT paper (Lim et al., 2021), where the lagged target was included in VSN 
and received weights of 0.30-0.70 across datasets.

As a result, this analysis covers EXOGENOUS FEATURES ONLY (VIX, Treasury_10Y, 
etc.), not the autoregressive signal. The model still uses lagged target 
values - they're just fed through a separate pathway and don't compete 
with other features in variable selection.

This is a design choice in pytorch-forecasting, not an error in experiment 
setup. The analysis answers: "Among exogenous features, which does the model 
select?" rather than "Among all features including lagged target, which matter?"

Key insight: TFT's forward() returns raw VSN weights in output dict:
- encoder_variables: [batch, encoder_len, 1, n_encoder_vars] - which features for history
- decoder_variables: [batch, decoder_len, 1, n_decoder_vars] - which features for future
- static_variables: [batch, 1, n_static_vars] - static feature importance

Usage:
    # Single experiment (outputs to experiment_dir/vsn_analysis/)
    python analyze_vsn_weights.py \\
        --experiment 00_baseline_exploration/sweep2_h16_drop_0.25
    
    # Entire phase
    python analyze_vsn_weights.py --phase 00_baseline_exploration --continue-on-error
    
    # Custom periods (e.g., pre/post Fed pivot)
    python analyze_vsn_weights.py \\
        --experiment 00_baseline_exploration/sweep2_h16_drop_0.25 \\
        --periods "2020-01-01:2021-12-31" "2022-01-01:2023-12-31" \\
        --period-labels "Pre-tightening" "Tightening"
    
    # With optional analyses
    python analyze_vsn_weights.py \\
        --experiment 00_baseline_exploration/sweep2_h16_drop_0.25 \\
        --correlate-vix --analyze-staleness

Author: Sam (EEE598 Deep Learning Project)
Date: November 2025
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
from typing import Dict, List, Optional, Tuple, Any

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
    log_path = os.path.join(output_dir, f'vsn_analysis_{timestamp}.log')
    
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
        description='Analyze TFT Variable Selection Network weights across market regimes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment selection (mutually exclusive: single experiment OR entire phase)
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument('--experiment', type=str, default=None,
                        help='Single experiment name (e.g., 00_baseline_exploration/sweep2_h16_drop_0.25)')
    exp_group.add_argument('--phase', type=str, default=None,
                        help='Process all experiments in a phase directory (e.g., 00_baseline_exploration)')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (if None, uses best from training)')
    parser.add_argument('--test-split', type=str, default=None,
                        help='Path to test CSV (if None, infers from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (if None, uses experiment_dir/vsn_analysis/)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference')
    
    # Period definition
    parser.add_argument('--periods', nargs='+', default=None,
                        help='Custom period ranges (format: "YYYY-MM-DD:YYYY-MM-DD")')
    parser.add_argument('--period-labels', nargs='+', default=None,
                        help='Labels for custom periods (must match --periods length)')
    
    # Analysis options
    parser.add_argument('--correlate-vix', action='store_true',
                        help='Correlate VSN weights with VIX levels')
    parser.add_argument('--analyze-staleness', action='store_true',
                        help='Analyze staleness feature interactions')
    parser.add_argument('--top-n-features', type=int, default=15,
                        help='Number of top features to show in plots')
    
    # Batch processing options
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments that already have vsn_analysis directory')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue processing other experiments if one fails')
    
    return parser.parse_args()


# ============================================================================
# CONFIGURATION LOADING (matches analyze_attention_by_period.py)
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
            ckpts = list(Path(ckpt_dir).glob('*.ckpt'))
            if ckpts:
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
# DATA LOADING (matches analyze_attention_by_period.py)
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
    
    train_split_path = test_split_path.replace('_test.csv', '_train.csv')
    if not os.path.exists(train_split_path):
        raise FileNotFoundError(f"Train split not found: {train_split_path}")
    
    train_df = pd.read_csv(train_split_path, index_col='Date', parse_dates=True)
    test_df = pd.read_csv(test_split_path, index_col='Date', parse_dates=True)
    
    # Check if staleness features are expected
    features_list = config.get('features', {}).get('all', [])
    has_staleness = any('days_since' in f or 'is_fresh' in f for f in features_list)
    
    if has_staleness:
        try:
            from data_utils import add_staleness_features
        except ImportError:
            try:
                from src.data_utils import add_staleness_features
            except ImportError:
                print("Warning: Could not import add_staleness_features")
                return train_df, test_df
        
        print("Adding staleness features...")
        train_df = add_staleness_features(train_df, verbose=False)
        test_df = add_staleness_features(test_df, verbose=False)
        
        staleness_cols = [c for c in train_df.columns if 'days_since' in c]
        for col in staleness_cols:
            train_df[col] = train_df[col] / 30.0
            test_df[col] = test_df[col] / 30.0
    
    return train_df, test_df


def prepare_test_dataset(train_df, test_df, config):
    """Prepare TimeSeriesDataSet for test data."""
    features_config = config.get('features', {})
    feature_list = features_config.get('all', [])
    
    # Store original test dates before reset_index
    test_dates = test_df.index.copy()
    
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df['time_idx'] = range(len(combined_df))
    combined_df['group'] = 'SP500'
    
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
    
    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        combined_df,
        predict=False,
        stop_randomization=True
    )
    
    test_start_idx = len(train_df)
    test_dataset.index = test_dataset.index[test_dataset.index['time'] >= test_start_idx]
    
    print(f"Test dataset size: {len(test_dataset.index)}")
    
    return test_dataset, test_dates, test_start_idx


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path, config):
    """Load trained TFT model from checkpoint (supports regime output and attention)."""
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
        print(f"  Model loaded on device: {device}")
        return model
    
    # Case 2: Regime output without dist loss
    if use_regime_output and not uses_dist_loss:
        print(f"  Detected regime output checkpoint, applying architecture modification...")
        
        # Load checkpoint dict first
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hparams = checkpoint['hyper_parameters']
        
        # Create model with baseline architecture
        model = TemporalFusionTransformer(**hparams)
        
        # Apply regime output architecture
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
        
        # Load weights
        model.load_state_dict(checkpoint['state_dict'])
        print(f"  Successfully loaded regime output state_dict")
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"  Model loaded on device: {device}")
        return model
    
    # Case 3: Distribution loss (with or without regime output/attention)
    print(f"  Loading distribution loss checkpoint (bypassing corrupted loss)...")
    
    import pickle
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
    except Exception as e:
        print(f"  Direct unpickling failed: {e}")
        raise RuntimeError(
            f"Cannot load checkpoint with monkey-patched loss. "
            f"Recommendation: Retrain without distribution loss."
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
    
    from loss_wrapper import add_distribution_penalties
    model = add_distribution_penalties(
        model,
        mean_weight=mean_weight,
        std_weight=std_weight,
        target_mean=training_config.get('dist_loss_target_mean', 0.0003),
        target_std=training_config.get('dist_loss_target_std', 0.01)
    )
    print(f"  Re-applied distribution penalties: mean_weight={mean_weight}, std_weight={std_weight}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Model loaded on device: {device}")
    
    return model


# ============================================================================
# VSN WEIGHT EXTRACTION
# ============================================================================

def extract_vsn_weights(model, test_dataset, batch_size=64):
    """
    Extract VSN weights for all test samples.
    
    Returns dict with:
        'encoder_vsn': [n_samples, n_encoder_vars] - time-averaged encoder VSN weights
        'encoder_vsn_per_timestep': [n_samples, encoder_len, n_encoder_vars] - full weights
        'decoder_vsn': [n_samples, n_decoder_vars] - time-averaged decoder VSN weights
        'static_vsn': [n_samples, n_static_vars] - static VSN weights
        'encoder_var_names': list of encoder variable names
        'decoder_var_names': list of decoder variable names
        'predictions': [n_samples]
        'actuals': [n_samples]
    """
    model.eval()
    device = next(model.parameters()).device
    dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    # Get variable names from model
    encoder_var_names = list(model.encoder_variables)
    decoder_var_names = list(model.decoder_variables)
    static_var_names = list(model.static_variables) if hasattr(model, 'static_variables') and model.static_variables else []
    
    print(f"\nExtracting VSN weights...")
    print(f"  Encoder variables ({len(encoder_var_names)}): {encoder_var_names[:5]}{'...' if len(encoder_var_names) > 5 else ''}")
    print(f"  Decoder variables ({len(decoder_var_names)}): {decoder_var_names[:5]}{'...' if len(decoder_var_names) > 5 else ''}")
    print(f"  Static variables ({len(static_var_names)}): {static_var_names}")
    
    all_encoder_vsn = []
    all_encoder_vsn_per_timestep = []
    all_decoder_vsn = []
    all_static_vsn = []
    all_predictions = []
    all_actuals = []
    all_encoder_lengths = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            # Move to device
            x = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
            
            # Forward pass - output contains raw VSN weights
            output = model(x)
            
            # Debug: print output structure on first batch
            if batch_idx == 0:
                print(f"\n  DEBUG: Output type: {type(output)}")
                if hasattr(output, '_fields'):
                    print(f"  DEBUG: Output fields (NamedTuple): {output._fields}")
                elif hasattr(output, 'keys'):
                    print(f"  DEBUG: Output keys (dict): {list(output.keys())}")
                elif hasattr(output, '__dict__'):
                    print(f"  DEBUG: Output attrs: {list(output.__dict__.keys())}")
                
                # Check for the VSN weights under different possible attribute names
                for attr in ['encoder_variables', 'static_variables', 'decoder_variables']:
                    if hasattr(output, attr):
                        val = getattr(output, attr)
                        print(f"  DEBUG: output.{attr} shape: {val.shape if hasattr(val, 'shape') else type(val)}")
                    elif isinstance(output, dict) and attr in output:
                        val = output[attr]
                        print(f"  DEBUG: output['{attr}'] shape: {val.shape if hasattr(val, 'shape') else type(val)}")
            
            # Extract encoder VSN weights
            # Shape: [batch, encoder_len, 1, n_vars] -> [batch, encoder_len, n_vars]
            # Try both dict-style and attribute-style access
            enc_vsn = None
            if hasattr(output, 'encoder_variables'):
                enc_vsn = output.encoder_variables
            elif isinstance(output, dict) and 'encoder_variables' in output:
                enc_vsn = output['encoder_variables']
            
            if enc_vsn is not None:
                if enc_vsn.ndim == 4:
                    enc_vsn = enc_vsn.squeeze(2)
                
                # Store per-timestep weights
                all_encoder_vsn_per_timestep.append(enc_vsn.cpu().numpy())
                
                # Time-averaged (accounting for variable encoder lengths)
                encoder_lengths = x['encoder_lengths'].cpu().numpy()
                batch_avg = []
                for i in range(enc_vsn.size(0)):
                    valid_len = int(encoder_lengths[i])
                    if valid_len > 0:
                        batch_avg.append(enc_vsn[i, :valid_len, :].mean(dim=0).cpu().numpy())
                    else:
                        batch_avg.append(enc_vsn[i].mean(dim=0).cpu().numpy())
                all_encoder_vsn.append(np.stack(batch_avg))
                all_encoder_lengths.append(encoder_lengths)
            
            # Extract decoder VSN weights
            dec_vsn = None
            if hasattr(output, 'decoder_variables'):
                dec_vsn = output.decoder_variables
            elif isinstance(output, dict) and 'decoder_variables' in output:
                dec_vsn = output['decoder_variables']
            
            if dec_vsn is not None:
                if dec_vsn.ndim == 4:
                    dec_vsn = dec_vsn.squeeze(2)
                # Average over decoder timesteps
                all_decoder_vsn.append(dec_vsn.mean(dim=1).cpu().numpy())
            
            # Extract static VSN weights
            static_vsn = None
            if hasattr(output, 'static_variables'):
                static_vsn = output.static_variables
            elif isinstance(output, dict) and 'static_variables' in output:
                static_vsn = output['static_variables']
            
            if static_vsn is not None:
                if static_vsn.ndim == 3:
                    static_vsn = static_vsn.squeeze(1)
                all_static_vsn.append(static_vsn.cpu().numpy())
            
            # Extract predictions
            if hasattr(output, 'prediction'):
                pred = output.prediction
            elif isinstance(output, dict) and 'prediction' in output:
                pred = output['prediction']
            else:
                pred = output
            
            if pred.ndim == 3:
                preds = pred[:, 0, 3]  # First step, median quantile
            else:
                preds = pred[:, 0]
            all_predictions.append(preds.cpu().numpy())
            
            # Extract actuals
            if isinstance(y, tuple):
                y_actual = y[0]
            else:
                y_actual = y
            all_actuals.append(y_actual[:, 0].cpu().numpy())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    # Concatenate
    results = {
        'encoder_vsn': np.concatenate(all_encoder_vsn) if all_encoder_vsn else np.array([]),
        'encoder_vsn_per_timestep': np.concatenate(all_encoder_vsn_per_timestep) if all_encoder_vsn_per_timestep else np.array([]),
        'decoder_vsn': np.concatenate(all_decoder_vsn) if all_decoder_vsn else np.array([]),
        'static_vsn': np.concatenate(all_static_vsn) if all_static_vsn else np.array([]),
        'encoder_var_names': encoder_var_names,
        'decoder_var_names': decoder_var_names,
        'static_var_names': static_var_names,
        'predictions': np.concatenate(all_predictions),
        'actuals': np.concatenate(all_actuals),
        'encoder_lengths': np.concatenate(all_encoder_lengths) if all_encoder_lengths else np.array([]),
    }
    
    print(f"\nExtraction complete:")
    print(f"  Samples: {len(results['predictions'])}")
    print(f"  Encoder VSN shape: {results['encoder_vsn'].shape}")
    if results['decoder_vsn'].size > 0:
        print(f"  Decoder VSN shape: {results['decoder_vsn'].shape}")
    
    return results


# ============================================================================
# PERIOD ANALYSIS
# ============================================================================

def create_periods(dates, custom_periods=None, custom_labels=None):
    """Split dates into analysis periods."""
    dates = pd.to_datetime(dates)
    
    if custom_periods is not None:
        periods = {}
        labels = custom_labels if custom_labels else [f"Period_{i+1}" for i in range(len(custom_periods))]
        
        for label, period_str in zip(labels, custom_periods):
            start_str, end_str = period_str.split(':')
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
            mask = (dates >= start) & (dates <= end)
            periods[label] = mask
            print(f"  {label}: {start_str} to {end_str} ({mask.sum()} samples)")
    else:
        years = sorted(dates.year.unique())
        periods = {}
        for year in years:
            mask = dates.year == year
            periods[str(year)] = mask
            print(f"  {year}: {mask.sum()} samples")
    
    return periods


def analyze_vsn_by_period(vsn_data, periods):
    """
    Analyze VSN weights for each period.
    
    Returns dict mapping period_label -> statistics
    """
    encoder_vsn = vsn_data['encoder_vsn']
    var_names = vsn_data['encoder_var_names']
    
    results = {}
    
    for period_name, mask in periods.items():
        period_vsn = encoder_vsn[mask]
        
        if len(period_vsn) == 0:
            print(f"Warning: No samples in period {period_name}")
            continue
        
        mean_weights = period_vsn.mean(axis=0)
        std_weights = period_vsn.std(axis=0)
        
        # Rank features by importance
        sorted_indices = np.argsort(mean_weights)[::-1]
        
        # Concentration metric (Herfindahl index)
        concentration = float((mean_weights ** 2).sum())
        
        results[period_name] = {
            'n_samples': int(mask.sum()),
            'mean_weights': mean_weights,
            'std_weights': std_weights,
            'concentration': concentration,
            'top_features': [(var_names[i], float(mean_weights[i])) 
                            for i in sorted_indices[:10]],
            'bottom_features': [(var_names[i], float(mean_weights[i])) 
                               for i in sorted_indices[-5:]],
        }
    
    return results


def compare_vsn_across_periods(period_stats, var_names):
    """Compare VSN patterns across periods."""
    period_names = list(period_stats.keys())
    
    if len(period_names) < 2:
        return {}
    
    comparisons = {}
    
    for i, period1 in enumerate(period_names[:-1]):
        for period2 in period_names[i+1:]:
            w1 = period_stats[period1]['mean_weights']
            w2 = period_stats[period2]['mean_weights']
            
            # Cosine similarity
            cos_sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-10)
            
            # L2 distance
            l2_dist = np.linalg.norm(w1 - w2)
            
            # Features with biggest changes
            weight_changes = w2 - w1
            top_increases = np.argsort(weight_changes)[-3:][::-1]
            top_decreases = np.argsort(weight_changes)[:3]
            
            comparisons[f"{period1}_vs_{period2}"] = {
                'cosine_similarity': float(cos_sim),
                'l2_distance': float(l2_dist),
                'concentration_change': period_stats[period2]['concentration'] - period_stats[period1]['concentration'],
                'top_increases': [(var_names[i], float(weight_changes[i])) for i in top_increases],
                'top_decreases': [(var_names[i], float(weight_changes[i])) for i in top_decreases],
            }
    
    return comparisons


# ============================================================================
# VIX CORRELATION ANALYSIS
# ============================================================================

def correlate_with_vix(vsn_data, test_df):
    """Correlate VSN weights with VIX levels."""
    # Find VIX column
    vix_col = None
    for col in ['VIX_Close', 'vix_close', 'VIX', 'VIXCLS']:
        if col in test_df.columns:
            vix_col = col
            break
    
    if vix_col is None:
        print("Warning: No VIX column found in data")
        return None
    
    print(f"\nCorrelating VSN weights with {vix_col}...")
    
    # Align VIX data with samples
    n_samples = len(vsn_data['predictions'])
    vix_values = test_df[vix_col].values[-n_samples:]
    
    if len(vix_values) != n_samples:
        print(f"Warning: VIX alignment issue. Expected {n_samples}, got {len(vix_values)}")
        return None
    
    encoder_vsn = vsn_data['encoder_vsn']
    var_names = vsn_data['encoder_var_names']
    
    results = {
        'vix_column': vix_col,
        'n_samples': n_samples,
        'correlations': {},
        'quartile_analysis': {},
    }
    
    # Compute correlations
    for i, var in enumerate(var_names):
        corr, p_value = stats.spearmanr(encoder_vsn[:, i], vix_values)
        results['correlations'][var] = {
            'spearman_r': float(corr),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
        }
    
    # Quartile analysis
    quartiles = pd.qcut(vix_values, q=4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
    
    for i, var in enumerate(var_names):
        quartile_means = {}
        for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
            mask = quartiles == q
            quartile_means[q] = float(encoder_vsn[mask, i].mean())
        results['quartile_analysis'][var] = quartile_means
    
    # Print top correlations
    print("\n  Top correlations with VIX:")
    sorted_corrs = sorted(results['correlations'].items(), 
                         key=lambda x: abs(x[1]['spearman_r']), reverse=True)[:5]
    for var, stats_dict in sorted_corrs:
        sig = "*" if stats_dict['significant'] else ""
        print(f"    {var}: r={stats_dict['spearman_r']:.3f}{sig}")
    
    return results


# ============================================================================
# STALENESS ANALYSIS
# ============================================================================

def analyze_staleness_interaction(vsn_data):
    """Analyze if VSN learns to use staleness features appropriately."""
    var_names = vsn_data['encoder_var_names']
    encoder_vsn = vsn_data['encoder_vsn']
    
    # Find staleness variables (days_since_*, *_staleness, *_is_fresh)
    staleness_vars = [
        v for v in var_names 
        if 'days_since' in v.lower() or 'staleness' in v.lower() or 'is_fresh' in v.lower()
    ]
    
    if not staleness_vars:
        print("\nNo staleness variables found in encoder variables.")
        return None
    
    print(f"\nAnalyzing staleness features: {staleness_vars}")
    
    results = {
        'staleness_vars': staleness_vars,
        'analysis': {},
    }
    
    for var in staleness_vars:
        idx = var_names.index(var)
        weights = encoder_vsn[:, idx]
        
        # Basic statistics
        analysis = {
            'mean_weight': float(weights.mean()),
            'std_weight': float(weights.std()),
            'min_weight': float(weights.min()),
            'max_weight': float(weights.max()),
        }
        
        # Try to find corresponding base variable
        base_var_candidates = [
            var.replace('days_since_', ''),
            var.replace('_days_since', ''),
            var.replace('staleness_', ''),
        ]
        
        for candidate in base_var_candidates:
            # Try exact match and common variants
            for try_var in [candidate, candidate.upper(), candidate.lower(), f"{candidate}_Close"]:
                if try_var in var_names:
                    base_idx = var_names.index(try_var)
                    base_weights = encoder_vsn[:, base_idx]
                    corr, p = stats.spearmanr(weights, base_weights)
                    analysis['base_variable'] = try_var
                    analysis['correlation_with_base'] = float(corr)
                    analysis['correlation_p_value'] = float(p)
                    break
            if 'base_variable' in analysis:
                break
        
        results['analysis'][var] = analysis
        print(f"  {var}: mean={analysis['mean_weight']:.4f}, std={analysis['std_weight']:.4f}")
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_vsn_heatmap(period_stats, var_names, output_path, top_n=15):
    """Heatmap of VSN weights by period and feature."""
    period_names = list(period_stats.keys())
    n_features = len(var_names)
    
    # Only show up to top_n features, but don't exceed actual feature count
    top_n = min(top_n, n_features)
    
    # Get top N features by overall importance
    all_weights = np.zeros(n_features)
    for stats in period_stats.values():
        all_weights += stats['mean_weights']
    top_indices = np.argsort(all_weights)[-top_n:][::-1]
    top_var_names = [var_names[i] for i in top_indices]
    
    # Build matrix
    matrix = np.zeros((len(period_names), top_n))
    for i, period in enumerate(period_names):
        for j, idx in enumerate(top_indices):
            matrix[i, j] = period_stats[period]['mean_weights'][idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.8), max(4, len(period_names) * 0.8)))
    
    sns.heatmap(
        matrix,
        xticklabels=top_var_names,
        yticklabels=period_names,
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Mean VSN Weight'},
        ax=ax
    )
    
    ax.set_title('Variable Selection Weights by Period', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Period', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved VSN heatmap: {output_path}")


def plot_vsn_comparison(period_stats, var_names, output_path, top_n=10):
    """Grouped bar chart comparing feature importance across periods."""
    period_names = list(period_stats.keys())
    n_periods = len(period_names)
    n_features = len(var_names)
    
    # Only show up to top_n features
    top_n = min(top_n, n_features)
    
    # Get top features
    all_weights = np.zeros(n_features)
    for stats in period_stats.values():
        all_weights += stats['mean_weights']
    top_indices = np.argsort(all_weights)[-top_n:][::-1]
    top_var_names = [var_names[i] for i in top_indices]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(max(10, top_n * 1.2), 6))
    
    x = np.arange(len(top_var_names))
    width = 0.8 / max(n_periods, 1)
    
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_periods, 1)))
    
    for i, period in enumerate(period_names):
        values = [period_stats[period]['mean_weights'][idx] for idx in top_indices]
        offset = (i - n_periods/2) * width + width/2
        ax.bar(x + offset, values, width, label=period, color=colors[i])
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Mean VSN Weight', fontsize=12)
    ax.set_title('Feature Selection by Period', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_var_names, rotation=45, ha='right')
    ax.legend(title='Period', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved VSN comparison: {output_path}")


def plot_vsn_concentration(period_stats, output_path):
    """Plot concentration (Herfindahl index) across periods."""
    period_names = list(period_stats.keys())
    concentrations = [period_stats[p]['concentration'] for p in period_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(len(period_names)), concentrations, marker='o', linewidth=2, markersize=8)
    ax.set_xticks(range(len(period_names)))
    ax.set_xticklabels(period_names, rotation=45, ha='right')
    ax.set_ylabel('Concentration (Herfindahl Index)', fontsize=12)
    ax.set_title('VSN Weight Concentration by Period\n(Higher = More Concentrated on Few Features)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved concentration plot: {output_path}")


def plot_vix_quartile_analysis(vix_results, output_path, top_n=8):
    """Plot how VSN weights change across VIX quartiles."""
    if vix_results is None:
        return
    
    var_names = list(vix_results['quartile_analysis'].keys())
    
    # Get features with largest Q1-Q4 differences
    q1_q4_diffs = []
    for var in var_names:
        qa = vix_results['quartile_analysis'][var]
        diff = abs(qa['Q4_high'] - qa['Q1_low'])
        q1_q4_diffs.append((var, diff))
    
    top_vars = [v[0] for v in sorted(q1_q4_diffs, key=lambda x: x[1], reverse=True)[:top_n]]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    quartiles = ['Q1_low', 'Q2', 'Q3', 'Q4_high']
    x = np.arange(len(quartiles))
    width = 0.8 / len(top_vars)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_vars)))
    
    for i, var in enumerate(top_vars):
        values = [vix_results['quartile_analysis'][var][q] for q in quartiles]
        offset = (i - len(top_vars)/2) * width + width/2
        ax.bar(x + offset, values, width, label=var, color=colors[i])
    
    ax.set_xlabel('VIX Quartile', fontsize=12)
    ax.set_ylabel('Mean VSN Weight', fontsize=12)
    ax.set_title('Feature Selection vs VIX Level\n(Features with Largest Q1-Q4 Differences)', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Low VIX', 'Q2', 'Q3', 'High VIX'])
    ax.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved VIX quartile analysis: {output_path}")


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(period_stats, comparisons, vix_results, staleness_results, var_names, output_dir):
    """Save analysis results to JSON."""
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'encoder_variables': var_names,
        'period_statistics': {},
        'period_comparisons': comparisons,
    }
    
    for period, stats in period_stats.items():
        results['period_statistics'][period] = {
            'n_samples': stats['n_samples'],
            'concentration': stats['concentration'],
            'top_features': stats['top_features'],
            'bottom_features': stats['bottom_features'],
            'mean_weights': stats['mean_weights'].tolist(),
            'std_weights': stats['std_weights'].tolist(),
        }
    
    if vix_results:
        results['vix_correlation'] = vix_results
    
    if staleness_results:
        results['staleness_analysis'] = staleness_results
    
    output_path = os.path.join(output_dir, 'vsn_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results: {output_path}")
    return results


def print_summary(period_stats, comparisons, vix_results):
    """Print human-readable summary."""
    print("\n" + "="*70)
    print("VSN ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nPer-Period Feature Importance:")
    print("-" * 70)
    for period, stats in period_stats.items():
        print(f"\n{period} ({stats['n_samples']} samples):")
        print(f"  Concentration: {stats['concentration']:.4f}")
        print(f"  Top 5 features:")
        for feat, weight in stats['top_features'][:5]:
            print(f"    {feat:30s} {weight:.4f}")
    
    if comparisons:
        print("\n" + "-" * 70)
        print("Period Comparisons:")
        print("-" * 70)
        for comp_name, metrics in comparisons.items():
            print(f"\n{comp_name}:")
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
            print(f"  L2 distance: {metrics['l2_distance']:.4f}")
            if metrics['top_increases']:
                print(f"  Biggest increases: {', '.join([f'{v[0]}({v[1]:+.3f})' for v in metrics['top_increases']])}")
            if metrics['top_decreases']:
                print(f"  Biggest decreases: {', '.join([f'{v[0]}({v[1]:+.3f})' for v in metrics['top_decreases']])}")
    
    if vix_results:
        print("\n" + "-" * 70)
        print("VIX Correlation (top 5 by magnitude):")
        print("-" * 70)
        sorted_corrs = sorted(vix_results['correlations'].items(),
                             key=lambda x: abs(x[1]['spearman_r']), reverse=True)[:5]
        for var, stats in sorted_corrs:
            sig = "*" if stats['significant'] else ""
            print(f"  {var:30s} r={stats['spearman_r']:.3f}{sig}")
    
    print("\n" + "="*70)


# ============================================================================
# SINGLE EXPERIMENT ANALYSIS
# ============================================================================

def analyze_single_experiment(experiment_name, args, output_dir=None):
    """
    Run VSN analysis on a single experiment.
    
    Parameters
    ----------
    experiment_name : str
        Experiment path relative to experiments/ (e.g., '00_baseline_exploration/sweep2_h16_drop_0.25')
    args : argparse.Namespace
        Command line arguments
    output_dir : str, optional
        Override output directory. If None, uses experiments/{experiment_name}/vsn_analysis/
    
    Returns
    -------
    bool
        True if analysis succeeded, False otherwise
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join('experiments', experiment_name, 'vsn_analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging for this experiment
    logger = setup_logging(output_dir)
    
    print(f"Experiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load configuration
        config = load_config(experiment_name)
        
        # Find checkpoint
        checkpoint_path = find_checkpoint(experiment_name, args.checkpoint)
        
        # Load data
        print("\nLoading data...")
        train_df, test_df = load_test_data(config, args.test_split)
        print(f"  Train samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        # Prepare dataset
        print("\nPreparing test dataset...")
        test_dataset, test_dates, test_start_idx = prepare_test_dataset(train_df, test_df, config)
        
        # Load model
        print("\nLoading model...")
        model = load_model(checkpoint_path, config)
        
        # Extract VSN weights
        vsn_data = extract_vsn_weights(model, test_dataset, args.batch_size)
        
        # Check extraction success
        if vsn_data['encoder_vsn'].size == 0:
            print("\n" + "!"*70)
            print("ERROR: Failed to extract VSN weights")
            print("!"*70)
            logger.close()
            return False
        
        # Get dates - test_dates is a DatetimeIndex from the original test_df
        n_samples = len(vsn_data['predictions'])
        # Account for encoder length: first prediction corresponds to encoder_length timesteps into test set
        max_encoder_length = config['architecture']['max_encoder_length']
        dates = test_dates[max_encoder_length:max_encoder_length + n_samples].values
        
        # Handle case where we have fewer dates than samples
        if len(dates) < n_samples:
            print(f"Warning: Date alignment issue. Have {len(dates)} dates for {n_samples} samples")
            dates = test_dates[-n_samples:].values
        
        print(f"\nAnalysis period: {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} to {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")
        
        # Create periods
        print("\nCreating temporal periods:")
        periods = create_periods(dates, args.periods, args.period_labels)
        
        # Analyze by period
        print("\nAnalyzing VSN weights by period...")
        var_names = vsn_data['encoder_var_names']
        period_stats = analyze_vsn_by_period(vsn_data, periods)
        
        # Compare periods
        print("\nComparing VSN patterns across periods...")
        comparisons = compare_vsn_across_periods(period_stats, var_names)
        
        # Optional: VIX correlation
        vix_results = None
        if args.correlate_vix:
            vix_results = correlate_with_vix(vsn_data, test_df)
        
        # Optional: Staleness analysis
        staleness_results = None
        if args.analyze_staleness:
            staleness_results = analyze_staleness_interaction(vsn_data)
        
        # Create visualizations
        print("\nCreating visualizations...")
        plot_vsn_heatmap(
            period_stats, var_names,
            os.path.join(output_dir, 'vsn_heatmap.png'),
            top_n=args.top_n_features
        )
        
        plot_vsn_comparison(
            period_stats, var_names,
            os.path.join(output_dir, 'vsn_comparison.png'),
            top_n=min(10, args.top_n_features)
        )
        
        plot_vsn_concentration(
            period_stats,
            os.path.join(output_dir, 'vsn_concentration.png')
        )
        
        if vix_results:
            plot_vix_quartile_analysis(
                vix_results,
                os.path.join(output_dir, 'vsn_vix_quartiles.png')
            )
        
        # Save results
        print("\nSaving results...")
        save_results(period_stats, comparisons, vix_results, staleness_results, var_names, output_dir)
        
        # Print summary
        print_summary(period_stats, comparisons, vix_results)
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"  - vsn_analysis_results.json")
        print(f"  - vsn_heatmap.png")
        print(f"  - vsn_comparison.png")
        print(f"  - vsn_concentration.png")
        if vix_results:
            print(f"  - vsn_vix_quartiles.png")
        
        logger.close()
        return True
        
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"ERROR analyzing {experiment_name}: {e}")
        print(f"{'!'*70}\n")
        import traceback
        traceback.print_exc()
        logger.close()
        return False


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def find_experiments_in_phase(phase_name):
    """
    Find all valid experiments in a phase directory.
    
    Parameters
    ----------
    phase_name : str
        Phase directory name (e.g., '00_baseline_exploration')
    
    Returns
    -------
    list of str
        Experiment names relative to experiments/ (e.g., '00_baseline_exploration/sweep2_h16_drop_0.25')
    """
    phase_dir = os.path.join('experiments', phase_name)
    
    if not os.path.exists(phase_dir):
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")
    
    experiments = []
    
    for item in sorted(os.listdir(phase_dir)):
        exp_path = os.path.join(phase_dir, item)
        
        # Check if it's a valid experiment directory (has config.json and checkpoints/)
        config_path = os.path.join(exp_path, 'config.json')
        checkpoints_dir = os.path.join(exp_path, 'checkpoints')
        
        if os.path.isdir(exp_path) and os.path.exists(config_path):
            # Check for at least one checkpoint
            if os.path.exists(checkpoints_dir) and any(f.endswith('.ckpt') for f in os.listdir(checkpoints_dir)):
                experiments.append(f"{phase_name}/{item}")
            else:
                print(f"  Skipping {item}: no checkpoints found")
        elif os.path.isdir(exp_path):
            # Check if it's a nested structure (unlikely but handle it)
            pass
    
    return experiments


def analyze_phase(phase_name, args):
    """
    Run VSN analysis on all experiments in a phase.
    
    Parameters
    ----------
    phase_name : str
        Phase directory name
    args : argparse.Namespace
        Command line arguments
    """
    print("="*70)
    print(f"BATCH VSN ANALYSIS: {phase_name}")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find all experiments
    print(f"\nScanning for experiments in experiments/{phase_name}/...")
    experiments = find_experiments_in_phase(phase_name)
    
    if not experiments:
        print(f"No valid experiments found in {phase_name}")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp}")
    
    # Process each experiment
    results_summary = {
        'succeeded': [],
        'failed': [],
        'skipped': [],
    }
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(experiments)}] Processing: {experiment}")
        print("="*70)
        
        # Check if should skip
        output_dir = os.path.join('experiments', experiment, 'vsn_analysis')
        if args.skip_existing and os.path.exists(output_dir):
            results_json = os.path.join(output_dir, 'vsn_analysis_results.json')
            if os.path.exists(results_json):
                print(f"Skipping (already analyzed): {experiment}")
                results_summary['skipped'].append(experiment)
                continue
        
        # Run analysis
        try:
            success = analyze_single_experiment(experiment, args)
            if success:
                results_summary['succeeded'].append(experiment)
            else:
                results_summary['failed'].append(experiment)
                if not args.continue_on_error:
                    print("\nStopping due to error (use --continue-on-error to override)")
                    break
        except Exception as e:
            print(f"Error processing {experiment}: {e}")
            results_summary['failed'].append(experiment)
            if not args.continue_on_error:
                print("\nStopping due to error (use --continue-on-error to override)")
                break
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(experiments)}")
    print(f"  Succeeded: {len(results_summary['succeeded'])}")
    print(f"  Failed: {len(results_summary['failed'])}")
    print(f"  Skipped: {len(results_summary['skipped'])}")
    
    if results_summary['failed']:
        print(f"\nFailed experiments:")
        for exp in results_summary['failed']:
            print(f"  - {exp}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    if args.phase:
        # Batch mode: process entire phase
        analyze_phase(args.phase, args)
    else:
        # Single experiment mode
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join('experiments', args.experiment, 'vsn_analysis')
        
        analyze_single_experiment(args.experiment, args, output_dir)


if __name__ == "__main__":
    main()