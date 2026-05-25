"""
Train Temporal Fusion Transformer on S&P 500 return prediction.

Includes full experiment tracking, checkpointing, and reproducibility controls.

Usage:
    # Basic usage with defaults
    python train_tft.py --experiment-name tft_baseline
    
    # Custom hyperparameters
    python train_tft.py --experiment-name tft_large \\
        --hidden-size 64 --attention-heads 4 --max-epochs 100
    
    # Different feature set
    python train_tft.py --experiment-name tft_macro \\
        --feature-set macro_heavy --frequency monthly
"""

import os
import platform
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import json
import argparse
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Custom components
from src.custom_losses import create_loss_from_args
from src.regime_output import replace_output_layer

# in same /train dir
from collapse_monitor import CollapseMonitor
from callbacks import EpochSummaryCallback

# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Temporal Fusion Transformer for S&P 500 return prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment configuration
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Name for this experiment (creates output directory)')
    parser.add_argument('--feature-set', type=str, default='core_proposal',
                        choices=['core_proposal', 'core_plus_credit', 'macro_heavy', 
                                 'market_only', 'kitchen_sink', 'core_dynamics'],
                        help='Feature set configuration')
    parser.add_argument('--frequency', type=str, default='daily',
                        choices=['daily', 'weekly', 'monthly'],
                        help='Data frequency')
    parser.add_argument('--alignment', type=str, default='vintage',
                        choices=['fixed', 'vintage'],
                        help='Release date alignment mode (vintage uses actual release dates)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # TFT architecture
    parser.add_argument('--max-encoder-length', type=int, default=20,
                        help='Lookback window length')
    parser.add_argument('--hidden-size', type=int, default=16,
                        help='Hidden layer size')
    parser.add_argument('--attention-heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.10,
                        help='Dropout rate')
    parser.add_argument('--hidden-continuous-size', type=int, default=16,
                        help='Hidden size for continuous features')
    
    # Training
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--gradient-clip', type=float, default=0.1,
                        help='Gradient clipping value')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Early stopping patience')

    # Include staleness features
    parser.add_argument('--staleness', action='store_true',
                        help='Enable staleness features (experimental)')    
    
    # Paths
    parser.add_argument('--splits-dir', type=str, default='data/splits',
                        help='Directory containing data splits')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Base directory for experiment outputs')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting existing experiment directory')

    # Collapse monitor
    parser.add_argument('--monitor-every-n-epochs', type=int, default=1,
                   help='How often to run collapse monitoring')

    # Checkpointing
    parser.add_argument('--checkpoint-every-n-epochs', type=int, default=1,
                   help='How often to save checkpoints (1=every epoch including epoch 0)')

    # Prediction diversity regularization (anti-collapse penalties)
    parser.add_argument('--dist-loss-mean-weight', type=float, default=0.0,
                    help='DEPRECATED: Anti-drift penalty weight (ignored)')
    parser.add_argument('--dist-loss-std-weight', type=float, default=0.0,
                    help='Variance-based anti-collapse penalty weight (0=disabled, typical: 0.1-0.2)')
    parser.add_argument('--collapse-threshold', type=float, default=0.005,
                    help='Minimum variance threshold for collapse penalty (default: 0.005 = 0.5%%)')
    parser.add_argument('--directional-weight', type=float, default=0.0,
                    help='Directional diversity penalty weight (0=disabled, typical: 0.1-0.2)')
    parser.add_argument('--directional-threshold', type=float, default=0.90,
                    help='Maximum directional bias threshold (default: 0.90 = 90%%)')
    parser.add_argument('--directional-window', type=int, default=30,
                    help='Rolling window size for directional bias calculation (default: 30)')
    parser.add_argument('--temporal-consistency-weight', type=float, default=0.0,
                    help='Temporal smoothness penalty weight (0=disabled, typical: 0.05-0.1)')
    
    # Magnitude-aware loss weighting (encourage larger predictions)
    parser.add_argument('--magnitude-weight-alpha', type=float, default=0.0,
                    help='Linear magnitude weighting coefficient (0=disabled, typical: 0.5-2.0). '
                         'Mutually exclusive with extreme-move-weight.')
    parser.add_argument('--extreme-move-weight', type=float, default=1.0,
                    help='Weight multiplier for extreme moves (1.0=disabled, typical: 2.0-5.0). '
                         'Mutually exclusive with magnitude-weight-alpha.')
    parser.add_argument('--extreme-move-percentile', type=int, default=95,
                    help='Percentile threshold for extreme moves (default: 95 = top 5%%)')

    # Regime-conditional output (Phase 5 modifications)
    parser.add_argument('--regime-output', action='store_true',
                        help='Enable regime-conditional output layer (MoE architecture)')
    parser.add_argument('--num-regimes', type=int, default=2,
                        help='Number of expert heads for regime-conditional output (default: 2)')
    parser.add_argument('--routing-mode', type=str, default='learned',
                        choices=['learned', 'disabled'],
                        help='Routing strategy: learned=MoE with learned router, disabled=single expert baseline')
    parser.add_argument('--routing-strategy', type=str, default='learned',
                    choices=['learned', 'vix_threshold'],
                    help='Routing strategy: learned (linear router) or vix_threshold (deterministic VIX)')
    parser.add_argument('--load-balance-weight', type=float, default=0.5,
                    help='Weight for load balancing auxiliary loss (prevents winner-takes-all routing)')
    parser.add_argument('--vix-threshold', type=float, default=25.0,
                    help='VIX threshold for 2-regime deterministic routing (only used with --routing-strategy vix_threshold)')
    parser.add_argument('--vix-threshold-low', type=float, default=None,
                    help='Lower VIX threshold for 3-regime routing (required when --num-regimes 3 with vix_threshold strategy)')
    parser.add_argument('--vix-threshold-high', type=float, default=None,
                    help='Upper VIX threshold for 3-regime routing (required when --num-regimes 3 with vix_threshold strategy)')
    parser.add_argument('--expert-hidden-size', type=int, default=0,
                    help='Expert hidden layer size. 0=linear experts (default), >0=MLP with that hidden size')
    parser.add_argument('--hard-routing-train', action='store_true',
                    help='Use hard routing during training (each sample only trains its assigned expert). '
                         'Requires --routing-strategy vix_threshold. Validation/test uses soft routing.')

    # Frozen backbone / transfer learning
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze all TFT parameters except output layer (for diagnostic/transfer learning)')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint to load before training (for transfer learning)')

    # Classification head (diagnostic auxiliary task)
    parser.add_argument('--classification', action='store_true',
                        help='Enable classification head alongside regression')
    parser.add_argument('--classification-mode', type=str, default='direction',
                        choices=['direction', 'direction_3class', 'regime_volatility', 'regime_volatility_3class'],
                        help='Classification target: direction=up/down, direction_3class=down/neutral/up, '
                             'regime_volatility=low/high VIX, regime_volatility_3class=low/med/high VIX')
    parser.add_argument('--classification-weight', type=float, default=1.0,
                        help='Weight for classification loss (beta)')
    parser.add_argument('--regression-weight', type=float, default=1.0,
                        help='Weight for regression loss (alpha). Set to 0 for pure classification.')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classification classes (2 for binary, 3 for 3-class modes)')
    parser.add_argument('--classification-thresholds', type=float, nargs='+', default=None,
                        help='Thresholds for multi-class classification (e.g., -0.01 0.01 for direction_3class)')

    # Regime attention args
    parser.add_argument('--regime-attention', action='store_true',
                        help='Enable regime-aware attention gating')
    parser.add_argument('--regime-attention-vix-threshold', type=float, default=25.0,
                        help='VIX threshold for regime switching (default: 25.0)')
    parser.add_argument('--regime-attention-grad-scale', type=float, default=100.0,
                        help='Gradient scaling factor for regime gates (default: 100.0)')
    parser.add_argument('--regime-gate-init', type=str, default='neutral',
                        choices=['neutral', 'separated'],
                        help='Gate initialization: neutral (0.5) or separated (0.38/0.62)')
    parser.add_argument('--gate-separation-weight', type=float, default=0.0,
                    help='Weight for regime gate separation reward (0.0 = disabled)')
    
    return parser.parse_args()


# ============================================================================
# FEATURE DEFINITIONS (based on frequency and feature set)
# ============================================================================

def get_features(feature_set, frequency, include_staleness=True):
    """
    Get feature lists based on configuration.
    Must match feature_configs.py exactly.
    
    Parameters:
    -----------
    feature_set : str
        Name of feature set from FEATURE_SETS
    frequency : str
        'daily' or 'monthly'
    include_staleness : bool
        If True, add staleness features for low-frequency variables
        
    Returns:
    --------
    dict with keys:
        'high_freq': list of high-frequency feature names
        'low_freq': list of low-frequency feature names
        'all': list of all feature names (including staleness if enabled)
        'staleness': list of staleness feature names (empty if not enabled)
    """
    from src.feature_configs import FEATURE_SETS, get_staleness_features
    
    config = FEATURE_SETS[feature_set]
    if config['features'] == 'all':
        # Kitchen sink - use everything available
        high_freq = ["VIX", "Treasury_10Y", "Yield_Spread"]
        low_freq = ["Inflation_YoY", "Unemployment", "Fed_Rate", 
                    "Consumer_Sentiment", "Industrial_Production"]
        all_features = high_freq + low_freq + ["SP500_Volatility"]
    else:
        # Use specified features from config
        all_features = [f for f in config['features'] if f != 'SP500_Returns']
        
        # Categorize as high/low frequency
        high_freq = [f for f in all_features if f in 
                     ["VIX", "Treasury_10Y", "Yield_Spread", "Credit_HY", "Credit_IG"]]
        low_freq = [f for f in all_features if f in 
                    ["Inflation_YoY", "Unemployment", "Fed_Rate", 
                     "Consumer_Sentiment", "Industrial_Production"]]
    
    # Add staleness features if requested
    staleness_info = get_staleness_features(all_features)
    staleness_features = staleness_info['staleness_features'] if include_staleness else []
    
    return {
        'high_freq': high_freq,
        'low_freq': low_freq,
        'all': all_features + staleness_features,
        'staleness': staleness_features,
    }


# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================

def set_all_seeds(seed):
    """Set seeds for complete reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# DATA LOADING
# ============================================================================

def load_splits(splits_dir, feature_set, frequency, alignment):
    """
    Load pre-created train/val/test splits.
    
    Expects new directory structure:
    data/splits/{alignment}/{feature_set}_{frequency}_{alignment}_{split}.csv
    
    Example: data/splits/vintage/core_proposal_daily_vintage_train.csv
    """
    # Construct paths with new naming convention
    splits_path = os.path.join(splits_dir, alignment)
    split_prefix = f"{feature_set}_{frequency}_{alignment}"
    
    train_path = os.path.join(splits_path, f"{split_prefix}_train.csv")
    val_path = os.path.join(splits_path, f"{split_prefix}_val.csv")
    test_path = os.path.join(splits_path, f"{split_prefix}_test.csv")
    
    # Check if files exist and provide helpful error
    for path, split_name in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {split_name} split at: {path}\n"
                f"Expected structure: {splits_dir}/{alignment}/{feature_set}_{frequency}_{alignment}_{split_name}.csv"
            )
    
    train = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    val = pd.read_csv(val_path, index_col='Date', parse_dates=True)
    test = pd.read_csv(test_path, index_col='Date', parse_dates=True)
    
    return train, val, test

def prepare_tft_data(train_df, val_df, args, features, add_staleness=True):
    """Prepare data in TimeSeriesDataSet format for TFT."""
    
    # add staleness
    if add_staleness and len(features['staleness']) > 0:
        print("\nAdding staleness features to data...")
        from src.data_utils import add_staleness_features
        
        # Use vintage=True if alignment is vintage, False if fixed
        use_vintage = (args.alignment == 'vintage')
        train_df = add_staleness_features(train_df, use_vintage=use_vintage, verbose=True)
        val_df = add_staleness_features(val_df, use_vintage=use_vintage, verbose=True)
    
    # DROP SOURCE COLUMNS (used for staleness detection only, not model features)
    source_cols_to_drop = []
    from src.feature_configs import FEATURE_METADATA
    for feature in train_df.columns:
        if feature in FEATURE_METADATA:
            continue  # Keep actual features
        # Check if this is a source column for another feature
        for feat, meta in FEATURE_METADATA.items():
            if meta.get('source_column') == feature:
                source_cols_to_drop.append(feature)
                break
    
    if source_cols_to_drop:
        print(f"\nDropping source columns (used for staleness only): {source_cols_to_drop}")
        train_df = train_df.drop(columns=source_cols_to_drop)
        val_df = val_df.drop(columns=source_cols_to_drop)
    
    print(f"\nFinal features for TFT: {list(train_df.columns)}")
    
    # Reset index and add required columns
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    
    # DEBUG: Print raw feature statistics
    print("\n" + "="*70)
    print("FEATURE STATISTICS - BEFORE NORMALIZATION")
    print("="*70)
    feature_cols = [c for c in train_df.columns if c in features['all']]
    for col in feature_cols:
        data = train_df[col]
        print(f"{col:30s}  mean={data.mean():8.4f}  std={data.std():8.4f}  "
              f"min={data.min():8.4f}  max={data.max():8.4f}")
    
    # NORMALIZE: Pre-normalize staleness features to [0, 1] range
    staleness_cols = [c for c in train_df.columns if 'days_since' in c]
    if staleness_cols:
        print("\n" + "="*70)
        print("NORMALIZING STALENESS FEATURES (log transform)")
        print("="*70)
        
        # Calculate max from actual data to handle both vintage and fixed alignments
        max_train = train_df[staleness_cols].max().max()
        max_val = val_df[staleness_cols].max().max()
        MAX_DAYS_STALE = np.ceil(max(max_train, max_val)) + 5  # Add 5-day buffer
        
        print(f"Max staleness observed: train={max_train:.0f}, val={max_val:.0f}")
        print(f"Using normalization max: {MAX_DAYS_STALE:.0f} days")
        print()
        
        for col in staleness_cols:
            # Log transform: log(1 + days) / log(1 + max_days)
            # Maps [0, max_days] to [0, 1] with compression of large values
            train_df[col] = np.log1p(train_df[col]) / np.log1p(MAX_DAYS_STALE)
            val_df[col] = np.log1p(val_df[col]) / np.log1p(MAX_DAYS_STALE)
        
        # Print after normalization
        print("\nSTALENESS FEATURES - AFTER MANUAL NORMALIZATION")
        for col in staleness_cols:
            data = train_df[col]
            print(f"{col:30s}  mean={data.mean():8.4f}  std={data.std():8.4f}  "
                  f"min={data.min():8.4f}  max={data.max():8.4f}")
    
    # Add time index (sequential integers)
    train_df['time_idx'] = range(len(train_df))
    val_df['time_idx'] = range(len(train_df), len(train_df) + len(val_df))
    
    # Add group identifier (single time series)
    train_df['group'] = 'SP500'
    val_df['group'] = 'SP500'
    
    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="SP500_Returns",
        group_ids=["group"],
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=1,  # Always predict 1 step ahead
        time_varying_known_reals=[],  # No known future inputs
        time_varying_unknown_reals=features['all'],
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    
    # DEBUG: Check what GroupNormalizer did (sample a batch)
    print("\n" + "="*70)
    print("FEATURES AFTER GROUPNORMALIZER (first batch, last timestep)")
    print("="*70)
    dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    x, y = next(iter(dataloader))
    
    # Extract feature values from batch
    encoder_cont = x['encoder_cont']  # Shape: [batch, time, features]
    # Get first sample, last timestep
    sample = encoder_cont[0, -1, :].cpu().numpy()
    
    for i, col in enumerate(features['all']):
        print(f"{col:30s}  normalized_value={sample[i]:8.4f}")
    print("="*70 + "\n")
    
    # Create validation dataset (uses training stats)
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=False,
        stop_randomization=True
    )

    # save VIX for regime o/p
    raw_vix_train = train_df['VIX'].values
    raw_vix_val = val_df['VIX'].values
    
    return training, validation, raw_vix_train, raw_vix_val

# ============================================================================
# MODEL SETUP
# ============================================================================

def create_model(training_dataset, args, raw_vix_train, raw_vix_val):
    """Initialize TFT model with EnhancedQuantileLoss."""
    
    # Create loss function with configured penalties
    loss_fn = create_loss_from_args(args)
    
    # Log which penalties are active
    active_penalties = []
    if args.dist_loss_std_weight > 0:
        threshold_pct = args.collapse_threshold * 100
        active_penalties.append(f"variance_collapse={args.dist_loss_std_weight} (threshold={threshold_pct:.1f}%)")
    if args.directional_weight > 0:
        dir_threshold_pct = args.directional_threshold * 100
        active_penalties.append(f"directional_diversity={args.directional_weight} (threshold={dir_threshold_pct:.0f}%)")
    if args.temporal_consistency_weight > 0:
        active_penalties.append(f"temporal_consistency={args.temporal_consistency_weight}")
    
    # Note: dist_loss_mean_weight is ignored (fixed distribution targets don't work with regime variation)
    if args.dist_loss_mean_weight > 0:
        print("\nNOTE: --dist-loss-mean-weight is ignored (use anti-collapse via --dist-loss-std-weight instead)")
    
    if active_penalties:
        print(f"\nActive loss penalties: {', '.join(active_penalties)}")
    else:
        print("\nUsing standard QuantileLoss (no penalties)")

    # Common TFT kwargs
    tft_kwargs = dict(
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_heads,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        output_size=7,
        loss=loss_fn,
        log_interval=-1 if args.classification else 10, # disable interpretation logging for classification
        reduce_on_plateau_patience=4,
    )
    
    # Create model - use ClassificationTFT if classification enabled
    if args.classification:
        from src.classification_tft import ClassificationTFT
        tft = ClassificationTFT.from_dataset(
            training_dataset,
            classification=True,
            classification_mode=args.classification_mode,
            classification_weight=args.classification_weight,
            regression_weight=args.regression_weight,
            num_classes=args.num_classes,
            classification_thresholds=args.classification_thresholds,
            **tft_kwargs
        )
        print(f"\n[CLASSIFICATION] Enabled: mode={args.classification_mode}, "
              f"classes={args.num_classes}, weights=(reg={args.regression_weight}, clf={args.classification_weight})")
    else:
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            **tft_kwargs
        )
    
    # REGIME CONDITIONED O/P
    if args.regime_output:
        # Validate 3-regime configuration
        if args.num_regimes == 3 and args.routing_strategy == 'vix_threshold':
            if args.vix_threshold_low is None or args.vix_threshold_high is None:
                raise ValueError(
                    "3-regime VIX routing requires both --vix-threshold-low and "
                    "--vix-threshold-high to be specified"
                )
        
        # Validate hard routing configuration
        if args.hard_routing_train and args.routing_strategy != 'vix_threshold':
            raise ValueError(
                "--hard-routing-train requires --routing-strategy vix_threshold"
            )
        
        from src.regime_output import replace_output_layer
        tft = replace_output_layer(
            model=tft,
            num_regimes=args.num_regimes,
            routing_mode=args.routing_mode,
            routing_strategy=args.routing_strategy,  
            vix_threshold=args.vix_threshold,
            vix_threshold_low=args.vix_threshold_low,
            vix_threshold_high=args.vix_threshold_high,
            load_balance_weight=args.load_balance_weight,
            expert_hidden_size=args.expert_hidden_size,
            hard_routing_train=args.hard_routing_train
        )
        routing_type = "HARD" if args.hard_routing_train else "soft"
        print(f"\n[REGIME OUTPUT] Enabled with {args.num_regimes} regimes, mode={args.routing_mode}, strategy={args.routing_strategy}, training={routing_type}")
        
        # Modify monkey-patch to extract and pass VIX
        original_training_step = tft.training_step
        
        def training_step_with_lb_and_vix(self, batch, batch_idx):
            # VIX extraction is handled by forward_with_vix wrapper
            # This wrapper just adds load balancing loss
            
            # Call original training_step
            loss_dict = original_training_step(batch, batch_idx)
            
            # Add load balancing loss
            if hasattr(self.output_layer, '_cached_lb_loss'):
                lb = self.output_layer._cached_lb_loss
                if lb is not None:
                    loss_dict['loss'] = loss_dict['loss'] + lb
            
            return loss_dict
        
        import types
        tft.training_step = types.MethodType(training_step_with_lb_and_vix, tft)
        
        # a lso need to patch TFT's forward to pass VIX to output layer
        original_forward = tft.forward
        
        def forward_with_vix(self, x):
            # Extract VIX if using VIX routing
            vix_values = None
            if hasattr(self.output_layer, 'routing_strategy'):
                if self.output_layer.routing_strategy == 'vix_threshold':
                    if 'decoder_time_idx' in x:
                        time_idx = x['decoder_time_idx'][:, 0]
                        if self.training:
                            vix_values = self._raw_vix_train[time_idx.cpu()].to(x['encoder_cont'].device)
                        else:
                            # Validation uses offset indices - subtract train length
                            offset_idx = time_idx.cpu() - len(self._raw_vix_train)
                            vix_values = self._raw_vix_val[offset_idx].to(x['encoder_cont'].device)  
            
            # store VIX on output_layer before forward
            if vix_values is not None:
                self.output_layer._vix_for_forward = vix_values
            
            result = original_forward(x)
            
            # Clear cache
            if hasattr(self.output_layer, '_vix_for_forward'):
                self.output_layer._vix_for_forward = None
            
            return result
        
        tft.forward = types.MethodType(forward_with_vix, tft)

    else:
        print(f"\n[REGIME OUTPUT] Disabled (using baseline output layer)")
   
    # REGIME ATTENTION
    if args.regime_attention:
        from src.regime_attention import replace_attention_module
        from train.regime_attention_training import patch_forward_for_regime
        
        tft = replace_attention_module(
            tft,
            regime_mode='vix_threshold',
            vix_threshold=args.regime_attention_vix_threshold,
            num_regimes=2,
            gate_grad_scale=args.regime_attention_grad_scale,
            gate_init=args.regime_gate_init
        )
        tft = patch_forward_for_regime(
            tft, 
            vix_feature_name='VIX',
            raw_vix_train=raw_vix_train,
            raw_vix_val=raw_vix_val
        )
        
        print(f"\n[REGIME ATTENTION] Enabled: vix_threshold={args.regime_attention_vix_threshold}")
        
        # Connect loss to model for gate separation penalty
        if args.gate_separation_weight > 0 and hasattr(tft, 'loss') and hasattr(tft.loss, 'set_model'):
            tft.loss.set_model(tft)
            print(f"[GATE SEPARATION] Enabled: weight={args.gate_separation_weight}")
    
    # DEBUG: Show what the model actually received
    print("\n" + "="*80)
    print("BASELINE MODEL FEATURE CONFIGURATION (after from_dataset)")
    print("="*80)
    print(f"tft.hparams.time_varying_reals_encoder: {tft.hparams.time_varying_reals_encoder}")
    print(f"  Count: {len(tft.hparams.time_varying_reals_encoder)}")
    print(f"\ntft.hparams.time_varying_reals_decoder: {tft.hparams.time_varying_reals_decoder}")
    print(f"  Count: {len(tft.hparams.time_varying_reals_decoder)}")
    print(f"\ntft.hparams.x_reals (all reals passed to model): {tft.hparams.x_reals}")
    print(f"  Count: {len(tft.hparams.x_reals)}")
    
    # Show encoder VSN input size
    if hasattr(tft, 'encoder_variable_selection'):
        enc_vsn = tft.encoder_variable_selection
        print(f"\nEncoder VSN configuration:")
        print(f"  num_inputs: {enc_vsn.num_inputs}")
        print(f"  input_sizes: {enc_vsn.input_sizes}")
        if hasattr(enc_vsn, 'flattened_grn') and hasattr(enc_vsn.flattened_grn, 'fc1'):
            fc1_in = enc_vsn.flattened_grn.fc1.in_features
            fc1_out = enc_vsn.flattened_grn.fc1.out_features
            print(f"  flattened_grn.fc1: in_features={fc1_in}, out_features={fc1_out}")
            print(f"  -> Params in fc1: {fc1_in * fc1_out} (should be 5*16*5=400 if 5 features)")
    print("="*80 + "\n")
    
    return tft

def freeze_backbone(model, verbose=True):
    """
    Freeze all TFT parameters except the output layer.
    
    Used for:
    1. Diagnostic: Test if hidden state contains usable regime signal
    2. Transfer learning: Train new output head on pretrained backbone
    
    Parameters
    ----------
    model : TemporalFusionTransformer
        Model to freeze
    verbose : bool
        Print parameter counts
    
    Returns
    -------
    model : TemporalFusionTransformer
        Model with frozen backbone (modified in-place)
    """
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if 'output_layer' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    if verbose:
        total = frozen_count + trainable_count
        print(f"\n[FREEZE BACKBONE] Parameter status:")
        print(f"  Frozen: {frozen_count:,} ({100*frozen_count/total:.1f}%)")
        print(f"  Trainable: {trainable_count:,} ({100*trainable_count/total:.1f}%)")
        print(f"  Total: {total:,}")
    
    return model

# ============================================================================
# EXPERIMENT TRACKING SETUP
# ============================================================================

def save_config(args, features, output_dir):
    """Save all hyperparameters and configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'experiment_name': args.experiment_name,
        'created_at': datetime.now().isoformat(),
        'random_seed': args.seed,
        'feature_set': args.feature_set,
        'frequency': args.frequency,
        'alignment': args.alignment,
        'data': {
            'splits_dir': args.splits_dir,
            'release_date_mode': args.alignment,  # For compatibility with evaluate_tft.py
            'split_prefix': f"{args.feature_set}_{args.frequency}_{args.alignment}",
            'train_size': len(pd.read_csv(os.path.join(
                args.splits_dir, 
                args.alignment,
                f"{args.feature_set}_{args.frequency}_{args.alignment}_train.csv"
            ))),
            'val_size': len(pd.read_csv(os.path.join(
                args.splits_dir,
                args.alignment, 
                f"{args.feature_set}_{args.frequency}_{args.alignment}_val.csv"
            ))),
        },
        'monitoring': {
            'monitor_every_n_epochs': args.monitor_every_n_epochs,
        },
        'architecture': {
            'max_encoder_length': args.max_encoder_length,
            'max_prediction_length': 1,
            'hidden_size': args.hidden_size,
            'attention_head_size': args.attention_heads,
            'dropout': args.dropout,
            'hidden_continuous_size': args.hidden_continuous_size,
        },
        'training': {
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'learning_rate': args.learning_rate,
            'gradient_clip_val': args.gradient_clip,
            'early_stop_patience': args.early_stop_patience,
        },
        'loss': {
            'type': 'EnhancedQuantileLoss',
            'dist_loss_mean_weight': args.dist_loss_mean_weight,
            'dist_loss_std_weight': args.dist_loss_std_weight,
            'collapse_threshold': getattr(args, 'collapse_threshold', 0.005),
            'directional_weight': getattr(args, 'directional_weight', 0.0),
            'directional_threshold': getattr(args, 'directional_threshold', 0.90),
            'temporal_consistency_weight': args.temporal_consistency_weight,
            'magnitude_weight_alpha': getattr(args, 'magnitude_weight_alpha', 0.0),
            'extreme_move_weight': getattr(args, 'extreme_move_weight', 1.0),
            'extreme_move_percentile': getattr(args, 'extreme_move_percentile', 95),
        },
        'regime_output': {
            'enabled': args.regime_output,
            'num_regimes': args.num_regimes if args.regime_output else None,
            'routing_mode': args.routing_mode if args.regime_output else None,
            'routing_strategy': getattr(args, 'routing_strategy', 'learned') if args.regime_output else None,
            'vix_threshold': args.vix_threshold if args.regime_output else None,
            'vix_threshold_low': getattr(args, 'vix_threshold_low', None) if args.regime_output else None,
            'vix_threshold_high': getattr(args, 'vix_threshold_high', None) if args.regime_output else None,
            'load_balance_weight': args.load_balance_weight if args.regime_output else None,
            'expert_hidden_size': getattr(args, 'expert_hidden_size', 0) if args.regime_output else None,
            'hard_routing_train': getattr(args, 'hard_routing_train', False) if args.regime_output else None,
        },
        'regime_attention': {
            'enabled': args.regime_attention,
            'vix_threshold': args.regime_attention_vix_threshold,
            'num_regimes': 2,
            'gate_grad_scale': args.regime_attention_grad_scale if args.regime_attention else None, 
        },
        'transfer_learning': {
            'freeze_backbone': getattr(args, 'freeze_backbone', False),
            'load_checkpoint': getattr(args, 'load_checkpoint', None),
        },
        'classification': {
            'enabled': getattr(args, 'classification', False),
            'mode': getattr(args, 'classification_mode', 'direction'),
            'weight': getattr(args, 'classification_weight', 1.0),
            'regression_weight': getattr(args, 'regression_weight', 1.0),
            'num_classes': getattr(args, 'num_classes', 2),
            'thresholds': getattr(args, 'classification_thresholds', None),
        },
        'features': features,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    return config

# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Main training loop."""
    # Parse arguments
    args = parse_args()
    
    # Set seeds for reproducibility
    set_all_seeds(args.seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    
    # Check for existing experiment
    if os.path.exists(output_dir) and not args.overwrite:
        print(f"\nERROR: Experiment directory already exists: {output_dir}")
        print("Options:")
        print("  1. Use --overwrite flag to overwrite")
        print("  2. Use a different --experiment-name")
        print("  3. Manually delete the directory")
        return None, None
    
    # Setup automatic logging to file
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    class Tee:
        """Redirect stdout/stderr to both console and file."""
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_handle = open(log_file, 'w', buffering=1)  # Line buffering
    import sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)

    # --- enable Tensor Core optimization for RTX 3070 ---
    torch.set_float32_matmul_precision('medium')  # high for even faster, but slightly less precise


    # --- device setup (macOS compatibility) ---
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        print("\n[INFO] MPS detected but upsample ops unsupported, using CPU for stability.")
        device = "cpu"
    else:
        device = "auto"  # let Lightning pick CUDA or CPU
        print(f"\n[INFO] Using {device.upper()} device selection.")


    # --- plot safety patch: prevent early-epoch plots from crashing on negative yerr ---
    try:
        from matplotlib.axes import _axes as _mpl_axes

        _orig_errorbar = _mpl_axes.Axes.errorbar
        def _safe_errorbar(self, x, y, yerr=None, *args, **kwargs):
            # matplotlib expects yerr as magnitudes; if PF hands us negatives, make them magnitudes
            if yerr is not None:
                yerr_arr = np.asarray(yerr)
                if (yerr_arr < 0).any():
                    yerr = np.abs(yerr_arr)
            return _orig_errorbar(self, x, y, yerr=yerr, *args, **kwargs)

        _mpl_axes.Axes.errorbar = _safe_errorbar
        print("\n[INFO] Patched matplotlib.Axes.errorbar to abs() negative yerr for robustness.")
    except Exception as e:
        print(f"\n[WARN] Could not patch matplotlib errorbar: {e}")
    
    # Get features for this configuration
    features = get_features(args.feature_set, args.frequency, include_staleness=args.staleness)
    
    print("="*70)
    print(f"Training TFT: {args.experiment_name}")
    print("="*70)
    print(f"Logging to: {log_file}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Save configuration
    config = save_config(args, features, output_dir)
    
    # Load data
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_splits(
        args.splits_dir, 
        args.feature_set, 
        args.frequency,
        args.alignment
    )
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Prepare TFT datasets
    print("\nPreparing TimeSeriesDataSet...")
    training, validation, raw_vix_train, raw_vix_val = prepare_tft_data(train_df, val_df, args, features)

    print(f"[DEBUG] VIX train values: {raw_vix_train[0:5]}")
    print(f"[DEBUG] VIX val values: {raw_vix_val[0:5]}")
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True, 
        batch_size=args.batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor = 2
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=args.batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True
    )

    print(f"Batches per epoch: {len(train_dataloader)}")
    print(f"[DEBUG] Validation batches: {len(val_dataloader)}")
    print(f"[DEBUG] Validation dataset size: {len(validation)}")
    print(f"[DEBUG] Batch size: {args.batch_size}")
    
    # Initialize model
    print("\nInitializing model...")
    tft = create_model(training, args, raw_vix_train, raw_vix_val)
    
    # Load checkpoint if specified (transfer learning)
    if args.load_checkpoint:
        if not os.path.exists(args.load_checkpoint):
            print(f"\nERROR: Checkpoint not found: {args.load_checkpoint}")
            return None, None
        
        print(f"\n[TRANSFER LEARNING] Loading checkpoint: {args.load_checkpoint}")
        
        # Load state dict from checkpoint
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out output_layer weights if we're using regime output
        # (we want fresh regime experts, not the old single output layer)
        if args.regime_output:
            state_dict = {k: v for k, v in state_dict.items() 
                         if 'output_layer' not in k}
            print(f"  Filtered out output_layer weights (using fresh regime experts)")
        
        # Load the state dict
        missing, unexpected = tft.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (expected if output_layer filtered)")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        
        print(f"  Backbone loaded successfully")
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        tft = freeze_backbone(tft, verbose=True)
    
    # store VIX on the model so training loop can access
    tft._raw_vix_train = torch.tensor(raw_vix_train, dtype=torch.float32)
    tft._raw_vix_val = torch.tensor(raw_vix_val, dtype=torch.float32)
    
    # Distribution penalties are now handled in create_model() via EnhancedQuantileLoss
    # Old monkey-patching code (Phase 3) removed - see loss_wrapper.py for reference
    
    print(f"Model parameters: {sum(p.numel() for p in tft.parameters()):,}")
    
    # Setup callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stop_patience,
        mode="min"
    )

    # add collapse monitor
    collapse_monitor = CollapseMonitor(
        val_dataloader=val_dataloader,
        log_dir=f'{output_dir}/collapse_monitoring',
        log_every_n_epochs=args.monitor_every_n_epochs
    )
        
    # Multiple checkpoints tracking different metrics
    checkpoint_val_loss = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='tft-epoch={epoch:02d}-valloss={val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=5,
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    
    checkpoint_pred_std = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='tft-epoch={epoch:02d}-predstd={val_pred_std:.4f}',
        monitor='val_pred_std',
        mode='max',  # Higher std = better diversity
        save_top_k=5,
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    
    checkpoint_unique = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='tft-epoch={epoch:02d}-unique={val_num_unique:.0f}',
        monitor='val_num_unique',
        mode='max',  # More unique predictions = better
        save_top_k=5,
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    
    # Financial performance checkpoints
    checkpoint_dir_acc = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='tft-epoch={epoch:02d}-diracc={val_dir_acc:.4f}',
        monitor='val_dir_acc',
        mode='max',  # Higher directional accuracy = better
        save_top_k=5,
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    
    checkpoint_sharpe = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='tft-epoch={epoch:02d}-sharpe={val_sharpe:.4f}',
        monitor='val_sharpe',
        mode='max',  # Higher Sharpe = better (more confident predictions)
        save_top_k=5,
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    
    checkpoint_composite = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='tft-epoch={epoch:02d}-composite={val_composite:.4f}',
        monitor='val_composite',
        mode='max',  # Higher = better directional accuracy with balanced predictions
        save_top_k=5,
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    
    
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    # Setup both loggers to get numeric metrics + figures
    csv_logger = CSVLogger("experiments", name=args.experiment_name)
    tb_logger = TensorBoardLogger("experiments", name=args.experiment_name)
    # Ensure tb_logger is first so figure logging is prioritized
    logger = [tb_logger, csv_logger]

    # build callback list
    callbacks = [
        early_stop,
        checkpoint_val_loss,
        checkpoint_pred_std,
        checkpoint_unique,
        checkpoint_dir_acc,
        checkpoint_sharpe,
        checkpoint_composite,
        EpochSummaryCallback(),
        collapse_monitor
    ]

    # Note: Anti-collapse penalty logging is integrated into CollapseMonitor
    
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator=device,
        devices=1,
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        deterministic="warn",
        strategy="auto",
        enable_progress_bar=False,
        enable_model_summary=True,
    )

    # Debug prints to inspect logger setup
    """
    print("Logger(s):", trainer.logger)
    if hasattr(trainer.logger, "loggers"):
        for lg in trainer.logger.loggers:
            exp = getattr(lg, "experiment", None)
            print(
                f"  Logger: {type(lg)}, experiment type: {type(exp)}, "
                f"has add_figure? {hasattr(exp, 'add_figure')}"
            )
    else:
        exp = getattr(trainer.logger, "experiment", None)
        print(
            f"  Single logger: {type(trainer.logger)}, experiment: {type(exp)}, "
            f"has add_figure? {hasattr(exp, 'add_figure')}"
        )
    """

    # Patch fallback for ExperimentWriter if it lacks add_figure
    if hasattr(trainer.logger, "loggers"):
        for lg in trainer.logger.loggers:
            exp = getattr(lg, "experiment", None)
            if exp is not None and not hasattr(exp, "add_figure"):
                def _dummy_add_figure(*args, **kwargs):
                    print("\n[WARN] add_figure called on experiment without support. Skipping.")
                setattr(exp, "add_figure", _dummy_add_figure)
                print(f"\n[INFO] Patched add_figure on {type(exp)}")
    else:
        exp = getattr(trainer.logger, "experiment", None)
        if exp is not None and not hasattr(exp, "add_figure"):
            def _dummy_add_figure(*args, **kwargs):
                print("\n[WARN] add_figure called on experiment without support. Skipping.")
            setattr(exp, "add_figure", _dummy_add_figure)
            print(f"\n[INFO] Patched add_figure on {type(exp)}")

    # Finally, run training
    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {output_dir}/checkpoints/")
    print(f"Logs will be saved to: {output_dir}/logs/")
    print()
    
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    if args.regime_attention and hasattr(tft.multihead_attn, 'regime_gates'):
        print(f"Final gate values: {torch.sigmoid(tft.multihead_attn.regime_gates)}") 

    print()
    print("\n" + "="*70)
    print("Training complete")
    print("="*70)
    print(f"\nBest val_loss checkpoint: {checkpoint_val_loss.best_model_path}")
    print(f"Best validation loss: {checkpoint_val_loss.best_model_score:.6f}")
    print(f"Checkpoints saved:")
    print(f"  - Best val_loss: {checkpoint_val_loss.best_model_path}")
    print(f"  - Best pred_std: {checkpoint_pred_std.best_model_path}")
    print(f"  - Best unique: {checkpoint_unique.best_model_path}")
    print(f"  - Best dir_acc: {checkpoint_dir_acc.best_model_path}")
    print(f"  - Best sharpe: {checkpoint_sharpe.best_model_path}")
    print(f"  - Best composite: {checkpoint_composite.best_model_path}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_file}")
    print("="*70)


    # --- Determine if early stopping triggered ---
    stopped_early = False
    for cb in trainer.callbacks:
        if isinstance(cb, EarlyStopping):
            # check attributes that may exist depending on version
            if getattr(cb, "stopped", False) or getattr(cb, "stopped_epoch", None) not in (None, 0):
                stopped_early = True
            break
    
    # Save final metrics
    metrics = {
        'best_val_loss_path': checkpoint_val_loss.best_model_path,
        'best_val_loss': float(checkpoint_val_loss.best_model_score) if checkpoint_val_loss.best_model_score is not None else None,
        'best_pred_std_path': checkpoint_pred_std.best_model_path,
        'best_pred_std': float(checkpoint_pred_std.best_model_score) if checkpoint_pred_std.best_model_score is not None else None,
        'best_unique_path': checkpoint_unique.best_model_path,
        'best_unique': float(checkpoint_unique.best_model_score) if checkpoint_unique.best_model_score is not None else None,
        'best_dir_acc_path': checkpoint_dir_acc.best_model_path,
        'best_dir_acc': float(checkpoint_dir_acc.best_model_score) if checkpoint_dir_acc.best_model_score is not None else None,
        'best_sharpe_path': checkpoint_sharpe.best_model_path,
        'best_sharpe': float(checkpoint_sharpe.best_model_score) if checkpoint_sharpe.best_model_score is not None else None,
        'best_composite_path': checkpoint_composite.best_model_path,
        'best_composite': float(checkpoint_composite.best_model_score) if checkpoint_composite.best_model_score is not None else None,
        'total_epochs': trainer.current_epoch,
        'early_stopped': stopped_early,
    }
    
    metrics_path = os.path.join(output_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Close log file and restore stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()
    
    return tft, trainer

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train()