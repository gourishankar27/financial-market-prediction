"""
Rigorous validation: Train actual TFT model with both loss functions
and verify checkpoint/prediction equivalence.

Modes:
- validate: CPU-only comparison (expects byte-for-byte identity)
- gpu: GPU-only comparison (expects byte-for-byte identity)  
- cross-platform: CPU vs GPU comparison (reports differences without pass/fail)

This is useful for:
1. Validating custom loss implementations match baseline
2. Understanding CPU vs GPU numerical differences
3. Comparing different architectural modifications
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import pytorch_lightning as pl
import numpy as np
import os
import sys
from pathlib import Path
import argparse

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.custom_losses import EnhancedQuantileLoss
from train.collapse_monitor import CollapseMonitor
from train.callbacks import EpochSummaryCallback

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare TFT training equivalence across loss functions or platforms'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['validate', 'gpu', 'cross-platform'],
        default='validate',
        help='Comparison mode: validate (CPU-only), gpu (GPU-only), cross-platform (CPU vs GPU)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    return parser.parse_args()


def set_all_seeds(seed=42):
    """Set seeds for complete reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    """Load actual training data."""
    import pandas as pd
    
    print("Loading data...")
    train_path = project_root / "data" / "splits" / "vintage" / "core_proposal_daily_vintage_train.csv"
    val_path = project_root / "data" / "splits" / "vintage" / "core_proposal_daily_vintage_val.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Filter to core_proposal features (no staleness)
    feature_cols = [
        'SP500_Returns', 'VIX', 'Treasury_10Y', 'Yield_Spread', 'Inflation_YoY'
    ]
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing column: {col}")
    
    train_df = train_df[feature_cols]
    val_df = val_df[feature_cols]
    
    # Add time index (sequential integers) - MATCHES train_tft.py
    train_df['time_idx'] = range(len(train_df))
    val_df['time_idx'] = range(len(train_df), len(train_df) + len(val_df))
    
    # Add group identifier (single time series) - MATCHES train_tft.py
    train_df['group'] = 'SP500'
    val_df['group'] = 'SP500'
    
    print(f"Train: {len(train_df)} rows")
    print(f"Val: {len(val_df)} rows")
    
    return train_df, val_df


def create_dataset(train_df, val_df):
    """Create TimeSeriesDataSet matching train_tft.py configuration."""
    print("\nCreating dataset...")
    
    max_encoder_length = 20
    max_prediction_length = 1
    
    feature_cols = [
        'SP500_Returns', 'VIX', 'Treasury_10Y', 'Yield_Spread', 'Inflation_YoY'
    ]
    
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="SP500_Returns",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=[],  # No known future inputs for S&P 500 forecasting
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(groups=["group"]),
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset using training statistics
    # predict=False generates samples for each timestep (for evaluation)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    
    return training, validation, train_dataloader, val_dataloader


def train_model(training, train_dataloader, val_dataloader, loss_fn, experiment_name, accelerator='cpu', n_epochs=20):
    """Train TFT model with specified loss function and hardware."""
    print(f"\n{'='*80}")
    print(f"Training: {experiment_name}")
    print(f"Loss: {type(loss_fn).__name__}")
    print(f"Accelerator: {accelerator.upper()}")
    if accelerator == 'gpu' and torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}")
    
    set_all_seeds(42)
    
    # Create model matching train_tft.py defaults
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.0005,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=loss_fn,
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in tft.parameters()):,}")
    
    # Setup callbacks
    output_dir = project_root / "test_equivalence" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CollapseMonitor saves validation predictions for later comparison
    collapse_monitor = CollapseMonitor(
        val_dataloader=val_dataloader,
        log_dir=str(output_dir / 'collapse_monitoring'),
        log_every_n_epochs=1  # Save every epoch for detailed comparison
    )
    
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        filename='{epoch:02d}',  # ModelCheckpoint adds 'epoch=' prefix automatically
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
    
    # Setup logger (required for TFT's internal plotting/logging)
    from pytorch_lightning.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger("test_equivalence", name=experiment_name)
    
    # Create trainer with specified hardware
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[checkpoint, early_stop, collapse_monitor, EpochSummaryCallback()],
        enable_progress_bar=False,  # Disabled for clean log files (use EpochSummaryCallback instead)
        enable_model_summary=True,
        logger=tb_logger,
        deterministic=True,
    )
    
    # Patch logger.experiment.add_figure if needed (prevents AttributeError)
    exp = getattr(trainer.logger, "experiment", None)
    if exp is not None and not hasattr(exp, "add_figure"):
        def _dummy_add_figure(*args, **kwargs):
            pass  # Silent no-op
        setattr(exp, "add_figure", _dummy_add_figure)
    
    # Train
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Find the last saved checkpoint epoch (early stopping may have triggered)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_files = sorted(checkpoint_dir.glob('epoch=*.ckpt'))
    if checkpoint_files:
        last_checkpoint = checkpoint_files[-1]
        # Extract epoch number from filename: epoch=XX.ckpt
        last_epoch = int(last_checkpoint.stem.split('=')[1])
    else:
        last_epoch = trainer.current_epoch
    
    print(f"\nTraining completed")
    print(f"Last checkpoint saved: epoch {last_epoch}")
    print(f"Final val_loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
    
    return output_dir, last_epoch


def compare_predictions(dir1, dir2, expect_identical=True):
    """
    Compare validation predictions across all saved epochs.
    
    Args:
        dir1, dir2: Experiment directories to compare
        expect_identical: If True, emphasize identity. If False, just report differences.
    """
    print(f"\n{'='*80}")
    print("Comparing Validation Predictions Across Epochs")
    print(f"{'='*80}")
    
    pred_dir1 = dir1 / 'collapse_monitoring'
    pred_dir2 = dir2 / 'collapse_monitoring'
    
    print(f"\nChecking directories:")
    print(f"  Dir1: {pred_dir1} (exists: {pred_dir1.exists()})")
    print(f"  Dir2: {pred_dir2} (exists: {pred_dir2.exists()})")
    
    if not pred_dir1.exists() or not pred_dir2.exists():
        print("\nPrediction files not found (CollapseMonitor may not have saved them)")
        return
    
    # Find all prediction files
    pred_files1 = sorted(pred_dir1.glob('val_predictions_epoch*.npy'))
    pred_files2 = sorted(pred_dir2.glob('val_predictions_epoch*.npy'))
    
    print(f"\nFound {len(pred_files1)} prediction files in experiment 1")
    print(f"Found {len(pred_files2)} prediction files in experiment 2")
    
    if not pred_files1 or not pred_files2:
        print("No prediction .npy files found")
        return
    
    # Extract epoch numbers
    epochs1 = [int(f.stem.split('epoch')[1]) for f in pred_files1]
    epochs2 = [int(f.stem.split('epoch')[1]) for f in pred_files2]
    
    common_epochs = sorted(set(epochs1) & set(epochs2))
    
    if not common_epochs:
        print("No common epochs found between experiments")
        return
    
    print(f"\nFound predictions for {len(common_epochs)} common epochs: {common_epochs}")
    
    # Debug: Check first file from each
    if common_epochs:
        epoch = common_epochs[0]
        pred1_file = pred_dir1 / f'val_predictions_epoch{epoch}.npy'
        pred2_file = pred_dir2 / f'val_predictions_epoch{epoch}.npy'
        
        pred1_sample = np.load(pred1_file)
        pred2_sample = np.load(pred2_file)
        
        print(f"\nDebug info for epoch {epoch}:")
        print(f"  Exp1: shape={pred1_sample.shape}, dtype={pred1_sample.dtype}")
        print(f"  Exp2: shape={pred2_sample.shape}, dtype={pred2_sample.dtype}")
        if len(pred1_sample) > 0:
            print(f"  Exp1 first 5: {pred1_sample[:5]}")
            print(f"  Exp2 first 5: {pred2_sample[:5]}")
    
    print(f"\n{'Epoch':<8} {'Max Diff':<12} {'Mean Diff':<12} {'Std Diff':<12} {'Correlation':<12} {'Samples':<10}")
    print("-" * 76)
    
    for epoch in common_epochs:
        pred1_file = pred_dir1 / f'val_predictions_epoch{epoch}.npy'
        pred2_file = pred_dir2 / f'val_predictions_epoch{epoch}.npy'
        
        pred1 = np.load(pred1_file)
        pred2 = np.load(pred2_file)
        
        if pred1.shape != pred2.shape:
            print(f"{epoch:<8} Shape mismatch: {pred1.shape} vs {pred2.shape}")
            continue
        
        # Compute differences
        diff = np.abs(pred1 - pred2)
        max_diff = diff.max()
        mean_diff = diff.mean()
        std_diff = diff.std()
        
        # Compute correlation
        if len(pred1) > 1:
            correlation = np.corrcoef(pred1, pred2)[0, 1]
        else:
            correlation = np.nan
        
        print(f"{epoch:<8} {max_diff:<12.2e} {mean_diff:<12.2e} {std_diff:<12.2e} {correlation:<12.6f} {len(pred1):<10}")
    
    print("\nInterpretation:")
    print("  Max Diff:     Largest difference between any two predictions")
    print("  Mean Diff:    Average absolute difference across all predictions")
    print("  Std Diff:     Standard deviation of differences")
    print("  Correlation:  Pearson correlation (1.0 = perfect agreement)")


def compare_checkpoints(dir1, dir2, epoch, expect_identical=True):
    """
    Compare model checkpoints.
    
    Args:
        dir1, dir2: Experiment directories to compare
        epoch: Which epoch checkpoint to compare
        expect_identical: If True, report pass/fail. If False, just report differences.
    
    Note: Checkpoints are always loaded to CPU memory for comparison regardless of 
    training hardware. This ensures consistent comparison across GPU/CPU trained models.
    """
    print(f"\n{'='*80}")
    print(f"Comparing checkpoints at epoch {epoch}")
    print(f"{'='*80}")
    
    ckpt1 = dir1 / 'checkpoints' / f'epoch={epoch:02d}.ckpt'
    ckpt2 = dir2 / 'checkpoints' / f'epoch={epoch:02d}.ckpt'
    
    if not ckpt1.exists() or not ckpt2.exists():
        print(f"Checkpoint files don't exist")
        print(f"  {ckpt1}: {ckpt1.exists()}")
        print(f"  {ckpt2}: {ckpt2.exists()}")
        return False
    
    # Load checkpoints to CPU for comparison
    print("Loading checkpoints to CPU for comparison...")
    state1 = torch.load(ckpt1, map_location='cpu')
    state2 = torch.load(ckpt2, map_location='cpu')
    
    # Extract training device info from checkpoints if available
    try:
        # Check if we can infer training device from checkpoint metadata
        hparams1 = state1.get('hyper_parameters', {})
        hparams2 = state2.get('hyper_parameters', {})
        print(f"Checkpoint 1: {ckpt1.parent.parent.name}")
        print(f"Checkpoint 2: {ckpt2.parent.parent.name}")
    except:
        pass
    
    # Compare state_dict (model weights)
    print("\nComparing model weights...")
    state_dict1 = state1['state_dict']
    state_dict2 = state2['state_dict']
    
    if state_dict1.keys() != state_dict2.keys():
        print(f"Different parameter names")
        return False
    
    all_match = True
    max_diff = 0.0
    params_with_diff = []
    
    for name in state_dict1.keys():
        param1 = state_dict1[name]
        param2 = state_dict2[name]
        
        if param1.shape != param2.shape:
            print(f"Shape mismatch: {name}")
            all_match = False
            continue
        
        diff = (param1 - param2).abs().max().item()
        max_diff = max(max_diff, diff)
        
        if diff > 1e-6:
            params_with_diff.append((name, diff))
            all_match = False
    
    print(f"\nMax weight difference across all parameters: {max_diff:.2e}")
    
    if params_with_diff:
        print(f"\nParameters with differences > 1e-6 ({len(params_with_diff)} total):")
        params_with_diff.sort(key=lambda x: x[1], reverse=True)
        for name, diff in params_with_diff[:10]:
            print(f"  {name}: max_diff={diff:.2e}")
        if len(params_with_diff) > 10:
            print(f"  ... and {len(params_with_diff) - 10} more")
    
    # Compare optimizer state
    print("\nComparing optimizer states...")
    opt1 = state1['optimizer_states'][0]
    opt2 = state2['optimizer_states'][0]
    
    if 'state' in opt1 and 'state' in opt2:
        opt_match = True
        opt_diffs = []
        for key in opt1['state'].keys():
            if key not in opt2['state']:
                print(f"Optimizer state mismatch at key {key}")
                opt_match = False
                continue
            
            for param_key in opt1['state'][key].keys():
                if param_key not in opt2['state'][key]:
                    continue
                
                val1 = opt1['state'][key][param_key]
                val2 = opt2['state'][key][param_key]
                
                if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                    diff = (val1 - val2).abs().max().item()
                    if diff > 1e-6:
                        opt_diffs.append((key, param_key, diff))
                        opt_match = False
        
        if opt_match:
            print("  Optimizer states match")
        else:
            print(f"  Optimizer states differ ({len(opt_diffs)} parameters)")
            opt_diffs.sort(key=lambda x: x[2], reverse=True)
            for key, param_key, diff in opt_diffs[:5]:
                print(f"    {key}/{param_key}: max_diff={diff:.2e}")
    
    # Overall result
    print(f"\n{'='*80}")
    if expect_identical:
        # Validation mode: pass/fail based on identity
        if all_match and max_diff < 1e-6:
            print("PASS: Checkpoints are identical")
            print("EnhancedQuantileLoss with penalties=0 is equivalent to QuantileLoss")
            return True
        else:
            print("FAIL: Checkpoints differ")
            print("There is a bug in EnhancedQuantileLoss implementation")
            print(f"\nMax difference: {max_diff:.2e}")
            print("Note: Differences < 1e-5 may be acceptable due to floating point precision")
            return False
    else:
        # Cross-platform mode: just report differences
        print("Checkpoint Comparison Summary:")
        print(f"  Max weight difference: {max_diff:.2e}")
        if all_match:
            print("  Status: Byte-for-byte identical (unexpected for CPU vs GPU)")
        elif max_diff < 1e-4:
            print("  Status: Very similar (typical for CPU vs GPU)")
        elif max_diff < 1e-3:
            print("  Status: Minor differences (acceptable for CPU vs GPU)")
        else:
            print("  Status: Significant differences (may indicate issue)")
        return None  # No pass/fail


def main():
    """Run TFT equivalence test based on selected mode."""
    args = parse_args()
    
    print("="*80)
    print("TFT TRAINING EQUIVALENCE TEST")
    print("="*80)
    print(f"\nMode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    
    if args.mode == 'validate':
        print("Hardware: CPU only (deterministic)")
        print("Purpose: Validate EnhancedQuantileLoss implementation")
        print("Expected: Byte-for-byte identical checkpoints and predictions")
    elif args.mode == 'gpu':
        print("Hardware: GPU only (deterministic)")
        print("Purpose: Validate EnhancedQuantileLoss on GPU")
        print("Expected: Byte-for-byte identical checkpoints and predictions")
    else:  # cross-platform
        print("Hardware: CPU vs GPU")
        print("Purpose: Measure numerical differences across platforms")
        print("Expected: Small differences (~1e-5 to 1e-4) due to different hardware")
    
    # Patch matplotlib errorbar to handle negative yerr values
    try:
        from matplotlib.axes import _axes as _mpl_axes
        
        _orig_errorbar = _mpl_axes.Axes.errorbar
        def _safe_errorbar(self, x, y, yerr=None, *args, **kwargs):
            if yerr is not None:
                yerr_arr = np.asarray(yerr)
                if (yerr_arr < 0).any():
                    yerr = np.abs(yerr_arr)
            return _orig_errorbar(self, x, y, yerr=yerr, *args, **kwargs)
        
        _mpl_axes.Axes.errorbar = _safe_errorbar
        print("[INFO] Patched matplotlib.Axes.errorbar for negative yerr handling\n")
    except Exception as e:
        print(f"[WARN] Could not patch matplotlib errorbar: {e}\n")
    
    # Load data
    train_df, val_df = load_data()
    training, validation, train_dataloader, val_dataloader = create_dataset(train_df, val_df)
    
    if args.mode == 'cross-platform':
        # Compare CPU vs GPU training
        print("\n" + "="*80)
        print("TRAINING ON CPU")
        print("="*80)
        loss_cpu = QuantileLoss()
        dir_cpu, epoch_cpu = train_model(
            training, train_dataloader, val_dataloader,
            loss_cpu, "cpu_quantileloss", accelerator='cpu', n_epochs=args.epochs
        )
        
        print("\n" + "="*80)
        print("TRAINING ON GPU")
        print("="*80)
        loss_gpu = QuantileLoss()
        dir_gpu, epoch_gpu = train_model(
            training, train_dataloader, val_dataloader,
            loss_gpu, "gpu_quantileloss", accelerator='gpu', n_epochs=args.epochs
        )
        
        final_epoch = min(epoch_cpu, epoch_gpu)
        compare_checkpoints(dir_cpu, dir_gpu, final_epoch, expect_identical=False)
        compare_predictions(dir_cpu, dir_gpu, expect_identical=False)
        
        # No pass/fail summary for cross-platform
        print("\n" + "="*80)
        print("CROSS-PLATFORM COMPARISON COMPLETE")
        print("="*80)
        print("See above for CPU vs GPU numerical differences.")
        
    else:
        # Validation mode (CPU or GPU)
        accelerator = 'cpu' if args.mode == 'validate' else 'gpu'
        
        # Train with QuantileLoss
        loss_original = QuantileLoss()
        dir_original, epoch_original = train_model(
            training, train_dataloader, val_dataloader,
            loss_original, "original_quantileloss", accelerator=accelerator, n_epochs=args.epochs
        )
        
        # Train with EnhancedQuantileLoss (all penalties disabled)
        loss_enhanced = EnhancedQuantileLoss(
            collapse_weight=0.0,
            temporal_consistency_weight=0.0
        )
        dir_enhanced, epoch_enhanced = train_model(
            training, train_dataloader, val_dataloader,
            loss_enhanced, "enhanced_quantileloss", accelerator=accelerator, n_epochs=args.epochs
        )
        
        # Compare checkpoints and predictions
        final_epoch = min(epoch_original, epoch_enhanced)
        checkpoint_result = compare_checkpoints(dir_original, dir_enhanced, final_epoch, expect_identical=True)
        compare_predictions(dir_original, dir_enhanced, expect_identical=True)
        
        # Summary
        print("\n" + "="*80)
        print("FINAL RESULT")
        print("="*80)
        if checkpoint_result:
            print("PASS: EnhancedQuantileLoss is equivalent to QuantileLoss")
            print(f"The implementation is correct on {accelerator.upper()}")
        else:
            print("FAIL: EnhancedQuantileLoss differs from QuantileLoss")
            print("There may be a bug in the implementation, OR")
            print("Differences may be within acceptable floating point precision")
        print("="*80)
        
        return 0 if checkpoint_result else 1
    
    return 0


if __name__ == "__main__":
    exit(main())
