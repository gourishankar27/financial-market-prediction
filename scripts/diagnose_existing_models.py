"""
Quick diagnostic tool for existing trained models.
Fixed version that uses evaluate_tft.py's exact data loading approach.

Usage:
    python scripts/diagnose_existing_models_fixed.py sweep2_h16_drop_0.25
    python scripts/diagnose_existing_models_fixed.py --compare sweep2_*
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


def load_model_and_config(experiment_name):
    """Load trained model and config."""
    exp_dir = Path(f'experiments/{experiment_name}')
    
    if not exp_dir.exists():
        return None, None
        
    # Load config
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        return None, None
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find best checkpoint
    checkpoint_dir = exp_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        return None, None
        
    checkpoints = list(checkpoint_dir.glob('*.ckpt'))
    if not checkpoints:
        return None, None
    
    # Find best by val_loss
    best_ckpt = None
    best_val_loss = float('inf')
    
    for ckpt in checkpoints:
        if 'val_loss' in ckpt.name:
            try:
                val_loss = float(ckpt.name.split('val_loss=')[1].split('.ckpt')[0])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = ckpt
            except:
                continue
    
    if best_ckpt is None:
        last_ckpt = checkpoint_dir / 'last.ckpt'
        if last_ckpt.exists():
            best_ckpt = last_ckpt
    
    if best_ckpt is None:
        return None, None
    
    # Load model
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
        model.eval()
    
    return model, config


def generate_predictions(model, dataloader):
    """
    Generate predictions using EXACT same logic as evaluate_tft.py.
    """
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            
            # Get predictions
            pred = model(x)
            
            # Extract prediction tensor (handle different return types)
            if hasattr(pred, 'prediction'):
                pred_tensor = pred.prediction
            elif isinstance(pred, tuple):
                pred_tensor = pred[0]
            elif isinstance(pred, dict) and 'prediction' in pred:
                pred_tensor = pred['prediction']
            else:
                pred_tensor = pred
            
            # Extract median quantile (index 3)
            point_pred = pred_tensor[:, 0, 3]
            
            # Extract actuals
            if isinstance(y, tuple):
                y_actual = y[0]
            else:
                y_actual = y
            
            predictions.append(point_pred.cpu().numpy())
            actuals.append(y_actual[:, 0].cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    print(f"   Mean absolute prediction: {np.mean(np.abs(predictions)):.6f}")
    print(f"   Predictions near zero (<0.01): {np.mean(np.abs(predictions) < 0.01)*100:.1f}%")
    
    return predictions, actuals


def diagnose_model(experiment_name, use_test=False, verbose=True, save=False):
    """
    Run diagnostics on a trained model.
    
    Args:
        use_test: If True, use test set. If False, use validation set.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"DIAGNOSING: {experiment_name}")
        print(f"{'='*70}")
    
    # Load model and config
    model, config = load_model_and_config(experiment_name)
    
    if model is None:
        print(f"Could not load model for {experiment_name}")
        return None
    
    diagnostics = {
        'experiment': experiment_name,
        'config': config,
    }
    
    # 1. Model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    diagnostics['total_params'] = total_params
    diagnostics['trainable_params'] = trainable_params
    
    if verbose:
        print(f"\n1. MODEL SIZE")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Hidden size: {config['architecture']['hidden_size']}")
    
    # 2. Weight statistics
    if verbose:
        print(f"\n2. WEIGHT STATISTICS")
    
    weight_stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            stats = {
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'norm': param.norm().item(),
            }
            weight_stats[name] = stats
            
            if verbose and any(key in name for key in ['lstm', 'output', 'variable_selection']):
                print(f"   {name[:50]:<50}")
                print(f"      norm: {stats['norm']:.4f}, std: {stats['std']:.4f}")
    
    diagnostics['weight_stats'] = weight_stats
    
    # 3. Prediction analysis - USE EVALUATE_TFT.PY APPROACH
    if verbose:
        print(f"\n3. PREDICTION ANALYSIS (using {'test' if use_test else 'validation'} set)")
    
    # Load data splits using evaluate_tft.py's approach
    prefix = f"{config['feature_set']}_{config['frequency']}"
    splits_dir = config['data']['splits_dir']
    
    train_df = pd.read_csv(f"{splits_dir}/{prefix}_train.csv", index_col='Date', parse_dates=True)
    
    if use_test:
        eval_df = pd.read_csv(f"{splits_dir}/{prefix}_test.csv", index_col='Date', parse_dates=True)
    else:
        eval_df = pd.read_csv(f"{splits_dir}/{prefix}_val.csv", index_col='Date', parse_dates=True)
    
    # Get features
    features = config['features']['all']
    has_staleness = any('days_since' in f or 'is_fresh' in f for f in features)
    
    # Prepare datasets EXACTLY like evaluate_tft.py
    train_df = train_df.reset_index()
    eval_df = eval_df.reset_index()
    
    if has_staleness:
        from src.data_utils import add_staleness_features
        
        #print("Detected staleness features in config, adding to data...")
        train_df = add_staleness_features(train_df, verbose=False)
        eval_df = add_staleness_features(eval_df, verbose=False)
    
    
    train_df['time_idx'] = range(len(train_df))
    train_df['group'] = 'SP500'
    
    # Combine for continuous time index
    combined_df = pd.concat([train_df, eval_df], ignore_index=True)
    combined_df['time_idx'] = range(len(combined_df))
    combined_df['group'] = 'SP500'
    
    # Create training dataset for normalization
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
    
    # Create eval dataset from combined data
    eval_dataset = TimeSeriesDataSet.from_dataset(
        training,
        combined_df,
        predict=False,
        stop_randomization=True
    )
    
    # Filter to eval period only
    eval_start_idx = len(train_df)
    eval_dataset.index = eval_dataset.index[eval_dataset.index['time'] >= eval_start_idx]
    
    # Create dataloader
    eval_dataloader = eval_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
    
    # Generate predictions
    print(f"For {experiment_name}")
    predictions, actuals = generate_predictions(model, eval_dataloader)
    
    # Compute statistics
    pred_stats = {
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'range': float(np.ptp(predictions)),
        'pct_positive': float(np.mean(predictions > 0) * 100),
        'pct_negative': float(np.mean(predictions < 0) * 100),
        'num_unique': int(len(np.unique(np.round(predictions, decimals=6)))),
    }
    
    diagnostics['predictions'] = pred_stats
    
    # Collapse detection
    collapsed = pred_stats['std'] < 0.05 or pred_stats['num_unique'] < 10
    always_positive = pred_stats['pct_positive'] > 99
    always_negative = pred_stats['pct_negative'] > 99
    
    diagnostics['collapsed'] = collapsed
    diagnostics['collapse_type'] = None
    if always_positive:
        diagnostics['collapse_type'] = 'always_positive'
    elif always_negative:
        diagnostics['collapse_type'] = 'always_negative'
    elif collapsed:
        diagnostics['collapse_type'] = 'low_diversity'
    
    if verbose:
        print(f"   Mean: {pred_stats['mean']:.6f}")
        print(f"   Std: {pred_stats['std']:.6f}")
        print(f"   Range: [{pred_stats['min']:.4f}, {pred_stats['max']:.4f}]")
        print(f"   Positive: {pred_stats['pct_positive']:.1f}%")
        print(f"   Negative: {pred_stats['pct_negative']:.1f}%")
        print(f"   Unique predictions: {pred_stats['num_unique']}")
        
        if collapsed:
            print(f"\n     COLLAPSED: {diagnostics['collapse_type']}")
        else:
            print(f"\n     Making varied predictions")
    
    if verbose:
        print(f"{'='*70}\n")
        
    # optionally save diagnosis
    if save:
        exp_dir = Path(f'experiments/{experiment_name}')
        diagnosis_path = exp_dir / 'collapse_diagnosis.json'
        
        # create simplified o/p (no weight_stats, just key info)
        save_data = {
            'experiment': diagnostics['experiment'],
            'collapsed': diagnostics['collapsed'],
            'collapse_type': diagnostics['collapse_type'],
            'predictions': diagnostics.get('predictions'),
            'hidden_size': diagnostics['config']['architecture']['hidden_size'],
            'total_params': diagnostics['total_params'],
            'diagnosed_at': pd.Timestamp.now().isoformat(),
        }
        
        with open(diagnosis_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        if verbose:
            print(f"\nSaved diagnosis to: {diagnosis_path}")
    
    return diagnostics


def compare_models(experiment_names, use_test=False, save=False):
    """Compare diagnostics across multiple models."""
    print(f"\n{'='*70}")
    print(f"COMPARING {len(experiment_names)} MODELS")
    print(f"{'='*70}\n")
    
    all_diagnostics = []
    
    for exp_name in experiment_names:
        diag = diagnose_model(exp_name, use_test=use_test, verbose=False, save=save)
        if diag:
            all_diagnostics.append(diag)
    
    if not all_diagnostics:
        print("No models successfully diagnosed")
        return
    
    # Print comparison table
    print(f"\n{'Experiment':<30} {'Hidden':>7} {'Params':>9} {'Pred_Std':>10} {'Collapsed':>10}")
    print(f"{'-'*70}")
    
    for diag in all_diagnostics:
        exp_short = diag['experiment'][:29]
        h_size = diag['config']['architecture']['hidden_size']
        params = diag['trainable_params']
        
        if 'predictions' in diag:
            pred_std = diag['predictions']['std']
            collapsed = "YES" if diag['collapsed'] else "NO"
        else:
            pred_std = None
            collapsed = "N/A"
        
        pred_std_str = f"{pred_std:.6f}" if pred_std is not None else "N/A"
        
        print(f"{exp_short:<30} {h_size:>7} {params:>9,} {pred_std_str:>10} {collapsed:>10}")
    
    print(f"{'-'*70}")
    
    # Summary
    if any('predictions' in d for d in all_diagnostics):
        with_preds = [d for d in all_diagnostics if 'predictions' in d]
        n_collapsed = sum(d['collapsed'] for d in with_preds)
        print(f"\nSummary: {n_collapsed}/{len(with_preds)} models collapsed")
        
        # By hidden size
        by_hidden = {}
        for d in with_preds:
            h = d['config']['architecture']['hidden_size']
            if h not in by_hidden:
                by_hidden[h] = []
            by_hidden[h].append(d)
        
        print(f"\nBy hidden size:")
        for h in sorted(by_hidden.keys()):
            models = by_hidden[h]
            n_collapsed_h = sum(m['collapsed'] for m in models)
            avg_std = np.mean([m['predictions']['std'] for m in models])
            print(f"  h={h}: {n_collapsed_h}/{len(models)} collapsed, avg_std={avg_std:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose trained TFT models')
    parser.add_argument('experiments', nargs='+', help='Experiment name(s) or pattern')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--use-test', action='store_true', help='Use test set instead of validation')
    parser.add_argument('--save', action='store_true', help='Save collapse diagnosis to JSON file')
    args = parser.parse_args()
    
    # Handle wildcards
    experiment_names = []
    for pattern in args.experiments:
        if '*' in pattern or '?' in pattern:
            exp_dir = Path('experiments')
            matches = [p.name for p in exp_dir.glob(pattern) if p.is_dir()]
            experiment_names.extend(sorted(matches))
        else:
            experiment_names.append(pattern)
    
    if not experiment_names:
        print("No experiments found matching pattern")
        return

    # Run diagnostics
    if args.compare or len(experiment_names) > 1:
        compare_models(experiment_names, use_test=args.use_test, save=args.save)
    else:
        diagnose_model(experiment_names[0], use_test=args.use_test, verbose=True, save=args.save)


if __name__ == '__main__':
    main()
