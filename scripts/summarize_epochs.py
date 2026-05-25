#!/usr/bin/env python3
"""
Summarize training duration and early stopping across experiments.

Reports:
  - Stopped epoch (from final_metrics.json or checkpoint filenames)
  - Max epochs configured
  - Whether early stopping triggered
  - Best validation metrics at stop

Usage:
    # Single phase
    python summarize_epochs.py experiments/test_weekly
    
    # Multiple phases
    python summarize_epochs.py experiments/00_baseline_exploration experiments/01_staleness_features
    
    # All experiments
    python summarize_epochs.py experiments/
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


def get_stopped_epoch_from_checkpoints(ckpt_dir: Path) -> int:
    """Infer stopped epoch from highest epoch in checkpoint filenames."""
    if not ckpt_dir.exists():
        return -1
    
    max_epoch = -1
    for ckpt in ckpt_dir.glob('*.ckpt'):
        match = re.search(r'epoch=(\d+)', ckpt.name)
        if match:
            epoch = int(match.group(1))
            max_epoch = max(max_epoch, epoch)
    
    return max_epoch


def get_stopped_epoch_from_log(exp_path: Path) -> int:
    """Parse training log for last epoch."""
    log_files = list(exp_path.glob('training_*.log'))
    if not log_files:
        return -1
    
    max_epoch = -1
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Find all epoch mentions
            for match in re.finditer(r'Epoch\s+(\d+):', content):
                epoch = int(match.group(1))
                max_epoch = max(max_epoch, epoch)
        except Exception:
            continue
    
    return max_epoch


def get_stopped_epoch_from_monitor(exp_path: Path) -> int:
    """Get stopped epoch from collapse monitor JSON."""
    monitor_path = exp_path / 'collapse_monitoring' / 'collapse_monitor_latest.json'
    if not monitor_path.exists():
        return -1
    
    try:
        with open(monitor_path) as f:
            data = json.load(f)
        epochs = data.get('epoch', [])
        return max(epochs) if epochs else -1
    except Exception:
        return -1


def analyze_experiment(exp_path: Path) -> dict:
    """
    Analyze a single experiment's training duration and early stopping.
    """
    result = {
        'experiment': exp_path.name,
        'path': str(exp_path),
    }
    
    # Try to load config
    config_path = exp_path / 'config.json'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            result['max_epochs'] = config.get('training', {}).get('max_epochs', -1)
            result['patience'] = config.get('training', {}).get('early_stopping_patience', -1)
            result['hidden_size'] = config.get('architecture', {}).get('hidden_size', -1)
            result['frequency'] = config.get('frequency', 'unknown')
        except Exception:
            pass
    
    # Try to load final_metrics
    final_metrics_path = exp_path / 'final_metrics.json'
    if final_metrics_path.exists():
        try:
            with open(final_metrics_path) as f:
                metrics = json.load(f)
            result['stopped_epoch'] = metrics.get('total_epochs', -1)
            result['best_val_loss'] = metrics.get('best_val_loss', np.nan)
            result['final_train_loss'] = metrics.get('final_train_loss', np.nan)
        except Exception:
            pass
    
    # Fallback: infer from checkpoints
    if result.get('stopped_epoch', -1) == -1:
        ckpt_epoch = get_stopped_epoch_from_checkpoints(exp_path / 'checkpoints')
        if ckpt_epoch > -1:
            result['stopped_epoch'] = ckpt_epoch
            result['epoch_source'] = 'checkpoints'
    
    # Fallback: infer from training log
    if result.get('stopped_epoch', -1) == -1:
        log_epoch = get_stopped_epoch_from_log(exp_path)
        if log_epoch > -1:
            result['stopped_epoch'] = log_epoch
            result['epoch_source'] = 'log'
    
    # Fallback: infer from collapse monitor
    if result.get('stopped_epoch', -1) == -1:
        monitor_epoch = get_stopped_epoch_from_monitor(exp_path)
        if monitor_epoch > -1:
            result['stopped_epoch'] = monitor_epoch
            result['epoch_source'] = 'monitor'
    
    # Determine if early stopping triggered
    if result.get('stopped_epoch', -1) > 0 and result.get('max_epochs', -1) > 0:
        result['early_stopped'] = result['stopped_epoch'] < result['max_epochs']
        result['pct_epochs_used'] = result['stopped_epoch'] / result['max_epochs'] * 100
    
    return result


def find_experiments(base_paths: list[Path]) -> list[Path]:
    """
    Find all experiment directories under given paths.
    
    An experiment directory has either config.json or checkpoints/ subdirectory.
    """
    experiments = []
    
    for base_path in base_paths:
        if not base_path.exists():
            print(f"Warning: {base_path} does not exist")
            continue
        
        # Check if base_path itself is an experiment
        if (base_path / 'config.json').exists() or (base_path / 'checkpoints').exists():
            experiments.append(base_path)
            continue
        
        # Search subdirectories
        for item in base_path.rglob('*'):
            if item.is_dir():
                if (item / 'config.json').exists() or (item / 'checkpoints').exists():
                    experiments.append(item)
    
    # Deduplicate and sort
    experiments = sorted(set(experiments))
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description='Summarize training epochs across experiments')
    parser.add_argument('paths', nargs='+', type=str,
                       help='Paths to search for experiments')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path')
    parser.add_argument('--sort-by', type=str, default='stopped_epoch',
                       choices=['stopped_epoch', 'experiment', 'pct_epochs_used', 'best_val_loss'],
                       help='Sort results by this column')
    parser.add_argument('--filter-phase', type=str, default=None,
                       help='Only include experiments from this phase')
    
    args = parser.parse_args()
    
    base_paths = [Path(p) for p in args.paths]
    experiments = find_experiments(base_paths)
    
    print(f"Found {len(experiments)} experiments")
    
    if not experiments:
        return
    
    # Analyze each experiment
    results = []
    for exp_path in experiments:
        result = analyze_experiment(exp_path)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Filter by phase if requested
    if args.filter_phase:
        df = df[df['path'].str.contains(args.filter_phase)]
    
    # Sort
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=(args.sort_by != 'stopped_epoch'))
    
    # Save CSV
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('results') / 'epoch_summary.csv'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("EPOCH SUMMARY")
    print(f"{'='*80}")
    
    # Display columns
    display_cols = ['experiment', 'stopped_epoch', 'max_epochs', 'pct_epochs_used', 
                   'early_stopped', 'best_val_loss', 'hidden_size', 'frequency']
    display_cols = [c for c in display_cols if c in df.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 40)
    
    print(df[display_cols].to_string(index=False))
    
    # Statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    
    if 'stopped_epoch' in df.columns:
        valid_epochs = df['stopped_epoch'].dropna()
        valid_epochs = valid_epochs[valid_epochs > 0]
        if len(valid_epochs) > 0:
            print(f"Stopped epochs: min={valid_epochs.min():.0f}, "
                  f"max={valid_epochs.max():.0f}, "
                  f"mean={valid_epochs.mean():.1f}, "
                  f"median={valid_epochs.median():.0f}, "
                  f"std={valid_epochs.std():.1f}")
    
    if 'early_stopped' in df.columns:
        early_stopped = df['early_stopped'].sum()
        total = df['early_stopped'].notna().sum()
        print(f"Early stopped: {early_stopped}/{total} ({early_stopped/total*100:.1f}%)")
    
    if 'pct_epochs_used' in df.columns:
        valid_pct = df['pct_epochs_used'].dropna()
        if len(valid_pct) > 0:
            print(f"Epochs used: mean={valid_pct.mean():.1f}%, "
                  f"median={valid_pct.median():.1f}%")


if __name__ == '__main__':
    main()
