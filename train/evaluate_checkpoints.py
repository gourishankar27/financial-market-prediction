#!/usr/bin/env python3
"""
Batch Checkpoint Evaluation

Evaluates all (or selected) checkpoints in an experiment and produces a comparison CSV.

Usage:
    # Evaluate all checkpoints
    python evaluate_checkpoints.py experiments/test_weekly/baseline_h16_daily
    
    # Evaluate specific checkpoints by pattern
    python evaluate_checkpoints.py experiments/test_weekly/baseline_h16_daily --pattern "diracc|sharpe"
    
    # Evaluate only top N by each metric type
    python evaluate_checkpoints.py experiments/test_weekly/baseline_h16_daily --top-per-metric 2
    
    # Quick mode (skip plots, faster)
    python evaluate_checkpoints.py experiments/test_weekly/baseline_h16_daily --quick

Output:
    {experiment}/checkpoint_comparison.csv - Summary table of all evaluated checkpoints
    {experiment}/eval_{checkpoint_name}/ - Individual evaluation outputs (unless --quick)
"""

import os
import sys
import re
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def find_checkpoints(exp_path: Path, pattern: str = None) -> list[Path]:
    """
    Find all checkpoint files in experiment directory.
    
    Args:
        exp_path: Path to experiment directory
        pattern: Optional regex pattern to filter checkpoints
        
    Returns:
        List of checkpoint paths
    """
    ckpt_dir = exp_path / 'checkpoints'
    if not ckpt_dir.exists():
        print(f"No checkpoints directory found at {ckpt_dir}")
        return []
    
    checkpoints = list(ckpt_dir.glob('*.ckpt'))
    
    # Filter out 'last' checkpoints (usually duplicates)
    checkpoints = [c for c in checkpoints if 'last' not in c.name.lower()]
    
    # Apply pattern filter if specified
    if pattern:
        regex = re.compile(pattern, re.IGNORECASE)
        checkpoints = [c for c in checkpoints if regex.search(c.name)]
    
    return sorted(checkpoints)


def parse_checkpoint_name(ckpt_path: Path) -> dict:
    """
    Extract metadata from checkpoint filename.
    
    Expected format: tft-epoch=epoch=XX-metric=val_metric=Y.YYYY.ckpt
    """
    name = ckpt_path.stem
    info = {'filename': name, 'path': str(ckpt_path)}
    
    # Extract epoch
    epoch_match = re.search(r'epoch=(\d+)', name)
    if epoch_match:
        info['epoch'] = int(epoch_match.group(1))
    
    # Extract metric type and value
    # Patterns: diracc, sharpe, valloss, predstd, unique
    metric_patterns = [
        (r'diracc=val_dir_acc=([0-9.]+)', 'val_dir_acc'),
        (r'sharpe=val_sharpe=([0-9.-]+)', 'val_sharpe'),
        (r'valloss=val_loss=([0-9.]+)', 'val_loss'),
        (r'predstd=val_pred_std=([0-9.]+)', 'val_pred_std'),
        (r'unique=val_num_unique=(\d+)', 'val_num_unique'),
    ]
    
    for pattern, metric_name in metric_patterns:
        match = re.search(pattern, name)
        if match:
            info['checkpoint_metric'] = metric_name
            info['checkpoint_value'] = float(match.group(1))
            break
    
    return info


def run_evaluation(exp_name: str, ckpt_path: Path, output_dir: Path, 
                   quick: bool = False) -> dict:
    """
    Run evaluate_tft.py on a single checkpoint.
    
    Args:
        exp_name: Experiment name for evaluate_tft.py
        ckpt_path: Path to checkpoint file
        output_dir: Directory to save evaluation outputs
        quick: If True, skip plot generation (not implemented in evaluate_tft.py yet)
        
    Returns:
        Dict of evaluation metrics, or None if evaluation failed
    """
    cmd = [
        sys.executable, 'train/evaluate_tft.py',
        '--experiment-name', exp_name,
        '--checkpoint', str(ckpt_path),
        '--output-dir', str(output_dir),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per checkpoint
        )
        
        if result.returncode != 0:
            print(f"  [FAILED] {ckpt_path.name}")
            print(f"    stderr: {result.stderr[:200]}...")
            return None
        
        # Load evaluation metrics
        metrics_path = output_dir / 'evaluation_metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        else:
            print(f"  [WARN] No metrics file at {metrics_path}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {ckpt_path.name}")
        return None
    except Exception as e:
        print(f"  [ERROR] {ckpt_path.name}: {e}")
        return None


def extract_key_metrics(eval_metrics: dict, ckpt_info: dict) -> dict:
    """
    Extract key metrics from evaluation results for comparison table.
    """
    row = {
        'checkpoint': ckpt_info.get('filename', 'unknown'),
        'epoch': ckpt_info.get('epoch', -1),
        'ckpt_metric': ckpt_info.get('checkpoint_metric', ''),
        'ckpt_value': ckpt_info.get('checkpoint_value', np.nan),
    }
    
    # Statistical metrics
    stat = eval_metrics.get('statistical_metrics', {})
    row['mse'] = stat.get('mse', np.nan)
    row['mae'] = stat.get('mae', np.nan)
    row['r2'] = stat.get('r2', np.nan)
    
    # Financial metrics
    fin = eval_metrics.get('financial_metrics', {})
    row['dir_acc'] = fin.get('directional_accuracy', np.nan)
    row['sharpe'] = fin.get('sharpe_ratio', np.nan)
    row['total_return'] = fin.get('total_return', np.nan)
    row['max_drawdown'] = fin.get('max_drawdown', np.nan)
    
    # Mode stats (at top level, not nested under collapse_diagnostics)
    mode_stats = eval_metrics.get('mode_stats', {})
    row['healthy_pct'] = mode_stats.get('healthy_pct', np.nan)
    row['degraded_pct'] = mode_stats.get('degraded_pct', np.nan)
    row['unidirectional_pct'] = mode_stats.get('unidirectional_pct', np.nan)
    row['weak_collapse_pct'] = mode_stats.get('weak_collapse_pct', np.nan)
    row['strong_collapse_pct'] = mode_stats.get('strong_collapse_pct', np.nan)
    
    # Prediction stats (at top level)
    pred_stats = eval_metrics.get('prediction_stats', {})
    row['pred_std'] = pred_stats.get('std', np.nan)
    row['pred_mean'] = pred_stats.get('mean', np.nan)
    row['pct_positive'] = pred_stats.get('pct_positive', np.nan)
    # Compute num_negative from percentage (backwards compatible - num_negative not in older JSONs)
    num_preds = pred_stats.get('num_predictions', 0)
    pct_neg = pred_stats.get('pct_negative', 0)
    row['num_negative'] = int(pct_neg * num_preds / 100) if num_preds > 0 else np.nan
    
    return row


def select_top_checkpoints(checkpoints: list[Path], top_n: int = 3) -> list[Path]:
    """
    Select top N checkpoints for each metric type.
    
    This avoids evaluating many similar checkpoints when you just want
    the best from each category. Also deduplicates by epoch.
    """
    by_metric = {}
    for ckpt in checkpoints:
        info = parse_checkpoint_name(ckpt)
        metric = info.get('checkpoint_metric', 'unknown')
        if metric not in by_metric:
            by_metric[metric] = []
        by_metric[metric].append((ckpt, info.get('checkpoint_value', 0), info.get('epoch', -1)))
    
    selected = []
    seen_epochs = set()
    for metric, ckpts in by_metric.items():
        # Sort by value (higher is better for most, lower for loss)
        reverse = metric not in ['val_loss']
        sorted_ckpts = sorted(ckpts, key=lambda x: x[1], reverse=reverse)
        
        count = 0
        for ckpt, val, epoch in sorted_ckpts:
            if count >= top_n:
                break
            if epoch not in seen_epochs:
                selected.append(ckpt)
                seen_epochs.add(epoch)
                count += 1
    
    return selected


def deduplicate_by_epoch(checkpoints: list[Path]) -> list[Path]:
    """
    Remove duplicate checkpoints for the same epoch.
    
    Keeps the first occurrence (arbitrary choice - they should have same weights).
    """
    seen_epochs = set()
    unique = []
    for ckpt in checkpoints:
        info = parse_checkpoint_name(ckpt)
        epoch = info.get('epoch', -1)
        if epoch not in seen_epochs:
            unique.append(ckpt)
            seen_epochs.add(epoch)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate checkpoints and compare results')
    parser.add_argument('exp_path', type=str,
                       help='Path to experiment directory')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Regex pattern to filter checkpoint names')
    parser.add_argument('--top-per-metric', type=int, default=None,
                       help='Only evaluate top N checkpoints per metric type')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (skip individual output dirs)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path (default: {exp}/checkpoint_comparison.csv)')
    
    args = parser.parse_args()
    
    exp_path = Path(args.exp_path)
    if not exp_path.exists():
        print(f"Error: {exp_path} does not exist")
        return
    
    # Determine experiment name for evaluate_tft.py
    # Handle both "experiments/phase/exp" and "phase/exp" formats
    if 'experiments' in exp_path.parts:
        exp_idx = exp_path.parts.index('experiments')
        exp_name = '/'.join(exp_path.parts[exp_idx + 1:])
    else:
        exp_name = str(exp_path)
    
    print(f"Experiment: {exp_name}")
    print(f"Path: {exp_path}")
    
    # Find checkpoints
    checkpoints = find_checkpoints(exp_path, args.pattern)
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Filter to top N per metric if requested
    if args.top_per_metric:
        checkpoints = select_top_checkpoints(checkpoints, args.top_per_metric)
        print(f"Selected {len(checkpoints)} checkpoints (top {args.top_per_metric} per metric, deduplicated by epoch)")
    else:
        # Always deduplicate by epoch
        orig_count = len(checkpoints)
        checkpoints = deduplicate_by_epoch(checkpoints)
        if len(checkpoints) < orig_count:
            print(f"Deduplicated: {orig_count} -> {len(checkpoints)} checkpoints (by epoch)")
    
    # Create evaluation output directory
    eval_base_dir = exp_path / 'evaluation'
    eval_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each checkpoint
    results = []
    for i, ckpt in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}] Evaluating: {ckpt.name}")
        
        ckpt_info = parse_checkpoint_name(ckpt)
        
        if args.quick:
            # In quick mode, use a shared temp dir (outputs overwritten)
            output_dir = eval_base_dir / 'temp'
        else:
            # Create separate output dir for each checkpoint
            safe_name = ckpt.stem.replace('=', '_').replace('.', '_')
            output_dir = eval_base_dir / safe_name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        eval_metrics = run_evaluation(exp_name, ckpt, output_dir, args.quick)
        
        if eval_metrics:
            row = extract_key_metrics(eval_metrics, ckpt_info)
            results.append(row)
            print(f"  dir_acc={row['dir_acc']:.4f}, sharpe={row['sharpe']:.4f}, "
                  f"healthy={row['healthy_pct']:.1f}%")
        else:
            # Still record the checkpoint with NaN metrics
            row = {'checkpoint': ckpt_info.get('filename'), 
                   'epoch': ckpt_info.get('epoch', -1),
                   'error': 'evaluation_failed'}
            results.append(row)
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Sort by epoch
    if 'epoch' in df.columns:
        df = df.sort_values('epoch')
    
    # Save results
    output_path = Path(args.output) if args.output else eval_base_dir / 'checkpoint_comparison.csv'
    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved comparison to: {output_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("CHECKPOINT COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    # Select columns to display
    display_cols = ['epoch', 'ckpt_metric', 'dir_acc', 'sharpe', 'total_return', 
                   'healthy_pct', 'unidirectional_pct', 'num_negative']
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 100 else f'{x:.1f}')
    
    print(df[display_cols].to_string(index=False))
    
    # Identify best checkpoints
    print(f"\n{'='*70}")
    print("BEST CHECKPOINTS BY METRIC")
    print(f"{'='*70}")
    
    if 'dir_acc' in df.columns and df['dir_acc'].notna().any():
        best_da = df.loc[df['dir_acc'].idxmax()]
        print(f"Best Dir Acc:    epoch {best_da['epoch']:.0f} ({best_da['dir_acc']:.4f})")
    
    if 'sharpe' in df.columns and df['sharpe'].notna().any():
        best_sharpe = df.loc[df['sharpe'].idxmax()]
        print(f"Best Sharpe:     epoch {best_sharpe['epoch']:.0f} ({best_sharpe['sharpe']:.4f})")
    
    if 'healthy_pct' in df.columns and df['healthy_pct'].notna().any():
        best_healthy = df.loc[df['healthy_pct'].idxmax()]
        print(f"Best Healthy %:  epoch {best_healthy['epoch']:.0f} ({best_healthy['healthy_pct']:.1f}%)")
    
    if 'total_return' in df.columns and df['total_return'].notna().any():
        best_return = df.loc[df['total_return'].idxmax()]
        print(f"Best Return:     epoch {best_return['epoch']:.0f} ({best_return['total_return']:.4f})")


if __name__ == '__main__':
    main()