#!/usr/bin/env python3
"""
Compare experiments across a phase by reading checkpoint_comparison.csv files.
Picks best checkpoint per experiment and outputs summary table.

Usage:
    python compare_experiments.py experiments/09_preliminary/
    python compare_experiments.py experiments/09_preliminary/ --metric sharpe
    python compare_experiments.py experiments/09_preliminary/ --baseline experiments/06a_weekly_sweep/h16_enc12_d015_bs16
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def load_experiment(exp_path: Path, metric: str = 'dir_acc', min_epoch: int = 10) -> dict:
    """Load best checkpoint from an experiment's checkpoint_comparison.csv.
    
    Args:
        exp_path: Path to experiment directory
        metric: Metric to select best checkpoint by
        min_epoch: Minimum epoch to consider (filter out early unstable checkpoints)
    """
    csv_path = exp_path / 'evaluation' / 'checkpoint_comparison.csv'
    
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        return None
    
    # Filter to epochs >= min_epoch
    df_filtered = df[df['epoch'] >= min_epoch]
    
    if df_filtered.empty:
        print(f"  Warning: {exp_path.name} has no checkpoints >= epoch {min_epoch}, skipping")
        return None
    
    # Pick best checkpoint by metric (higher is better for most metrics)
    if metric in ['mse', 'mae', 'weak_collapse_pct', 'strong_collapse_pct']:
        best_idx = df_filtered[metric].idxmin()
    else:
        best_idx = df_filtered[metric].idxmax()
    
    best = df_filtered.loc[best_idx]
    
    return {
        'experiment': exp_path.name,
        'epoch': int(best.get('epoch', 0)),
        'dir_acc': best.get('dir_acc', 0),
        'sharpe': best.get('sharpe', 0),
        'total_return': best.get('total_return', 0),
        'healthy_pct': best.get('healthy_pct', 0),
        'unidirectional_pct': best.get('unidirectional_pct', 0),
        'weak_collapse_pct': best.get('weak_collapse_pct', 0),
        'strong_collapse_pct': best.get('strong_collapse_pct', 0),
        'pred_std': best.get('pred_std', 0),
        'pct_positive': best.get('pct_positive', 0),
        'mae': best.get('mae', 0),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare experiments in a phase')
    parser.add_argument('phase_dir', type=str, help='Path to phase directory')
    parser.add_argument('--metric', type=str, default='dir_acc',
                        help='Metric to select best checkpoint (default: dir_acc)')
    parser.add_argument('--baseline', type=str, action='append', default=[],
                        help='Baseline experiment paths to include (can specify multiple)')
    parser.add_argument('--min-epoch', type=int, default=10,
                        help='Minimum epoch to consider for best checkpoint (default: 10)')
    parser.add_argument('--csv', type=str, help='Output CSV path')
    args = parser.parse_args()
    
    phase_dir = Path(args.phase_dir)
    
    if not phase_dir.exists():
        print(f"Error: {phase_dir} does not exist")
        sys.exit(1)
    
    # Collect all experiments in phase
    results = []
    
    # Add baselines first
    for baseline in args.baseline:
        baseline_path = Path(baseline)
        if baseline_path.exists():
            result = load_experiment(baseline_path, args.metric, args.min_epoch)
            if result:
                result['experiment'] = f"[BASE] {baseline_path.name}"
                results.append(result)
        else:
            print(f"Warning: Baseline {baseline} not found")
    
    # Add phase experiments
    for exp_path in sorted(phase_dir.iterdir()):
        if exp_path.is_dir():
            result = load_experiment(exp_path, args.metric, args.min_epoch)
            if result:
                results.append(result)
    
    if not results:
        print("No experiments found with checkpoint_comparison.csv")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate collapse total
    df['collapse_total'] = df['weak_collapse_pct'] + df['strong_collapse_pct']
    
    # Format for display
    display_cols = [
        'experiment', 'epoch', 'dir_acc', 'sharpe', 'pred_std', 
        'pct_positive', 'healthy_pct', 'unidirectional_pct', 'collapse_total', 'mae'
    ]
    
    df_display = df[display_cols].copy()
    
    # Format numbers
    df_display['dir_acc'] = df_display['dir_acc'].apply(lambda x: f"{x*100:.1f}%" if x <= 1 else f"{x:.1f}%")
    df_display['sharpe'] = df_display['sharpe'].apply(lambda x: f"{x:.2f}")
    df_display['pred_std'] = df_display['pred_std'].apply(lambda x: f"{x:.4f}")
    df_display['pct_positive'] = df_display['pct_positive'].apply(lambda x: f"{x:.1f}%")
    df_display['healthy_pct'] = df_display['healthy_pct'].apply(lambda x: f"{x:.1f}%")
    df_display['unidirectional_pct'] = df_display['unidirectional_pct'].apply(lambda x: f"{x:.1f}%")
    df_display['collapse_total'] = df_display['collapse_total'].apply(lambda x: f"{x:.1f}%")
    df_display['mae'] = df_display['mae'].apply(lambda x: f"{x:.3f}")
    
    # Rename columns for display
    df_display.columns = ['Experiment', 'Epoch', 'DirAcc', 'Sharpe', 'PredStd', 
                          'Pct+', 'Healthy%', 'Unidir%', 'Collapse%', 'MAE']
    
    print(f"\n{'='*90}")
    print(f"EXPERIMENT COMPARISON (best by {args.metric}, min_epoch={args.min_epoch})")
    print(f"{'='*90}\n")
    print(df_display.to_string(index=False))
    print()
    
    # Save to CSV if requested
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved to: {args.csv}")
    
    # Quick insights
    print(f"{'='*90}")
    print("INSIGHTS")
    print(f"{'='*90}")
    
    best_dir = df.loc[df['dir_acc'].idxmax()]
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    best_healthy = df.loc[df['healthy_pct'].idxmax()]
    lowest_unidir = df.loc[df['unidirectional_pct'].idxmin()]
    lowest_collapse = df.loc[(df['weak_collapse_pct'] + df['strong_collapse_pct']).idxmin()]
    
    print(f"  Best DirAcc:       {best_dir['experiment']} ({best_dir['dir_acc']*100:.1f}%)")
    print(f"  Best Sharpe:       {best_sharpe['experiment']} ({best_sharpe['sharpe']:.2f})")
    print(f"  Best Healthy%:     {best_healthy['experiment']} ({best_healthy['healthy_pct']:.1f}%)")
    print(f"  Lowest Unidir%:    {lowest_unidir['experiment']} ({lowest_unidir['unidirectional_pct']:.1f}%)")
    print(f"  Lowest Collapse%:  {lowest_collapse['experiment']} ({lowest_collapse['weak_collapse_pct'] + lowest_collapse['strong_collapse_pct']:.1f}%)")
    print()


if __name__ == '__main__':
    main()