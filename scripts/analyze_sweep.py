#!/usr/bin/env python3
"""
Analyze overnight sweep results - aggregate by factors.

Usage:
    python analyze_sweep.py experiments/10_overnight_sweep/
"""

import argparse
import pandas as pd
from pathlib import Path
import re
import sys


def parse_experiment_name(name: str) -> dict:
    """Parse experiment name like 'weekly_3q_h1_s42' into factors."""
    # Pattern: {freq}_{quant}_{horizon}_{seed}
    match = re.match(r'(weekly|daily)_(\d+q)_h(\d+)_s(\d+)', name)
    if match:
        return {
            'frequency': match.group(1),
            'quantiles': match.group(2),
            'horizon': int(match.group(3)),
            'seed': int(match.group(4))
        }
    return None


def load_sweep_results(sweep_dir: Path, min_epoch: int = 20) -> pd.DataFrame:
    """Load all checkpoint_comparison.csv files from sweep."""
    results = []
    
    for exp_path in sorted(sweep_dir.iterdir()):
        if not exp_path.is_dir():
            continue
            
        csv_path = exp_path / 'evaluation' / 'checkpoint_comparison.csv'
        if not csv_path.exists():
            print(f"  Warning: {exp_path.name} missing evaluation")
            continue
        
        factors = parse_experiment_name(exp_path.name)
        if not factors:
            continue
        
        df = pd.read_csv(csv_path)
        df_filtered = df[df['epoch'] >= min_epoch]
        
        if df_filtered.empty:
            print(f"  Warning: {exp_path.name} no checkpoints >= epoch {min_epoch}")
            continue
        
        # Get best by dir_acc
        best = df_filtered.loc[df_filtered['dir_acc'].idxmax()]
        
        results.append({
            'experiment': exp_path.name,
            **factors,
            'epoch': int(best['epoch']),
            'dir_acc': best['dir_acc'],
            'sharpe': best['sharpe'],
            'healthy_pct': best['healthy_pct'],
            'unidirectional_pct': best['unidirectional_pct'],
            'weak_collapse_pct': best['weak_collapse_pct'],
            'strong_collapse_pct': best['strong_collapse_pct'],
            'pred_std': best['pred_std'],
            'pct_positive': best['pct_positive'],
            'mae': best['mae'],
        })
    
    return pd.DataFrame(results)


def print_factor_analysis(df: pd.DataFrame):
    """Print aggregated analysis by each factor."""
    
    df['collapse_pct'] = df['weak_collapse_pct'] + df['strong_collapse_pct']
    
    metrics = ['dir_acc', 'sharpe', 'healthy_pct', 'unidirectional_pct', 'collapse_pct', 'pred_std']
    
    print("\n" + "="*90)
    print("FACTOR ANALYSIS (mean ± std across seeds)")
    print("="*90)
    
    # By Frequency
    print("\n>>> BY FREQUENCY")
    freq_agg = df.groupby('frequency')[metrics].agg(['mean', 'std'])
    for freq in ['weekly', 'daily']:
        if freq in freq_agg.index:
            row = freq_agg.loc[freq]
            print(f"  {freq:8s}: DirAcc={row['dir_acc']['mean']*100:.1f}±{row['dir_acc']['std']*100:.1f}%  "
                  f"Sharpe={row['sharpe']['mean']:.2f}±{row['sharpe']['std']:.2f}  "
                  f"Healthy={row['healthy_pct']['mean']:.1f}%  "
                  f"Collapse={row['collapse_pct']['mean']:.1f}%")
    
    # By Quantiles
    print("\n>>> BY QUANTILES")
    quant_agg = df.groupby('quantiles')[metrics].agg(['mean', 'std'])
    for quant in ['3q', '7q']:
        if quant in quant_agg.index:
            row = quant_agg.loc[quant]
            print(f"  {quant:8s}: DirAcc={row['dir_acc']['mean']*100:.1f}±{row['dir_acc']['std']*100:.1f}%  "
                  f"Sharpe={row['sharpe']['mean']:.2f}±{row['sharpe']['std']:.2f}  "
                  f"Healthy={row['healthy_pct']['mean']:.1f}%  "
                  f"Collapse={row['collapse_pct']['mean']:.1f}%")
    
    # By Horizon
    print("\n>>> BY HORIZON")
    horizon_agg = df.groupby('horizon')[metrics].agg(['mean', 'std'])
    for h in [1, 3, 5]:
        if h in horizon_agg.index:
            row = horizon_agg.loc[h]
            print(f"  h{h:7d}: DirAcc={row['dir_acc']['mean']*100:.1f}±{row['dir_acc']['std']*100:.1f}%  "
                  f"Sharpe={row['sharpe']['mean']:.2f}±{row['sharpe']['std']:.2f}  "
                  f"Healthy={row['healthy_pct']['mean']:.1f}%  "
                  f"Collapse={row['collapse_pct']['mean']:.1f}%")
    
    # By Frequency × Quantiles (most important interaction)
    print("\n>>> BY FREQUENCY × QUANTILES")
    fq_agg = df.groupby(['frequency', 'quantiles'])[metrics].agg(['mean', 'std'])
    for (freq, quant), row in fq_agg.iterrows():
        print(f"  {freq}_{quant}: DirAcc={row['dir_acc']['mean']*100:.1f}±{row['dir_acc']['std']*100:.1f}%  "
              f"Sharpe={row['sharpe']['mean']:.2f}±{row['sharpe']['std']:.2f}  "
              f"Healthy={row['healthy_pct']['mean']:.1f}%  "
              f"Collapse={row['collapse_pct']['mean']:.1f}%")
    
    # By Frequency × Horizon
    print("\n>>> BY FREQUENCY × HORIZON")
    fh_agg = df.groupby(['frequency', 'horizon'])[metrics].agg(['mean', 'std'])
    for (freq, h), row in fh_agg.iterrows():
        print(f"  {freq}_h{h}: DirAcc={row['dir_acc']['mean']*100:.1f}±{row['dir_acc']['std']*100:.1f}%  "
              f"Sharpe={row['sharpe']['mean']:.2f}±{row['sharpe']['std']:.2f}  "
              f"Healthy={row['healthy_pct']['mean']:.1f}%  "
              f"Collapse={row['collapse_pct']['mean']:.1f}%")


def print_top_experiments(df: pd.DataFrame, n: int = 5):
    """Print top N experiments by different metrics."""
    
    df['collapse_pct'] = df['weak_collapse_pct'] + df['strong_collapse_pct']
    
    print("\n" + "="*90)
    print(f"TOP {n} EXPERIMENTS")
    print("="*90)
    
    # Top by dir_acc
    print(f"\n>>> By Directional Accuracy")
    top_dir = df.nlargest(n, 'dir_acc')[['experiment', 'dir_acc', 'sharpe', 'healthy_pct', 'collapse_pct']]
    for _, row in top_dir.iterrows():
        print(f"  {row['experiment']:30s} DirAcc={row['dir_acc']*100:.1f}%  Sharpe={row['sharpe']:.2f}  Healthy={row['healthy_pct']:.1f}%")
    
    # Top by Sharpe
    print(f"\n>>> By Sharpe Ratio")
    top_sharpe = df.nlargest(n, 'sharpe')[['experiment', 'dir_acc', 'sharpe', 'healthy_pct', 'collapse_pct']]
    for _, row in top_sharpe.iterrows():
        print(f"  {row['experiment']:30s} Sharpe={row['sharpe']:.2f}  DirAcc={row['dir_acc']*100:.1f}%  Healthy={row['healthy_pct']:.1f}%")
    
    # Top by Healthy%
    print(f"\n>>> By Healthy %")
    top_healthy = df.nlargest(n, 'healthy_pct')[['experiment', 'dir_acc', 'sharpe', 'healthy_pct', 'collapse_pct']]
    for _, row in top_healthy.iterrows():
        print(f"  {row['experiment']:30s} Healthy={row['healthy_pct']:.1f}%  DirAcc={row['dir_acc']*100:.1f}%  Collapse={row['collapse_pct']:.1f}%")
    
    # Lowest collapse
    print(f"\n>>> By Lowest Collapse")
    top_stable = df.nsmallest(n, 'collapse_pct')[['experiment', 'dir_acc', 'sharpe', 'healthy_pct', 'collapse_pct']]
    for _, row in top_stable.iterrows():
        print(f"  {row['experiment']:30s} Collapse={row['collapse_pct']:.1f}%  DirAcc={row['dir_acc']*100:.1f}%  Healthy={row['healthy_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze sweep results by factors')
    parser.add_argument('sweep_dir', type=str, help='Path to sweep directory')
    parser.add_argument('--min-epoch', type=int, default=20, help='Min epoch for best checkpoint')
    parser.add_argument('--top', type=int, default=5, help='Number of top experiments to show')
    parser.add_argument('--csv', type=str, help='Output CSV path')
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    
    print(f"Loading results from: {sweep_dir}")
    df = load_sweep_results(sweep_dir, args.min_epoch)
    
    if df.empty:
        print("No results found!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} experiments")
    
    # Factor analysis
    print_factor_analysis(df)
    
    # Top experiments
    print_top_experiments(df, args.top)
    
    # Save if requested
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved to: {args.csv}")
    
    print()


if __name__ == '__main__':
    main()
