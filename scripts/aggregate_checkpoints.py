#!/usr/bin/env python3
"""
Aggregate and analyze sweep results across multiple experiments.

Works with any phase that uses evaluate_checkpoints.py to generate
checkpoint_comparison.csv files.

Usage:
    # Analyze a sweep phase
    python aggregate_experiments.py experiments/06a_weekly_sweep
    
    # Compare multiple phases
    python aggregate_experiments.py experiments/06a_weekly_sweep experiments/02b_vintage_sweep
    
    # Filter to specific checkpoint metric
    python aggregate_experiments.py experiments/06a_weekly_sweep --ckpt-metric val_dir_acc
    
    # Custom output
    python aggregate_experiments.py experiments/06a_weekly_sweep --output results/weekly_analysis.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re


# ============================================================================
# DATA LOADING
# ============================================================================

def load_checkpoint_comparison(exp_dir):
    """Load checkpoint_comparison.csv from an experiment directory."""
    # Check both possible locations
    csv_path = exp_dir / 'evaluation' / 'checkpoint_comparison.csv'
    if not csv_path.exists():
        csv_path = exp_dir / 'checkpoint_comparison.csv'
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    df['experiment'] = exp_dir.name
    df['phase'] = exp_dir.parent.name
    return df


def load_phase_results(phase_dir):
    """Load all checkpoint comparisons from a phase directory or single experiment."""
    phase_dir = Path(phase_dir)
    
    # Check if this is a single experiment (has evaluation/ or checkpoints/ directly)
    if (phase_dir / 'evaluation').exists() or (phase_dir / 'checkpoints').exists():
        df = load_checkpoint_comparison(phase_dir)
        if df is not None:
            df['phase'] = phase_dir.parent.name  # Parent is the phase
            return df
        return None
    
    # Otherwise treat as phase directory containing multiple experiments
    all_dfs = []
    for exp_dir in sorted(phase_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        df = load_checkpoint_comparison(exp_dir)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        return None
    
    return pd.concat(all_dfs, ignore_index=True)


def parse_experiment_name(exp_name):
    """Extract hyperparameters from experiment name."""
    params = {}
    
    # Common patterns: h16, enc8, d015, bs32, drop0.10, etc.
    patterns = [
        (r'h(\d+)', 'hidden_size'),
        (r'enc(\d+)', 'encoder_length'),
        (r'd(\d{2,3})', 'dropout'),  # d015 or d10
        (r'drop([\d.]+)', 'dropout'),  # drop0.10
        (r'bs(\d+)', 'batch_size'),
        (r'lr([\d.e-]+)', 'learning_rate'),
    ]
    
    for pattern, param_name in patterns:
        match = re.search(pattern, exp_name, re.IGNORECASE)
        if match:
            val = match.group(1)
            # Convert dropout like "015" to 0.15
            if param_name == 'dropout' and val.isdigit() and len(val) >= 2:
                val = float(val) / 100
            else:
                try:
                    val = float(val) if '.' in str(val) or 'e' in str(val).lower() else int(val)
                except:
                    pass
            params[param_name] = val
    
    return params


# ============================================================================
# ANALYSIS
# ============================================================================

def get_best_per_experiment(df, metric='dir_acc', higher_is_better=True):
    """Get best checkpoint for each experiment based on a metric."""
    if higher_is_better:
        idx = df.groupby('experiment')[metric].idxmax()
    else:
        idx = df.groupby('experiment')[metric].idxmin()
    
    return df.loc[idx].copy()


def get_best_overall(df, metric='dir_acc', n=10, higher_is_better=True):
    """Get top N checkpoints across all experiments."""
    if higher_is_better:
        return df.nlargest(n, metric)
    else:
        return df.nsmallest(n, metric)


def analyze_by_checkpoint_metric(df):
    """Analyze which checkpoint selection metric produces best test results."""
    if 'ckpt_metric' not in df.columns:
        return None
    
    metrics = ['dir_acc', 'sharpe', 'healthy_pct', 'total_return']
    metrics = [m for m in metrics if m in df.columns]
    
    return df.groupby('ckpt_metric')[metrics].agg(['mean', 'std']).round(4)


def analyze_by_hyperparameter(df):
    """Analyze how hyperparameters affect performance."""
    # Extract hyperparameters from experiment names
    params_list = []
    for exp in df['experiment'].unique():
        params = parse_experiment_name(exp)
        params['experiment'] = exp
        params_list.append(params)
    
    params_df = pd.DataFrame(params_list)
    
    if params_df.empty or len(params_df.columns) <= 1:
        return None
    
    # Merge with best-per-experiment results
    best = get_best_per_experiment(df)
    merged = best.merge(params_df, on='experiment')
    
    # Compute correlations between hyperparams and metrics
    hp_cols = [c for c in params_df.columns if c != 'experiment']
    metric_cols = ['dir_acc', 'sharpe', 'healthy_pct']
    metric_cols = [m for m in metric_cols if m in merged.columns]
    
    if not hp_cols or not metric_cols:
        return None
    
    correlations = {}
    for hp in hp_cols:
        if merged[hp].dtype in [np.float64, np.int64]:
            correlations[hp] = {}
            for metric in metric_cols:
                corr = merged[hp].corr(merged[metric])
                correlations[hp][metric] = round(corr, 3)
    
    return pd.DataFrame(correlations).T


def summarize_phase(df):
    """Generate summary statistics for a phase."""
    best = get_best_per_experiment(df)
    
    summary = {
        'num_experiments': len(best),
        'num_checkpoints_total': len(df),
    }
    
    for metric in ['dir_acc', 'sharpe', 'healthy_pct', 'total_return', 'num_negative']:
        if metric in best.columns:
            summary[f'{metric}_mean'] = best[metric].mean()
            summary[f'{metric}_std'] = best[metric].std()
            summary[f'{metric}_max'] = best[metric].max()
            summary[f'{metric}_min'] = best[metric].min()
    
    return summary


# ============================================================================
# REPORTING
# ============================================================================

def print_report(df, phase_name):
    """Print analysis report for a phase."""
    print(f"\n{'='*70}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*70}")
    
    # Summary
    summary = summarize_phase(df)
    print(f"\nSUMMARY:")
    print(f"  Experiments: {summary['num_experiments']}")
    print(f"  Total checkpoints evaluated: {summary['num_checkpoints_total']}")
    
    # Best per experiment
    print(f"\nBEST CHECKPOINT PER EXPERIMENT (by dir_acc):")
    print("-"*70)
    best = get_best_per_experiment(df)
    cols = ['experiment', 'epoch', 'ckpt_metric', 'dir_acc', 'sharpe', 'healthy_pct', 'num_negative']
    cols = [c for c in cols if c in best.columns]
    print(best[cols].sort_values('dir_acc', ascending=False).to_string(index=False))
    
    # Top 5 overall
    print(f"\nTOP 5 CHECKPOINTS OVERALL:")
    print("-"*70)
    top5 = get_best_overall(df, n=5)
    print(top5[cols].to_string(index=False))
    
    # By checkpoint metric
    print(f"\nPERFORMANCE BY CHECKPOINT SELECTION METRIC:")
    print("-"*70)
    by_ckpt = analyze_by_checkpoint_metric(df)
    if by_ckpt is not None:
        print(by_ckpt.to_string())
    
    # By hyperparameter
    print(f"\nHYPERPARAMETER CORRELATIONS WITH METRICS:")
    print("-"*70)
    hp_corr = analyze_by_hyperparameter(df)
    if hp_corr is not None and not hp_corr.empty:
        print(hp_corr.to_string())
    else:
        print("  (Could not extract hyperparameters from experiment names)")
    
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and analyze sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('phases', nargs='+',
                        help='Path(s) to phase directories')
    parser.add_argument('--ckpt-metric', type=str, default=None,
                        help='Filter to specific checkpoint metric (e.g., val_dir_acc)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path for combined best-per-experiment results')
    parser.add_argument('--output-all', type=str, default=None,
                        help='Output CSV path for all checkpoint results')
    parser.add_argument('--metric', type=str, default='dir_acc',
                        help='Metric to use for selecting best (default: dir_acc)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top results to show (default: 10)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Load all phases
    all_dfs = []
    for phase_path in args.phases:
        phase_path = Path(phase_path)
        df = load_phase_results(phase_path)
        
        if df is None:
            print(f"Warning: No results found in {phase_path}")
            continue
        
        print(f"Loaded: {phase_path.name} ({df['experiment'].nunique()} experiments, {len(df)} checkpoints)")
        all_dfs.append(df)
    
    if not all_dfs:
        print("No results loaded!")
        return
    
    # Combine all results
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Filter by checkpoint metric if specified
    if args.ckpt_metric:
        combined = combined[combined['ckpt_metric'] == args.ckpt_metric]
        print(f"Filtered to ckpt_metric={args.ckpt_metric}: {len(combined)} checkpoints")
    
    # Print report for each phase
    if not args.quiet:
        for phase_path in args.phases:
            phase_name = Path(phase_path).name
            phase_df = combined[combined['phase'] == phase_name]
            if not phase_df.empty:
                print_report(phase_df, phase_name)
        
        # Cross-phase comparison if multiple
        if len(args.phases) > 1:
            print(f"\n{'='*70}")
            print("CROSS-PHASE COMPARISON")
            print(f"{'='*70}")
            
            phase_summaries = []
            for phase_path in args.phases:
                phase_name = Path(phase_path).name
                phase_df = combined[combined['phase'] == phase_name]
                if not phase_df.empty:
                    summary = summarize_phase(phase_df)
                    summary['phase'] = phase_name
                    phase_summaries.append(summary)
            
            if phase_summaries:
                summary_df = pd.DataFrame(phase_summaries)
                cols = ['phase', 'num_experiments', 'dir_acc_mean', 'dir_acc_std', 
                        'sharpe_mean', 'healthy_pct_mean']
                cols = [c for c in cols if c in summary_df.columns]
                print(summary_df[cols].to_string(index=False))
    
    # Save outputs
    if args.output:
        best = get_best_per_experiment(combined, metric=args.metric)
        best.to_csv(args.output, index=False)
        print(f"\nSaved best-per-experiment to: {args.output}")
    
    if args.output_all:
        combined.to_csv(args.output_all, index=False)
        print(f"Saved all checkpoints to: {args.output_all}")


if __name__ == "__main__":
    main()