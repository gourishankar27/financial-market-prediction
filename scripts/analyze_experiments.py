#!/usr/bin/env python3
"""
analyze_experiments.py

Phase-agnostic experiment analysis and ranking tool.
Supports single or multiple phases, flexible metric sorting, and filtering.

Usage:
    # Single phase
    python analyze_experiments.py --phases 02b_vintage_sweep

    # Multiple phases
    python analyze_experiments.py --phases 02_vintage_baseline 02b_vintage_sweep

    # Custom sorting priority
    python analyze_experiments.py --phases 02b_vintage_sweep \
        --sort-by dir_acc healthy_pct sharpe_ratio --top 10

    # Filter out collapsed models
    python analyze_experiments.py --phases 02b_vintage_sweep \
        --no-collapse --min-healthy 40

    # Save results
    python analyze_experiments.py --phases 02b_vintage_sweep \
        --output results/top_configs.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import json


def load_experiments(phases):
    """Load and combine experiments from multiple phases."""
    dfs = []
    for phase in phases:
        exp_dir = Path('experiments') / phase
        summary_file = exp_dir / 'experiments_summary.csv'
        
        if not summary_file.exists():
            print(f"Warning: {summary_file} not found, skipping {phase}")
            continue
        
        df = pd.read_csv(summary_file)
        df['phase'] = phase
        dfs.append(df)
        print(f"Loaded {len(df)} experiments from {phase}")
    
    if not dfs:
        print("Error: No valid experiment data found")
        sys.exit(1)
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def filter_experiments(df, args):
    """Apply filtering criteria."""
    original_count = len(df)
    
    # Only evaluated experiments
    if args.evaluated_only:
        df = df[df['evaluated'] == True]
    
    # Collapse filtering
    if args.no_collapse:
        df = df[df['has_any_collapse'] == False]
    elif args.no_strong_collapse:
        df = df[df['has_strong_collapse'] == False]
    
    # Threshold filtering
    if args.min_healthy is not None:
        df = df[df['healthy_pct'] >= args.min_healthy]
    
    if args.min_dir_acc is not None:
        df = df[df['dir_acc'] >= args.min_dir_acc]
    
    if args.max_problematic is not None:
        df = df[df['problematic_pct'] <= args.max_problematic]
    
    # Remove monthly experiments if requested
    if args.no_monthly:
        df = df[~df['experiment_name'].str.contains('monthly', case=False, na=False)]
    
    filtered_count = len(df)
    if filtered_count < original_count:
        print(f"Filtered: {original_count} → {filtered_count} experiments")
    
    return df


def rank_experiments(df, sort_by, ascending):
    """Sort experiments by specified metrics."""
    # Validate metrics exist
    missing = [m for m in sort_by if m not in df.columns]
    if missing:
        print(f"Warning: Metrics not found: {missing}")
        sort_by = [m for m in sort_by if m in df.columns]
    
    if not sort_by:
        print("Error: No valid sort metrics")
        sys.exit(1)
    
    # Sort (all metrics descending by default except losses)
    ascending_flags = [ascending] * len(sort_by) if isinstance(ascending, bool) else ascending
    df_sorted = df.sort_values(by=sort_by, ascending=ascending_flags)
    
    return df_sorted


def print_summary(df, top_n, display_cols):
    """Print formatted summary table."""
    print("\n" + "="*100)
    print(f"TOP {top_n} EXPERIMENTS")
    print("="*100)
    
    display_df = df[display_cols].head(top_n)
    
    # Format percentages and floats
    for col in display_df.columns:
        if 'pct' in col or col.endswith('_pct'):
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        elif col in ['dir_acc', 'sharpe_ratio', 'alpha', 'auc_roc', 'learning_rate']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        elif col in ['test_mse', 'test_rmse', 'test_mae']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print()


def print_detailed_comparison(df, top_n):
    """Print detailed metrics including temporal quality breakdown."""
    print("\n" + "="*100)
    print(f"DETAILED ANALYSIS - TOP {top_n}")
    print("="*100)
    
    for idx, row in df.head(top_n).iterrows():
        phase = row['phase']
        exp_name = row['experiment_name']
        
        # Load full metrics
        eval_path = Path(f'experiments/{phase}/{exp_name}/evaluation/evaluation_metrics.json')
        if not eval_path.exists():
            print(f"\n{exp_name}: evaluation metrics not found")
            continue
            
        with open(eval_path) as f:
            m = json.load(f)
        
        print(f"\n{exp_name}:")
        print(f"  Config: h={row['hidden_size']}, dropout={row['dropout']}, lr={row.get('learning_rate', 'N/A')}")
        print(f"  Phase: {phase}")
        
        print(f"\n  Temporal Quality:")
        print(f"    Healthy:        {m['mode_stats']['healthy_pct']:6.1f}% ({m['mode_stats']['healthy_days']:4d} days)")
        print(f"    Degraded:       {m['mode_stats']['degraded_pct']:6.1f}% ({m['mode_stats']['degraded_days']:4d} days)")
        print(f"    Unidirectional: {m['mode_stats']['unidirectional_pct']:6.1f}% ({m['mode_stats']['unidirectional_days']:4d} days)")
        print(f"    Weak collapse:  {m['mode_stats']['weak_collapse_pct']:6.1f}% ({m['mode_stats']['weak_collapse_days']:4d} days)")
        print(f"    Strong collapse:{m['mode_stats']['strong_collapse_pct']:6.1f}% ({m['mode_stats']['strong_collapse_days']:4d} days)")
        
        print(f"\n  Performance Metrics:")
        print(f"    Directional accuracy: {m['financial_metrics']['directional_accuracy']:.4f}")
        print(f"    Sharpe ratio:         {m['financial_metrics']['sharpe_ratio']:.4f}")
        print(f"    AUC-ROC:              {m['financial_metrics']['auc_roc']:.4f}")
        print(f"    Hit rate:             {m['financial_metrics']['hit_rate']:.4f}")
        print(f"    Num trades:           {m['financial_metrics']['num_trades']}")
        
        print(f"\n  Classification Metrics:")
        print(f"    Precision:            {m['financial_metrics']['precision']:.4f}")
        print(f"    Recall:               {m['financial_metrics']['recall']:.4f}")
        print(f"    F1 score:             {m['financial_metrics']['f1_score']:.4f}")
        
        print(f"\n  Statistical Metrics:")
        print(f"    RMSE:                 {m['statistical_metrics']['rmse']:.6f}")
        print(f"    MAE:                  {m['statistical_metrics']['mae']:.6f}")
        print(f"    R²:                   {m['statistical_metrics']['r2']:.4f}")
        print(f"    Mean error:           {m['statistical_metrics']['mean_error']:.6f}")
        
        print(f"\n  Residual Diagnostics:")
        print(f"    Normality p-value:    {m['residual_diagnostics']['normality_p_value']:.4f}")
        print(f"    ACF lag-1:            {m['residual_diagnostics']['acf_lag1']:.4f}")
        print(f"    Skewness:             {m['residual_diagnostics']['skewness']:.4f}")
        print(f"    Kurtosis:             {m['residual_diagnostics']['kurtosis']:.4f}")
        
        print("\n" + "-"*100)
    
    print()



def print_statistics(df, phases):
    """Print summary statistics across phases."""
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    if len(phases) > 1:
        print("\nBy Phase:")
        phase_stats = df.groupby('phase').agg({
            'experiment_name': 'count',
            'has_strong_collapse': 'sum',
            'has_any_collapse': 'sum',
            'dir_acc': ['mean', 'std', 'max'],
            'healthy_pct': ['mean', 'max']
        }).round(4)
        print(phase_stats)
    
    print(f"\nOverall:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Strong collapse: {df['has_strong_collapse'].sum()} ({df['has_strong_collapse'].sum()/len(df)*100:.1f}%)")
    print(f"  Any collapse: {df['has_any_collapse'].sum()} ({df['has_any_collapse'].sum()/len(df)*100:.1f}%)")
    print(f"\nMetric Ranges:")
    for metric in ['dir_acc', 'sharpe_ratio', 'healthy_pct', 'auc_roc']:
        if metric in df.columns:
            print(f"  {metric:15s}: {df[metric].min():.4f} - {df[metric].max():.4f} (mean: {df[metric].mean():.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and rank TFT experiments across phases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Phase selection
    parser.add_argument('--phases', nargs='+', required=True,
                        help='Phase directories to analyze (e.g., 02_vintage_baseline 02b_vintage_sweep)')
    
    # Sorting
    parser.add_argument('--sort-by', nargs='+', 
                        default=['composite_score', 'healthy_pct', 'dir_acc'],
                        help='Metrics to sort by (default: composite_score healthy_pct dir_acc)')
    parser.add_argument('--ascending', action='store_true',
                        help='Sort ascending instead of descending')
    
    # Filtering
    parser.add_argument('--evaluated-only', action='store_true',
                        help='Only include evaluated experiments')
    parser.add_argument('--no-collapse', action='store_true',
                        help='Exclude any experiments with collapse')
    parser.add_argument('--no-strong-collapse', action='store_true',
                        help='Exclude experiments with strong collapse only')
    parser.add_argument('--no-monthly', action='store_true',
                        help='Exclude monthly experiments')
    parser.add_argument('--min-healthy', type=float,
                        help='Minimum healthy_pct threshold')
    parser.add_argument('--min-dir-acc', type=float,
                        help='Minimum directional accuracy threshold')
    parser.add_argument('--max-problematic', type=float,
                        help='Maximum problematic_pct threshold')
    
    # Display
    parser.add_argument('--top', type=int, default=15,
                        help='Number of top experiments to display (default: 15)')
    parser.add_argument('--display-cols', nargs='+',
                        default=['experiment_name', 'phase', 'hidden_size', 'dropout', 
                                'learning_rate', 'dir_acc', 'healthy_pct', 'sharpe_ratio',
                                'auc_roc', 'has_strong_collapse'],
                        help='Columns to display in summary')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed metrics including temporal quality breakdown')
    
    # Output
    parser.add_argument('--output', type=str,
                        help='Save results to CSV file')
    parser.add_argument('--save-top', type=int,
                        help='Number of top experiments to save (default: all filtered)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, skip ranking')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading experiments from phases: {', '.join(args.phases)}")
    df = load_experiments(args.phases)
    
    # Filter
    df_filtered = filter_experiments(df, args)
    
    if len(df_filtered) == 0:
        print("Error: No experiments remain after filtering")
        sys.exit(1)
    
    # Print statistics
    print_statistics(df_filtered, args.phases)
    
    if not args.stats_only:
        # Rank
        df_ranked = rank_experiments(df_filtered, args.sort_by, args.ascending)
        
        # Display
        print_summary(df_ranked, args.top, args.display_cols)
        
        # Display detailed if requested
        if args.detailed:
            print_detailed_comparison(df_ranked, args.top)
        
        # Save
        if args.output:
            save_n = args.save_top if args.save_top else len(df_ranked)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df_ranked.head(save_n).to_csv(output_path, index=False)
            print(f"Saved top {save_n} experiments to {output_path}")


if __name__ == '__main__':
    main()