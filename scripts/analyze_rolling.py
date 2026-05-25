#!/usr/bin/env python3
"""
Analyze rolling evaluation results and compare to fixed-split baselines.

Usage:
    # Analyze single rolling experiment
    python scripts/analyze_rolling.py experiments/06b_rolling/daily_baseline
    
    # Compare rolling to fixed-split baseline
    python scripts/analyze_rolling.py experiments/06b_rolling/daily_baseline \
        --compare experiments/02b_vintage_sweep/baseline_h16_drop0.10
    
    # Compare multiple rolling experiments
    python scripts/analyze_rolling.py experiments/06b_rolling/daily_baseline \
        experiments/06b_rolling/weekly_baseline \
        --output reports/rolling_comparison.csv
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# DATA LOADING
# ============================================================================

def load_rolling_results(experiment_dir):
    """Load rolling evaluation results from experiment directory."""
    experiment_dir = Path(experiment_dir)
    
    # Try full results CSV first
    full_csv = experiment_dir / 'rolling_results_full.csv'
    if full_csv.exists():
        df = pd.read_csv(full_csv)
        
        # Check if metrics are nested (JSON strings) and flatten them
        nested_cols = ['prediction_stats', 'statistical_metrics', 'financial_metrics', 
                       'residual_diagnostics', 'mode_stats']
        
        for col in nested_cols:
            if col in df.columns:
                try:
                    # Parse JSON/dict strings and expand to columns
                    expanded = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    expanded_df = pd.json_normalize(expanded)
                    # Prefix columns to avoid conflicts (except for common metrics)
                    if col == 'financial_metrics':
                        # Keep these as-is (most commonly used)
                        df = pd.concat([df, expanded_df], axis=1)
                    elif col == 'mode_stats':
                        # Keep these as-is too
                        df = pd.concat([df, expanded_df], axis=1)
                    elif col == 'statistical_metrics':
                        df = pd.concat([df, expanded_df], axis=1)
                    elif col == 'prediction_stats':
                        # Prefix to avoid confusion
                        expanded_df.columns = [f'pred_{c}' for c in expanded_df.columns]
                        df = pd.concat([df, expanded_df], axis=1)
                    # Drop the original nested column
                    df = df.drop(columns=[col])
                except Exception as e:
                    print(f"Warning: Could not expand {col}: {e}")
        
        df['experiment'] = experiment_dir.name
        return df
    
    # Fall back to loading individual fold metrics
    folds = []
    for fold_dir in sorted(experiment_dir.glob('fold_*')):
        metrics_path = fold_dir / 'evaluation' / 'evaluation_metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            # Flatten nested dicts
            flat_metrics = flatten_metrics(metrics)
            flat_metrics['fold_id'] = fold_dir.name
            flat_metrics['test_year'] = int(fold_dir.name.split('_')[1])
            folds.append(flat_metrics)
    
    if not folds:
        raise FileNotFoundError(f"No fold results found in {experiment_dir}")
    
    df = pd.DataFrame(folds)
    df['experiment'] = experiment_dir.name
    return df


def flatten_metrics(metrics, prefix=''):
    """Flatten nested metric dictionaries."""
    flat = {}
    for key, value in metrics.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, f"{new_key}_"))
        elif isinstance(value, (int, float, bool)):
            flat[new_key] = value
    return flat


def load_fixed_split_results(experiment_dir):
    """Load fixed-split evaluation results."""
    experiment_dir = Path(experiment_dir)
    metrics_path = experiment_dir / 'evaluation' / 'evaluation_metrics.json'
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"No evaluation metrics found at {metrics_path}")
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    flat = flatten_metrics(metrics)
    flat['experiment'] = experiment_dir.name
    return flat


# ============================================================================
# ANALYSIS
# ============================================================================

def compute_summary_stats(df, metrics=None):
    """Compute summary statistics across folds."""
    if metrics is None:
        metrics = [
            'directional_accuracy', 'sharpe_ratio', 'total_return',
            'healthy_pct', 'unidirectional_pct', 'strong_collapse_pct'
        ]
    
    # Filter to metrics that exist
    metrics = [m for m in metrics if m in df.columns]
    
    summary = df[metrics].agg(['mean', 'std', 'min', 'max']).T
    summary['cv'] = summary['std'] / summary['mean'].abs()
    summary['range'] = summary['max'] - summary['min']
    
    return summary


def analyze_regime_performance(df):
    """Analyze performance by market regime/year."""
    # Define known regimes
    regimes = {
        2016: 'Post-election rally',
        2017: 'Low vol bull',
        2018: 'Vol spike / correction',
        2019: 'Recovery',
        2020: 'COVID crash & recovery',
        2021: 'Meme stocks / inflation',
        2022: 'Bear market / rate hikes',
        2023: 'AI rally',
        2024: 'Continued rally',
    }
    
    if 'test_year' not in df.columns:
        return None
    
    regime_df = df.copy()
    regime_df['regime'] = regime_df['test_year'].map(regimes)
    
    return regime_df[['test_year', 'regime', 'directional_accuracy', 
                       'sharpe_ratio', 'healthy_pct', 'total_return']]


def compare_to_fixed(rolling_df, fixed_metrics):
    """Compare rolling results to fixed-split baseline."""
    comparison = {}
    
    key_metrics = ['directional_accuracy', 'sharpe_ratio', 'total_return', 
                   'healthy_pct', 'unidirectional_pct']
    
    for metric in key_metrics:
        if metric in rolling_df.columns and metric in fixed_metrics:
            rolling_mean = rolling_df[metric].mean()
            rolling_std = rolling_df[metric].std()
            fixed_val = fixed_metrics[metric]
            
            comparison[metric] = {
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
                'fixed': fixed_val,
                'diff': rolling_mean - fixed_val,
                'diff_pct': (rolling_mean - fixed_val) / abs(fixed_val) * 100 if fixed_val != 0 else 0
            }
    
    return pd.DataFrame(comparison).T


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_fold_performance(df, output_path=None):
    """Plot performance metrics across folds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('directional_accuracy', 'Directional Accuracy', axes[0, 0]),
        ('sharpe_ratio', 'Sharpe Ratio', axes[0, 1]),
        ('healthy_pct', 'Healthy %', axes[1, 0]),
        ('total_return', 'Total Return', axes[1, 1]),
    ]
    
    for metric, title, ax in metrics:
        if metric not in df.columns:
            continue
            
        x = df['test_year'] if 'test_year' in df.columns else range(len(df))
        y = df[metric]
        
        ax.bar(x, y, alpha=0.7, edgecolor='black')
        ax.axhline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.3f}')
        ax.fill_between([min(x)-0.5, max(x)+0.5], 
                        y.mean() - y.std(), y.mean() + y.std(),
                        alpha=0.2, color='red', label=f'±1 std: {y.std():.3f}')
        ax.set_xlabel('Test Year')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f"Rolling Evaluation: {df['experiment'].iloc[0]}", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rolling_comparison(dfs, output_path=None, presentation=False, name_map=None, market_baselines=None):
    """Compare multiple rolling experiments.
    
    Args:
        dfs: List of DataFrames with rolling results
        output_path: Path to save figure
        presentation: If True, use simplified 2-panel layout
        name_map: Dict mapping experiment names to display names
        market_baselines: Dict mapping experiment names to market positive rates (for dir acc)
    """
    if presentation:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        metrics = [
            ('directional_accuracy', 'Directional Accuracy'),
            ('sharpe_ratio', 'Sharpe Ratio'),
        ]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = [
            ('directional_accuracy', 'Directional Accuracy'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('healthy_pct', 'Healthy %'),
        ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#5C946E']  # Presentation colors
    
    for (metric, title), ax in zip(metrics, axes):
        data = []
        labels = []
        exp_names = []  # Track original names for baseline lookup
        for df in dfs:
            if metric in df.columns:
                exp_name = df['experiment'].iloc[0]
                display_name = name_map.get(exp_name, exp_name) if name_map else exp_name
                data.append(df[metric].values)
                labels.append(display_name)
                exp_names.append(exp_name)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Style boxes
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            
            ax.set_ylabel(title, fontsize=12 if presentation else 10)
            ax.set_title(title, fontsize=14 if presentation else 12, 
                        fontweight='bold' if presentation else 'normal')
            ax.grid(axis='y', alpha=0.3)
            
            # Reference lines
            if metric == 'directional_accuracy':
                ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
                
                # Add market baseline lines per boxplot
                if market_baselines:
                    n_boxes = len(exp_names)
                    for i, exp_name in enumerate(exp_names):
                        if exp_name in market_baselines:
                            baseline = market_baselines[exp_name]
                            # Draw short horizontal line spanning just this boxplot
                            ax.hlines(baseline, i + 0.6, i + 1.4, colors='#E63946', 
                                     linestyles=':', linewidth=2, label='Market +rate' if i == 0 else '')
                            # Annotation: left side for last box, right side for others
                            if i == n_boxes - 1:
                                ax.annotate(f'{baseline:.1%}', xy=(i + 0.58, baseline), 
                                           fontsize=9, color='#E63946', va='center', ha='right')
                            else:
                                ax.annotate(f'{baseline:.1%}', xy=(i + 1.42, baseline), 
                                           fontsize=9, color='#E63946', va='center')
                    
            elif metric == 'sharpe_ratio':
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            
            if not presentation:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add legend for directional accuracy panel
            if metric == 'directional_accuracy' and market_baselines:
                ax.legend(loc='lower left', fontsize=9)
    
    plt.suptitle("Rolling Evaluation Comparison", fontsize=14, 
                 fontweight='bold' if presentation else 'normal')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200 if presentation else 150, 
                    bbox_inches='tight', facecolor='white')
        print(f"Saved comparison plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze rolling evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('experiments', nargs='+',
                        help='Path(s) to rolling experiment directories')
    parser.add_argument('--compare', type=str, default=None,
                        help='Fixed-split experiment to compare against')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path for combined results')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--presentation', action='store_true',
                        help='Simplified plot for presentation (2 panels, clean names)')
    parser.add_argument('--name-map', type=str, default=None,
                        help='JSON mapping of experiment names, e.g. \'{"daily_h16_baseline":"Daily"}\'')
    parser.add_argument('--market-baseline', type=str, default=None,
                        help='JSON mapping of experiment names to market positive rates, e.g. \'{"daily_h16_baseline":0.539}\'')


    
    args = parser.parse_args()
    
    # Load rolling results
    all_dfs = []
    for exp_path in args.experiments:
        try:
            df = load_rolling_results(exp_path)
            all_dfs.append(df)
            print(f"Loaded: {exp_path} ({len(df)} folds)")
        except Exception as e:
            print(f"Warning: Could not load {exp_path}: {e}")
    
    if not all_dfs:
        print("No results loaded!")
        return
    
    # Analyze each experiment
    for df in all_dfs:
        exp_name = df['experiment'].iloc[0]
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp_name}")
        print('='*70)
        
        # Summary stats
        print("\nSUMMARY STATISTICS:")
        print("-"*50)
        summary = compute_summary_stats(df)
        print(summary.to_string())
        
        # Regime analysis
        regime_df = analyze_regime_performance(df)
        if regime_df is not None:
            print("\n\nPERFORMANCE BY REGIME:")
            print("-"*50)
            print(regime_df.to_string(index=False))
        
        # Plot
        if not args.no_plots:
            plot_dir = Path(args.plot_dir) if args.plot_dir else Path(args.experiments[0])
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_fold_performance(df, plot_dir / f'{exp_name}_fold_performance.png')
    
    # Compare to fixed split if provided
    if args.compare:
        try:
            fixed = load_fixed_split_results(args.compare)
            print(f"\n{'='*70}")
            print("COMPARISON TO FIXED SPLIT")
            print('='*70)
            print(f"Fixed experiment: {args.compare}")
            
            for df in all_dfs:
                print(f"\n{df['experiment'].iloc[0]} vs Fixed:")
                print("-"*50)
                comparison = compare_to_fixed(df, fixed)
                print(comparison.to_string())
        except Exception as e:
            print(f"Warning: Could not load fixed comparison: {e}")
    
    # Compare multiple rolling experiments
    if len(all_dfs) > 1:
        print(f"\n{'='*70}")
        print("CROSS-EXPERIMENT COMPARISON")
        print('='*70)
        
        comparison_rows = []
        for df in all_dfs:
            row = {'experiment': df['experiment'].iloc[0]}
            for metric in ['directional_accuracy', 'sharpe_ratio', 'healthy_pct']:
                if metric in df.columns:
                    row[f'{metric}_mean'] = df[metric].mean()
                    row[f'{metric}_std'] = df[metric].std()
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        print(comparison_df.to_string(index=False))
        
        if not args.no_plots:
            # Parse name map if provided
            name_map = None
            if args.name_map:
                import json as json_module
                name_map = json_module.loads(args.name_map)
            
            # Parse market baselines if provided
            market_baselines = None
            if args.market_baseline:
                import json as json_module
                market_baselines = json_module.loads(args.market_baseline)
            
            plot_dir = Path(args.plot_dir) if args.plot_dir else Path(args.experiments[0]).parent
            suffix = '_presentation' if args.presentation else ''
            plot_rolling_comparison(
                all_dfs, 
                plot_dir / f'rolling_comparison{suffix}.png',
                presentation=args.presentation,
                name_map=name_map,
                market_baselines=market_baselines
            )
    
    # Save combined results
    if args.output:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved combined results to: {args.output}")


if __name__ == "__main__":
    main()