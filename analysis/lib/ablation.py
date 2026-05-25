"""
Ablation analysis utilities for experiment comparison.

Usage:
    from lib.ablation import ablate, compare_groups
    
    # Compare impact of a single dimension
    ablate(df, vary='hard_routing_train', metrics=['directional_accuracy'])
    
    # Compare with filters
    ablate(df, vary='expert_type', 
           filter={'routing_strategy': 'vix_threshold'},
           metrics=['pred_std', 'healthy_pct'])
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path


# Default metrics for different analysis types
DEFAULT_PERFORMANCE_METRICS = ['directional_accuracy', 'sharpe_ratio', 'healthy_pct']
DEFAULT_BEHAVIOR_METRICS = ['pred_std', 'final_expert_weight_cosine', 'final_routing_entropy']
DEFAULT_TRAINING_METRICS = ['best_val_loss', 'total_epochs']


def ablate(
    df: pd.DataFrame,
    vary: str,
    metrics: Optional[List[str]] = None,
    filter: Optional[Dict[str, any]] = None,
    show_n: bool = True,
    plot: bool = True,
    figsize: Tuple[int, int] = (10, 4)
) -> pd.DataFrame:
    """
    Ablation analysis: vary one dimension, measure impact on metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Experiment data from load_experiments()
    vary : str
        Column name to vary (e.g., 'hard_routing_train', 'expert_type')
    metrics : list of str, optional
        Metrics to compare. Defaults to performance metrics.
    filter : dict, optional
        Filter conditions to apply first (e.g., {'routing_strategy': 'vix_threshold'})
    show_n : bool, default=True
        Show sample counts in output
    plot : bool, default=True
        Generate bar plot
    figsize : tuple, default=(10, 4)
        Figure size for plot
        
    Returns
    -------
    pd.DataFrame
        Summary table with mean/std/n for each metric by varied dimension
    """
    if metrics is None:
        metrics = DEFAULT_PERFORMANCE_METRICS
    
    # Validate
    if vary not in df.columns:
        raise ValueError(f"Column '{vary}' not in dataframe. Available: {list(df.columns)}")
    
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        raise ValueError(f"No valid metrics. Requested: {metrics}, Available: {list(df.columns)}")
    
    # Apply filters
    filtered_df = _apply_filter(df, filter)
    
    if len(filtered_df) == 0:
        print(f"Warning: No experiments match filter {filter}")
        return pd.DataFrame()
    
    # Group by the varied dimension
    groups = filtered_df.groupby(vary, dropna=False)
    
    # Compute stats
    results = []
    for name, group in groups:
        row = {'value': name, 'n': len(group)}
        for metric in available_metrics:
            values = group[metric].dropna()
            row[f'{metric}_mean'] = values.mean() if len(values) > 0 else np.nan
            row[f'{metric}_std'] = values.std() if len(values) > 1 else 0
            row[f'{metric}_n'] = len(values)
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Print summary
    _print_ablation_summary(result_df, vary, available_metrics, filter, show_n)
    
    # Plot
    if plot and len(result_df) > 0:
        _plot_ablation(result_df, vary, available_metrics, figsize)
    
    return result_df


def compare_groups(
    df: pd.DataFrame,
    group_a: Dict[str, any],
    group_b: Dict[str, any],
    metrics: Optional[List[str]] = None,
    names: Tuple[str, str] = ('Group A', 'Group B')
) -> pd.DataFrame:
    """
    Direct comparison between two experiment groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        Experiment data
    group_a, group_b : dict
        Filter conditions for each group
    metrics : list of str, optional
        Metrics to compare
    names : tuple of str
        Display names for groups
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    if metrics is None:
        metrics = DEFAULT_PERFORMANCE_METRICS + DEFAULT_BEHAVIOR_METRICS
    
    df_a = _apply_filter(df, group_a)
    df_b = _apply_filter(df, group_b)
    
    results = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        vals_a = df_a[metric].dropna()
        vals_b = df_b[metric].dropna()
        
        results.append({
            'metric': metric,
            f'{names[0]}_mean': vals_a.mean() if len(vals_a) > 0 else np.nan,
            f'{names[0]}_std': vals_a.std() if len(vals_a) > 1 else 0,
            f'{names[0]}_n': len(vals_a),
            f'{names[1]}_mean': vals_b.mean() if len(vals_b) > 0 else np.nan,
            f'{names[1]}_std': vals_b.std() if len(vals_b) > 1 else 0,
            f'{names[1]}_n': len(vals_b),
            'diff': (vals_a.mean() - vals_b.mean()) if len(vals_a) > 0 and len(vals_b) > 0 else np.nan
        })
    
    result_df = pd.DataFrame(results)
    
    # Print
    print(f"\n{'='*60}")
    print(f"Comparison: {names[0]} vs {names[1]}")
    print(f"{'='*60}")
    print(f"{names[0]} filter: {group_a} (n={len(df_a)})")
    print(f"{names[1]} filter: {group_b} (n={len(df_b)})")
    print(f"{'-'*60}")
    print(result_df.to_string(index=False))
    
    return result_df


def correlation_matrix(
    df: pd.DataFrame,
    config_cols: Optional[List[str]] = None,
    metric_cols: Optional[List[str]] = None,
    plot: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> pd.DataFrame:
    """
    Compute correlations between config choices and outcomes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Experiment data
    config_cols : list of str, optional
        Config columns to include
    metric_cols : list of str, optional
        Metric columns to include
    plot : bool, default=True
        Generate heatmap
    figsize : tuple
        Figure size
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    if config_cols is None:
        config_cols = [
            'num_regimes', 'dropout', 'load_balance_weight', 
            'expert_hidden_size', 'hard_routing_train', 'vix_threshold',
            'directional_weight', 'dist_loss_std_weight'
        ]
    
    if metric_cols is None:
        metric_cols = [
            'directional_accuracy', 'sharpe_ratio', 'healthy_pct',
            'pred_std', 'best_val_loss'
        ]
    
    # Filter to available columns
    config_cols = [c for c in config_cols if c in df.columns]
    metric_cols = [c for c in metric_cols if c in df.columns]
    
    # Convert bools to int for correlation
    subset = df[config_cols + metric_cols].copy()
    for col in subset.columns:
        if subset[col].dtype == bool:
            subset[col] = subset[col].astype(int)
    
    # Compute correlation
    corr = subset.corr()
    
    # Extract config vs metric correlations
    config_metric_corr = corr.loc[config_cols, metric_cols]
    
    if plot:
        _plot_correlation_heatmap(config_metric_corr, figsize)
    
    return config_metric_corr


def rank_experiments(
    df: pd.DataFrame,
    by: str = 'directional_accuracy',
    ascending: bool = False,
    top_n: int = 10,
    display_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Rank experiments by a metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Experiment data
    by : str
        Metric to rank by
    ascending : bool, default=False
        Sort order (False = highest first)
    top_n : int, default=10
        Number of experiments to show
    display_cols : list of str, optional
        Additional columns to display
        
    Returns
    -------
    pd.DataFrame
        Top N experiments
    """
    if by not in df.columns:
        raise ValueError(f"Column '{by}' not found")
    
    sorted_df = df.sort_values(by, ascending=ascending, na_position='last')
    
    # Default display columns
    if display_cols is None:
        display_cols = ['experiment_name', 'routing_label', 'expert_type']
    
    # Add the ranking metric
    cols = display_cols + [by]
    
    # Add a few other key metrics if available
    for extra in ['healthy_pct', 'pred_std', 'sharpe_ratio']:
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    
    cols = [c for c in cols if c in df.columns]
    
    result = sorted_df.head(top_n)[cols].copy()
    
    print(f"\nTop {top_n} experiments by {by}:")
    print(result.to_string(index=False))
    
    return result


def _apply_filter(df: pd.DataFrame, filter: Optional[Dict[str, any]]) -> pd.DataFrame:
    """Apply filter conditions to dataframe."""
    if filter is None:
        return df.copy()
    
    mask = pd.Series(True, index=df.index)
    for col, val in filter.items():
        if col not in df.columns:
            print(f"Warning: Filter column '{col}' not found, skipping")
            continue
        
        if isinstance(val, list):
            mask &= df[col].isin(val)
        else:
            mask &= (df[col] == val)
    
    return df[mask].copy()


def _print_ablation_summary(
    result_df: pd.DataFrame, 
    vary: str, 
    metrics: List[str],
    filter: Optional[Dict],
    show_n: bool
):
    """Print formatted ablation summary."""
    print(f"\n{'='*60}")
    print(f"Ablation: {vary}")
    if filter:
        print(f"Filter: {filter}")
    print(f"{'='*60}")
    
    for _, row in result_df.iterrows():
        val_str = str(row['value'])
        n_str = f" (n={row['n']})" if show_n else ""
        print(f"\n{vary} = {val_str}{n_str}")
        
        for metric in metrics:
            mean = row.get(f'{metric}_mean', np.nan)
            std = row.get(f'{metric}_std', 0)
            if pd.notna(mean):
                print(f"  {metric}: {mean:.4f} +/- {std:.4f}")


def _plot_ablation(
    result_df: pd.DataFrame,
    vary: str,
    metrics: List[str],
    figsize: Tuple[int, int]
):
    """Generate ablation bar plot."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    x_labels = [str(v) for v in result_df['value']]
    x = np.arange(len(x_labels))
    
    for ax, metric in zip(axes, metrics):
        means = result_df[f'{metric}_mean'].values
        stds = result_df[f'{metric}_std'].values
        
        bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by {vary}')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            if pd.notna(mean):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def _plot_correlation_heatmap(corr: pd.DataFrame, figsize: Tuple[int, int]):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)
    
    # Annotations
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            if pd.notna(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Config vs Outcome Correlations')
    plt.tight_layout()
    plt.show()