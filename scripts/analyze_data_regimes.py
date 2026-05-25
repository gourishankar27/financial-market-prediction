#!/usr/bin/env python3
"""
Analyze regime characteristics in S&P 500 returns data.

This script provides comprehensive statistical analysis and visualization of
return distributions across train/val/test splits to:
1. Identify regime changes and non-stationarity
2. Justify adaptive modeling approaches over hardcoded priors
3. Generate publication-quality figures and tables for reports

Usage:
    python analyze_data_regimes.py --data_dir data/splits/vintage --output_dir results/regime_analysis
    python analyze_data_regimes.py --splits test --no-figures  # Quick stats only
    python analyze_data_regimes.py --mark-events fed  # Mark Fed policy changes
    python analyze_data_regimes.py --mark-events all  # Mark all significant events
    python analyze_data_regimes.py --combined-plot   # Generate single combined timeline
    python analyze_data_regimes.py --paper-format    # IEEE paper optimized output
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Event definitions for market regime analysis
MARKET_EVENTS = {
    'fed': [
        ('1999-06-30', 'Y2K rate hikes begin', 'blue'),
        ('2001-01-03', 'Emergency rate cuts begin', 'blue'),
        ('2004-06-30', 'Rate hike cycle begins', 'blue'),
        ('2007-09-18', 'Crisis rate cuts begin', 'blue'),
        ('2008-12-16', 'Fed rate to 0%', 'blue'),
        ('2013-12-18', 'Taper tantrum begins', 'blue'),
        ('2015-12-16', 'First rate hike post-2008', 'blue'),
        ('2020-03-15', 'Emergency rate cut to 0%', 'blue'),
        ('2022-03-16', 'Rate hike cycle begins', 'blue'),
        ('2024-09-18', 'Fed cuts rates 50bp', 'blue'),
    ],
    'crises': [
        ('2000-03-10', 'Dot-com peak', 'orange'),
        ('2001-09-11', '9/11 attacks', 'red'),
        ('2002-10-09', 'Dot-com bottom', 'orange'),
        ('2007-10-09', 'Market peak pre-GFC', 'orange'),
        ('2008-03-14', 'Bear Stearns collapse', 'red'),
        ('2008-09-15', 'Lehman bankruptcy', 'red'),
        ('2009-03-09', 'GFC market bottom', 'red'),
        ('2010-05-06', 'Flash crash', 'orange'),
        ('2011-08-05', 'US debt downgrade', 'red'),
        ('2015-08-24', 'China devaluation shock', 'orange'),
        ('2016-06-23', 'Brexit vote', 'red'),
        ('2020-03-23', 'COVID market bottom', 'red'),
        ('2022-02-24', 'Ukraine invasion', 'red'),
        ('2025-08-05', 'Market selloff begins', 'red'),
    ],
    'regime_shifts': [
        ('1998-08-31', 'LTCM crisis', 'purple'),
        ('2003-03-12', 'Iraq war begins', 'purple'),
        ('2013-05-22', 'Taper tantrum speech', 'purple'),
        ('2018-12-24', 'Powell pivot rumors', 'purple'),
        ('2020-11-09', 'Vaccine announcement', 'green'),
        ('2022-01-01', 'Fed pivot expectations', 'purple'),
        ('2024-11-05', 'Trump election', 'purple'),
    ],
    # Curated subset for IEEE paper - major events only, well-spaced
    'paper': [
        ('2000-03-10', 'Dot-com Peak', 'red'),
        ('2008-09-15', 'Lehman', 'red'),
        ('2020-03-23', 'COVID', 'red'),
        ('2022-03-16', 'Fed Hikes', 'blue'),
    ]
}


def get_events_for_marking(event_type: str) -> List[Tuple[str, str, str]]:
    """Get list of events to mark based on user selection."""
    if event_type == 'all':
        events = []
        for key, event_list in MARKET_EVENTS.items():
            if key != 'paper':  # Don't include paper subset in 'all'
                events.extend(event_list)
        return events
    elif event_type in MARKET_EVENTS:
        return MARKET_EVENTS[event_type]
    else:
        return []


def load_split_data(data_dir: Path, split: str) -> pd.DataFrame:
    """Load a single split's data."""
    filepath = data_dir / f"core_proposal_daily_vintage_{split}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Split file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def compute_split_statistics(returns: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive statistics for a return series."""
    return {
        'count': len(returns),
        'mean': float(np.mean(returns)),
        'std': float(np.std(returns, ddof=1)),
        'min': float(np.min(returns)),
        'max': float(np.max(returns)),
        'p05': float(np.percentile(returns, 5)),
        'p25': float(np.percentile(returns, 25)),
        'p50': float(np.percentile(returns, 50)),
        'p75': float(np.percentile(returns, 75)),
        'p95': float(np.percentile(returns, 95)),
        'skewness': float(stats.skew(returns)),
        'kurtosis': float(stats.kurtosis(returns)),
    }


def compute_rolling_statistics(
    returns: np.ndarray, window: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean and std for regime detection."""
    n = len(returns)
    rolling_mean = np.full(n, np.nan)
    rolling_std = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_data = returns[i - window + 1:i + 1]
        rolling_mean[i] = np.mean(window_data)
        rolling_std[i] = np.std(window_data, ddof=1)
    
    return rolling_mean, rolling_std


def analyze_rolling_windows(returns: np.ndarray, window: int = 30) -> Dict[str, float]:
    """Compute statistics on rolling window behavior."""
    n = len(returns)
    n_windows = n - window + 1
    
    window_means = []
    window_stds = []
    
    for i in range(n_windows):
        window_data = returns[i:i+window]
        window_means.append(np.mean(window_data))
        window_stds.append(np.std(window_data, ddof=1))
    
    window_means = np.array(window_means)
    window_stds = np.array(window_stds)
    
    return {
        'num_windows': n_windows,
        'mean_of_means': float(np.mean(window_means)),
        'std_of_means': float(np.std(window_means, ddof=1)),
        'min_mean': float(np.min(window_means)),
        'max_mean': float(np.max(window_means)),
        'mean_of_stds': float(np.mean(window_stds)),
        'std_of_stds': float(np.std(window_stds, ddof=1)),
        'min_std': float(np.min(window_stds)),
        'max_std': float(np.max(window_stds)),
    }


def analyze_directional_streaks(returns: np.ndarray) -> Dict[str, float]:
    """Analyze streaks of consecutive positive or negative returns."""
    signs = np.sign(returns)
    signs[signs == 0] = 1
    
    streak_lengths = []
    current_streak = 1
    
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1]:
            current_streak += 1
        else:
            streak_lengths.append(current_streak)
            current_streak = 1
    streak_lengths.append(current_streak)
    
    streak_lengths = np.array(streak_lengths)
    
    window = 30
    rolling_pct_positive = []
    
    for i in range(len(returns)):
        if i < window - 1:
            rolling_pct_positive.append(np.nan)
        else:
            start_idx = i - window + 1
            window_returns = returns[start_idx:i+1]
            pct_pos = np.mean(window_returns > 0)
            rolling_pct_positive.append(pct_pos)
    
    rolling_pct_positive = np.array(rolling_pct_positive)
    valid_pct = rolling_pct_positive[~np.isnan(rolling_pct_positive)]
    
    highly_positive_windows = np.sum(valid_pct > 0.9)
    highly_negative_windows = np.sum(valid_pct < 0.1)
    
    return {
        'min_streak': int(np.min(streak_lengths)),
        'mean_streak': float(np.mean(streak_lengths)),
        'median_streak': float(np.median(streak_lengths)),
        'max_streak': int(np.max(streak_lengths)),
        'pct_pos_returns': float(np.mean(returns > 0) * 100),
        'highly_pos_windows': int(highly_positive_windows),
        'highly_neg_windows': int(highly_negative_windows),
        'mean_window_pct_pos': float(np.mean(valid_pct) * 100),
        'std_window_pct_pos': float(np.std(valid_pct) * 100),
    }


def analyze_naive_baselines(returns: np.ndarray) -> Dict[str, float]:
    """Analyze naive baseline prediction strategies."""
    always_positive = np.ones_like(returns)
    dir_acc_positive = np.mean((returns > 0) == (always_positive > 0))
    
    always_zero = np.zeros_like(returns)
    dir_acc_zero = np.mean((returns > 0) == (always_zero > 0))
    
    mean_pred = np.full_like(returns, np.mean(returns))
    dir_acc_mean = np.mean((returns > 0) == (mean_pred > 0))
    
    momentum_pred = np.concatenate([[0], returns[:-1]])
    dir_acc_momentum = np.mean((returns[1:] > 0) == (momentum_pred[1:] > 0))
    
    reversion_pred = -momentum_pred
    dir_acc_reversion = np.mean((returns[1:] > 0) == (reversion_pred[1:] > 0))
    
    window = 30
    random_pred = np.random.randn(len(returns)) * np.std(returns)
    
    rolling_corr_random = []
    for i in range(len(returns)):
        if i < window - 1:
            rolling_corr_random.append(np.nan)
        else:
            start_idx = i - window + 1
            window_actual = returns[start_idx:i+1]
            window_pred = random_pred[start_idx:i+1]
            if np.std(window_actual) > 0 and np.std(window_pred) > 0:
                corr = np.corrcoef(window_actual, window_pred)[0, 1]
                rolling_corr_random.append(corr)
            else:
                rolling_corr_random.append(np.nan)
    
    rolling_corr_random = np.array(rolling_corr_random)
    valid_corr = rolling_corr_random[~np.isnan(rolling_corr_random)]
    
    return {
        'dir_acc_always_positive': float(dir_acc_positive * 100),
        'dir_acc_always_zero': float(dir_acc_zero * 100),
        'dir_acc_historical_mean': float(dir_acc_mean * 100),
        'dir_acc_momentum': float(dir_acc_momentum * 100),
        'dir_acc_mean_reversion': float(dir_acc_reversion * 100),
        'random_walk_mean_corr': float(np.mean(valid_corr)),
        'random_walk_std_corr': float(np.std(valid_corr)),
    }


def detect_regimes_vix(df: pd.DataFrame, low_thresh: float = 15, high_thresh: float = 25) -> np.ndarray:
    """Classify regimes based on VIX levels."""
    if 'VIX' not in df.columns:
        return np.array(['unknown'] * len(df))
    
    vix = df['VIX'].values
    regimes = np.empty(len(vix), dtype=object)
    regimes[vix < low_thresh] = 'low_vol'
    regimes[(vix >= low_thresh) & (vix < high_thresh)] = 'medium_vol'
    regimes[vix >= high_thresh] = 'high_vol'
    
    return regimes


def print_statistics_table(stats_dict: Dict[str, Dict[str, float]]) -> None:
    """Print formatted statistics table."""
    print("\n" + "="*80)
    print("SPLIT STATISTICS SUMMARY")
    print("="*80)
    
    splits = list(stats_dict.keys())
    print(f"{'Metric':<20}", end='')
    for split in splits:
        print(f"{split.upper():<18}", end='')
    print()
    print("-"*80)
    
    metrics = [
        ('count', 'Count', '14.0f'),
        ('mean', 'Mean', '14.6f'),
        ('std', 'Std Dev', '14.6f'),
        ('min', 'Min', '14.6f'),
        ('max', 'Max', '14.6f'),
        ('p05', '5th pct', '14.6f'),
        ('p50', 'Median', '14.6f'),
        ('p95', '95th pct', '14.6f'),
        ('skewness', 'Skewness', '14.4f'),
        ('kurtosis', 'Kurtosis', '14.4f'),
    ]
    
    for key, label, fmt in metrics:
        print(f"{label:<20}", end='')
        for split in splits:
            print(f"{stats_dict[split][key]:>{fmt}}  ", end='')
        print()
    
    print("="*80 + "\n")


def print_rolling_window_table(window_stats_dict: Dict[str, Dict[str, float]]) -> None:
    """Print 30-day rolling window statistics table."""
    print("\n" + "="*80)
    print("30-DAY ROLLING WINDOW STATISTICS")
    print("="*80)
    
    splits = list(window_stats_dict.keys())
    print(f"{'Metric':<20}", end='')
    for split in splits:
        print(f"{split.upper():<18}", end='')
    print()
    print("-"*80)
    
    metrics = [
        ('num_windows', 'Num Windows', '14.0f'),
        ('mean_of_means', 'Mean of means', '14.6f'),
        ('std_of_means', 'Std of means', '14.6f'),
        ('min_mean', 'Min mean', '14.6f'),
        ('max_mean', 'Max mean', '14.6f'),
        ('mean_of_stds', 'Mean of stds', '14.6f'),
        ('std_of_stds', 'Std of stds', '14.6f'),
        ('min_std', 'Min std', '14.6f'),
        ('max_std', 'Max std', '14.6f'),
    ]
    
    for key, label, fmt in metrics:
        print(f"{label:<20}", end='')
        for split in splits:
            print(f"{window_stats_dict[split][key]:>{fmt}}  ", end='')
        print()
    
    print("="*80 + "\n")


def print_directional_streak_table(streak_stats_dict: Dict[str, Dict[str, float]]) -> None:
    """Print directional streak analysis table."""
    print("\n" + "="*80)
    print("DIRECTIONAL BEHAVIOR ANALYSIS")
    print("="*80)
    
    splits = list(streak_stats_dict.keys())
    print(f"{'Metric':<25}", end='')
    for split in splits:
        print(f"{split.upper():<18}", end='')
    print()
    print("-"*80)
    
    metrics = [
        ('pct_pos_returns', '% Positive returns', '14.2f'),
        ('min_streak', 'Min streak (days)', '14.0f'),
        ('mean_streak', 'Mean streak (days)', '14.2f'),
        ('median_streak', 'Median streak (days)', '14.2f'),
        ('max_streak', 'Max streak (days)', '14.0f'),
        ('mean_window_pct_pos', 'Mean 30d % positive', '14.2f'),
        ('std_window_pct_pos', 'Std 30d % positive', '14.2f'),
        ('highly_pos_windows', '30d windows >90% pos', '14.0f'),
        ('highly_neg_windows', '30d windows >90% neg', '14.0f'),
    ]
    
    for key, label, fmt in metrics:
        print(f"{label:<25}", end='')
        for split in splits:
            print(f"{streak_stats_dict[split][key]:>{fmt}}  ", end='')
        print()
    
    print("="*80 + "\n")
    print("Note: Streaks measure consecutive days with same sign.")
    print("      Highly directional windows have >90% returns in one direction.")
    print()


def print_baseline_performance_table(baseline_stats_dict: Dict[str, Dict[str, float]]) -> None:
    """Print naive baseline performance table."""
    print("\n" + "="*80)
    print("NAIVE BASELINE PERFORMANCE")
    print("="*80)
    
    splits = list(baseline_stats_dict.keys())
    print(f"{'Strategy':<30}", end='')
    for split in splits:
        print(f"{split.upper():<18}", end='')
    print()
    print("-"*80)
    
    metrics = [
        ('dir_acc_always_positive', 'Always positive (equity drift)', '14.2f'),
        ('dir_acc_always_zero', 'Always zero (no change)', '14.2f'),
        ('dir_acc_historical_mean', 'Predict historical mean', '14.2f'),
        ('dir_acc_momentum', 'Momentum (t-1 return)', '14.2f'),
        ('dir_acc_mean_reversion', 'Mean reversion (-t-1)', '14.2f'),
        ('random_walk_mean_corr', 'Random walk mean corr', '14.4f'),
        ('random_walk_std_corr', 'Random walk std corr', '14.4f'),
    ]
    
    for key, label, fmt in metrics:
        print(f"{label:<30}", end='')
        for split in splits:
            print(f"{baseline_stats_dict[split][key]:>{fmt}}  ", end='')
        print()
    
    print("="*80 + "\n")
    print("Note: Directional accuracy (%) shows sign prediction accuracy.")
    print("      Models should exceed these baselines to be useful.")
    print()


def print_markdown_table(stats_dict: Dict[str, Dict[str, float]]) -> str:
    """Generate markdown-formatted table."""
    splits = list(stats_dict.keys())
    
    lines = []
    lines.append("| Metric | " + " | ".join(s.capitalize() for s in splits) + " |")
    lines.append("|--------|" + "|".join(["--------:"] * len(splits)) + "|")
    
    metrics = [
        ('count', 'Count', '.0f'),
        ('mean', 'Mean', '.6f'),
        ('std', 'Std Dev', '.6f'),
        ('min', 'Min', '.6f'),
        ('max', 'Max', '.6f'),
        ('p05', '5th percentile', '.6f'),
        ('p50', 'Median', '.6f'),
        ('p95', '95th percentile', '.6f'),
        ('skewness', 'Skewness', '.4f'),
        ('kurtosis', 'Kurtosis', '.4f'),
    ]
    
    for key, label, fmt in metrics:
        values = [f"{stats_dict[split][key]:{fmt}}" for split in splits]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def plot_distribution_comparison(data_dict: Dict[str, np.ndarray], output_path: Path) -> None:
    """Create overlaid histogram and Q-Q plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    for split, returns in data_dict.items():
        ax.hist(returns, bins=50, alpha=0.5, label=split.capitalize(), 
                color=colors.get(split, 'gray'), density=True)
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison Across Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for idx, (split, returns) in enumerate(data_dict.items()):
        if idx >= 3:
            break
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        ax = axes[row, col]
        
        stats.probplot(returns, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {split.capitalize()}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution comparison: {output_path}")
    plt.close()


def plot_rolling_volatility(
    df: pd.DataFrame, split_name: str, window: int, output_path: Path,
    mark_events: List[Tuple[str, str, str]] = None
) -> None:
    """Plot rolling volatility with regime coloring."""
    returns = df['SP500_Returns'].values
    dates = df['Date'].values
    _, rolling_std = compute_rolling_statistics(returns, window)
    
    regimes = detect_regimes_vix(df)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax = axes[0]
    regime_colors = {'low_vol': 'green', 'medium_vol': 'yellow', 'high_vol': 'red'}
    
    for regime in ['low_vol', 'medium_vol', 'high_vol']:
        mask = regimes == regime
        ax.scatter(dates[mask], returns[mask], c=regime_colors[regime], 
                  alpha=0.6, s=10, label=regime.replace('_', ' ').title())
    
    if mark_events:
        date_range = (pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]))
        for event_date_str, event_label, event_color in mark_events:
            event_date = pd.Timestamp(event_date_str)
            if date_range[0] <= event_date <= date_range[1]:
                ax.axvline(x=event_date, color=event_color, linestyle='--', 
                          linewidth=1.5, alpha=0.8)
                y_pos = ax.get_ylim()[1] * 0.85
                ax.text(event_date, y_pos, event_label, 
                       rotation=90, verticalalignment='top', horizontalalignment='right',
                       fontsize=8, alpha=0.9, color=event_color, fontweight='bold')
    
    ax.set_ylabel('Daily Returns')
    ax.set_title(f'{split_name.capitalize()} Split: Returns Colored by VIX Regime')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    ax = axes[1]
    ax.plot(dates, rolling_std, color='darkblue', linewidth=1.5)
    
    if mark_events:
        for event_date_str, event_label, event_color in mark_events:
            event_date = pd.Timestamp(event_date_str)
            if date_range[0] <= event_date <= date_range[1]:
                ax.axvline(x=event_date, color=event_color, linestyle='--', 
                          linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{window}-Day Rolling Std Dev')
    ax.set_title(f'{window}-Day Rolling Volatility')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved rolling volatility plot: {output_path}")
    plt.close()


def plot_combined_timeline(
    df_dict: Dict[str, pd.DataFrame], 
    window: int, 
    output_path: Path,
    mark_events: List[Tuple[str, str, str]] = None,
    figsize: Tuple[float, float] = (10, 5)
) -> None:
    """
    Plot all splits concatenated in a single timeline.
    Optimized for IEEE double-column paper format.
    """
    # Concatenate all splits in order
    all_dfs = []
    split_boundaries = []
    
    for split in ['train', 'val', 'test']:
        if split in df_dict:
            df = df_dict[split].copy()
            all_dfs.append(df)
            split_boundaries.append({
                'name': split,
                'start_date': df['Date'].iloc[0],
                'end_date': df['Date'].iloc[-1]
            })
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    returns = combined_df['SP500_Returns'].values
    dates = combined_df['Date'].values
    _, rolling_std = compute_rolling_statistics(returns, window)
    
    regimes = detect_regimes_vix(combined_df)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Returns plot
    ax = axes[0]
    regime_colors = {'low_vol': '#2ecc71', 'medium_vol': '#f1c40f', 'high_vol': '#e74c3c'}
    
    for regime in ['low_vol', 'medium_vol', 'high_vol']:
        mask = regimes == regime
        ax.scatter(dates[mask], returns[mask], c=regime_colors[regime], 
                  alpha=0.5, s=5, label=regime.replace('_', ' ').title(), rasterized=True)
    
    # Mark split boundaries
    split_colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    for i, boundary in enumerate(split_boundaries):
        if i < len(split_boundaries) - 1:
            next_start = split_boundaries[i + 1]['start_date']
        else:
            next_start = pd.Timestamp(dates[-1])
        
        ax.axvspan(boundary['start_date'], next_start, 
                   alpha=0.08, color=split_colors[boundary['name']])
        
        # Add split label at bottom of upper plot
        mid_date = boundary['start_date'] + (pd.Timestamp(next_start) - pd.Timestamp(boundary['start_date'])) / 2
        ax.text(mid_date, -11, boundary['name'].upper(), 
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=split_colors[boundary['name']], alpha=0.9)
    
    # Mark events
    if mark_events:
        date_range = (pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]))
        for event_date_str, event_label, event_color in mark_events:
            event_date = pd.Timestamp(event_date_str)
            if date_range[0] <= event_date <= date_range[1]:
                ax.axvline(x=event_date, color=event_color, linestyle='--', 
                          linewidth=1.2, alpha=0.8)
                ax.text(event_date, 10, event_label, 
                       rotation=90, verticalalignment='top', horizontalalignment='right',
                       fontsize=14, alpha=0.9, color=event_color, fontweight='bold')
    
    ax.set_ylabel('Daily Returns (%)', fontsize=14)
    ax.set_title('S&P 500 Returns by VIX Regime (1991-2025)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=14, markerscale=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.3)
    ax.set_ylim(-13, 13)
    ax.tick_params(axis='both', labelsize=16)
    
    # Rolling volatility plot
    ax = axes[1]
    ax.plot(dates, rolling_std, color='darkblue', linewidth=1.0)
    ax.fill_between(dates, 0, rolling_std, alpha=0.3, color='darkblue')
    
    if mark_events:
        for event_date_str, event_label, event_color in mark_events:
            event_date = pd.Timestamp(event_date_str)
            if date_range[0] <= event_date <= date_range[1]:
                ax.axvline(x=event_date, color=event_color, linestyle='--', 
                          linewidth=1.2, alpha=0.8)
    
    for i, boundary in enumerate(split_boundaries):
        if i < len(split_boundaries) - 1:
            next_start = split_boundaries[i + 1]['start_date']
        else:
            next_start = pd.Timestamp(dates[-1])
        ax.axvspan(boundary['start_date'], next_start, 
                   alpha=0.08, color=split_colors[boundary['name']])
    
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel(f'{window}-Day Rolling Std', fontsize=16)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined timeline plot: {output_path}")
    plt.close()


def plot_regime_statistics(data_dict: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot statistics by regime across splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    regime_stats = []
    for split_name, df in data_dict.items():
        regimes = detect_regimes_vix(df)
        returns = df['SP500_Returns'].values
        
        for regime in ['low_vol', 'medium_vol', 'high_vol']:
            mask = regimes == regime
            if mask.sum() > 0:
                regime_returns = returns[mask]
                regime_stats.append({
                    'split': split_name,
                    'regime': regime,
                    'mean': np.mean(regime_returns),
                    'std': np.std(regime_returns, ddof=1),
                    'count': mask.sum()
                })
    
    regime_df = pd.DataFrame(regime_stats)
    
    ax = axes[0]
    for split in regime_df['split'].unique():
        split_data = regime_df[regime_df['split'] == split]
        ax.plot(split_data['regime'], split_data['mean'], marker='o', label=split.capitalize())
    ax.set_xlabel('Regime')
    ax.set_ylabel('Mean Return')
    ax.set_title('Mean Returns by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for split in regime_df['split'].unique():
        split_data = regime_df[regime_df['split'] == split]
        ax.plot(split_data['regime'], split_data['std'], marker='o', label=split.capitalize())
    ax.set_xlabel('Regime')
    ax.set_ylabel('Std Dev')
    ax.set_title('Volatility by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved regime statistics plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze regime characteristics in financial returns data'
    )
    parser.add_argument('--data_dir', type=str, default='data/splits/vintage')
    parser.add_argument('--output_dir', type=str, default='results/regime_analysis')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--mark-events', type=str, 
                        choices=['fed', 'crises', 'regime_shifts', 'all', 'paper'],
                        default=None)
    parser.add_argument('--combined-plot', action='store_true',
                        help='Generate single combined timeline of all splits')
    parser.add_argument('--paper-format', action='store_true',
                        help='Use IEEE paper formatting (smaller, curated events)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not args.no_figures:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    events_to_mark = None
    if args.mark_events:
        events_to_mark = get_events_for_marking(args.mark_events)
        print(f"\nMarking {len(events_to_mark)} events of type: {args.mark_events}")
    
    if args.paper_format and not args.mark_events:
        events_to_mark = get_events_for_marking('paper')
        print(f"\nUsing paper-optimized events ({len(events_to_mark)} events)")
    
    print(f"\nLoading data from {data_dir}...")
    data_dict = {}
    df_dict = {}
    
    for split in args.splits:
        try:
            df = load_split_data(data_dir, split)
            df_dict[split] = df
            data_dict[split] = df['SP500_Returns'].values
            print(f"  {split}: {len(df)} days, {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
    
    if not data_dict:
        print("Error: No valid splits found.")
        sys.exit(1)
    
    print("\nComputing statistics...")
    stats_dict = {}
    window_stats_dict = {}
    streak_stats_dict = {}
    baseline_stats_dict = {}
    
    for split, returns in data_dict.items():
        stats_dict[split] = compute_split_statistics(returns)
        window_stats_dict[split] = analyze_rolling_windows(returns, args.window)
        streak_stats_dict[split] = analyze_directional_streaks(returns)
        baseline_stats_dict[split] = analyze_naive_baselines(returns)
    
    print_rolling_window_table(window_stats_dict)
    print_directional_streak_table(streak_stats_dict)
    print_baseline_performance_table(baseline_stats_dict)
    print_statistics_table(stats_dict)
    
    if not args.no_figures:
        md_table = print_markdown_table(stats_dict)
        md_path = output_dir / 'statistics_table.md'
        with open(md_path, 'w') as f:
            f.write("# S&P 500 Returns: Split Statistics\n\n")
            f.write(md_table)
            f.write("\n")
        print(f"Saved markdown table: {md_path}")
        
        json_path = output_dir / 'statistics.json'
        with open(json_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Saved statistics JSON: {json_path}")
    
    if not args.no_figures:
        print("\nGenerating figures...")
        
        plot_distribution_comparison(data_dict, output_dir / 'distribution_comparison.png')
        
        if args.combined_plot or args.paper_format:
            figsize = (7, 3.5) if args.paper_format else (14, 6)
            plot_combined_timeline(
                df_dict, args.window,
                output_dir / 'combined_timeline.png',
                mark_events=events_to_mark,
                figsize=figsize
            )
        else:
            for split, df in df_dict.items():
                plot_rolling_volatility(
                    df, split, args.window,
                    output_dir / f'rolling_volatility_{split}.png',
                    mark_events=events_to_mark
                )
        
        plot_regime_statistics(df_dict, output_dir / 'regime_statistics.png')
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
