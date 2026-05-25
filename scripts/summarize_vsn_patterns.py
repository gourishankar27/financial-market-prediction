#!/usr/bin/env python3
"""
Summarize VSN (Variable Selection Network) patterns across multiple experiments.

Aggregates VSN weight analysis results to identify:
- Cross-experiment feature importance consistency
- Regime-dependent feature selection shifts
- Correlation between VSN patterns and model collapse/performance
- Feature concentration trends across market conditions

Usage:
    # All experiments in a phase
    python summarize_vsn_patterns.py experiments/02b_vintage_sweep/
    
    # Multiple phases
    python summarize_vsn_patterns.py experiments/00_baseline_exploration/ experiments/02b_vintage_sweep/
    
    # Compare specific periods (e.g., Fed pivot analysis)
    python summarize_vsn_patterns.py experiments/ --compare-periods "2021" "2022" "2023"
    
    # Correlate with collapse metrics
    python summarize_vsn_patterns.py experiments/ --experiments-csv results/experiments_summary.csv

Output:
    reports/
    ├── vsn_summary.csv                    # Per-experiment summary stats
    ├── vsn_feature_importance.csv         # Cross-experiment feature rankings
    ├── vsn_regime_shifts.csv              # Detected feature selection shifts
    ├── vsn_with_collapse.csv              # Merged with collapse metrics
    ├── vsn_summary_report.txt             # Human-readable analysis
    ├── vsn_feature_importance_heatmap.png # Feature importance by period
    ├── vsn_concentration_by_phase.png     # Concentration trends
    └── vsn_regime_shift_timeline.png      # When feature selection changed
"""

import os
import sys
import json
import argparse
import glob
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# DATA LOADING
# ============================================================================

def find_vsn_results(paths: List[str]) -> Dict[str, Path]:
    """
    Recursively find all VSN analysis result files.
    
    Parameters
    ----------
    paths : list of str
        Paths to search (experiments/, single phase dir, etc.)
    
    Returns
    -------
    dict : {experiment_name: result_file_path}
    """
    results = {}
    phase_pattern = re.compile(r'^\d{2}[a-z]?_')  # 00_, 01_, 02b_, etc.
    
    for path_pattern in paths:
        expanded_paths = glob.glob(path_pattern, recursive=True)
        if not expanded_paths:
            expanded_paths = [path_pattern]
        
        for path_str in expanded_paths:
            path = Path(path_str)
            
            if not path.exists():
                print(f"Warning: Path does not exist: {path}")
                continue
            
            is_phase_dir = phase_pattern.match(path.name) is not None
            
            if path.is_file() and path.name == 'vsn_analysis_results.json':
                # Direct file provided
                exp_name = path.parent.parent.name
                if is_phase_dir:
                    exp_name = f"{path.name}/{exp_name}"
                results[exp_name] = path
            else:
                # Directory - search recursively for vsn_analysis/vsn_analysis_results.json
                for result_file in path.rglob('vsn_analysis/vsn_analysis_results.json'):
                    # Extract experiment name from path structure
                    try:
                        exp_path = result_file.relative_to(path)
                        parts = exp_path.parts
                        # Remove vsn_analysis/ and filename
                        if len(parts) >= 2:
                            exp_name = '/'.join(parts[:-2])
                        else:
                            exp_name = result_file.parent.parent.name
                        
                        if is_phase_dir:
                            exp_name = f"{path.name}/{exp_name}"
                        
                        results[exp_name] = result_file
                    except ValueError:
                        # Fallback
                        exp_name = result_file.parent.parent.name
                        results[exp_name] = result_file
    
    return results


def load_vsn_results(result_files: Dict[str, Path]) -> Dict[str, dict]:
    """Load all VSN analysis results."""
    data = {}
    
    for exp_name, file_path in result_files.items():
        try:
            with open(file_path, 'r') as f:
                data[exp_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    return data


def extract_experiment_metadata(exp_name: str) -> dict:
    """
    Extract hyperparameters from experiment name.
    
    Expected patterns:
    - baseline_h16_drop0.10
    - sweep2_h16_drop_0.25
    - staleness_h20_drop0.15
    """
    metadata = {
        'phase': None,
        'hidden_size': None,
        'dropout': None,
        'has_staleness': False,
    }
    
    # Extract phase (first component if path-like)
    if '/' in exp_name:
        parts = exp_name.split('/')
        metadata['phase'] = parts[0]
        exp_base = parts[-1]
    else:
        exp_base = exp_name
    
    # Hidden size
    h_match = re.search(r'h(\d+)', exp_base)
    if h_match:
        metadata['hidden_size'] = int(h_match.group(1))
    
    # Dropout
    drop_match = re.search(r'drop[_]?(0\.\d+)', exp_base)
    if drop_match:
        metadata['dropout'] = float(drop_match.group(1))
    
    # Staleness features
    metadata['has_staleness'] = 'staleness' in exp_base.lower()
    
    return metadata


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def compute_experiment_summary(vsn_data: Dict[str, dict]) -> pd.DataFrame:
    """
    Compute per-experiment summary statistics.
    
    Returns DataFrame with:
    - experiment_name, phase, hidden_size, dropout
    - n_periods, n_features
    - avg_concentration, std_concentration, min/max_concentration
    - concentration_trend (slope over time)
    - top_feature_overall (most weighted feature across all periods)
    - feature_stability (how consistent is top feature across periods)
    """
    rows = []
    
    for exp_name, data in vsn_data.items():
        period_stats = data.get('period_statistics', {})
        if not period_stats:
            continue
        
        periods = sorted(period_stats.keys())
        metadata = extract_experiment_metadata(exp_name)
        
        # Get feature names
        feature_names = data.get('encoder_variables', [])
        n_features = len(feature_names)
        
        # Concentration values
        concentrations = [period_stats[p]['concentration'] for p in periods]
        
        # Concentration trend
        if len(concentrations) > 1:
            x = np.arange(len(concentrations))
            concentration_trend = np.polyfit(x, concentrations, 1)[0]
        else:
            concentration_trend = 0.0
        
        # Aggregate feature weights across all periods
        aggregated_weights = np.zeros(n_features)
        top_features_per_period = []
        
        for period in periods:
            weights = np.array(period_stats[period]['mean_weights'])
            aggregated_weights += weights
            
            # Track top feature each period
            if len(period_stats[period]['top_features']) > 0:
                top_feat = period_stats[period]['top_features'][0][0]
                top_features_per_period.append(top_feat)
        
        # Overall top feature
        if n_features > 0 and len(feature_names) == len(aggregated_weights):
            top_idx = np.argmax(aggregated_weights)
            top_feature = feature_names[top_idx]
        else:
            top_feature = 'unknown'
        
        # Feature stability: what fraction of periods had same top feature?
        if top_features_per_period:
            from collections import Counter
            feat_counts = Counter(top_features_per_period)
            most_common_count = feat_counts.most_common(1)[0][1]
            feature_stability = most_common_count / len(top_features_per_period)
        else:
            feature_stability = 0.0
        
        rows.append({
            'experiment_name': exp_name,
            'phase': metadata['phase'],
            'hidden_size': metadata['hidden_size'],
            'dropout': metadata['dropout'],
            'has_staleness': metadata['has_staleness'],
            'n_periods': len(periods),
            'n_features': n_features,
            'avg_concentration': float(np.mean(concentrations)),
            'std_concentration': float(np.std(concentrations)),
            'min_concentration': float(np.min(concentrations)),
            'max_concentration': float(np.max(concentrations)),
            'concentration_trend': float(concentration_trend),
            'top_feature': top_feature,
            'feature_stability': float(feature_stability),
        })
    
    return pd.DataFrame(rows)


def compute_feature_importance_matrix(vsn_data: Dict[str, dict]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a matrix of feature importance across experiments and periods.
    
    Returns:
    - DataFrame: rows = (experiment, period), cols = features
    - List of all unique feature names
    """
    # Collect all unique features
    all_features = set()
    for data in vsn_data.values():
        all_features.update(data.get('encoder_variables', []))
    all_features = sorted(all_features)
    
    rows = []
    for exp_name, data in vsn_data.items():
        period_stats = data.get('period_statistics', {})
        feature_names = data.get('encoder_variables', [])
        
        for period, stats in period_stats.items():
            row = {
                'experiment': exp_name,
                'period': period,
                'n_samples': stats.get('n_samples', 0),
                'concentration': stats.get('concentration', 0),
            }
            
            # Map weights to features
            weights = stats.get('mean_weights', [])
            for i, feat in enumerate(feature_names):
                if i < len(weights):
                    row[feat] = weights[i]
            
            # Fill missing features with 0
            for feat in all_features:
                if feat not in row:
                    row[feat] = 0.0
            
            rows.append(row)
    
    return pd.DataFrame(rows), all_features


def compute_cross_experiment_feature_rankings(importance_df: pd.DataFrame, 
                                               feature_cols: List[str]) -> pd.DataFrame:
    """
    Rank features by importance across all experiments.
    
    Returns DataFrame with:
    - feature_name
    - mean_weight, std_weight, median_weight
    - rank (by mean weight)
    - consistency (1 - normalized std, higher = more consistent)
    - n_experiments (how many experiments included this feature)
    """
    if not feature_cols:
        return pd.DataFrame()
    
    rows = []
    for feat in feature_cols:
        if feat not in importance_df.columns:
            continue
        
        weights = importance_df[feat].dropna()
        if len(weights) == 0:
            continue
        
        mean_w = weights.mean()
        std_w = weights.std()
        
        # Consistency: inverse of coefficient of variation (capped)
        if mean_w > 0:
            cv = std_w / mean_w
            consistency = 1.0 / (1.0 + cv)  # Transform to 0-1 scale
        else:
            consistency = 0.0
        
        rows.append({
            'feature': feat,
            'mean_weight': float(mean_w),
            'std_weight': float(std_w),
            'median_weight': float(weights.median()),
            'min_weight': float(weights.min()),
            'max_weight': float(weights.max()),
            'consistency': float(consistency),
            'n_observations': len(weights),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('mean_weight', ascending=False)
        df['rank'] = range(1, len(df) + 1)
    
    return df


# ============================================================================
# REGIME SHIFT DETECTION
# ============================================================================

def detect_feature_selection_shifts(vsn_data: Dict[str, dict],
                                     period_pairs: Optional[List[Tuple[str, str]]] = None,
                                     cosine_threshold: float = 0.95,
                                     l2_threshold: float = 0.03,
                                     concentration_threshold: float = 0.05,
                                     top_feature_change: bool = True) -> List[dict]:
    """
    Detect significant shifts in feature selection between periods.
    
    Signals:
    1. Cosine similarity between weight vectors < threshold
    2. L2 distance between weight vectors > threshold
    3. Concentration change > threshold
    4. Top feature changed
    
    Returns list of detected shifts with metadata.
    """
    shifts = []
    
    for exp_name, data in vsn_data.items():
        period_stats = data.get('period_statistics', {})
        comparisons = data.get('period_comparisons', {})
        
        periods = sorted(period_stats.keys())
        
        # Determine which pairs to analyze
        if period_pairs:
            pairs_to_check = [(p1, p2) for p1, p2 in period_pairs 
                             if p1 in periods and p2 in periods]
        else:
            # All consecutive pairs
            pairs_to_check = [(periods[i], periods[i+1]) for i in range(len(periods)-1)]
        
        for period1, period2 in pairs_to_check:
            stats1 = period_stats[period1]
            stats2 = period_stats[period2]
            
            signals_fired = []
            
            # Get comparison metrics if available
            pair_key = f"{period1}_vs_{period2}"
            comp = comparisons.get(pair_key, {})
            
            # Signal 1: Cosine similarity
            cos_sim = comp.get('cosine_similarity')
            if cos_sim is not None and cos_sim < cosine_threshold:
                signals_fired.append('cosine')
            
            # Signal 2: L2 distance
            l2_dist = comp.get('l2_distance')
            if l2_dist is not None and l2_dist > l2_threshold:
                signals_fired.append('l2')
            
            # Signal 3: Concentration change
            conc1 = stats1.get('concentration', 0)
            conc2 = stats2.get('concentration', 0)
            conc_change = abs(conc2 - conc1)
            if conc_change > concentration_threshold:
                signals_fired.append('concentration')
            
            # Signal 4: Top feature changed
            top1 = stats1.get('top_features', [[None]])[0][0] if stats1.get('top_features') else None
            top2 = stats2.get('top_features', [[None]])[0][0] if stats2.get('top_features') else None
            if top_feature_change and top1 and top2 and top1 != top2:
                signals_fired.append('top_feature')
            
            if signals_fired:
                shifts.append({
                    'experiment': exp_name,
                    'period_from': period1,
                    'period_to': period2,
                    'n_signals': len(signals_fired),
                    'signals': ','.join(signals_fired),
                    'cosine_similarity': cos_sim,
                    'l2_distance': l2_dist,
                    'concentration_from': conc1,
                    'concentration_to': conc2,
                    'concentration_change': conc2 - conc1,
                    'top_feature_from': top1,
                    'top_feature_to': top2,
                    'samples_from': stats1.get('n_samples', 0),
                    'samples_to': stats2.get('n_samples', 0),
                })
    
    return shifts


def summarize_shifts_by_period(shifts: List[dict]) -> pd.DataFrame:
    """
    Aggregate shifts by period transition.
    
    E.g., how many experiments showed a shift from 2021→2022?
    """
    if not shifts:
        return pd.DataFrame()
    
    df = pd.DataFrame(shifts)
    
    # Group by period transition
    grouped = df.groupby(['period_from', 'period_to']).agg({
        'experiment': 'count',
        'n_signals': 'mean',
        'cosine_similarity': 'mean',
        'l2_distance': 'mean',
        'concentration_change': 'mean',
    }).reset_index()
    
    grouped.columns = ['period_from', 'period_to', 'n_experiments', 
                       'avg_signals', 'avg_cosine', 'avg_l2', 'avg_conc_change']
    
    return grouped.sort_values('n_experiments', ascending=False)


# ============================================================================
# COLLAPSE CORRELATION
# ============================================================================

def load_collapse_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Load experiment summary CSV with collapse metrics."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Warning: Failed to load collapse data: {e}")
        return None


def correlate_vsn_with_collapse(summary_df: pd.DataFrame, 
                                 collapse_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Merge VSN summary with collapse metrics and compute correlations.
    
    Returns:
    - merged_df: Combined data
    - corr_matrix: Correlation matrix of VSN vs collapse metrics
    """
    if collapse_df is None or summary_df.empty:
        return None, None
    
    # Try to match on experiment name
    # May need to normalize names (strip phase prefix, etc.)
    
    # Create normalized name for matching
    summary_df = summary_df.copy()
    summary_df['match_name'] = summary_df['experiment_name'].apply(
        lambda x: x.split('/')[-1] if '/' in x else x
    )
    
    collapse_df = collapse_df.copy()
    if 'experiment' in collapse_df.columns:
        collapse_df['match_name'] = collapse_df['experiment'].apply(
            lambda x: x.split('/')[-1] if '/' in str(x) else str(x)
        )
    elif 'experiment_name' in collapse_df.columns:
        collapse_df['match_name'] = collapse_df['experiment_name'].apply(
            lambda x: x.split('/')[-1] if '/' in str(x) else str(x)
        )
    else:
        return None, None
    
    merged = summary_df.merge(collapse_df, on='match_name', how='inner', suffixes=('', '_collapse'))
    
    if merged.empty:
        print("Warning: No experiments matched between VSN and collapse data")
        return None, None
    
    # Compute correlations between VSN metrics and collapse metrics
    vsn_cols = ['avg_concentration', 'std_concentration', 'concentration_trend', 'feature_stability']
    collapse_cols = [c for c in merged.columns if any(x in c.lower() for x in 
                     ['collapse', 'dir_acc', 'sharpe', 'pred_std', 'composite'])]
    
    if not collapse_cols:
        return merged, None
    
    # Filter to numeric columns
    numeric_cols = [c for c in vsn_cols + collapse_cols if c in merged.columns and 
                    pd.api.types.is_numeric_dtype(merged[c])]
    
    if len(numeric_cols) < 2:
        return merged, None
    
    corr_matrix = merged[numeric_cols].corr()
    
    return merged, corr_matrix


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_feature_importance_heatmap(importance_df: pd.DataFrame, 
                                     feature_cols: List[str],
                                     output_dir: Path,
                                     top_n: int = 10):
    """
    Heatmap of feature importance by period (averaged across experiments).
    """
    if importance_df.empty or not feature_cols:
        return
    
    # Average weights by period across experiments
    period_means = importance_df.groupby('period')[feature_cols].mean()
    
    if period_means.empty:
        return
    
    # Select top features by overall mean
    overall_means = period_means.mean().sort_values(ascending=False)
    top_features = overall_means.head(top_n).index.tolist()
    
    plot_df = period_means[top_features]
    
    fig, ax = plt.subplots(figsize=(max(10, len(top_features) * 0.8), 
                                     max(6, len(plot_df) * 0.5)))
    
    sns.heatmap(plot_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Mean VSN Weight'})
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Period')
    ax.set_title('Feature Importance by Period (Averaged Across Experiments)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_path = output_dir / 'vsn_feature_importance_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_concentration_by_phase(summary_df: pd.DataFrame, output_dir: Path):
    """Box plot of concentration by phase. Only generates if multiple phases present."""
    if summary_df.empty or 'phase' not in summary_df.columns:
        return
    
    # Remove None phases
    plot_df = summary_df[summary_df['phase'].notna()].copy()
    
    if plot_df.empty:
        return
    
    # Skip if only one phase - plot is meaningless
    n_phases = plot_df['phase'].nunique()
    if n_phases < 2:
        print(f"  Skipping concentration_by_phase plot (only {n_phases} phase)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Order phases
    phase_order = sorted(plot_df['phase'].unique())
    
    sns.boxplot(data=plot_df, x='phase', y='avg_concentration', 
                order=phase_order, ax=ax)
    
    # Overlay points
    sns.stripplot(data=plot_df, x='phase', y='avg_concentration',
                  order=phase_order, color='black', alpha=0.5, size=4, ax=ax)
    
    ax.set_xlabel('Phase')
    ax.set_ylabel('Average Concentration (Herfindahl Index)')
    ax.set_title('Feature Concentration by Phase')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_path = output_dir / 'vsn_concentration_by_phase.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_concentration_over_time(importance_df: pd.DataFrame, output_dir: Path):
    """Line plot of concentration over periods, by phase."""
    if importance_df.empty or 'concentration' not in importance_df.columns:
        return
    
    # Extract phase from experiment name
    df = importance_df.copy()
    df['phase'] = df['experiment'].apply(lambda x: x.split('/')[0] if '/' in x else 'unknown')
    
    # Average by period and phase
    period_phase = df.groupby(['period', 'phase'])['concentration'].mean().reset_index()
    
    if period_phase.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    phases = sorted(period_phase['phase'].unique())
    for phase in phases:
        phase_data = period_phase[period_phase['phase'] == phase].sort_values('period')
        ax.plot(phase_data['period'], phase_data['concentration'], 
                marker='o', label=phase, linewidth=2)
    
    ax.set_xlabel('Period')
    ax.set_ylabel('Average Concentration')
    ax.set_title('Feature Concentration Over Time by Phase')
    ax.legend(title='Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_path = output_dir / 'vsn_concentration_timeline.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_shift_frequency(shifts_by_period: pd.DataFrame, output_dir: Path):
    """Bar chart of shift frequency by period transition."""
    if shifts_by_period.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create transition labels
    shifts_by_period = shifts_by_period.copy()
    shifts_by_period['transition'] = shifts_by_period['period_from'] + ' → ' + shifts_by_period['period_to']
    
    # Sort by period_from
    shifts_by_period = shifts_by_period.sort_values('period_from')
    
    bars = ax.bar(shifts_by_period['transition'], shifts_by_period['n_experiments'])
    
    # Color by average signals
    if 'avg_signals' in shifts_by_period.columns:
        colors = plt.cm.Reds(shifts_by_period['avg_signals'] / shifts_by_period['avg_signals'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    ax.set_xlabel('Period Transition')
    ax.set_ylabel('Number of Experiments with Shift')
    ax.set_title('Feature Selection Shifts by Period')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_path = output_dir / 'vsn_shift_frequency.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_correlation_heatmap(corr_matrix: Optional[pd.DataFrame], output_dir: Path):
    """Correlation heatmap between VSN metrics and collapse metrics."""
    if corr_matrix is None or corr_matrix.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, mask=mask, ax=ax,
                square=True)
    
    ax.set_title('VSN Metrics vs Collapse/Performance Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_path = output_dir / 'vsn_collapse_correlation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================================
# TEXT REPORT
# ============================================================================

def generate_text_report(summary_df: pd.DataFrame,
                         feature_rankings: pd.DataFrame,
                         shifts: List[dict],
                         shifts_by_period: pd.DataFrame,
                         merged_df: Optional[pd.DataFrame],
                         output_dir: Path,
                         thresholds: dict):
    """Generate human-readable summary report."""
    report_path = output_dir / 'vsn_summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VSN PATTERN ANALYSIS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        # Overview
        f.write("OVERVIEW\n")
        f.write("-"*70 + "\n")
        f.write(f"Total experiments analyzed: {len(summary_df)}\n")
        if 'phase' in summary_df.columns:
            phase_counts = summary_df['phase'].value_counts()
            f.write("By phase:\n")
            for phase, count in phase_counts.items():
                f.write(f"  {phase}: {count}\n")
        f.write("\n")
        
        # Shift detection thresholds
        f.write("SHIFT DETECTION THRESHOLDS\n")
        f.write("-"*70 + "\n")
        for key, val in thresholds.items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        # Feature rankings
        if not feature_rankings.empty:
            f.write("CROSS-EXPERIMENT FEATURE IMPORTANCE\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Rank':<6} {'Feature':<25} {'Mean Wt':<10} {'Std':<10} {'Consistency':<12}\n")
            f.write("-"*70 + "\n")
            for _, row in feature_rankings.head(15).iterrows():
                f.write(f"{row['rank']:<6} {row['feature']:<25} {row['mean_weight']:.4f}    "
                       f"{row['std_weight']:.4f}    {row['consistency']:.3f}\n")
            f.write("\n")
            f.write("Note: Consistency = 1/(1+CV), higher = more stable across experiments/periods\n\n")
        
        # Concentration stats
        f.write("CONCENTRATION STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Average concentration: {summary_df['avg_concentration'].mean():.4f} "
               f"(± {summary_df['avg_concentration'].std():.4f})\n")
        f.write(f"Range: {summary_df['min_concentration'].min():.4f} - "
               f"{summary_df['max_concentration'].max():.4f}\n")
        
        # By phase if available
        if 'phase' in summary_df.columns:
            f.write("\nBy phase:\n")
            phase_stats = summary_df.groupby('phase')['avg_concentration'].agg(['mean', 'std'])
            for phase, row in phase_stats.iterrows():
                f.write(f"  {phase}: {row['mean']:.4f} (± {row['std']:.4f})\n")
        f.write("\n")
        
        # Feature stability
        f.write("FEATURE STABILITY\n")
        f.write("-"*70 + "\n")
        f.write("Feature stability = fraction of periods where same feature was #1\n")
        f.write(f"Average stability: {summary_df['feature_stability'].mean():.3f}\n")
        
        # Most stable experiments
        stable_exps = summary_df.nlargest(5, 'feature_stability')
        f.write("\nMost stable feature selection:\n")
        for _, row in stable_exps.iterrows():
            f.write(f"  {row['experiment_name']}: {row['feature_stability']:.2f} "
                   f"(top: {row['top_feature']})\n")
        f.write("\n")
        
        # Regime shifts
        if shifts:
            f.write("REGIME SHIFTS DETECTED\n")
            f.write("-"*70 + "\n")
            f.write(f"Total shifts detected: {len(shifts)}\n")
            
            # Count by signal type
            all_signals = []
            for s in shifts:
                all_signals.extend(s['signals'].split(','))
            from collections import Counter
            signal_counts = Counter(all_signals)
            f.write("By signal type:\n")
            for sig, count in signal_counts.most_common():
                f.write(f"  {sig}: {count}\n")
            f.write("\n")
            
            # Summary by period transition
            if not shifts_by_period.empty:
                f.write("Shifts by period transition:\n")
                for _, row in shifts_by_period.head(10).iterrows():
                    f.write(f"  {row['period_from']} → {row['period_to']}: "
                           f"{int(row['n_experiments'])} experiments "
                           f"(avg {row['avg_signals']:.1f} signals)\n")
            f.write("\n")
            
            # Notable shifts (high signal count)
            high_signal_shifts = [s for s in shifts if s['n_signals'] >= 3]
            if high_signal_shifts:
                f.write("Strong shifts (3+ signals):\n")
                for s in sorted(high_signal_shifts, key=lambda x: -x['n_signals'])[:10]:
                    f.write(f"  {s['experiment']}: {s['period_from']}→{s['period_to']} "
                           f"[{s['signals']}]\n")
                    if s.get('top_feature_from') and s.get('top_feature_to'):
                        f.write(f"    Top feature: {s['top_feature_from']} → {s['top_feature_to']}\n")
            f.write("\n")
        else:
            f.write("REGIME SHIFTS: None detected with current thresholds\n\n")
        
        # Correlation with collapse
        if merged_df is not None and len(merged_df) > 0:
            f.write("CORRELATION WITH COLLAPSE METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Matched experiments: {len(merged_df)}\n\n")
            
            # Report key correlations
            def safe_corr(df, col1, col2):
                if col1 in df.columns and col2 in df.columns:
                    valid = df[[col1, col2]].dropna()
                    if len(valid) > 2 and valid[col1].std() > 0 and valid[col2].std() > 0:
                        return valid[col1].corr(valid[col2])
                return None
            
            correlations = [
                ('avg_concentration', 'dir_acc', 'Concentration vs Directional Accuracy'),
                ('avg_concentration', 'composite_score', 'Concentration vs Composite Score'),
                ('feature_stability', 'dir_acc', 'Feature Stability vs Directional Accuracy'),
                ('concentration_trend', 'dir_acc', 'Concentration Trend vs Directional Accuracy'),
            ]
            
            for col1, col2, label in correlations:
                corr = safe_corr(merged_df, col1, col2)
                if corr is not None:
                    f.write(f"{label}: {corr:.3f}\n")
            f.write("\n")
        
        # Notable experiments
        f.write("NOTABLE EXPERIMENTS\n")
        f.write("-"*70 + "\n")
        
        # Highest concentration
        f.write("Highest concentration (focused on few features):\n")
        for _, row in summary_df.nlargest(5, 'avg_concentration').iterrows():
            f.write(f"  {row['experiment_name']}: {row['avg_concentration']:.4f}\n")
        f.write("\n")
        
        # Lowest concentration
        f.write("Lowest concentration (diffuse attention):\n")
        for _, row in summary_df.nsmallest(5, 'avg_concentration').iterrows():
            f.write(f"  {row['experiment_name']}: {row['avg_concentration']:.4f}\n")
        f.write("\n")
        
        # Strongest concentration increase over time
        f.write("Largest concentration increase over time:\n")
        for _, row in summary_df.nlargest(5, 'concentration_trend').iterrows():
            f.write(f"  {row['experiment_name']}: {row['concentration_trend']:+.4f}/period\n")
        
    print(f"Saved: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Summarize VSN patterns across experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('paths', nargs='+',
                        help='Paths to search for VSN results (supports globs)')
    parser.add_argument('--compare-periods', nargs='*', default=None,
                        help='Specific periods to compare for shift detection')
    parser.add_argument('--cosine-threshold', type=float, default=0.95,
                        help='Cosine similarity threshold for shift detection')
    parser.add_argument('--l2-threshold', type=float, default=0.03,
                        help='L2 distance threshold for shift detection')
    parser.add_argument('--concentration-threshold', type=float, default=0.05,
                        help='Concentration change threshold for shift detection')
    parser.add_argument('--min-signals', type=int, default=1,
                        help='Minimum signals to report a shift')
    parser.add_argument('--output', type=str, default='reports',
                        help='Output directory for reports and plots')
    parser.add_argument('--experiments-csv', type=str, default=None,
                        help='Path to experiments summary CSV for collapse correlation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = {
        'cosine': args.cosine_threshold,
        'l2': args.l2_threshold,
        'concentration': args.concentration_threshold,
        'min_signals': args.min_signals,
    }
    
    print("="*70)
    print("VSN PATTERN ANALYSIS")
    print("="*70)
    print(f"\nShift detection thresholds:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")
    
    # Find results
    print("\nSearching for VSN analysis results...")
    result_files = find_vsn_results(args.paths)
    print(f"Found {len(result_files)} experiments with VSN results")
    
    if not result_files:
        print("No VSN results found. Run analyze_vsn_weights.py first.")
        return
    
    # Load data
    print("\nLoading VSN data...")
    vsn_data = load_vsn_results(result_files)
    print(f"Successfully loaded {len(vsn_data)} experiments")
    
    # Compute summaries
    print("\nComputing experiment summaries...")
    summary_df = compute_experiment_summary(vsn_data)
    
    print("Building feature importance matrix...")
    importance_df, feature_cols = compute_feature_importance_matrix(vsn_data)
    
    print("Computing cross-experiment feature rankings...")
    feature_rankings = compute_cross_experiment_feature_rankings(importance_df, feature_cols)
    
    # Detect shifts
    print("\nDetecting feature selection shifts...")
    if args.compare_periods:
        periods = args.compare_periods
        pairs = [(periods[i], periods[i+1]) for i in range(len(periods)-1)]
        print(f"Comparing specific period pairs: {pairs}")
    else:
        pairs = None
        print("Comparing all consecutive periods")
    
    shifts = detect_feature_selection_shifts(
        vsn_data, pairs,
        args.cosine_threshold, args.l2_threshold, args.concentration_threshold
    )
    shifts = [s for s in shifts if s['n_signals'] >= args.min_signals]
    shifts_by_period = summarize_shifts_by_period(shifts)
    
    print(f"Detected {len(shifts)} shifts across {len(set(s['experiment'] for s in shifts))} experiments")
    
    # Correlate with collapse
    print("\nCorrelating with collapse metrics...")
    collapse_df = load_collapse_data(args.experiments_csv)
    merged_df, corr_matrix = correlate_vsn_with_collapse(summary_df, collapse_df)
    if merged_df is not None:
        print(f"Matched {len(merged_df)} experiments with collapse data")
    
    # Save CSVs
    print("\nSaving results...")
    summary_df.to_csv(output_dir / 'vsn_summary.csv', index=False)
    print(f"Saved: {output_dir / 'vsn_summary.csv'}")
    
    if not feature_rankings.empty:
        feature_rankings.to_csv(output_dir / 'vsn_feature_importance.csv', index=False)
        print(f"Saved: {output_dir / 'vsn_feature_importance.csv'}")
    
    if shifts:
        pd.DataFrame(shifts).to_csv(output_dir / 'vsn_regime_shifts.csv', index=False)
        print(f"Saved: {output_dir / 'vsn_regime_shifts.csv'}")
    
    if merged_df is not None:
        merged_df.to_csv(output_dir / 'vsn_with_collapse.csv', index=False)
        print(f"Saved: {output_dir / 'vsn_with_collapse.csv'}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_feature_importance_heatmap(importance_df, feature_cols, output_dir)
    plot_concentration_by_phase(summary_df, output_dir)
    plot_concentration_over_time(importance_df, output_dir)
    plot_shift_frequency(shifts_by_period, output_dir)
    plot_correlation_heatmap(corr_matrix, output_dir)
    
    # Text report
    print("\nGenerating text report...")
    generate_text_report(summary_df, feature_rankings, shifts, shifts_by_period, 
                         merged_df, output_dir, thresholds)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - vsn_summary.csv")
    print(f"  - vsn_feature_importance.csv")
    if shifts:
        print(f"  - vsn_regime_shifts.csv")
    print(f"  - vsn_summary_report.txt")
    print(f"  - vsn_feature_importance_heatmap.png")
    # Only list concentration_by_phase if multiple phases
    n_phases = summary_df['phase'].nunique() if 'phase' in summary_df.columns else 0
    if n_phases >= 2:
        print(f"  - vsn_concentration_by_phase.png")
    print(f"  - vsn_concentration_timeline.png")
    if shifts:
        print(f"  - vsn_shift_frequency.png")
    if corr_matrix is not None:
        print(f"  - vsn_collapse_correlation.png")


if __name__ == '__main__':
    main()