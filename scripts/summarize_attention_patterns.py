"""
Summarize attention patterns across multiple experiments.

Flexible analysis tool that:
- Works with any period granularity (yearly, monthly, custom)
- Detects attention shifts between consecutive periods
- Correlates attention metrics with model collapse
- Generates summary reports and visualizations

Usage:
    # All experiments
    python scripts/summarize_attention_patterns.py experiments/
    
    # Single phase
    python scripts/summarize_attention_patterns.py experiments/00_baseline_exploration/
    
    # Multiple specific paths
    python scripts/summarize_attention_patterns.py \\
        experiments/00_baseline_exploration/ \\
        experiments/04_staleness_attention/
    
    # Focus on specific period comparisons
    python scripts/summarize_attention_patterns.py experiments/ \\
        --compare-periods "2021" "2022" "2023" "2024"
    
    # Detailed per-experiment reports
    python scripts/summarize_attention_patterns.py experiments/ --detailed
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine


# ============================================================================
# DATA LOADING
# ============================================================================

def find_attention_results(paths):
    """
    Recursively find all attention analysis result files.
    
    Parameters:
    -----------
    paths : list of str or Path
        Paths to search (experiments/, single phase dir, single experiment, etc.)
    
    Returns:
    --------
    dict: {experiment_name: result_file_path}
    """
    import re
    results = {}
    phase_pattern = re.compile(r'^\d{2}[a-z]?_')  # Pattern for 00_, 01_, 02_, 02b_, etc.
    
    for path_pattern in paths:
        # Expand globs
        expanded_paths = glob.glob(path_pattern, recursive=True)
        if not expanded_paths:
            expanded_paths = [path_pattern]
        
        for path_str in expanded_paths:
            path = Path(path_str)
            
            if not path.exists():
                print(f"Warning: Path does not exist: {path}")
                continue
            
            # Check if this path itself is a phase directory
            is_phase_dir = phase_pattern.match(path.name)
            
            # Search for attention_analysis_*/attention_analysis_results.json
            if path.is_file() and path.name == 'attention_analysis_results.json':
                # Direct file provided
                exp_name = path.parent.parent.name
                if is_phase_dir:
                    exp_name = f"{path.name}/{exp_name}"
                results[exp_name] = path
            else:
                # Directory - search recursively
                for result_file in path.rglob('attention_analysis_*/attention_analysis_results.json'):
                    # Extract experiment name from path
                    # e.g., experiments/00_baseline/sweep2_h16/attention_analysis_year/results.json
                    exp_path = result_file.relative_to(path).parts
                    if len(exp_path) >= 3:
                        exp_name = '/'.join(exp_path[:-2])  # Remove attention_analysis_*/ and filename
                    else:
                        exp_name = result_file.parent.parent.name
                    
                    # If we're searching within a phase directory, prepend the phase name
                    if is_phase_dir:
                        exp_name = f"{path.name}/{exp_name}"
                    
                    results[exp_name] = result_file
    
    return results


def load_attention_results(result_files):
    """
    Load all attention analysis results.
    
    Returns:
    --------
    dict: {experiment_name: parsed_json_data}
    """
    data = {}
    
    for exp_name, file_path in result_files.items():
        try:
            with open(file_path, 'r') as f:
                data[exp_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    return data


def infer_period_type(periods):
    """
    Infer period granularity from period names.
    
    Returns: 'yearly' | 'custom' | 'mixed'
    """
    if not periods:
        return 'unknown'
    
    # Check if all are 4-digit years
    if all(p.isdigit() and len(p) == 4 for p in periods):
        return 'yearly'
    
    # Check for common custom patterns
    if any('-' in p or '_' in p for p in periods):
        return 'custom'
    
    return 'mixed'


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def compute_summary_stats(attention_data):
    """
    Compute per-experiment summary statistics.
    
    Returns:
    --------
    DataFrame with columns:
        - experiment_name
        - period_type
        - n_periods
        - avg_entropy
        - std_entropy
        - min_entropy
        - max_entropy
        - avg_concentration
        - entropy_trend (slope of entropy over time)
        - top_timestep_overall (most attended timestep across all periods)
    """
    rows = []
    
    for exp_name, data in attention_data.items():
        period_stats = data.get('period_statistics', {})
        
        if not period_stats:
            continue
        
        periods = sorted(period_stats.keys())
        period_type = infer_period_type(periods)
        
        # Extract entropy values
        entropies = [period_stats[p]['entropy_mean'] for p in periods]
        concentrations = [period_stats[p]['attention_concentration'] for p in periods]
        
        # Compute entropy trend (linear fit)
        if len(entropies) > 1:
            x = np.arange(len(entropies))
            entropy_trend = np.polyfit(x, entropies, 1)[0]  # Slope
        else:
            entropy_trend = 0.0
        
        # Find most attended timestep overall
        all_attention = np.zeros(len(period_stats[periods[0]]['mean_attention']))
        for period in periods:
            all_attention += np.array(period_stats[period]['mean_attention'])
        top_timestep_idx = np.argmax(all_attention)
        
        rows.append({
            'experiment_name': exp_name,
            'period_type': period_type,
            'n_periods': len(periods),
            'avg_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'min_entropy': float(np.min(entropies)),
            'max_entropy': float(np.max(entropies)),
            'avg_concentration': float(np.mean(concentrations)),
            'entropy_trend': float(entropy_trend),
            'top_timestep': top_timestep_idx,
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# ATTENTION SHIFT DETECTION
# ============================================================================

def detect_attention_shifts(attention_data, compare_periods=None, 
                          cosine_threshold=0.96, l2_threshold=0.025,
                          entropy_threshold=0.05, top_timestep_threshold=5):
    """
    Detect significant attention pattern changes between periods using multiple signals.
    
    Parameters:
    -----------
    attention_data : dict
        Loaded attention results
    compare_periods : list of (period1, period2) tuples, optional
        Specific period pairs to compare. If None, compare all consecutive periods.
    cosine_threshold : float
        Cosine similarity threshold below which a shift is detected
    l2_threshold : float
        L2 distance threshold above which a shift is detected
    entropy_threshold : float
        Absolute entropy change threshold above which a shift is detected
    top_timestep_threshold : int
        Position change in top-attended timestep above which a shift is detected
    
    Returns:
    --------
    dict: {experiment_name: [shift_info_dicts]}
    """
    shifts = {}
    
    for exp_name, data in attention_data.items():
        period_stats = data.get('period_statistics', {})
        comparisons = data.get('period_comparisons', {})
        
        if not period_stats or not comparisons:
            continue
        
        exp_shifts = []
        periods = sorted(period_stats.keys())
        
        if compare_periods is None:
            # Compare all consecutive periods
            pairs_to_check = [(periods[i], periods[i+1]) for i in range(len(periods)-1)]
        else:
            # Only compare specified periods (if they exist)
            pairs_to_check = [
                (p1, p2) for p1, p2 in compare_periods 
                if p1 in periods and p2 in periods
            ]
        
        for p1, p2 in pairs_to_check:
            # Look for comparison in period_comparisons
            comp_key = f"{p1}_vs_{p2}"
            if comp_key not in comparisons:
                continue
            
            comp = comparisons[comp_key]
            
            # Extract metrics
            cosine_sim = comp['cosine_similarity']
            l2_distance = comp['l2_distance']
            entropy_change = comp['entropy_change']
            concentration_change = comp['concentration_change']
            
            # Find top timesteps for each period
            attn1 = np.array(period_stats[p1]['mean_attention'])
            attn2 = np.array(period_stats[p2]['mean_attention'])
            
            top_idx1 = np.argmax(attn1)
            top_idx2 = np.argmax(attn2)
            top_timestep_delta = abs(top_idx2 - top_idx1)
            
            # Evaluate each signal
            signals = {
                'cosine': cosine_sim < cosine_threshold,
                'l2': l2_distance > l2_threshold,
                'entropy': abs(entropy_change) > entropy_threshold,
                'top_timestep': top_timestep_delta > top_timestep_threshold,
            }
            
            n_signals_fired = sum(signals.values())
            shift_detected = n_signals_fired > 0  # At least one signal
            
            exp_shifts.append({
                'period1': p1,
                'period2': p2,
                'cosine_similarity': cosine_sim,
                'l2_distance': l2_distance,
                'entropy_change': entropy_change,
                'concentration_change': concentration_change,
                'top_timestep1': f"t-{len(attn1) - top_idx1 - 1}",
                'top_timestep2': f"t-{len(attn2) - top_idx2 - 1}",
                'top_timestep_delta': int(top_timestep_delta),
                'signal_cosine': signals['cosine'],
                'signal_l2': signals['l2'],
                'signal_entropy': signals['entropy'],
                'signal_top_timestep': signals['top_timestep'],
                'n_signals_fired': n_signals_fired,
                'shift_detected': shift_detected,
            })
        
        if exp_shifts:
            shifts[exp_name] = exp_shifts
    
    return shifts


def summarize_shifts(shift_data, min_signals=1):
    """
    Create a summary DataFrame of all detected shifts.
    
    Parameters:
    -----------
    shift_data : dict
        Output from detect_attention_shifts
    min_signals : int
        Minimum number of signals that must fire to include in summary
    
    Returns:
    --------
    DataFrame with one row per shift meeting the minimum signal threshold
    """
    rows = []
    
    for exp_name, shifts in shift_data.items():
        for shift in shifts:
            if shift['n_signals_fired'] >= min_signals:
                rows.append({
                    'experiment_name': exp_name,
                    'period1': shift['period1'],
                    'period2': shift['period2'],
                    'cosine_similarity': shift['cosine_similarity'],
                    'l2_distance': shift['l2_distance'],
                    'entropy_change': shift['entropy_change'],
                    'concentration_change': shift['concentration_change'],
                    'top_timestep1': shift['top_timestep1'],
                    'top_timestep2': shift['top_timestep2'],
                    'top_timestep_delta': shift['top_timestep_delta'],
                    'signal_cosine': shift['signal_cosine'],
                    'signal_l2': shift['signal_l2'],
                    'signal_entropy': shift['signal_entropy'],
                    'signal_top_timestep': shift['signal_top_timestep'],
                    'n_signals_fired': shift['n_signals_fired'],
                })
    
    return pd.DataFrame(rows)


# ============================================================================
# CORRELATION WITH COLLAPSE
# ============================================================================

def load_collapse_data(experiments_csv='results/experiments_summary.csv'):
    """
    Load collapse information from experiments summary.
    
    Returns:
    --------
    DataFrame with experiment_name, phase, and collapse metrics
    """
    if not os.path.exists(experiments_csv):
        print(f"Warning: {experiments_csv} not found, skipping collapse correlation")
        return None
    
    try:
        df = pd.read_csv(experiments_csv)
        
        # Extract relevant columns (handle different naming conventions)
        cols = ['experiment_name']
        
        # CRITICAL: Keep 'phase' column for merging
        if 'phase' in df.columns:
            cols.append('phase')
        
        # Performance metrics (try multiple names)
        for col_options in [
            ['dir_acc', 'dir_accuracy', 'directional_accuracy'],
            ['sharpe_ratio', 'sharpe'],
            ['test_rmse', 'rmse'],
            ['test_mae', 'mae'],
        ]:
            for col in col_options:
                if col in df.columns and col not in cols:
                    cols.append(col)
                    break
        
        # Collapse metrics - updated with 5-mode evaluation
        for col in ['has_any_collapse', 'has_strong_collapse', 'has_degradation', 'has_unidirectional',
                    'healthy_pct', 'degraded_pct', 'unidirectional_pct', 'weak_collapse_pct', 'strong_collapse_pct',
                    'problematic_pct', 'pred_std', 'composite_score',
                    # Legacy columns (may not exist)
                    'has_collapse', 'collapsed', 'collapse_type', 'degraded_pct', 'collapse_pct']:
            if col in df.columns and col not in cols:
                cols.append(col)
        
        return df[cols]
    
    except Exception as e:
        print(f"Warning: Failed to load {experiments_csv}: {e}")
        return None


def correlate_attention_collapse(summary_df, collapse_df):
    """
    Merge attention summary with collapse data and compute correlations.
    
    Returns:
    --------
    merged DataFrame and correlation matrix
    """
    if collapse_df is None:
        return None, None
    
    # Extract phase prefix from attention experiment_name
    # e.g., "01_staleness_features_fixed/sweep_stale_h18" -> "01_staleness_features_fixed"
    # e.g., "00_baseline_exploration/sweep2_h16" -> "00_baseline_exploration"
    summary_df = summary_df.copy()
    summary_df['phase_prefix'] = summary_df['experiment_name'].str.split('/').str[0]
    
    # Extract experiment name (without phase prefix) for matching
    # e.g., "02b_vintage_sweep/baseline_h16_drop0.10" -> "baseline_h16_drop0.10"
    summary_df['exp_name_only'] = summary_df['experiment_name'].str.split('/').str[-1]
    
    # Check if collapse_df has a 'phase' column to merge on
    if 'phase' not in collapse_df.columns:
        print(f"\nWARNING: No 'phase' column in collapse data for merging")
        print(f"  Available columns: {collapse_df.columns.tolist()}")
        return None, None
    
    # Debug: Check experiment name formats
    print(f"\nDEBUG: Merging attention and collapse data")
    print(f"  Attention experiments (first 3): {summary_df['experiment_name'].head(3).tolist()}")
    print(f"  Attention exp_name_only (first 3): {summary_df['exp_name_only'].head(3).tolist()}")
    print(f"  Attention phase prefixes (first 3): {summary_df['phase_prefix'].head(3).tolist()}")
    print(f"  Collapse experiments (first 3): {collapse_df['experiment_name'].head(3).tolist()}")
    print(f"  Collapse phase values (first 3): {collapse_df['phase'].head(3).tolist()}")
    
    # Merge on BOTH phase and experiment name
    merged = summary_df.merge(
        collapse_df, 
        left_on=['phase_prefix', 'exp_name_only'], 
        right_on=['phase', 'experiment_name'], 
        how='inner',
        suffixes=('_attention', '_collapse')
    )
    
    print(f"  Merge result: {len(merged)} rows (from {len(summary_df)} attention + {len(collapse_df)} collapse)")
    
    if len(merged) == 0:
        print(f"  WARNING: No matching experiments! Check name formats.")
        return None, None
    
    # Compute correlations for numeric columns
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    
    # Drop columns with all NaN or zero variance
    valid_cols = []
    for col in numeric_cols:
        if merged[col].notna().sum() > 1 and merged[col].std() > 0:
            valid_cols.append(col)
    
    if not valid_cols:
        print("  WARNING: No valid numeric columns for correlation")
        return merged, None
    
    corr_matrix = merged[valid_cols].corr()
    
    print(f"  Computed correlations for {len(valid_cols)} numeric columns")
    
    return merged, corr_matrix


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_entropy_distribution(summary_df, output_dir):
    """Plot entropy distributions by phase with clear statistical comparison."""
    # Extract phase from experiment_name (first part before /)
    summary_df['phase'] = summary_df['experiment_name'].str.split('/').str[0]
    
    phases = sorted(summary_df['phase'].unique())
    n_phases = len(phases)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Compute statistics
    stats = []
    for phase in phases:
        phase_data = summary_df[summary_df['phase'] == phase]
        stats.append({
            'phase': phase,
            'n': len(phase_data),
            'entropy_mean': phase_data['avg_entropy'].mean(),
            'entropy_std': phase_data['avg_entropy'].std(),
            'entropy_median': phase_data['avg_entropy'].median(),
            'conc_mean': phase_data['avg_concentration'].mean(),
            'conc_std': phase_data['avg_concentration'].std(),
            'conc_median': phase_data['avg_concentration'].median(),
        })
    
    # Plot 1: Entropy mean comparison
    ax = axes[0, 0]
    means = [s['entropy_mean'] for s in stats]
    stds = [s['entropy_std'] for s in stats]
    bars = ax.bar(range(n_phases), means, yerr=stds, capsize=10, 
                   color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=2)
    
    # Annotate with n and mean±std
    for i, (bar, s) in enumerate(zip(bars, stats)):
        height = bar.get_height()
        ax.text(i, height + stds[i] + 0.02, 
                f"n={s['n']}\nμ={s['entropy_mean']:.2f}±{s['entropy_std']:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(n_phases))
    ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')
    ax.set_title('Entropy: Mean ± Std Dev', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(means) + max(stds) + 0.3)
    
    # Plot 2: Entropy distribution (violin)
    ax = axes[0, 1]
    data_by_phase = [summary_df[summary_df['phase'] == p]['avg_entropy'].values for p in phases]
    parts = ax.violinplot(data_by_phase, positions=range(n_phases), 
                          showmeans=True, showmedians=True, widths=0.6)
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.5)
        pc.set_edgecolor('darkblue')
    
    ax.set_xticks(range(n_phases))
    ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Average Entropy', fontsize=12, fontweight='bold')
    ax.set_title('Entropy Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Concentration mean comparison
    ax = axes[1, 0]
    means = [s['conc_mean'] for s in stats]
    stds = [s['conc_std'] for s in stats]
    bars = ax.bar(range(n_phases), means, yerr=stds, capsize=10,
                   color='coral', alpha=0.7, edgecolor='darkred', linewidth=2)
    
    for i, (bar, s) in enumerate(zip(bars, stats)):
        height = bar.get_height()
        ax.text(i, height + stds[i] + 0.002,
                f"n={s['n']}\nμ={s['conc_mean']:.3f}±{s['conc_std']:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(n_phases))
    ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Average Concentration', fontsize=12, fontweight='bold')
    ax.set_title('Concentration: Mean ± Std Dev', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(means) + max(stds) + 0.01)
    
    # Plot 4: Concentration distribution (violin)
    ax = axes[1, 1]
    data_by_phase = [summary_df[summary_df['phase'] == p]['avg_concentration'].values for p in phases]
    parts = ax.violinplot(data_by_phase, positions=range(n_phases),
                          showmeans=True, showmedians=True, widths=0.6)
    for pc in parts['bodies']:
        pc.set_facecolor('coral')
        pc.set_alpha(0.5)
        pc.set_edgecolor('darkred')
    
    ax.set_xticks(range(n_phases))
    ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Average Concentration', fontsize=12, fontweight='bold')
    ax.set_title('Concentration Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'attention_metrics_by_phase.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_shift_timeline(shift_df, output_dir):
    """Visualize when shifts occur across experiments."""
    if shift_df.empty:
        print("No shifts detected, skipping timeline plot")
        return
    
    # Filter to consecutive period transitions only
    # Assumes periods are numeric years or can be sorted
    def is_consecutive(p1, p2):
        try:
            # Try parsing as years
            y1, y2 = int(p1), int(p2)
            return y2 == y1 + 1
        except:
            # For custom periods, can't determine consecutiveness
            return True
    
    consecutive_shifts = shift_df[
        shift_df.apply(lambda row: is_consecutive(row['period1'], row['period2']), axis=1)
    ]
    
    if consecutive_shifts.empty:
        print("No consecutive period shifts detected, skipping timeline plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by period transition
    shift_counts = consecutive_shifts.groupby(['period1', 'period2']).size()
    
    # Get all unique periods in order
    all_periods = sorted(set(
        consecutive_shifts['period1'].tolist() + 
        consecutive_shifts['period2'].tolist()
    ))
    
    # Create bar chart instead of heatmap for clarity
    transitions = []
    counts = []
    
    for p1, p2 in zip(all_periods[:-1], all_periods[1:]):
        transition_key = (p1, p2)
        if transition_key in shift_counts.index:
            transitions.append(f"{p1}→{p2}")
            counts.append(shift_counts[transition_key])
    
    if not transitions:
        print("No transition data to plot")
        return
    
    # Create bar plot
    bars = ax.bar(range(len(transitions)), counts, color='darkred', alpha=0.7, edgecolor='black')
    
    # Annotate bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(transitions, rotation=0, ha='center', fontsize=12)
    ax.set_xlabel('Period Transition', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Experiments', fontsize=13, fontweight='bold')
    ax.set_title('Attention Shifts: Consecutive Period Transitions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'attention_shift_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_correlation_heatmap(corr_matrix, output_dir):
    """Plot correlation between attention metrics and collapse."""
    if corr_matrix is None or corr_matrix.empty:
        return
    
    # Filter to relevant correlations
    attention_metrics = ['avg_entropy', 'std_entropy', 'avg_concentration', 'entropy_trend']
    collapse_metrics = [
        # Binary flags
        'has_any_collapse', 'has_strong_collapse', 'has_degradation', 'has_unidirectional',
        # Percentages
        'healthy_pct', 'degraded_pct', 'unidirectional_pct', 'weak_collapse_pct', 'strong_collapse_pct',
        'problematic_pct',
        # Performance
        'dir_acc', 'sharpe_ratio', 'pred_std', 'composite_score',
        'test_rmse', 'test_mae',
        # Legacy (fallback)
        'has_collapse', 'collapsed'
    ]
    
    available_attention = [m for m in attention_metrics if m in corr_matrix.columns]
    available_collapse = [m for m in collapse_metrics if m in corr_matrix.columns]
    
    if not available_attention or not available_collapse:
        print(f"Insufficient metrics for correlation plot")
        print(f"  Available attention metrics: {available_attention}")
        print(f"  Available collapse metrics: {available_collapse}")
        return
    
    # Extract submatrix
    sub_corr = corr_matrix.loc[available_attention, available_collapse]
    
    # Debug: print actual correlation values
    print(f"\nCorrelation matrix shape: {sub_corr.shape}")
    print(f"Correlation range: [{sub_corr.min().min():.4f}, {sub_corr.max().max():.4f}]")
    print(f"Non-NaN values: {sub_corr.notna().sum().sum()}")
    
    # Check if all NaN
    if sub_corr.isna().all().all():
        print("Warning: All correlation values are NaN, skipping plot")
        return
    
    # Check if correlations are all weak
    max_abs_corr = sub_corr.abs().max().max()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Adjust color scale based on actual correlation range
    if max_abs_corr < 0.3:
        vmin, vmax = -0.3, 0.3
        center = 0
        print(f"Note: All correlations are weak (max |r| = {max_abs_corr:.3f})")
    else:
        vmin, vmax = -1, 1
        center = 0
    
    # Force annotation display and handle NaN
    sns.heatmap(sub_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=center, vmin=vmin, vmax=vmax, ax=ax,
                cbar_kws={'label': 'Correlation'},
                linewidths=1, linecolor='gray',
                mask=sub_corr.isna(),  # Mask NaN values
                annot_kws={'size': 10})
    
    ax.set_title('Attention Metrics vs Model Performance\n(Weak correlations indicate attention patterns are not strongly predictive of collapse)', 
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Performance Metrics', fontsize=11, labelpad=10)
    ax.set_ylabel('Attention Metrics', fontsize=11, labelpad=10)
    
    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'attention_collapse_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


# ============================================================================
# REPORTING
# ============================================================================

def generate_text_report(summary_df, shift_df, merged_df, output_dir, thresholds):
    """Generate a text report summarizing key findings."""
    report_path = Path(output_dir) / 'attention_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ATTENTION PATTERN ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total experiments analyzed: {len(summary_df)}\n")
        f.write(f"Period types: {summary_df['period_type'].value_counts().to_dict()}\n")
        f.write(f"\nEntropy statistics:\n")
        f.write(f"  Mean: {summary_df['avg_entropy'].mean():.4f}\n")
        f.write(f"  Std:  {summary_df['avg_entropy'].std():.4f}\n")
        f.write(f"  Min:  {summary_df['avg_entropy'].min():.4f}\n")
        f.write(f"  Max:  {summary_df['avg_entropy'].max():.4f}\n")
        f.write(f"\nConcentration statistics:\n")
        f.write(f"  Mean: {summary_df['avg_concentration'].mean():.4f}\n")
        f.write(f"  Std:  {summary_df['avg_concentration'].std():.4f}\n")
        f.write("\n")
        
        # Attention shifts with multi-signal analysis
        f.write("ATTENTION SHIFT DETECTION (MULTI-SIGNAL)\n")
        f.write("-"*70 + "\n")
        f.write("Shift Detection Signals:\n")
        f.write(f"  - Cosine similarity < {thresholds['cosine']:.3f} (direction change)\n")
        f.write(f"  - L2 distance > {thresholds['l2']:.3f} (magnitude change)\n")
        f.write(f"  - Entropy change > ±{thresholds['entropy']:.3f} (focus change)\n")
        f.write(f"  - Top timestep shift > {thresholds['top_timestep']} positions (strategy change)\n\n")
        
        if not shift_df.empty:
            f.write(f"Experiments with detected shifts: {shift_df['experiment_name'].nunique()}\n")
            f.write(f"Total shifts detected: {len(shift_df)}\n\n")
            
            # Signal effectiveness
            f.write("Signal Firing Statistics:\n")
            for signal in ['signal_cosine', 'signal_l2', 'signal_entropy', 'signal_top_timestep']:
                count = shift_df[signal].sum()
                pct = 100 * count / len(shift_df)
                signal_name = signal.replace('signal_', '').replace('_', ' ').title()
                f.write(f"  {signal_name}: {count} ({pct:.1f}%)\n")
            f.write("\n")
            
            # Multi-signal shifts (more confident)
            multi_signal = shift_df[shift_df['n_signals_fired'] >= 2]
            f.write(f"High-confidence shifts (2+ signals): {len(multi_signal)}\n")
            if len(multi_signal) > 0:
                strong_signal = shift_df[shift_df['n_signals_fired'] >= 3]
                f.write(f"Very strong shifts (3+ signals): {len(strong_signal)}\n")
            f.write("\n")
            
            # Most common shift periods
            shift_transitions = shift_df.groupby(['period1', 'period2']).size().sort_values(ascending=False)
            f.write("Most common shift transitions:\n")
            for (p1, p2), count in shift_transitions.head(10).items():
                # Get primary signals for this transition
                trans_shifts = shift_df[(shift_df['period1'] == p1) & (shift_df['period2'] == p2)]
                signals_fired = []
                for sig in ['signal_cosine', 'signal_l2', 'signal_entropy', 'signal_top_timestep']:
                    if trans_shifts[sig].sum() > count * 0.5:  # More than 50% of shifts
                        signals_fired.append(sig.replace('signal_', ''))
                f.write(f"  {p1} → {p2}: {count} experiments")
                if signals_fired:
                    f.write(f" (primary: {', '.join(signals_fired)})")
                f.write("\n")
            f.write("\n")
            
            # Notable individual shifts (3+ signals)
            strong_shifts = shift_df[shift_df['n_signals_fired'] >= 3].sort_values('n_signals_fired', ascending=False)
            if len(strong_shifts) > 0:
                f.write("Notable Shifts (3+ signals fired):\n")
                for _, row in strong_shifts.head(15).iterrows():
                    signals = []
                    if row['signal_cosine']:
                        signals.append('cosine')
                    if row['signal_l2']:
                        signals.append('l2')
                    if row['signal_entropy']:
                        signals.append('entropy')
                    if row['signal_top_timestep']:
                        signals.append(f"top_timestep(Δ{row['top_timestep_delta']})")
                    
                    f.write(f"  {row['experiment_name']} ({row['period1']}→{row['period2']}): ")
                    f.write(f"{', '.join(signals)}\n")
                f.write("\n")
            
            # Experiments with most shifts
            shifts_per_exp = shift_df['experiment_name'].value_counts()
            f.write("Experiments with most shifts:\n")
            for exp, count in shifts_per_exp.head(10).items():
                f.write(f"  {exp}: {count} shifts\n")
            f.write("\n")
        else:
            f.write("No significant attention shifts detected with current thresholds.\n\n")
        
        # Correlations with collapse
        if merged_df is not None:
            f.write("ATTENTION-COLLAPSE CORRELATIONS\n")
            f.write("-"*70 + "\n")
            
            # Helper function to safely compute correlation
            def safe_corr(df, col1, col2):
                try:
                    # Ensure both columns are numeric (converts bool to 0/1)
                    df_numeric = df[[col1, col2]].copy()
                    df_numeric[col1] = pd.to_numeric(df_numeric[col1], errors='coerce')
                    df_numeric[col2] = pd.to_numeric(df_numeric[col2], errors='coerce')
                    df_numeric = df_numeric.dropna()
                    if len(df_numeric) < 2:
                        return None
                    # Check for zero variance
                    if df_numeric[col1].std() == 0 or df_numeric[col2].std() == 0:
                        return None
                    corr_matrix = df_numeric.corr()
                    if corr_matrix.shape == (2, 2):
                        return corr_matrix.iloc[0, 1]
                    return None
                except:
                    return None
            
            # Try different column name variations - use new 5-mode collapse metrics
            correlations_computed = []
            
            # Binary collapse flags
            if 'has_any_collapse' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'has_any_collapse')
                if corr is not None:
                    f.write(f"Entropy vs Has Any Collapse: {corr:.3f}\n")
                    correlations_computed.append('has_any_collapse')
            
            if 'has_strong_collapse' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'has_strong_collapse')
                if corr is not None:
                    f.write(f"Entropy vs Has Strong Collapse: {corr:.3f}\n")
                    correlations_computed.append('has_strong_collapse')
            
            if 'has_degradation' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'has_degradation')
                if corr is not None:
                    f.write(f"Entropy vs Has Degradation: {corr:.3f}\n")
                    correlations_computed.append('has_degradation')
            
            # Percentage metrics (continuous - most informative)
            if 'problematic_pct' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'problematic_pct')
                if corr is not None:
                    f.write(f"Entropy vs Problematic %: {corr:.3f}\n")
                    correlations_computed.append('problematic_pct')
            
            if 'healthy_pct' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'healthy_pct')
                if corr is not None:
                    f.write(f"Entropy vs Healthy %: {corr:.3f}\n")
                    correlations_computed.append('healthy_pct')
            
            if 'weak_collapse_pct' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'weak_collapse_pct')
                if corr is not None:
                    f.write(f"Entropy vs Weak Collapse %: {corr:.3f}\n")
                    correlations_computed.append('weak_collapse_pct')
            
            if 'strong_collapse_pct' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'strong_collapse_pct')
                if corr is not None:
                    f.write(f"Entropy vs Strong Collapse %: {corr:.3f}\n")
                    correlations_computed.append('strong_collapse_pct')
            
            # Performance metrics
            if 'dir_acc' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'dir_acc')
                if corr is not None:
                    f.write(f"Entropy vs Directional Accuracy: {corr:.3f}\n")
                    correlations_computed.append('dir_acc')
            
            if 'sharpe_ratio' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'sharpe_ratio')
                if corr is not None:
                    f.write(f"Entropy vs Sharpe Ratio: {corr:.3f}\n")
                    correlations_computed.append('sharpe_ratio')
            
            if 'pred_std' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'pred_std')
                if corr is not None:
                    f.write(f"Entropy vs Prediction Std: {corr:.3f}\n")
                    correlations_computed.append('pred_std')
            
            if 'composite_score' in merged_df.columns:
                corr = safe_corr(merged_df, 'avg_entropy', 'composite_score')
                if corr is not None:
                    f.write(f"Entropy vs Composite Score: {corr:.3f}\n")
                    correlations_computed.append('composite_score')
            
            if not correlations_computed:
                f.write("Note: No valid correlations (columns may have zero variance)\n")
            
            f.write("\n")
        
        # Interesting experiments
        f.write("NOTABLE EXPERIMENTS\n")
        f.write("-"*70 + "\n")
        
        # Highest entropy
        top_entropy = summary_df.nlargest(5, 'avg_entropy')
        f.write("Highest average entropy (most diffuse attention):\n")
        for _, row in top_entropy.iterrows():
            f.write(f"  {row['experiment_name']}: {row['avg_entropy']:.4f}\n")
        f.write("\n")
        
        # Lowest entropy
        low_entropy = summary_df.nsmallest(5, 'avg_entropy')
        f.write("Lowest average entropy (most focused attention):\n")
        for _, row in low_entropy.iterrows():
            f.write(f"  {row['experiment_name']}: {row['avg_entropy']:.4f}\n")
        f.write("\n")
        
        # Strongest entropy decrease over time
        decreasing = summary_df.nsmallest(5, 'entropy_trend')
        f.write("Strongest entropy decrease over time:\n")
        for _, row in decreasing.iterrows():
            f.write(f"  {row['experiment_name']}: {row['entropy_trend']:.4f}\n")
        f.write("\n")
    
    print(f"Saved: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Summarize attention patterns across experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('paths', nargs='+',
                        help='Paths to search for attention results (supports globs)')
    parser.add_argument('--compare-periods', nargs='*', default=None,
                        help='Specific periods to compare (e.g., "2021" "2022" "2023")')
    parser.add_argument('--cosine-threshold', type=float, default=0.96,
                        help='Cosine similarity threshold for shift detection')
    parser.add_argument('--l2-threshold', type=float, default=0.025,
                        help='L2 distance threshold for shift detection')
    parser.add_argument('--entropy-threshold', type=float, default=0.05,
                        help='Absolute entropy change threshold for shift detection')
    parser.add_argument('--top-timestep-threshold', type=int, default=5,
                        help='Top timestep position change threshold for shift detection')
    parser.add_argument('--min-signals', type=int, default=1,
                        help='Minimum number of signals that must fire to report a shift')
    parser.add_argument('--min-samples', type=int, default=10,
                        help='Minimum samples per period to include in analysis')
    parser.add_argument('--output', type=str, default='reports',
                        help='Output directory for reports and plots')
    parser.add_argument('--experiments-csv', type=str, default='results/experiments_summary.csv',
                        help='Path to experiments summary CSV for collapse correlation')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed per-phase breakdowns')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store thresholds for reporting
    thresholds = {
        'cosine': args.cosine_threshold,
        'l2': args.l2_threshold,
        'entropy': args.entropy_threshold,
        'top_timestep': args.top_timestep_threshold,
    }
    
    print("="*70)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*70)
    print(f"\nShift detection thresholds:")
    print(f"  Cosine similarity: < {args.cosine_threshold}")
    print(f"  L2 distance: > {args.l2_threshold}")
    print(f"  Entropy change: > ±{args.entropy_threshold}")
    print(f"  Top timestep shift: > {args.top_timestep_threshold} positions")
    print(f"  Minimum signals required: {args.min_signals}")
    
    # Find all attention results
    print("\nSearching for attention analysis results...")
    result_files = find_attention_results(args.paths)
    print(f"Found {len(result_files)} experiments with attention results")
    
    if not result_files:
        print("No attention results found. Exiting.")
        return
    
    # Load data
    print("\nLoading attention data...")
    attention_data = load_attention_results(result_files)
    print(f"Successfully loaded {len(attention_data)} experiments")
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_df = compute_summary_stats(attention_data)
    
    # Detect attention shifts
    print("\nDetecting attention shifts...")
    if args.compare_periods:
        # Parse period pairs
        periods = args.compare_periods
        pairs = [(periods[i], periods[i+1]) for i in range(len(periods)-1)]
        print(f"Comparing specific period pairs: {pairs}")
        shift_data = detect_attention_shifts(
            attention_data, pairs,
            args.cosine_threshold, args.l2_threshold,
            args.entropy_threshold, args.top_timestep_threshold
        )
    else:
        print("Comparing all consecutive periods")
        shift_data = detect_attention_shifts(
            attention_data, None,
            args.cosine_threshold, args.l2_threshold,
            args.entropy_threshold, args.top_timestep_threshold
        )
    
    shift_df = summarize_shifts(shift_data, args.min_signals)
    print(f"Detected {len(shift_df)} attention shifts across {shift_df['experiment_name'].nunique() if not shift_df.empty else 0} experiments")
    if not shift_df.empty:
        print(f"  Shifts with 2+ signals: {len(shift_df[shift_df['n_signals_fired'] >= 2])}")
        print(f"  Shifts with 3+ signals: {len(shift_df[shift_df['n_signals_fired'] >= 3])}")
    
    # Correlate with collapse data
    print("\nCorrelating with model collapse metrics...")
    collapse_df = load_collapse_data(args.experiments_csv)
    merged_df, corr_matrix = correlate_attention_collapse(summary_df, collapse_df)
    
    # Save summary CSV
    print("\nSaving results...")
    summary_csv = output_dir / 'attention_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")
    
    if not shift_df.empty:
        shift_csv = output_dir / 'attention_shifts.csv'
        shift_df.to_csv(shift_csv, index=False)
        print(f"Saved: {shift_csv}")
    
    if merged_df is not None:
        merged_csv = output_dir / 'attention_with_collapse.csv'
        merged_df.to_csv(merged_csv, index=False)
        print(f"Saved: {merged_csv}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_entropy_distribution(summary_df, output_dir)
    plot_shift_timeline(shift_df, output_dir)
    plot_correlation_heatmap(corr_matrix, output_dir)
    
    # Generate text report
    print("\nGenerating text report...")
    generate_text_report(summary_df, shift_df, merged_df, output_dir, thresholds)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - attention_summary.csv")
    if not shift_df.empty:
        print(f"  - attention_shifts.csv")
    print(f"  - attention_analysis_report.txt")
    print(f"  - attention_metrics_by_phase.png")
    if not shift_df.empty:
        print(f"  - attention_shift_timeline.png")
    if corr_matrix is not None:
        print(f"  - attention_collapse_correlation.png")


if __name__ == '__main__':
    main()