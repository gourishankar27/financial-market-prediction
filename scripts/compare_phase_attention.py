#!/usr/bin/env python3
"""
Compare attention patterns between two experimental phases.

Designed for A/B testing architectural modifications to attention mechanisms.
Focuses on statistical differences rather than temporal regime detection.

Usage:
    python compare_phase_attention.py \\
        --baseline experiments/00_baseline_exploration/ \\
        --treatment experiments/04_staleness_attention/ \\
        --output reports/attention_comparison/
        
    python compare_phase_attention.py \\
        --baseline experiments/02b_vintage_sweep/ \\
        --treatment experiments/05_mixed_frequency_attention/ \\
        --baseline-label "Vintage Baseline" \\
        --treatment-label "Mixed-Freq Attention" \\
        --output reports/mf_vs_baseline/
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine


def load_attention_results(phase_dir):
    """Load all attention analysis results from a phase directory."""
    results = {}
    phase_path = Path(phase_dir)
    
    for exp_dir in phase_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Look for attention_analysis_*/attention_analysis_results.json
        attention_files = list(exp_dir.glob('attention_analysis_*/attention_analysis_results.json'))
        
        if attention_files:
            with open(attention_files[0]) as f:
                results[exp_dir.name] = json.load(f)
    
    return results


def extract_metrics(attention_data):
    """Extract key metrics from attention results."""
    metrics = []
    
    for exp_name, data in attention_data.items():
        period_stats = data.get('period_statistics', {})
        
        if not period_stats:
            continue
        
        periods = sorted(period_stats.keys())
        
        # Aggregate across periods
        entropies = [period_stats[p]['entropy_mean'] for p in periods]
        concentrations = [period_stats[p]['attention_concentration'] for p in periods]
        
        # Get mean attention patterns
        attention_patterns = [np.array(period_stats[p]['mean_attention']) for p in periods]
        
        # Temporal stability - variance in attention pattern across periods
        if len(attention_patterns) > 1:
            stacked = np.stack(attention_patterns)
            temporal_variance = np.mean(np.var(stacked, axis=0))
        else:
            temporal_variance = 0.0
        
        # Top timestep consistency - how often does the same timestep dominate?
        top_timesteps = [np.argmax(pattern) for pattern in attention_patterns]
        top_timestep_consistency = len(set(top_timesteps)) / len(top_timesteps) if top_timesteps else 1.0
        
        metrics.append({
            'experiment': exp_name,
            'avg_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies),
            'entropy_range': np.max(entropies) - np.min(entropies),
            'avg_concentration': np.mean(concentrations),
            'std_concentration': np.std(concentrations),
            'temporal_variance': temporal_variance,
            'top_timestep_diversity': top_timestep_consistency,
            'mean_attention_pattern': np.mean(attention_patterns, axis=0) if attention_patterns else None,
        })
    
    return pd.DataFrame(metrics)


def statistical_comparison(baseline_df, treatment_df, metric):
    """Perform statistical tests comparing baseline and treatment."""
    baseline_vals = baseline_df[metric].values
    treatment_vals = treatment_df[metric].values
    
    # T-test
    t_stat, p_value = stats.ttest_ind(baseline_vals, treatment_vals)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(baseline_vals) + np.var(treatment_vals)) / 2)
    cohens_d = (np.mean(treatment_vals) - np.mean(baseline_vals)) / pooled_std if pooled_std > 0 else 0
    
    # Mann-Whitney U (non-parametric alternative)
    u_stat, u_p = stats.mannwhitneyu(baseline_vals, treatment_vals, alternative='two-sided')
    
    return {
        'metric': metric,
        'baseline_mean': np.mean(baseline_vals),
        'baseline_std': np.std(baseline_vals),
        'treatment_mean': np.mean(treatment_vals),
        'treatment_std': np.std(treatment_vals),
        'difference': np.mean(treatment_vals) - np.mean(baseline_vals),
        'percent_change': 100 * (np.mean(treatment_vals) - np.mean(baseline_vals)) / np.mean(baseline_vals) if np.mean(baseline_vals) != 0 else 0,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'u_statistic': u_stat,
        'u_p_value': u_p,
        'significant': p_value < 0.05,
    }


def compare_attention_patterns(baseline_df, treatment_df):
    """Compare mean attention patterns between phases."""
    # Filter to only experiments with valid patterns
    baseline_valid = baseline_df[baseline_df['mean_attention_pattern'].notna()].copy()
    treatment_valid = treatment_df[treatment_df['mean_attention_pattern'].notna()].copy()
    
    if len(baseline_valid) == 0 or len(treatment_valid) == 0:
        print("Warning: No valid attention patterns found")
        return None
    
    # Get pattern lengths
    baseline_valid['pattern_length'] = baseline_valid['mean_attention_pattern'].apply(len)
    treatment_valid['pattern_length'] = treatment_valid['mean_attention_pattern'].apply(len)
    
    # Find common encoder length (most frequent)
    all_lengths = pd.concat([baseline_valid['pattern_length'], treatment_valid['pattern_length']])
    common_length = all_lengths.mode()[0] if len(all_lengths) > 0 else None
    
    if common_length is None:
        print("Warning: No common encoder length found")
        return None
    
    # Filter to common length
    baseline_filtered = baseline_valid[baseline_valid['pattern_length'] == common_length]
    treatment_filtered = treatment_valid[treatment_valid['pattern_length'] == common_length]
    
    print(f"  Using encoder length {common_length}")
    print(f"  Baseline: {len(baseline_filtered)} experiments")
    print(f"  Treatment: {len(treatment_filtered)} experiments")
    
    if len(baseline_filtered) == 0 or len(treatment_filtered) == 0:
        print("Warning: No experiments with common encoder length")
        return None
    
    baseline_patterns = np.stack(baseline_filtered['mean_attention_pattern'].values)
    treatment_patterns = np.stack(treatment_filtered['mean_attention_pattern'].values)
    
    baseline_mean = np.mean(baseline_patterns, axis=0)
    treatment_mean = np.mean(treatment_patterns, axis=0)
    
    # Cosine similarity
    cos_sim = 1 - cosine(baseline_mean, treatment_mean)
    
    # L2 distance
    l2_dist = np.linalg.norm(baseline_mean - treatment_mean)
    
    # Timestep-wise differences
    timestep_diffs = treatment_mean - baseline_mean
    
    return {
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'baseline_pattern': baseline_mean,
        'treatment_pattern': treatment_mean,
        'timestep_differences': timestep_diffs,
        'encoder_length': common_length,
        'n_baseline': len(baseline_filtered),
        'n_treatment': len(treatment_filtered),
    }


def plot_effect_sizes(stats_results, output_path, baseline_label="Baseline", treatment_label="Treatment"):
    """Plot effect sizes for all metrics with significance markers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = [r['metric'].replace('_', ' ').title() for r in stats_results]
    effect_sizes = [r['cohens_d'] for r in stats_results]
    significant = [r['significant'] for r in stats_results]
    
    # Color by significance and direction
    colors = []
    for es, sig in zip(effect_sizes, significant):
        if not sig:
            colors.append('lightgray')
        elif es > 0:
            colors.append('green')
        else:
            colors.append('red')
    
    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, effect_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add magnitude lines
    ax.axvline(0, color='black', linewidth=2)
    ax.axvline(0.2, color='gray', linestyle='--', alpha=0.3, label='Small (0.2)')
    ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.3, label='Medium (0.5)')
    ax.axvline(-0.5, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(0.8, color='gray', linestyle='-.', alpha=0.3, label='Large (0.8)')
    ax.axvline(-0.8, color='gray', linestyle='-.', alpha=0.3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
    ax.set_title(f'Effect Sizes: {treatment_label} vs {baseline_label}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Annotate values
    for i, (bar, es, sig) in enumerate(zip(bars, effect_sizes, significant)):
        label = f"{es:.2f}{'*' if sig else ''}"
        x_pos = es + (0.05 if es > 0 else -0.05)
        ha = 'left' if es > 0 else 'right'
        ax.text(x_pos, i, label, va='center', ha=ha, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metric_comparison(baseline_df, treatment_df, metric, label, output_path,
                           baseline_label="Baseline", treatment_label="Treatment", 
                           stats_result=None):
    """Plot single metric comparison with statistical annotation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    baseline_vals = baseline_df[metric].dropna().values
    treatment_vals = treatment_df[metric].dropna().values
    
    # Violin plot with embedded box plot
    parts = ax.violinplot([baseline_vals, treatment_vals], positions=[1, 2], 
                          widths=0.7, showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.6)
        pc.set_edgecolor('darkblue')
        pc.set_linewidth(1.5)
    
    # Overlay box plot
    bp = ax.boxplot([baseline_vals, treatment_vals], positions=[1, 2], widths=0.3,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    # Add means
    means = [baseline_vals.mean(), treatment_vals.mean()]
    ax.scatter([1, 2], means, color='green', s=150, zorder=3, marker='D', 
              edgecolor='darkgreen', linewidth=2, label='Mean')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels([baseline_label, treatment_label], fontsize=12, fontweight='bold')
    ax.set_ylabel(label, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Add statistical annotation if provided
    if stats_result is not None:
        y_max = max(baseline_vals.max(), treatment_vals.max())
        y_range = y_max - min(baseline_vals.min(), treatment_vals.min())
        y_text = y_max + 0.05 * y_range
        
        sig_text = f"p={stats_result['p_value']:.3f}"
        if stats_result['significant']:
            sig_text += " ***"
            color = 'green'
        else:
            sig_text += " (n.s.)"
            color = 'gray'
        
        ax.text(1.5, y_text, sig_text, ha='center', fontsize=11, 
               fontweight='bold', color=color)
        
        # Add effect size
        effect_text = f"d={stats_result['cohens_d']:.2f}"
        ax.text(1.5, y_text - 0.08 * y_range, effect_text, ha='center', 
               fontsize=10, style='italic')
    
    title = f'{label}: {baseline_label} vs {treatment_label}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_attention_patterns(pattern_comparison, output_path, 
                            baseline_label="Baseline", treatment_label="Treatment"):
    """Plot attention pattern comparison with difference analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    baseline_pattern = pattern_comparison['baseline_pattern']
    treatment_pattern = pattern_comparison['treatment_pattern']
    diffs = pattern_comparison['timestep_differences']
    
    timesteps = np.arange(len(baseline_pattern))
    width = 0.35
    
    # Pattern comparison
    ax1.bar(timesteps - width/2, baseline_pattern, width, 
           label=baseline_label, alpha=0.8, color='steelblue', edgecolor='darkblue', linewidth=1.5)
    ax1.bar(timesteps + width/2, treatment_pattern, width, 
           label=treatment_label, alpha=0.8, color='coral', edgecolor='darkred', linewidth=1.5)
    
    ax1.set_xlabel('Encoder Timestep (t-N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Attention Weight', fontsize=12, fontweight='bold')
    ax1.set_title(f'Attention Patterns (n={pattern_comparison["n_baseline"]}+{pattern_comparison["n_treatment"]})', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Timestep labels (show every 5th)
    ax1.set_xticks(timesteps[::5])
    ax1.set_xticklabels([f't-{len(baseline_pattern)-i}' for i in timesteps[::5]])
    
    # Difference plot with significance
    colors = ['darkred' if d < -0.005 else 'darkgreen' if d > 0.005 else 'lightgray' 
              for d in diffs]
    ax2.bar(timesteps, diffs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=2)
    ax2.axhline(0.005, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(-0.005, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Encoder Timestep (t-N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Δ Attention Weight', fontsize=12, fontweight='bold')
    ax2.set_title(f'{treatment_label} - {baseline_label}', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax2.set_xticks(timesteps[::5])
    ax2.set_xticklabels([f't-{len(baseline_pattern)-i}' for i in timesteps[::5]])
    
    # Add cosine similarity annotation
    cos_text = f'Cosine Similarity: {pattern_comparison["cosine_similarity"]:.3f}'
    ax1.text(0.02, 0.98, cos_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comprehensive_comparison(baseline_df, treatment_df, stats_results, pattern_comparison, 
                                  output_dir, baseline_label="Baseline", treatment_label="Treatment"):
    """Generate comprehensive comparison with all metrics."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Effect sizes overview (top left)
    ax_effect = fig.add_subplot(gs[0, 0])
    metrics_short = [r['metric'].replace('avg_', '').replace('std_', 'σ ').replace('_', ' ').title()[:15] 
                     for r in stats_results]
    effect_sizes = [r['cohens_d'] for r in stats_results]
    significant = [r['significant'] for r in stats_results]
    
    colors = ['green' if sig and es > 0 else 'red' if sig and es < 0 else 'lightgray' 
              for es, sig in zip(effect_sizes, significant)]
    
    y_pos = np.arange(len(metrics_short))
    ax_effect.barh(y_pos, effect_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax_effect.axvline(0, color='black', linewidth=2)
    ax_effect.axvline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax_effect.axvline(-0.5, color='gray', linestyle='--', alpha=0.3)
    ax_effect.set_yticks(y_pos)
    ax_effect.set_yticklabels(metrics_short, fontsize=9)
    ax_effect.set_xlabel("Cohen's d", fontsize=10, fontweight='bold')
    ax_effect.set_title('Effect Sizes', fontsize=11, fontweight='bold')
    ax_effect.grid(True, alpha=0.3, axis='x')
    
    # Individual metric comparisons (violin + box)
    metrics_to_plot = [
        ('avg_entropy', 'Average Entropy', gs[0, 1]),
        ('avg_concentration', 'Concentration', gs[0, 2]),
        ('temporal_variance', 'Temporal Variance', gs[1, 0]),
        ('std_entropy', 'Entropy Variability', gs[1, 1]),
    ]
    
    for metric, label, position in metrics_to_plot:
        ax = fig.add_subplot(position)
        
        baseline_vals = baseline_df[metric].dropna().values
        treatment_vals = treatment_df[metric].dropna().values
        
        # Compact violin + box
        parts = ax.violinplot([baseline_vals, treatment_vals], positions=[1, 2], 
                             widths=0.6, showmeans=False, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.5)
        
        bp = ax.boxplot([baseline_vals, treatment_vals], positions=[1, 2], widths=0.25,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='white', alpha=0.9),
                       medianprops=dict(color='red', linewidth=2))
        
        means = [baseline_vals.mean(), treatment_vals.mean()]
        ax.scatter([1, 2], means, color='green', s=100, zorder=3, marker='D')
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels([baseline_label[:8], treatment_label[:8]], fontsize=9)
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')
        
        # Add p-value
        stat = [r for r in stats_results if r['metric'] == metric][0]
        if stat['significant']:
            ax.set_title(f"{label} (p={stat['p_value']:.3f}***)", 
                        fontsize=10, fontweight='bold', color='green')
        else:
            ax.set_title(f"{label} (p={stat['p_value']:.2f})", 
                        fontsize=10)
    
    # Recency bias metric (calculated on the fly)
    ax_recency = fig.add_subplot(gs[1, 2])
    
    def calc_recency_bias(df):
        """Calculate mean attention on last 5 timesteps."""
        biases = []
        for pattern in df['mean_attention_pattern'].dropna():
            if len(pattern) >= 5:
                biases.append(np.mean(pattern[-5:]))
        return np.array(biases)
    
    baseline_recency = calc_recency_bias(baseline_df)
    treatment_recency = calc_recency_bias(treatment_df)
    
    if len(baseline_recency) > 0 and len(treatment_recency) > 0:
        parts = ax_recency.violinplot([baseline_recency, treatment_recency], 
                                     positions=[1, 2], widths=0.6)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.5)
        
        bp = ax_recency.boxplot([baseline_recency, treatment_recency], 
                               positions=[1, 2], widths=0.25, patch_artist=True,
                               boxprops=dict(facecolor='white', alpha=0.9),
                               medianprops=dict(color='red', linewidth=2))
        
        means_r = [baseline_recency.mean(), treatment_recency.mean()]
        ax_recency.scatter([1, 2], means_r, color='green', s=100, zorder=3, marker='D')
        
        ax_recency.set_xticks([1, 2])
        ax_recency.set_xticklabels([baseline_label[:8], treatment_label[:8]], fontsize=9)
        ax_recency.set_ylabel('Recency Bias', fontsize=10, fontweight='bold')
        ax_recency.set_title('Attention on Recent Data (t-1 to t-5)', fontsize=10)
        ax_recency.grid(True, alpha=0.2, axis='y')
    
    # Attention patterns (bottom 2/3)
    if pattern_comparison is not None:
        ax_pattern = fig.add_subplot(gs[2, :2])
        ax_diff = fig.add_subplot(gs[2, 2])
        
        baseline_pattern = pattern_comparison['baseline_pattern']
        treatment_pattern = pattern_comparison['treatment_pattern']
        diffs = pattern_comparison['timestep_differences']
        timesteps = np.arange(len(baseline_pattern))
        width = 0.35
        
        ax_pattern.bar(timesteps - width/2, baseline_pattern, width, 
                      label=baseline_label, alpha=0.8, color='steelblue', edgecolor='darkblue')
        ax_pattern.bar(timesteps + width/2, treatment_pattern, width, 
                      label=treatment_label, alpha=0.8, color='coral', edgecolor='darkred')
        
        ax_pattern.set_xlabel('Encoder Timestep', fontsize=11, fontweight='bold')
        ax_pattern.set_ylabel('Mean Attention', fontsize=11, fontweight='bold')
        ax_pattern.set_title('Mean Attention Patterns', fontsize=12, fontweight='bold')
        ax_pattern.legend(fontsize=10)
        ax_pattern.grid(True, alpha=0.3, axis='y')
        
        # Difference
        colors_diff = ['darkred' if d < -0.005 else 'darkgreen' if d > 0.005 else 'gray' 
                      for d in diffs]
        ax_diff.bar(timesteps, diffs, color=colors_diff, alpha=0.7)
        ax_diff.axhline(0, color='black', linestyle='-', linewidth=1.5)
        ax_diff.set_xlabel('Timestep', fontsize=11, fontweight='bold')
        ax_diff.set_ylabel('Δ Attention', fontsize=11, fontweight='bold')
        ax_diff.set_title(f'{treatment_label} - {baseline_label}', fontsize=12, fontweight='bold')
        ax_diff.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'comprehensive_comparison.png'}")


def generate_report(stats_results, pattern_comparison, output_dir,
                   baseline_label="Baseline", treatment_label="Treatment"):
    """Generate text report."""
    
    report_path = output_dir / 'comparison_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ATTENTION MECHANISM PHASE COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Baseline:  {baseline_label}\n")
        f.write(f"Treatment: {treatment_label}\n\n")
        
        f.write("STATISTICAL COMPARISONS\n")
        f.write("-"*70 + "\n\n")
        
        for result in stats_results:
            f.write(f"{result['metric'].upper().replace('_', ' ')}\n")
            f.write(f"  Baseline:  {result['baseline_mean']:.4f} ± {result['baseline_std']:.4f}\n")
            f.write(f"  Treatment: {result['treatment_mean']:.4f} ± {result['treatment_std']:.4f}\n")
            f.write(f"  Difference: {result['difference']:+.4f} ({result['percent_change']:+.1f}%)\n")
            f.write(f"  t-test: t={result['t_statistic']:.3f}, p={result['p_value']:.4f}\n")
            f.write(f"  Effect size (Cohen's d): {result['cohens_d']:.3f}\n")
            
            if result['significant']:
                if abs(result['cohens_d']) > 0.8:
                    magnitude = "LARGE"
                elif abs(result['cohens_d']) > 0.5:
                    magnitude = "MEDIUM"
                else:
                    magnitude = "SMALL"
                f.write(f"  *** SIGNIFICANT ({magnitude} effect) ***\n")
            else:
                f.write(f"  Not significant (p > 0.05)\n")
            
            f.write("\n")
        
        f.write("\nATTENTION PATTERN SIMILARITY\n")
        f.write("-"*70 + "\n")
        
        if pattern_comparison is not None:
            f.write(f"Cosine Similarity: {pattern_comparison['cosine_similarity']:.4f}\n")
            f.write(f"L2 Distance: {pattern_comparison['l2_distance']:.4f}\n")
            f.write(f"Compared experiments: {pattern_comparison['n_baseline']} baseline + {pattern_comparison['n_treatment']} treatment\n")
            f.write(f"Encoder length: {pattern_comparison['encoder_length']}\n")
            
            if pattern_comparison['cosine_similarity'] > 0.95:
                f.write("→ Patterns are VERY SIMILAR (minimal architectural impact)\n")
            elif pattern_comparison['cosine_similarity'] > 0.85:
                f.write("→ Patterns are SIMILAR (modest architectural impact)\n")
            else:
                f.write("→ Patterns are DIFFERENT (substantial architectural impact)\n")
            
            f.write("\n")
            
            # Top changed timesteps
            diffs = pattern_comparison['timestep_differences']
            top_increased = np.argsort(diffs)[-5:][::-1]
            top_decreased = np.argsort(diffs)[:5]
            
            f.write("\nTOP ATTENTION SHIFTS\n")
            f.write("-"*70 + "\n")
            f.write("Timesteps with increased attention in treatment:\n")
            for idx in top_increased:
                f.write(f"  t-{len(diffs)-idx}: {diffs[idx]:+.4f}\n")
            
            f.write("\nTimesteps with decreased attention in treatment:\n")
            for idx in top_decreased:
                f.write(f"  t-{len(diffs)-idx}: {diffs[idx]:+.4f}\n")
        else:
            f.write("Pattern comparison not available (no common encoder length)\n")
    
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare attention patterns between experimental phases'
    )
    
    parser.add_argument('--baseline', required=True,
                       help='Baseline phase directory (e.g., experiments/00_baseline_exploration/)')
    parser.add_argument('--treatment', required=True,
                       help='Treatment phase directory (e.g., experiments/04_staleness_attention/)')
    parser.add_argument('--baseline-label', default='Baseline',
                       help='Label for baseline phase in plots')
    parser.add_argument('--treatment-label', default='Treatment',
                       help='Label for treatment phase in plots')
    parser.add_argument('--output', default='reports/phase_comparison',
                       help='Output directory')
    parser.add_argument('--plot-only', nargs='+', 
                       choices=['effect_sizes', 'entropy', 'concentration', 'temporal_variance', 
                               'patterns', 'comprehensive'],
                       help='Generate only specific plots (default: all)')
    parser.add_argument('--skip-comprehensive', action='store_true',
                       help='Skip the large comprehensive plot')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PHASE COMPARISON: {args.baseline_label} vs {args.treatment_label}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading baseline attention data...")
    baseline_data = load_attention_results(args.baseline)
    print(f"  Found {len(baseline_data)} experiments")
    
    print("Loading treatment attention data...")
    treatment_data = load_attention_results(args.treatment)
    print(f"  Found {len(treatment_data)} experiments")
    
    if not baseline_data or not treatment_data:
        print("\nError: No attention results found in one or both phases")
        print("Run batch_analyze_attention.py first to generate attention analysis")
        return
    
    # Extract metrics
    print("\nExtracting metrics...")
    baseline_df = extract_metrics(baseline_data)
    treatment_df = extract_metrics(treatment_data)
    
    # Statistical comparisons
    print("\nPerforming statistical tests...")
    stats_results = []
    for metric in ['avg_entropy', 'std_entropy', 'avg_concentration', 
                   'temporal_variance', 'top_timestep_diversity']:
        result = statistical_comparison(baseline_df, treatment_df, metric)
        stats_results.append(result)
        
        sig = "***" if result['significant'] else "   "
        print(f"  {sig} {metric}: Δ={result['difference']:+.4f} (p={result['p_value']:.4f})")
    
    # Pattern comparison
    print("\nComparing attention patterns...")
    pattern_comparison = compare_attention_patterns(baseline_df, treatment_df)
    
    if pattern_comparison is not None:
        print(f"  Cosine similarity: {pattern_comparison['cosine_similarity']:.4f}")
        print(f"  L2 distance: {pattern_comparison['l2_distance']:.4f}")
    else:
        print("  Skipping pattern comparison (no common encoder length)")
    
    # Generate outputs
    print("\nGenerating visualizations...")
    
    # Determine which plots to generate
    if args.plot_only:
        plots_to_generate = args.plot_only
    else:
        plots_to_generate = ['effect_sizes', 'entropy', 'concentration', 
                            'temporal_variance', 'patterns', 'comprehensive']
    
    # Effect sizes
    if 'effect_sizes' in plots_to_generate:
        plot_effect_sizes(stats_results, output_dir / 'effect_sizes.png',
                         args.baseline_label, args.treatment_label)
    
    # Individual metrics
    metric_map = {
        'entropy': ('avg_entropy', 'Average Entropy'),
        'concentration': ('avg_concentration', 'Average Concentration'),
        'temporal_variance': ('temporal_variance', 'Temporal Variance'),
    }
    
    for plot_name, (metric, label) in metric_map.items():
        if plot_name in plots_to_generate:
            stat = [r for r in stats_results if r['metric'] == metric][0]
            plot_metric_comparison(baseline_df, treatment_df, metric, label,
                                 output_dir / f'{plot_name}_comparison.png',
                                 args.baseline_label, args.treatment_label, stat)
    
    # Attention patterns
    if 'patterns' in plots_to_generate and pattern_comparison is not None:
        plot_attention_patterns(pattern_comparison, 
                               output_dir / 'attention_patterns.png',
                               args.baseline_label, args.treatment_label)
    
    # Comprehensive view
    if 'comprehensive' in plots_to_generate and not args.skip_comprehensive:
        plot_comprehensive_comparison(baseline_df, treatment_df, stats_results,
                                     pattern_comparison, output_dir,
                                     args.baseline_label, args.treatment_label)
    
    print("\nGenerating report...")
    generate_report(stats_results, pattern_comparison, output_dir,
                   args.baseline_label, args.treatment_label)
    
    # Save detailed metrics
    baseline_df['phase'] = args.baseline_label
    treatment_df['phase'] = args.treatment_label
    combined = pd.concat([baseline_df, treatment_df], ignore_index=True)
    
    metrics_csv = output_dir / 'detailed_metrics.csv'
    combined.to_csv(metrics_csv, index=False)
    print(f"Saved: {metrics_csv}")
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    
    # List generated files
    generated_files = ['comparison_report.txt', 'detailed_metrics.csv']
    if args.plot_only:
        for plot_type in args.plot_only:
            if plot_type == 'effect_sizes':
                generated_files.append('effect_sizes.png')
            elif plot_type == 'comprehensive':
                generated_files.append('comprehensive_comparison.png')
            elif plot_type == 'patterns':
                generated_files.append('attention_patterns.png')
            else:
                generated_files.append(f'{plot_type}_comparison.png')
    else:
        generated_files.extend([
            'effect_sizes.png',
            'entropy_comparison.png',
            'concentration_comparison.png',
            'temporal_variance_comparison.png',
            'attention_patterns.png',
        ])
        if not args.skip_comprehensive:
            generated_files.append('comprehensive_comparison.png')
    
    for f in generated_files:
        if (output_dir / f).exists():
            print(f"  - {f}")


if __name__ == '__main__':
    main()