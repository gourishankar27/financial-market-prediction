#!/usr/bin/env python3
"""
Unified Training Analysis Dashboard

Merges training log metrics (val_loss, train_loss) with collapse monitoring JSON
to enable correlation analysis and early stopping criterion selection.

Features:
  - Parses training_*.log files for per-epoch loss values
  - Merges with collapse_monitor JSON by epoch
  - Correlation matrix between all tracked metrics
  - Multi-panel time series with shared x-axis
  - Early stopping analysis (identifies best epoch by various criteria)

Usage:
    python analyze_training.py experiments/phase/exp_name
    python analyze_training.py experiments/phase/exp_name --output figures/
    python analyze_training.py experiments/phase/exp_name --correlation-only
"""

import json
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


@dataclass
class TrainingMetrics:
    """Container for merged training metrics."""
    epoch: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    # From collapse monitor JSON
    prediction_std: list[float] = field(default_factory=list)
    prediction_range: list[float] = field(default_factory=list)
    prediction_mean: list[float] = field(default_factory=list)
    pct_positive: list[float] = field(default_factory=list)
    pct_negative: list[float] = field(default_factory=list)
    directional_accuracy: list[float] = field(default_factory=list)
    prediction_sharpe: list[float] = field(default_factory=list)
    attention_entropy: list[float] = field(default_factory=list)
    # Gradient norms (aggregated)
    grad_lstm_encoder: list[float] = field(default_factory=list)
    grad_lstm_decoder: list[float] = field(default_factory=list)
    grad_output_layer: list[float] = field(default_factory=list)
    # VSN
    vsn_encoder_std: list[float] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        data = {}
        for field_name in self.__dataclass_fields__:
            values = getattr(self, field_name)
            if values:  # Only include non-empty fields
                data[field_name] = values
        
        # Ensure all arrays same length (truncate to shortest)
        if data:
            min_len = min(len(v) for v in data.values())
            data = {k: v[:min_len] for k, v in data.items()}
        
        return pd.DataFrame(data)


def parse_training_log(log_path: Path) -> dict[int, dict[str, float]]:
    """
    Parse training log for per-epoch val_loss and train_loss.
    
    Returns:
        Dict mapping epoch -> {'val_loss': float, 'train_loss': float}
    """
    results = {}
    
    # Patterns for different log formats
    val_pattern = re.compile(r'Epoch\s+(\d+):\s*val_loss=([0-9.]+)')
    train_pattern = re.compile(r'Epoch\s+(\d+):\s*train_loss=([0-9.]+)')
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract val_loss
    for match in val_pattern.finditer(content):
        epoch = int(match.group(1))
        val_loss = float(match.group(2))
        if epoch not in results:
            results[epoch] = {}
        results[epoch]['val_loss'] = val_loss
    
    # Extract train_loss
    for match in train_pattern.finditer(content):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        if epoch not in results:
            results[epoch] = {}
        results[epoch]['train_loss'] = train_loss
    
    return results


def load_collapse_monitor(monitor_path: Path) -> dict:
    """Load collapse monitor JSON."""
    with open(monitor_path, 'r') as f:
        return json.load(f)


def merge_metrics(exp_path: Path) -> Optional[TrainingMetrics]:
    """
    Merge training log and collapse monitor data.
    
    Args:
        exp_path: Path to experiment directory
        
    Returns:
        TrainingMetrics object or None if data not found
    """
    # Find training log
    log_files = list(exp_path.glob('training_*.log'))
    if not log_files:
        print(f"Warning: No training log found in {exp_path}")
        return None
    log_path = log_files[0]  # Use first if multiple
    
    # Find collapse monitor JSON (prefer 'latest', fall back to highest epoch)
    monitor_dir = exp_path / 'collapse_monitoring'
    if not monitor_dir.exists():
        print(f"Warning: No collapse_monitoring dir in {exp_path}")
        return None
    
    latest_path = monitor_dir / 'collapse_monitor_latest.json'
    if latest_path.exists():
        monitor_path = latest_path
    else:
        # Find highest epoch file
        epoch_files = list(monitor_dir.glob('collapse_monitor_epoch*.json'))
        if not epoch_files:
            print(f"Warning: No collapse monitor JSON in {monitor_dir}")
            return None
        monitor_path = sorted(epoch_files)[-1]
    
    # Parse both sources
    log_data = parse_training_log(log_path)
    monitor_data = load_collapse_monitor(monitor_path)
    
    # Create merged metrics
    metrics = TrainingMetrics()
    
    epochs = monitor_data.get('epoch', [])
    n_epochs = len(epochs)
    
    for i, epoch in enumerate(epochs):
        metrics.epoch.append(epoch)
        
        # From log file
        if epoch in log_data:
            metrics.val_loss.append(log_data[epoch].get('val_loss', np.nan))
            metrics.train_loss.append(log_data[epoch].get('train_loss', np.nan))
        else:
            metrics.val_loss.append(np.nan)
            metrics.train_loss.append(np.nan)
        
        # From collapse monitor
        if i < len(monitor_data.get('prediction_std', [])):
            metrics.prediction_std.append(monitor_data['prediction_std'][i])
        if i < len(monitor_data.get('prediction_range', [])):
            metrics.prediction_range.append(monitor_data['prediction_range'][i])
        if i < len(monitor_data.get('prediction_mean', [])):
            metrics.prediction_mean.append(monitor_data['prediction_mean'][i])
        if i < len(monitor_data.get('pct_positive', [])):
            metrics.pct_positive.append(monitor_data['pct_positive'][i])
        if i < len(monitor_data.get('pct_negative', [])):
            metrics.pct_negative.append(monitor_data['pct_negative'][i])
        if i < len(monitor_data.get('directional_accuracy', [])):
            metrics.directional_accuracy.append(monitor_data['directional_accuracy'][i])
        if i < len(monitor_data.get('prediction_sharpe', [])):
            metrics.prediction_sharpe.append(monitor_data['prediction_sharpe'][i])
        if i < len(monitor_data.get('attention_entropy', [])):
            metrics.attention_entropy.append(monitor_data['attention_entropy'][i])
        
        # Gradient norms (need to aggregate by layer group)
        grad_norms = monitor_data.get('gradient_norms', {})
        for layer_key, attr_name in [('lstm_encoder', 'grad_lstm_encoder'),
                                      ('lstm_decoder', 'grad_lstm_decoder'),
                                      ('output_layer', 'grad_output_layer')]:
            matching = [k for k in grad_norms.keys() if layer_key in k]
            if matching:
                vals = [grad_norms[k][i] for k in matching 
                       if i < len(grad_norms[k])]
                if vals:
                    getattr(metrics, attr_name).append(np.mean(vals))
        
        # VSN output std
        vsn_data = monitor_data.get('vsn_output_std', {})
        if 'encoder' in vsn_data and i < len(vsn_data['encoder']):
            metrics.vsn_encoder_std.append(vsn_data['encoder'][i])
    
    return metrics


def compute_composite_score(df: pd.DataFrame, 
                            weights: dict[str, float] = None) -> pd.Series:
    """
    Compute composite early stopping score.
    
    Lower is better for: val_loss, |prediction_mean|, |pct_positive - 50|
    Higher is better for: prediction_std, directional_accuracy
    
    Returns normalized composite score (lower = better).
    """
    if weights is None:
        weights = {
            'val_loss': 0.3,
            'prediction_std': -0.2,  # Negative because higher is better
            'directional_accuracy': -0.3,  # Negative because higher is better
            'sign_balance': 0.2,  # |pct_positive - 50|
        }
    
    scores = pd.DataFrame(index=df.index)
    
    # Normalize each metric to [0, 1]
    if 'val_loss' in df.columns:
        vl = df['val_loss']
        scores['val_loss'] = (vl - vl.min()) / (vl.max() - vl.min() + 1e-10)
    
    if 'prediction_std' in df.columns:
        ps = df['prediction_std']
        # Invert so higher std = lower score
        scores['prediction_std'] = 1 - (ps - ps.min()) / (ps.max() - ps.min() + 1e-10)
    
    if 'directional_accuracy' in df.columns:
        da = df['directional_accuracy']
        # Invert so higher accuracy = lower score
        scores['directional_accuracy'] = 1 - (da - da.min()) / (da.max() - da.min() + 1e-10)
    
    if 'pct_positive' in df.columns:
        # Distance from 50% (balanced)
        sign_balance = np.abs(df['pct_positive'] - 50)
        scores['sign_balance'] = sign_balance / 50  # Normalize to [0, 1]
    
    # Weighted sum
    composite = pd.Series(0.0, index=df.index)
    for metric, weight in weights.items():
        if metric in scores.columns:
            composite += np.abs(weight) * scores[metric]
    
    return composite


def plot_training_dashboard(metrics: TrainingMetrics, 
                            output_path: Path,
                            exp_name: str = '') -> None:
    """
    Create comprehensive training dashboard with 6 panels.
    """
    df = metrics.to_dataframe()
    if df.empty:
        print("No data to plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    epochs = df['epoch'].values
    
    # Panel 1: Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    if 'val_loss' in df.columns:
        ax1.plot(epochs, df['val_loss'], 'b-', label='Val Loss', linewidth=2)
    if 'train_loss' in df.columns:
        ax1.plot(epochs, df['train_loss'], 'r--', label='Train Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Prediction diversity
    ax2 = fig.add_subplot(gs[0, 1])
    if 'prediction_std' in df.columns:
        ax2.plot(epochs, df['prediction_std'], 'g-', label='Pred Std', linewidth=2)
    if 'prediction_range' in df.columns:
        ax2.plot(epochs, df['prediction_range'], 'g--', label='Pred Range', alpha=0.7)
    ax2.axhline(y=0.05, color='red', linestyle=':', label='Collapse threshold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.set_title('Prediction Diversity', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Panel 3: Directional accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    if 'directional_accuracy' in df.columns:
        ax3.plot(epochs, df['directional_accuracy'] * 100, 'purple', linewidth=2)
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax3.axhline(y=55, color='green', linestyle=':', alpha=0.5, label='Target')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Directional Accuracy', fontweight='bold')
    ax3.set_ylim([45, 60])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Sign distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if 'pct_positive' in df.columns and 'pct_negative' in df.columns:
        ax4.fill_between(epochs, 0, df['pct_positive'], alpha=0.5, label='% Positive', color='green')
        ax4.fill_between(epochs, 100, 100 - df['pct_negative'], alpha=0.5, label='% Negative', color='red')
        ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Percentage')
    ax4.set_title('Sign Distribution', fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Gradient norms
    ax5 = fig.add_subplot(gs[1, 1])
    grad_cols = ['grad_lstm_encoder', 'grad_lstm_decoder', 'grad_output_layer']
    colors = ['blue', 'orange', 'green']
    for col, color in zip(grad_cols, colors):
        if col in df.columns and len(df[col]) > 0:
            ax5.plot(epochs[:len(df[col])], df[col], label=col.replace('grad_', ''), 
                    color=color, linewidth=1.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Gradient Norm')
    ax5.set_title('Gradient Flow', fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Panel 6: Attention entropy + VSN std
    ax6 = fig.add_subplot(gs[1, 2])
    ax6_twin = ax6.twinx()
    if 'attention_entropy' in df.columns:
        ln1 = ax6.plot(epochs, df['attention_entropy'], 'b-', label='Attn Entropy', linewidth=2)
    if 'vsn_encoder_std' in df.columns and len(df['vsn_encoder_std']) > 0:
        ln2 = ax6_twin.plot(epochs[:len(df['vsn_encoder_std'])], df['vsn_encoder_std'], 
                           'r--', label='VSN Std', linewidth=1.5)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Attention Entropy', color='blue')
    ax6_twin.set_ylabel('VSN Encoder Std', color='red')
    ax6.set_title('Attention & Feature Selection', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Panel 7: Composite score + early stopping analysis
    ax7 = fig.add_subplot(gs[2, :2])
    composite = compute_composite_score(df)
    ax7.plot(epochs, composite, 'k-', linewidth=2, label='Composite Score')
    
    # Mark best epochs by different criteria
    if 'val_loss' in df.columns:
        best_val = df['val_loss'].idxmin()
        ax7.axvline(x=df.loc[best_val, 'epoch'], color='blue', linestyle='--', 
                   alpha=0.7, label=f'Best Val Loss (e{int(df.loc[best_val, "epoch"])})')
    
    if 'directional_accuracy' in df.columns:
        best_da = df['directional_accuracy'].idxmax()
        ax7.axvline(x=df.loc[best_da, 'epoch'], color='purple', linestyle=':', 
                   alpha=0.7, label=f'Best Dir Acc (e{int(df.loc[best_da, "epoch"])})')
    
    best_composite = composite.idxmin()
    ax7.axvline(x=df.loc[best_composite, 'epoch'], color='green', linestyle='-', 
               alpha=0.7, linewidth=2, label=f'Best Composite (e{int(df.loc[best_composite, "epoch"])})')
    
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Composite Score (lower = better)')
    ax7.set_title('Early Stopping Analysis', fontweight='bold')
    ax7.legend(loc='upper right', fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Panel 8: Summary stats table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Compute summary
    summary_text = "Early Stopping Recommendations\n"
    summary_text += "=" * 35 + "\n\n"
    
    if 'val_loss' in df.columns:
        best_val_epoch = int(df.loc[df['val_loss'].idxmin(), 'epoch'])
        best_val_loss = df['val_loss'].min()
        summary_text += f"Best Val Loss: {best_val_loss:.4f} @ epoch {best_val_epoch}\n"
    
    if 'directional_accuracy' in df.columns:
        best_da_epoch = int(df.loc[df['directional_accuracy'].idxmax(), 'epoch'])
        best_da = df['directional_accuracy'].max()
        summary_text += f"Best Dir Acc: {best_da*100:.2f}% @ epoch {best_da_epoch}\n"
    
    best_comp_epoch = int(df.loc[composite.idxmin(), 'epoch'])
    summary_text += f"Best Composite: epoch {best_comp_epoch}\n\n"
    
    # Collapse status
    if 'prediction_std' in df.columns:
        final_std = df['prediction_std'].iloc[-1]
        status = "COLLAPSED" if final_std < 0.02 else ("DEGRADED" if final_std < 0.05 else "HEALTHY")
        summary_text += f"Final Pred Std: {final_std:.4f}\n"
        summary_text += f"Status: {status}\n\n"
    
    # Sign bias
    if 'pct_positive' in df.columns:
        final_pos = df['pct_positive'].iloc[-1]
        bias = "UNIDIRECTIONAL" if final_pos > 95 or final_pos < 5 else "BALANCED"
        summary_text += f"Final % Positive: {final_pos:.1f}%\n"
        summary_text += f"Sign Balance: {bias}\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
            fontsize=10, family='monospace', verticalalignment='top')
    
    fig.suptitle(f'Training Analysis: {exp_name}', fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved dashboard: {output_path}")
    plt.close()


def plot_correlation_matrix(metrics: TrainingMetrics, 
                            output_path: Path,
                            exp_name: str = '') -> None:
    """
    Create correlation matrix between all tracked metrics.
    """
    df = metrics.to_dataframe()
    if df.empty or len(df.columns) < 3:
        print("Insufficient data for correlation matrix")
        return
    
    # Remove epoch column for correlation
    df_corr = df.drop(columns=['epoch'], errors='ignore')
    
    # Compute correlation matrix
    corr = df_corr.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
               center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
               ax=ax, annot_kws={'size': 8})
    
    ax.set_title(f'Metric Correlations: {exp_name}', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved correlation matrix: {output_path}")
    plt.close()
    
    # Print key correlations to console
    print("\nKey correlations with val_loss:")
    if 'val_loss' in corr.columns:
        val_corrs = corr['val_loss'].drop('val_loss').sort_values()
        for metric, r in val_corrs.items():
            print(f"  {metric}: {r:.3f}")
    
    # Warn about constant columns
    constant_cols = [c for c in df_corr.columns if df_corr[c].std() < 1e-10]
    if constant_cols:
        print(f"\nWarning: Constant columns (collapsed model?): {constant_cols}")


def plot_lagged_crosscorrelation(metrics: TrainingMetrics,
                                  output_path: Path,
                                  exp_name: str = '',
                                  max_lag: int = 10) -> None:
    """
    Time-lagged cross-correlation analysis.
    
    Shows whether metrics at epoch N predict val_loss at epoch N+k.
    Positive lag = metric leads val_loss (useful for early stopping).
    Negative lag = metric lags val_loss (reactive, not predictive).
    """
    df = metrics.to_dataframe()
    if df.empty or 'val_loss' not in df.columns:
        print("Insufficient data for lagged cross-correlation")
        return
    
    # Key metrics to analyze as potential leading indicators
    candidate_metrics = [
        'prediction_std', 'prediction_range', 'directional_accuracy',
        'pct_positive', 'grad_output_layer', 'attention_entropy',
        'prediction_sharpe', 'vsn_encoder_std'
    ]
    
    available_metrics = [m for m in candidate_metrics if m in df.columns 
                        and df[m].std() > 1e-10]  # Skip constant columns
    
    if not available_metrics:
        print("No variable metrics available for cross-correlation")
        return
    
    lags = range(-max_lag, max_lag + 1)
    
    fig, axes = plt.subplots(2, (len(available_metrics) + 1) // 2, 
                             figsize=(14, 8), squeeze=False)
    axes = axes.flatten()
    
    val_loss = df['val_loss'].values
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        metric_vals = df[metric].values
        
        correlations = []
        for lag in lags:
            if lag >= 0:
                # Positive lag: metric leads (metric[:-lag] vs val_loss[lag:])
                if lag == 0:
                    corr = np.corrcoef(metric_vals, val_loss)[0, 1]
                else:
                    corr = np.corrcoef(metric_vals[:-lag], val_loss[lag:])[0, 1]
            else:
                # Negative lag: metric lags (metric[-lag:] vs val_loss[:lag])
                corr = np.corrcoef(metric_vals[-lag:], val_loss[:lag])[0, 1]
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # Plot
        colors = ['green' if l > 0 else 'gray' if l == 0 else 'red' for l in lags]
        ax.bar(lags, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Mark peak correlation
        peak_idx = np.argmax(np.abs(correlations))
        peak_lag = list(lags)[peak_idx]
        peak_corr = correlations[peak_idx]
        ax.annotate(f'peak: lag={peak_lag}\nr={peak_corr:.2f}', 
                   xy=(peak_lag, peak_corr), fontsize=8,
                   ha='center', va='bottom' if peak_corr > 0 else 'top')
        
        ax.set_xlabel('Lag (epochs)')
        ax.set_ylabel('Correlation')
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold', fontsize=10)
        ax.set_ylim([-1.1, 1.1])
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Lagged Cross-Correlation with val_loss: {exp_name}\n'
                 f'(green=metric leads, red=metric lags)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved lagged cross-correlation: {output_path}")
    plt.close()


def plot_dual_axis_overlay(metrics: TrainingMetrics,
                           output_path: Path,
                           exp_name: str = '') -> None:
    """
    Dual-axis time series overlay.
    
    Shows val_loss alongside key metrics on shared timeline,
    with best epoch markers for each criterion.
    """
    df = metrics.to_dataframe()
    if df.empty or 'val_loss' not in df.columns:
        print("Insufficient data for dual-axis overlay")
        return
    
    epochs = df['epoch'].values
    
    # Select key metrics for overlay
    overlay_configs = [
        ('prediction_std', 'Prediction Std', 'green', False),
        ('directional_accuracy', 'Dir Accuracy', 'purple', True),  # True = higher is better
        ('prediction_sharpe', 'Pred Sharpe', 'orange', True),
        ('grad_output_layer', 'Output Grad', 'red', False),
    ]
    
    available_configs = [(m, l, c, h) for m, l, c, h in overlay_configs 
                        if m in df.columns and df[m].std() > 1e-10]
    
    if not available_configs:
        print("No variable metrics for dual-axis overlay")
        return
    
    fig, axes = plt.subplots(len(available_configs), 1, figsize=(12, 3 * len(available_configs)),
                             sharex=True)
    if len(available_configs) == 1:
        axes = [axes]
    
    val_loss = df['val_loss'].values
    best_val_epoch = epochs[np.argmin(val_loss)]
    
    for ax, (metric, label, color, higher_better) in zip(axes, available_configs):
        metric_vals = df[metric].values
        
        # Primary axis: val_loss
        ln1 = ax.plot(epochs, val_loss, 'b-', linewidth=2, label='Val Loss', alpha=0.8)
        ax.set_ylabel('Val Loss', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Secondary axis: metric
        ax2 = ax.twinx()
        ln2 = ax2.plot(epochs, metric_vals, color=color, linewidth=2, 
                      label=label, linestyle='--', alpha=0.8)
        ax2.set_ylabel(label, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Best epoch markers
        ax.axvline(x=best_val_epoch, color='blue', linestyle=':', alpha=0.7)
        
        if higher_better:
            best_metric_epoch = epochs[np.argmax(metric_vals)]
        else:
            best_metric_epoch = epochs[np.argmin(metric_vals)]
        ax2.axvline(x=best_metric_epoch, color=color, linestyle=':', alpha=0.7)
        
        # Legend
        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Val Loss vs {label}', fontweight='bold', fontsize=10)
        
        # Annotate best epochs
        ax.annotate(f'best val: e{int(best_val_epoch)}', 
                   xy=(best_val_epoch, val_loss[np.argmin(val_loss)]),
                   fontsize=8, color='blue', ha='left')
        ax2.annotate(f'best {label.lower()}: e{int(best_metric_epoch)}',
                    xy=(best_metric_epoch, metric_vals[np.argmax(metric_vals) if higher_better else np.argmin(metric_vals)]),
                    fontsize=8, color=color, ha='left')
    
    axes[-1].set_xlabel('Epoch')
    fig.suptitle(f'Dual-Axis Metric Comparison: {exp_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved dual-axis overlay: {output_path}")
    plt.close()


def plot_rolling_correlation(metrics: TrainingMetrics,
                             output_path: Path,
                             exp_name: str = '',
                             window: int = 10) -> None:
    """
    Rolling window correlation analysis.
    
    Shows how val_loss correlation with each metric changes during training.
    Useful for identifying phase transitions (early vs late training dynamics).
    """
    df = metrics.to_dataframe()
    if df.empty or 'val_loss' not in df.columns or len(df) < window + 5:
        print(f"Insufficient data for rolling correlation (need > {window + 5} epochs)")
        return
    
    epochs = df['epoch'].values
    
    candidate_metrics = [
        'prediction_std', 'directional_accuracy', 'grad_output_layer',
        'attention_entropy', 'prediction_sharpe'
    ]
    
    available_metrics = [m for m in candidate_metrics if m in df.columns
                        and df[m].std() > 1e-10]
    
    if not available_metrics:
        print("No variable metrics for rolling correlation")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    val_loss = df['val_loss'].values
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_metrics)))
    
    for metric, color in zip(available_metrics, colors):
        metric_vals = df[metric].values
        
        # Compute rolling correlation
        rolling_corrs = []
        rolling_epochs = []
        for i in range(window, len(df)):
            window_val = val_loss[i-window:i]
            window_metric = metric_vals[i-window:i]
            
            if np.std(window_metric) > 1e-10:  # Skip if window has no variance
                corr = np.corrcoef(window_val, window_metric)[0, 1]
            else:
                corr = np.nan
            rolling_corrs.append(corr)
            rolling_epochs.append(epochs[i])
        
        ax.plot(rolling_epochs, rolling_corrs, color=color, linewidth=2,
               label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Rolling Correlation (window={window})')
    ax.set_title(f'Rolling Correlation with Val Loss: {exp_name}', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.1, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved rolling correlation: {output_path}")
    plt.close()


def plot_scatter_matrix(metrics: TrainingMetrics,
                        output_path: Path,
                        exp_name: str = '') -> None:
    """
    Scatter matrix of key metrics.
    
    Reveals non-linear relationships that correlation matrix misses.
    Points colored by epoch to show training progression.
    """
    df = metrics.to_dataframe()
    if df.empty:
        print("Insufficient data for scatter matrix")
        return
    
    # Select key metrics
    key_metrics = ['val_loss', 'prediction_std', 'directional_accuracy', 
                   'grad_output_layer', 'prediction_sharpe']
    
    available = [m for m in key_metrics if m in df.columns and df[m].std() > 1e-10]
    
    if len(available) < 3:
        print("Too few variable metrics for scatter matrix")
        return
    
    plot_df = df[['epoch'] + available].copy()
    
    # Create scatter matrix
    n_metrics = len(available)
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(12, 12))
    
    epoch_colors = plt.cm.viridis(df['epoch'].values / df['epoch'].max())
    
    for i, m1 in enumerate(available):
        for j, m2 in enumerate(available):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(df[m1], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_ylabel('Count' if j == 0 else '')
            elif i > j:
                # Lower triangle: scatter with epoch coloring
                scatter = ax.scatter(df[m2], df[m1], c=df['epoch'], 
                                    cmap='viridis', alpha=0.6, s=20, edgecolor='none')
            else:
                # Upper triangle: correlation value
                corr = df[[m1, m2]].corr().iloc[0, 1]
                ax.text(0.5, 0.5, f'r = {corr:.2f}', transform=ax.transAxes,
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       color='green' if abs(corr) > 0.5 else 'gray')
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Labels
            if i == n_metrics - 1:
                ax.set_xlabel(m2.replace('_', '\n'), fontsize=8)
            if j == 0:
                ax.set_ylabel(m1.replace('_', '\n'), fontsize=8)
    
    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=0, vmax=df['epoch'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Epoch', fontsize=10)
    
    fig.suptitle(f'Metric Scatter Matrix: {exp_name}\n(color = epoch progression)', 
                fontsize=12, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved scatter matrix: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training metrics and identify optimal early stopping')
    parser.add_argument('exp_path', type=str, 
                       help='Path to experiment directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: {exp_path}/analysis/)')
    parser.add_argument('--correlation-only', action='store_true',
                       help='Only generate correlation matrix')
    parser.add_argument('--csv', action='store_true',
                       help='Also save merged metrics to CSV')
    
    args = parser.parse_args()
    
    exp_path = Path(args.exp_path)
    if not exp_path.exists():
        print(f"Error: {exp_path} does not exist")
        return
    
    exp_name = exp_path.name
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = exp_path / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and merge metrics
    print(f"Loading metrics from: {exp_path}")
    metrics = merge_metrics(exp_path)
    
    if metrics is None:
        print("Failed to load metrics")
        return
    
    df = metrics.to_dataframe()
    print(f"Loaded {len(df)} epochs with {len(df.columns)} metrics")
    
    # Save CSV if requested
    if args.csv:
        csv_path = output_dir / 'merged_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
    
    # Generate plots
    if args.correlation_only:
        plot_correlation_matrix(metrics, output_dir / 'correlation_matrix.png', exp_name)
    else:
        plot_training_dashboard(metrics, output_dir / 'training_dashboard.png', exp_name)
        plot_correlation_matrix(metrics, output_dir / 'correlation_matrix.png', exp_name)
        plot_lagged_crosscorrelation(metrics, output_dir / 'lagged_crosscorr.png', exp_name)
        plot_dual_axis_overlay(metrics, output_dir / 'dual_axis_overlay.png', exp_name)
        plot_rolling_correlation(metrics, output_dir / 'rolling_correlation.png', exp_name)
        plot_scatter_matrix(metrics, output_dir / 'scatter_matrix.png', exp_name)
    
    print(f"\nAnalysis complete. Output in: {output_dir}")


if __name__ == '__main__':
    main()
