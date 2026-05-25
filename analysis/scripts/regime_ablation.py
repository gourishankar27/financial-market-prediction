#!/usr/bin/env python3
"""
Regime Output Ablation Analysis

Systematic comparison of regime-conditional output modifications.
Outputs tables to terminal and saves plots to output directory.

Usage:
    python regime_ablation.py --csv experiments_comparison.csv
    python regime_ablation.py --csv experiments_comparison.csv --output-dir results/
    python regime_ablation.py --csv experiments_comparison.csv --no-plots
"""

import sys
import argparse
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt

from lib import (
    load_experiments,
    ablate,
    compare_groups,
    correlation_matrix,
    rank_experiments
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Regime output ablation analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to experiments comparison CSV')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output plots')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    return parser.parse_args()


def section_header(title):
    """Print a section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_ablation_save_plot(df, vary, metrics, filter=None, output_dir=None, filename=None):
    """Run ablation and optionally save the plot."""
    result = ablate(df, vary=vary, metrics=metrics, filter=filter, plot=False)
    
    if output_dir and filename and len(result) > 0:
        # Generate plot manually for saving
        n_metrics = len(metrics)
        available_metrics = [m for m in metrics if f'{m}_mean' in result.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(1, len(available_metrics), figsize=(4*len(available_metrics), 4))
            if len(available_metrics) == 1:
                axes = [axes]
            
            x_labels = [str(v) for v in result['value']]
            x = np.arange(len(x_labels))
            
            for ax, metric in zip(axes, available_metrics):
                means = result[f'{metric}_mean'].values
                stds = result[f'{metric}_std'].values
                
                bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7, color='steelblue')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric}')
                
                for bar, mean in zip(bars, means):
                    if pd.notna(mean):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.suptitle(f'Ablation: {vary}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            save_path = Path(output_dir) / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved: {save_path}")
    
    return result


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    section_header("LOADING DATA")
    
    df_all = load_experiments(args.csv, filter_evaluated=True)
    print(f"Total evaluated experiments: {len(df_all)}")
    
    df = load_experiments(args.csv, filter_evaluated=True, filter_regime=True)
    print(f"Regime experiments: {len(df)}")
    
    if len(df) == 0:
        print("No regime experiments found!")
        return
    
    # Overview
    section_header("EXPERIMENT OVERVIEW")
    
    print("\nBy routing_strategy:")
    print(df['routing_strategy'].value_counts().to_string())
    
    print("\nBy num_regimes:")
    print(df['num_regimes'].value_counts().to_string())
    
    print("\nBy expert_type:")
    print(df['expert_type'].value_counts().to_string())
    
    if 'hard_routing_train' in df.columns:
        print("\nBy hard_routing_train:")
        print(df['hard_routing_train'].value_counts().to_string())
    
    # Top performers
    section_header("TOP PERFORMERS")
    
    print("\n--- By Directional Accuracy ---")
    rank_experiments(df, by='directional_accuracy', top_n=10)
    
    print("\n--- By Healthy Percentage ---")
    rank_experiments(df, by='healthy_pct', top_n=10)
    
    print("\n--- By Prediction Std (non-collapsed) ---")
    rank_experiments(df, by='pred_std', top_n=10)
    
    # Ablations
    section_header("ABLATION: ROUTING STRATEGY")
    run_ablation_save_plot(
        df, vary='routing_strategy',
        metrics=['directional_accuracy', 'healthy_pct', 'pred_std'],
        output_dir=None if args.no_plots else output_dir,
        filename='ablation_routing_strategy.png'
    )
    
    section_header("ABLATION: NUMBER OF REGIMES")
    run_ablation_save_plot(
        df, vary='num_regimes',
        metrics=['directional_accuracy', 'healthy_pct', 'pred_std'],
        output_dir=None if args.no_plots else output_dir,
        filename='ablation_num_regimes.png'
    )
    
    section_header("ABLATION: EXPERT ARCHITECTURE")
    run_ablation_save_plot(
        df, vary='expert_type',
        metrics=['directional_accuracy', 'healthy_pct', 'pred_std'],
        output_dir=None if args.no_plots else output_dir,
        filename='ablation_expert_type.png'
    )
    
    section_header("ABLATION: DROPOUT")
    run_ablation_save_plot(
        df, vary='dropout',
        metrics=['directional_accuracy', 'healthy_pct', 'best_val_loss'],
        output_dir=None if args.no_plots else output_dir,
        filename='ablation_dropout.png'
    )
    
    # Conditional ablations
    if 'hard_routing_train' in df.columns:
        df_vix = df[df['routing_strategy'] == 'vix_threshold']
        if len(df_vix) > 0:
            section_header("ABLATION: HARD ROUTING (VIX strategy only)")
            run_ablation_save_plot(
                df, vary='hard_routing_train',
                filter={'routing_strategy': 'vix_threshold'},
                metrics=['directional_accuracy', 'final_expert_weight_cosine', 'pred_std'],
                output_dir=None if args.no_plots else output_dir,
                filename='ablation_hard_routing.png'
            )
    
    df_learned = df[df['routing_strategy'] == 'learned']
    if len(df_learned) > 0 and 'load_balance_weight' in df.columns:
        section_header("ABLATION: LOAD BALANCE WEIGHT (Learned strategy only)")
        run_ablation_save_plot(
            df, vary='load_balance_weight',
            filter={'routing_strategy': 'learned'},
            metrics=['directional_accuracy', 'healthy_pct', 'pred_std'],
            output_dir=None if args.no_plots else output_dir,
            filename='ablation_load_balance.png'
        )
    
    # VIX threshold ablation (if multiple thresholds)
    if 'vix_threshold' in df.columns:
        vix_thresholds = df[df['routing_strategy'] == 'vix_threshold']['vix_threshold'].dropna().unique()
        if len(vix_thresholds) > 1:
            section_header("ABLATION: VIX THRESHOLD (VIX 2-regime only)")
            run_ablation_save_plot(
                df, vary='vix_threshold',
                filter={'routing_strategy': 'vix_threshold', 'num_regimes': 2},
                metrics=['directional_accuracy', 'healthy_pct', 'pred_std'],
                output_dir=None if args.no_plots else output_dir,
                filename='ablation_vix_threshold.png'
            )
    
    # Correlation analysis
    section_header("CORRELATION ANALYSIS")
    
    corr = correlation_matrix(df, plot=False)
    print("\nConfig vs Outcome Correlations:")
    print(corr.to_string())
    
    if not args.no_plots:
        # Save correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.index)
        
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
        
        save_path = output_dir / 'correlation_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved: {save_path}")
    
    # Collapse analysis
    section_header("COLLAPSE ANALYSIS")
    
    if 'healthy_pct' in df.columns:
        print("\nMode distribution across experiments (% of test days):")
        mode_cols = ['healthy_pct', 'degraded_pct', 'unidirectional_pct', 'weak_collapse_pct', 'strong_collapse_pct']
        mode_cols = [c for c in mode_cols if c in df.columns]
        
        for col in mode_cols:
            vals = df[col].dropna()
            print(f"  {col:25s}: mean={vals.mean():5.1f}%, min={vals.min():5.1f}%, max={vals.max():5.1f}%")
        
        print("\nExperiments with >5% collapse days:")
        print(f"  Strong collapse: {df['has_strong_collapse'].sum()} / {len(df)}")
        print(f"  Weak collapse:   {df['has_weak_collapse'].sum()} / {len(df)}")
        print(f"  Any collapse:    {df['has_any_collapse'].sum()} / {len(df)}")
        print(f"  Mostly healthy:  {df['mostly_healthy'].sum()} / {len(df)}")
        
        print("\nHealthy % by routing_strategy:")
        print(df.groupby('routing_strategy')['healthy_pct'].agg(['mean', 'std', 'count']).round(1).to_string())
        
        print("\nHealthy % by expert_type:")
        print(df.groupby('expert_type')['healthy_pct'].agg(['mean', 'std', 'count']).round(1).to_string())
        
        print("\nHealthy % by num_regimes:")
        print(df.groupby('num_regimes')['healthy_pct'].agg(['mean', 'std', 'count']).round(1).to_string())
        
        # Best and worst by health
        print("\nMost healthy experiments:")
        top_healthy = df.nlargest(5, 'healthy_pct')[['experiment_name', 'healthy_pct', 'directional_accuracy', 'pred_std']]
        print(top_healthy.to_string(index=False))
        
        print("\nLeast healthy experiments:")
        bottom_healthy = df.nsmallest(5, 'healthy_pct')[['experiment_name', 'healthy_pct', 'directional_accuracy', 'pred_std']]
        print(bottom_healthy.to_string(index=False))
    
    elif 'pred_std' in df.columns:
        # Fallback if no evaluation mode stats
        print("\nPrediction std distribution:")
        print(f"  Min:    {df['pred_std'].min():.4f}")
        print(f"  Max:    {df['pred_std'].max():.4f}")
        print(f"  Mean:   {df['pred_std'].mean():.4f}")
        print(f"  Median: {df['pred_std'].median():.4f}")
    
    # Direct comparisons
    section_header("DIRECT COMPARISONS")
    
    # Learned vs VIX (2-regime)
    compare_groups(
        df,
        group_a={'routing_strategy': 'learned', 'num_regimes': 2},
        group_b={'routing_strategy': 'vix_threshold', 'num_regimes': 2},
        names=('Learned 2R', 'VIX 2R')
    )
    
    # Hard vs Soft routing (VIX only)
    if 'hard_routing_train' in df.columns:
        df_vix = df[df['routing_strategy'] == 'vix_threshold']
        if len(df_vix[df_vix['hard_routing_train'] == True]) > 0:
            compare_groups(
                df,
                group_a={'routing_strategy': 'vix_threshold', 'hard_routing_train': True},
                group_b={'routing_strategy': 'vix_threshold', 'hard_routing_train': False},
                names=('Hard Routing', 'Soft Routing')
            )
    
    # Linear vs MLP
    compare_groups(
        df,
        group_a={'expert_type': 'linear'},
        group_b={'expert_type': 'mlp_16'},
        names=('Linear', 'MLP-16')
    )
    
    # Summary
    section_header("SUMMARY")
    
    print(f"\nTotal experiments analyzed: {len(df)}")
    print(f"Output directory: {output_dir.absolute()}")
    
    if not args.no_plots:
        plots = list(output_dir.glob('*.png'))
        print(f"Plots saved: {len(plots)}")
        for p in sorted(plots):
            print(f"  - {p.name}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()