"""
Analyze collapse monitoring data from capacity sweep.

Generates:
- Summary table of when collapse occurs
- Plots of prediction std over training
- Gradient flow analysis
- Identification of collapse threshold
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def load_monitoring_data(experiment_name):
    """Load collapse monitoring data for an experiment."""
    monitor_path = Path(f'experiments/{experiment_name}/collapse_monitoring/collapse_monitor_latest.json')
    
    if not monitor_path.exists():
        return None
        
    with open(monitor_path, 'r') as f:
        data = json.load(f)
    
    return data


def detect_collapse(data, threshold=0.05):
    """
    Determine if and when model collapsed.
    
    Returns:
        dict with collapse info: {'collapsed': bool, 'epoch': int or None, 'final_std': float}
    """
    if data is None:
        return {'collapsed': None, 'epoch': None, 'final_std': None}
    
    pred_stds = data['prediction_std']
    epochs = data['epoch']
    
    # Model collapsed if final std < threshold
    final_std = pred_stds[-1]
    collapsed = final_std < threshold
    
    # Find when collapse happened (first time std drops below threshold)
    collapse_epoch = None
    if collapsed:
        for i, std in enumerate(pred_stds):
            if std < threshold:
                collapse_epoch = epochs[i]
                break
    
    return {
        'collapsed': collapsed,
        'epoch': collapse_epoch,
        'final_std': final_std,
        'min_std': min(pred_stds),
        'max_std': max(pred_stds)
    }


def analyze_all_experiments():
    """Analyze all capacity sweep experiments."""
    experiments_dir = Path('experiments')
    
    results = []
    
    # Find all capacity_* experiments
    for exp_dir in sorted(experiments_dir.glob('capacity_*')):
        exp_name = exp_dir.name
        
        # Load config to get hidden size
        config_path = exp_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            continue
            
        # Load monitoring data
        monitor_data = load_monitoring_data(exp_name)
        collapse_info = detect_collapse(monitor_data)
        
        # Extract key info
        result = {
            'experiment': exp_name,
            'hidden_size': config['architecture']['hidden_size'],
            'dropout': config['architecture']['dropout'],
            'learning_rate': config['training']['learning_rate'],
            'seed': config.get('random_seed', 42),
            'collapsed': collapse_info['collapsed'],
            'collapse_epoch': collapse_info['epoch'],
            'final_pred_std': collapse_info['final_std'],
            'min_pred_std': collapse_info['min_std'],
            'max_pred_std': collapse_info['max_std'],
        }
        
        # Get parameter count from config if available
        if monitor_data:
            # Estimate from TFT structure
            h = config['architecture']['hidden_size']
            # Rough estimate: ~h^2 * scaling_factor
            result['est_params'] = estimate_tft_params(h)
        
        results.append(result)
    
    return pd.DataFrame(results)


def estimate_tft_params(hidden_size):
    """Rough estimate of TFT parameter count based on hidden size."""
    # From the logs we saw:
    # h8: 7.3K, h16: 22.6K, h24: 41.8K, h32: 67K
    # Approximately quadratic in hidden_size
    # Empirical fit: params ≈ 88 * h^2 + 50 * h
    return int(88 * hidden_size**2 + 50 * hidden_size)


def plot_collapse_trajectories(df):
    """Plot prediction std over training for different hidden sizes."""
    experiments_dir = Path('experiments')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Group by hidden size
    for idx, hidden_size in enumerate(sorted(df['hidden_size'].unique())):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        # Get all experiments with this hidden size
        exp_subset = df[df['hidden_size'] == hidden_size]
        
        for _, row in exp_subset.iterrows():
            exp_name = row['experiment']
            monitor_data = load_monitoring_data(exp_name)
            
            if monitor_data is None:
                continue
                
            epochs = monitor_data['epoch']
            pred_stds = monitor_data['prediction_std']
            
            # Color by collapse status
            color = 'red' if row['collapsed'] else 'green'
            alpha = 0.7 if 'baseline' in exp_name else 0.4
            
            ax.plot(epochs, pred_stds, marker='o', alpha=alpha, 
                   color=color, label=exp_name.replace('capacity_', ''))
        
        ax.axhline(y=0.05, color='orange', linestyle='--', 
                  label='Collapse threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Prediction Std Dev')
        ax.set_title(f'Hidden Size = {hidden_size}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('experiments/capacity_analysis_trajectories.png', dpi=150)
    print("Saved: experiments/capacity_analysis_trajectories.png")
    plt.close()


def plot_capacity_threshold(df):
    """Plot collapse rate vs hidden size."""
    # Group by hidden size and compute collapse rate
    collapse_by_size = df.groupby('hidden_size').agg({
        'collapsed': ['sum', 'count', 'mean'],
        'final_pred_std': 'mean',
        'est_params': 'first'
    }).reset_index()
    
    collapse_by_size.columns = ['hidden_size', 'n_collapsed', 'n_total', 
                                 'collapse_rate', 'avg_final_std', 'est_params']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Collapse rate vs hidden size
    ax1.bar(collapse_by_size['hidden_size'], 
            collapse_by_size['collapse_rate'],
            color=['green' if r < 0.5 else 'red' 
                   for r in collapse_by_size['collapse_rate']])
    ax1.axhline(y=0.5, color='orange', linestyle='--', 
               label='50% collapse threshold')
    ax1.set_xlabel('Hidden Size')
    ax1.set_ylabel('Collapse Rate')
    ax1.set_title('Collapse Rate by Model Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final std vs parameters
    train_samples = 6066  # From our data
    collapse_by_size['samples_per_param'] = train_samples / collapse_by_size['est_params']
    
    colors = ['green' if r < 0.5 else 'red' 
              for r in collapse_by_size['collapse_rate']]
    
    ax2.scatter(collapse_by_size['samples_per_param'],
               collapse_by_size['avg_final_std'],
               s=200, c=colors, alpha=0.6)
    
    for _, row in collapse_by_size.iterrows():
        ax2.annotate(f"h={row['hidden_size']}", 
                    (row['samples_per_param'], row['avg_final_std']),
                    fontsize=8)
    
    ax2.axhline(y=0.05, color='orange', linestyle='--', 
               label='Collapse threshold')
    ax2.set_xlabel('Samples per Parameter')
    ax2.set_ylabel('Final Prediction Std')
    ax2.set_title('Capacity vs Collapse')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('experiments/capacity_threshold_analysis.png', dpi=150)
    print("Saved: experiments/capacity_threshold_analysis.png")
    plt.close()


def print_summary_table(df):
    """Print summary table of results."""
    print("\n" + "="*80)
    print("CAPACITY ANALYSIS SUMMARY")
    print("="*80)
    
    # Sort by hidden size then experiment name
    df_sorted = df.sort_values(['hidden_size', 'experiment'])
    
    print("\nIndividual Experiments:")
    print("-"*80)
    print(f"{'Experiment':<35} {'h':>3} {'drop':>5} {'lr':>7} "
          f"{'Collapsed':>10} {'Final_Std':>10} {'At_Epoch':>9}")
    print("-"*80)
    
    for _, row in df_sorted.iterrows():
        exp_short = row['experiment'].replace('capacity_', '')[:34]
        collapsed_str = "YES" if row['collapsed'] else "NO"
        epoch_str = str(row['collapse_epoch']) if row['collapse_epoch'] else "N/A"
        std_str = f"{row['final_pred_std']:.6f}" if row['final_pred_std'] else "N/A"
        
        print(f"{exp_short:<35} {row['hidden_size']:>3} "
              f"{row['dropout']:>5.2f} {row['learning_rate']:>7.4f} "
              f"{collapsed_str:>10} {std_str:>10} {epoch_str:>9}")
    
    print("-"*80)
    
    # Summary by hidden size
    print("\nCollapse Rate by Hidden Size:")
    print("-"*80)
    print(f"{'Hidden Size':>12} {'Experiments':>12} {'Collapsed':>10} "
          f"{'Rate':>8} {'Avg Std':>10}")
    print("-"*80)
    
    for h_size in sorted(df['hidden_size'].unique()):
        subset = df[df['hidden_size'] == h_size]
        n_total = len(subset)
        n_collapsed = subset['collapsed'].sum()
        rate = n_collapsed / n_total if n_total > 0 else 0
        avg_std = subset['final_pred_std'].mean()
        
        print(f"{h_size:>12} {n_total:>12} {n_collapsed:>10} "
              f"{rate:>8.1%} {avg_std:>10.6f}")
    
    print("-"*80)
    
    # Key findings
    print("\nKey Findings:")
    print("-"*80)
    
    # Find threshold
    working_sizes = df[df['collapsed'] == False]['hidden_size'].unique()
    collapsed_sizes = df[df['collapsed'] == True]['hidden_size'].unique()
    
    if len(working_sizes) > 0 and len(collapsed_sizes) > 0:
        max_working = max(working_sizes)
        min_collapsed = min(collapsed_sizes)
        
        print(f"  Largest working hidden size: {max_working}")
        print(f"  Smallest collapsed hidden size: {min_collapsed}")
        
        if max_working < min_collapsed:
            print(f"  → Collapse threshold between h={max_working} and h={min_collapsed}")
    
    # Regularization effects
    h24_experiments = df[df['hidden_size'] == 24]
    if len(h24_experiments) > 1:
        print(f"\n  h=24 experiments: {len(h24_experiments)} total")
        print(f"  Baseline (drop=0.15): {h24_experiments[h24_experiments['dropout'] == 0.15]['collapsed'].values}")
        high_dropout = h24_experiments[h24_experiments['dropout'] > 0.15]
        if len(high_dropout) > 0:
            print(f"  Higher dropout: {high_dropout['collapsed'].sum()}/{len(high_dropout)} collapsed")
    
    print("="*80)


def main():
    """Run complete capacity analysis."""
    print("Analyzing capacity sweep experiments...")
    
    # Load and analyze all experiments
    df = analyze_all_experiments()
    
    if len(df) == 0:
        print("No experiments found. Run sweep_capacity_analysis.sh first.")
        return
    
    # Print summary
    print_summary_table(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_collapse_trajectories(df)
    plot_capacity_threshold(df)
    
    # Save results
    df.to_csv('experiments/capacity_analysis_results.csv', index=False)
    print("\nSaved: experiments/capacity_analysis_results.csv")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

