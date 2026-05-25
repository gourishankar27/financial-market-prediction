"""
Plotting utility for collapse monitoring data.

Creates publication-quality figures from monitoring logs.

Usage:
    # Single experiment (auto-saves to experiments/{exp}/monitoring/)
    python scripts/plot_monitoring_data.py experiment_name
    
    # Single experiment with custom output
    python scripts/plot_monitoring_data.py experiment_name --output-dir figures/
    
    # Multiple experiments (comparison)
    python scripts/plot_monitoring_data.py exp1 exp2 exp3
    
    # All experiments in a phase (each saved to its own monitoring/ subdir)
    python scripts/plot_monitoring_data.py --phase 02b_vintage_sweep
    
    # Comprehensive summary
    python scripts/plot_monitoring_data.py --phase 02b_vintage_sweep --comprehensive
    
    # Paper-quality gradient overlay plot
    python scripts/plot_monitoring_data.py experiment_name --gradient-overlay
    python scripts/plot_monitoring_data.py experiment_name --gradient-overlay --paper-format
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_monitoring_log(experiment_name):
    """Load collapse monitoring data."""
    log_path = Path(f'experiments/{experiment_name}/collapse_monitoring/collapse_monitor_latest.json')
    
    if not log_path.exists():
        phase_dirs = ['00_baseline_exploration', '01_staleness_features', 
                      '01_staleness_features_fixed', '02_custom_tft',
                      '02_vintage_baseline', '02b_vintage_sweep']
        for phase in phase_dirs:
            alt_path = Path(f'experiments/{phase}/{experiment_name}/collapse_monitoring/collapse_monitor_latest.json')
            if alt_path.exists():
                log_path = alt_path
                break
        
        if not log_path.exists():
            return None
    
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_prediction_diversity(experiments, output_path='prediction_diversity.png'):
    """Plot prediction std over training for multiple experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    
    for idx, exp_name in enumerate(experiments):
        data = load_monitoring_log(exp_name)
        if data is None:
            print(f"Warning: No data for {exp_name}")
            continue
        
        epochs = data['epoch']
        pred_std = data['prediction_std']
        pred_range = data['prediction_range']
        
        collapsed = pred_std[-1] < 0.05
        linestyle = '--' if collapsed else '-'
        marker = 'x' if collapsed else 'o'
        
        label = exp_name.replace('capacity_', '').replace('monitor_', '')
        
        ax1.plot(epochs, pred_std, linestyle=linestyle, marker=marker,
                markersize=4, alpha=0.7, color=colors[idx], label=label)
        ax2.plot(epochs, pred_range, linestyle=linestyle, marker=marker,
                markersize=4, alpha=0.7, color=colors[idx], label=label)
    
    ax1.axhline(y=0.05, color='red', linestyle=':', linewidth=2, 
               label='Collapse threshold', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Prediction Std Dev', fontsize=12)
    ax1.set_title('Prediction Diversity Over Training', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Prediction Range', fontsize=12)
    ax2.set_title('Prediction Range Over Training', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gradient_flow(experiments, output_path='gradient_flow.png'):
    """Plot gradient norms over training (4-panel version)."""
    has_attention_gradients = False
    has_attention_entropy = False
    
    for exp_name in experiments:
        data = load_monitoring_log(exp_name)
        if data is None:
            continue
        if any('multihead_attention' in k for k in data.get('gradient_norms', {}).keys()):
            has_attention_gradients = True
        if 'attention_entropy' in data and any(e is not None for e in data['attention_entropy']):
            has_attention_entropy = True
    
    if has_attention_gradients and has_attention_entropy:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        layer_groups = {
            'LSTM Encoder': 'lstm_encoder',
            'LSTM Decoder': 'lstm_decoder', 
            'Attention (Gradients)': 'multihead_attention',
            'Attention Entropy': 'attention_entropy',
            'Output Layer': 'output_layer',
        }
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        if has_attention_gradients:
            layer_groups = {
                'LSTM Encoder': 'lstm_encoder',
                'LSTM Decoder': 'lstm_decoder',
                'Attention (Gradients)': 'multihead_attention',
                'Output Layer': 'output_layer',
            }
        else:
            layer_groups = {
                'LSTM Encoder': 'lstm_encoder',
                'LSTM Decoder': 'lstm_decoder',
                'Attention Entropy': 'attention_entropy',
                'Output Layer': 'output_layer',
            }
    
    axes = axes.flatten()
    
    for idx, (group_name, layer_key) in enumerate(layer_groups.items()):
        ax = axes[idx]
        
        if group_name == 'Attention Entropy':
            ax.set_title('Attention Entropy', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Entropy (nats)')
            
            for exp_name in experiments:
                data = load_monitoring_log(exp_name)
                if data is None or 'attention_entropy' not in data:
                    continue
                
                epochs = data['epoch']
                entropy = data['attention_entropy']
                
                valid_data = [(e, ent) for e, ent in zip(epochs, entropy) if ent is not None]
                if not valid_data:
                    continue
                
                valid_epochs, valid_entropy = zip(*valid_data)
                
                collapsed = data['prediction_std'][-1] < 0.05
                linestyle = '--' if collapsed else '-'
                label = exp_name.replace('capacity_', '').replace('monitor_', '')
                ax.plot(valid_epochs, valid_entropy, linestyle=linestyle,
                       alpha=0.7, label=label)
            
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            if not has_attention_gradients:
                ax.text(0.5, 0.95, '(Baseline: head-averaged)', 
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=8, style='italic', alpha=0.7)
            else:
                ax.text(0.5, 0.95, '(Custom: per-head averaged)', 
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=8, style='italic', alpha=0.7)
        
        else:
            for exp_name in experiments:
                data = load_monitoring_log(exp_name)
                if data is None:
                    continue
                
                epochs = data['epoch']
                matching_layers = [k for k in data['gradient_norms'].keys() 
                                 if layer_key in k]
                
                if not matching_layers:
                    continue
                
                avg_norms = []
                for epoch_idx in range(len(epochs)):
                    norms = [data['gradient_norms'][layer][epoch_idx] 
                            for layer in matching_layers
                            if epoch_idx < len(data['gradient_norms'][layer])]
                    if norms:
                        avg_norms.append(np.mean(norms))
                    else:
                        avg_norms.append(np.nan)
                
                collapsed = data['prediction_std'][-1] < 0.05
                linestyle = '--' if collapsed else '-'
                
                label = exp_name.replace('capacity_', '').replace('monitor_', '')
                ax.plot(epochs[:len(avg_norms)], avg_norms, linestyle=linestyle,
                       alpha=0.7, label=label)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title(f'{group_name}', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    if len(axes) > len(layer_groups):
        axes[-1].axis('off')
    
    plt.suptitle('Gradient Flow Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def extract_layer_gradients(data):
    """Extract gradient norms for each layer group from monitoring data."""
    gradient_norms = data.get('gradient_norms', {})
    epochs = data['epoch']
    
    layers = {
        'lstm_encoder': [],
        'lstm_decoder': [],
        'output_layer': [],
        'attention': [],
    }
    
    for epoch_idx in range(len(epochs)):
        encoder_grads = []
        decoder_grads = []
        output_grads = []
        attention_grads = []
        
        for key, values in gradient_norms.items():
            if epoch_idx < len(values):
                val = values[epoch_idx]
                if val is not None:
                    if 'lstm_encoder' in key:
                        encoder_grads.append(val)
                    elif 'lstm_decoder' in key:
                        decoder_grads.append(val)
                    elif 'output' in key or 'fc_out' in key:
                        output_grads.append(val)
                    elif 'attention' in key or 'multihead' in key:
                        attention_grads.append(val)
        
        layers['lstm_encoder'].append(np.mean(encoder_grads) if encoder_grads else np.nan)
        layers['lstm_decoder'].append(np.mean(decoder_grads) if decoder_grads else np.nan)
        layers['output_layer'].append(np.mean(output_grads) if output_grads else np.nan)
        layers['attention'].append(np.mean(attention_grads) if attention_grads else np.nan)
    
    return epochs, layers


def plot_gradient_flow_overlay(
    experiments, 
    output_path='gradient_flow_overlay.png',
    paper_format=False,
    show_entropy=False  # Default off - cleaner
):
    """
    Create overlaid gradient flow plot for publication.
    
    All layer gradient norms on single plot, with output layer highlighted
    to show collapse phenomenon.
    """
    # Match combined timeline aesthetics
    if paper_format:
        figsize = (10, 4)
    else:
        figsize = (12, 5)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors matching combined timeline style - output layer in warning red
    colors = {
        'lstm_encoder': '#2563eb',      # Blue
        'lstm_decoder': '#16a34a',      # Green  
        'output_layer': '#dc2626',      # Red (problem layer)
        'attention': '#7c3aed',         # Purple
    }
    
    labels = {
        'lstm_encoder': 'LSTM Encoder',
        'lstm_decoder': 'LSTM Decoder',
        'output_layer': 'Output Layer',
        'attention': 'Attention',
    }
    
    for exp_idx, exp_name in enumerate(experiments):
        data = load_monitoring_log(exp_name)
        if data is None:
            print(f"Warning: No data for {exp_name}")
            continue
        
        epochs, layer_grads = extract_layer_gradients(data)
        
        # Plot non-output layers first (thinner, semi-transparent)
        for layer_name in ['lstm_encoder', 'lstm_decoder', 'attention']:
            grads = layer_grads[layer_name]
            if all(np.isnan(grads)):
                continue
            
            label = labels[layer_name] if exp_idx == 0 else None
            ax.plot(epochs, grads, 
                    color=colors[layer_name],
                    linestyle='-',
                    linewidth=2.0,
                    alpha=0.7,
                    zorder=5,
                    label=label)
        
        # Plot output layer last (thicker, full opacity, on top)
        output_grads = layer_grads['output_layer']
        if not all(np.isnan(output_grads)):
            label = labels['output_layer'] if exp_idx == 0 else None
            ax.plot(epochs, output_grads, 
                    color=colors['output_layer'],
                    linestyle='-',
                    linewidth=3.5,
                    alpha=1.0,
                    zorder=10,
                    label=label)
    
    # Formatting to match combined timeline
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Gradient Norm', fontsize=14)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    ax.set_title('Gradient Flow During Training', fontsize=16, fontweight='bold')
    
    # Legend outside plot area on right
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_sign_distribution(experiments, output_path='sign_distribution.png'):
    """Plot evolution of prediction sign distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for exp_name in experiments:
        data = load_monitoring_log(exp_name)
        if data is None:
            continue
        
        epochs = data['epoch']
        pct_pos = data['pct_positive']
        pct_neg = data['pct_negative']
        
        collapsed = data['prediction_std'][-1] < 0.05
        linestyle = '--' if collapsed else '-'
        marker = 'x' if collapsed else 'o'
        
        label = exp_name.replace('capacity_', '').replace('monitor_', '')
        
        ax1.plot(epochs, pct_pos, linestyle=linestyle, marker=marker,
                markersize=4, alpha=0.7, label=label)
        ax2.plot(epochs, pct_neg, linestyle=linestyle, marker=marker,
                markersize=4, alpha=0.7, label=label)
    
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=95, color='red', linestyle=':', alpha=0.5, label='Collapse threshold')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=95, color='red', linestyle=':', alpha=0.5, label='Collapse threshold')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('% Positive Predictions', fontsize=12)
    ax1.set_title('Positive Predictions Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('% Negative Predictions', fontsize=12)
    ax2.set_title('Negative Predictions Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comprehensive_summary(experiments, output_path='comprehensive_summary.png'):
    """Create a comprehensive 6-panel summary figure."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    
    summary_data = []
    
    for idx, exp_name in enumerate(experiments):
        data = load_monitoring_log(exp_name)
        if data is None:
            continue
        
        epochs = data['epoch']
        collapsed = data['prediction_std'][-1] < 0.05
        linestyle = '--' if collapsed else '-'
        marker = 'x' if collapsed else 'o'
        label = exp_name.replace('capacity_', '').replace('monitor_', '')[:20]
        
        ax1.plot(epochs, data['prediction_std'], linestyle=linestyle, 
                marker=marker, markersize=3, alpha=0.7, color=colors[idx], label=label)
        ax2.plot(epochs, data['prediction_range'], linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, color=colors[idx])
        ax3.plot(epochs, data['num_unique_predictions'], linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, color=colors[idx])
        ax4.plot(epochs, data['pct_positive'], linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, color=colors[idx])
        ax5.plot(epochs, data['pct_negative'], linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, color=colors[idx])
        
        summary_data.append({
            'experiment': label,
            'final_std': data['prediction_std'][-1],
            'collapsed': collapsed,
            'collapse_epoch': next((i for i, s in enumerate(data['prediction_std']) 
                                  if s < 0.05), None) if collapsed else None
        })
    
    ax1.axhline(y=0.05, color='red', linestyle=':', alpha=0.7)
    ax1.set_ylabel('Prediction Std', fontsize=10)
    ax1.set_title('Prediction Diversity', fontweight='bold')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_ylabel('Prediction Range', fontsize=10)
    ax2.set_title('Value Range', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax3.set_ylabel('Unique Predictions', fontsize=10)
    ax3.set_title('Prediction Uniqueness', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4.axhline(y=95, color='red', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('% Positive', fontsize=10)
    ax4.set_title('Positive Predictions', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5.axhline(y=95, color='red', linestyle=':', alpha=0.7)
    ax5.set_xlabel('Epoch', fontsize=10)
    ax5.set_ylabel('% Negative', fontsize=10)
    ax5.set_title('Negative Predictions', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    ax6.axis('off')
    ax6.text(0.5, 0.95, 'Collapse Summary', ha='center', va='top',
            fontsize=12, fontweight='bold', transform=ax6.transAxes)
    
    table_text = "Experiment          Final Std  Status\n"
    table_text += "-" * 45 + "\n"
    for item in summary_data:
        status = "COLLAPSED" if item['collapsed'] else "Working"
        table_text += f"{item['experiment']:<20} {item['final_std']:.4f}  {status}\n"
    
    ax6.text(0.1, 0.85, table_text, ha='left', va='top',
            fontsize=8, family='monospace', transform=ax6.transAxes)
    
    fig.suptitle('Collapse Investigation - Comprehensive Summary', 
                fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot collapse monitoring data')
    parser.add_argument('experiments', nargs='*', help='Experiment names (or use --phase)')
    parser.add_argument('--phase', type=str, default=None,
                       help='Process all experiments in a phase directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory for output plots')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Generate comprehensive summary figure')
    parser.add_argument('--gradient-overlay', action='store_true',
                       help='Generate single overlaid gradient flow plot (paper-ready)')
    parser.add_argument('--paper-format', action='store_true',
                       help='Use IEEE paper formatting (smaller figures)')
    parser.add_argument('--no-entropy', action='store_true',
                       help='Omit attention entropy from gradient overlay')
    
    args = parser.parse_args()
    
    if args.phase:
        phase_path = Path('experiments') / args.phase
        if not phase_path.exists():
            print(f"Error: Phase directory not found: {phase_path}")
            return
        
        experiments = []
        for exp_dir in sorted(phase_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            monitor_path = exp_dir / 'collapse_monitoring' / 'collapse_monitor_latest.json'
            if monitor_path.exists():
                experiments.append(f"{args.phase}/{exp_dir.name}")
        
        if not experiments:
            print(f"No experiments with monitoring data found in {phase_path}")
            return
        
        print(f"Found {len(experiments)} experiments in {args.phase}")
        
        for exp_full_path in experiments:
            exp_name = exp_full_path.split('/')[-1]
            print(f"\nProcessing: {exp_name}")
            
            exp_output_dir = Path('experiments') / exp_full_path / 'monitoring'
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                if args.comprehensive:
                    plot_comprehensive_summary([exp_full_path], 
                                             exp_output_dir / 'collapse_comprehensive.png')
                elif args.gradient_overlay:
                    suffix = '_paper' if args.paper_format else ''
                    plot_gradient_flow_overlay(
                        [exp_full_path], 
                        exp_output_dir / f'gradient_flow_overlay{suffix}.png',
                        paper_format=args.paper_format,
                        show_entropy=not args.no_entropy
                    )
                else:
                    plot_prediction_diversity([exp_full_path], 
                                            exp_output_dir / 'collapse_diversity.png')
                    plot_sign_distribution([exp_full_path], 
                                         exp_output_dir / 'collapse_signs.png')
                    plot_gradient_flow([exp_full_path], 
                                     exp_output_dir / 'collapse_gradients.png')
                
                print(f"  Saved plots to: {exp_output_dir}")
            except Exception as e:
                print(f"  Error plotting {exp_name}: {e}")
        
        print(f"\nProcessed {len(experiments)} experiments")
        
    else:
        if not args.experiments:
            print("Error: Must specify experiments or use --phase")
            parser.print_help()
            return
        
        experiments = []
        for pattern in args.experiments:
            if '*' in pattern:
                exp_dir = Path('experiments')
                matches = [p.name for p in exp_dir.glob(pattern) if p.is_dir()]
                experiments.extend(sorted(matches))
            else:
                experiments.append(pattern)
        
        if not experiments:
            print("No experiments found")
            return
        
        print(f"Plotting {len(experiments)} experiments...")
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif len(experiments) == 1:
            exp_path = experiments[0]
            if '/' in exp_path:
                output_dir = Path('experiments') / exp_path / 'monitoring'
            else:
                found = False
                for phase_dir in ['00_baseline_exploration', '01_staleness_features',
                                 '01_staleness_features_fixed', '02_custom_tft',
                                 '02_vintage_baseline', '02b_vintage_sweep']:
                    test_path = Path('experiments') / phase_dir / exp_path
                    if test_path.exists():
                        output_dir = test_path / 'monitoring'
                        found = True
                        break
                if not found:
                    output_dir = Path('experiments') / exp_path / 'monitoring'
        else:
            output_dir = Path('experiments')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.comprehensive:
            plot_comprehensive_summary(experiments, output_dir / 'collapse_comprehensive.png')
        elif args.gradient_overlay:
            suffix = '_paper' if args.paper_format else ''
            plot_gradient_flow_overlay(
                experiments,
                output_dir / f'gradient_flow_overlay{suffix}.png',
                paper_format=args.paper_format,
                show_entropy=not args.no_entropy
            )
        else:
            plot_prediction_diversity(experiments, output_dir / 'collapse_diversity.png')
            plot_sign_distribution(experiments, output_dir / 'collapse_signs.png')
            plot_gradient_flow(experiments, output_dir / 'collapse_gradients.png')
        
        print("\nDone! Check output in:", output_dir)


if __name__ == '__main__':
    main()
