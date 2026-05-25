#!/usr/bin/env python3
"""
Plot regime attention gate evolution during training.

Usage:
    python scripts/plot_gate_evolution.py experiments/test_regime_attention_weekly_full_enc12
    python scripts/plot_gate_evolution.py experiments/test_regime_attention_weekly_full_enc12 --presentation
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def plot_gate_evolution(experiment_dir, output_path=None, presentation=False):
    """Plot regime attention gate values over training epochs."""
    
    experiment_dir = Path(experiment_dir)
    monitor_path = experiment_dir / 'collapse_monitoring' / 'collapse_monitor_latest.json'
    
    with open(monitor_path) as f:
        history = json.load(f)
    
    gates = history['regime_attention_gate_values']
    # gates[epoch][regime][head]
    
    epochs = range(len(gates))
    
    # Extract gate values
    low_vol_h0 = [g[0][0] for g in gates]
    low_vol_h1 = [g[0][1] for g in gates]
    high_vol_h0 = [g[1][0] for g in gates]
    high_vol_h1 = [g[1][1] for g in gates]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6) if presentation else (8, 5))
    
    # Colors: blue for low-vol, red/orange for high-vol
    low_vol_color = '#2E86AB'
    high_vol_color = '#E63946'
    
    lw = 2.5 if presentation else 2
    
    # Plot lines: solid for head 0, dashed for head 1
    ax.plot(epochs, low_vol_h0, color=low_vol_color, linestyle='-', linewidth=lw, 
            label='Low Vol - Head 1')
    ax.plot(epochs, low_vol_h1, color=low_vol_color, linestyle='--', linewidth=lw, 
            label='Low Vol - Head 2')
    ax.plot(epochs, high_vol_h0, color=high_vol_color, linestyle='-', linewidth=lw, 
            label='High Vol - Head 1')
    ax.plot(epochs, high_vol_h1, color=high_vol_color, linestyle='--', linewidth=lw, 
            label='High Vol - Head 2')
    
    # Reference line at 0.5 (initialization / neutral)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Neutral (init)')
    
    if presentation:
        # Expand y-axis for less squished look
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        ax.set_ylim(ymin - 0.15 * y_range, ymax + 0.15 * y_range)
        
        # Shade background regions
        ylim = ax.get_ylim()
        ax.axhspan(0.5, ylim[1], alpha=0.08, color=high_vol_color, zorder=0)
        ax.axhspan(ylim[0], 0.5, alpha=0.08, color=low_vol_color, zorder=0)
        
        # Annotation arrows pointing to final values (shorter, with labels nearby)
        final_epoch = len(epochs) - 1
        
        # Arrow to high-vol (if it ends above 0.5)
        high_vol_final = max(high_vol_h0[-1], high_vol_h1[-1])
        if high_vol_final > 0.5:
            arrow_x = final_epoch * 0.75
            ax.annotate('Amplifies', xy=(arrow_x + final_epoch * 0.08, high_vol_final), 
                       xytext=(arrow_x, high_vol_final + 0.015),
                       fontsize=10, color=high_vol_color, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=high_vol_color, lw=1.5))
        
        # Arrow to low-vol (if it ends below 0.5)
        low_vol_final = min(low_vol_h0[-1], low_vol_h1[-1])
        if low_vol_final < 0.5:
            arrow_x = final_epoch * 0.75
            ax.annotate('Dampens', xy=(arrow_x + final_epoch * 0.08, low_vol_final), 
                       xytext=(arrow_x, low_vol_final - 0.02),
                       fontsize=10, color=low_vol_color, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=low_vol_color, lw=1.5))
    
    ax.set_xlabel('Epoch', fontsize=12 if presentation else 10)
    ax.set_ylabel('Gate Value', fontsize=12 if presentation else 10)
    ax.set_title('Learned Attention Gates by Regime', 
                 fontsize=14 if presentation else 12, fontweight='bold' if presentation else 'normal')
    ax.legend(loc='best', fontsize=10 if presentation else 9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = experiment_dir / 'gate_evolution.png'
    
    plt.savefig(output_path, dpi=200 if presentation else 150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot regime attention gate evolution')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for plot')
    parser.add_argument('--presentation', action='store_true', 
                        help='Use presentation style (larger, with annotations)')
    
    args = parser.parse_args()
    plot_gate_evolution(args.experiment_dir, args.output, args.presentation)


if __name__ == "__main__":
    main()