"""
List available checkpoints for a trained experiment.

Usage:
    # List all experiments
    python scripts/list_checkpoints.py
    
    # List checkpoints for specific experiment
    python scripts/list_checkpoints.py exp004
"""

import os
import sys
import json
import argparse


def list_experiments():
    """List all available experiments in experiments/ directory."""
    exp_dir = "experiments"
    
    if not os.path.exists(exp_dir):
        print(f"No experiments directory found at: {exp_dir}")
        return
    
    experiments = [d for d in os.listdir(exp_dir) 
                   if os.path.isdir(os.path.join(exp_dir, d))]
    
    if not experiments:
        print(f"No experiments found in {exp_dir}/")
        return
    
    print("\n" + "="*70)
    print("AVAILABLE EXPERIMENTS")
    print("="*70)
    
    for exp in sorted(experiments):
        exp_path = os.path.join(exp_dir, exp)
        
        # Check if it has a config (indicates valid experiment)
        config_path = os.path.join(exp_path, 'config.json')
        metrics_path = os.path.join(exp_path, 'final_metrics.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            feature_set = config.get('feature_set', 'unknown')
            
            # Check if training completed
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Extract epoch from best checkpoint filename
                best_path = metrics.get('best_model_path', '')
                epoch_str = "?"
                if 'epoch=' in best_path:
                    # Extract epoch number from filename like "tft-epoch=02-val_loss=0.1234.ckpt"
                    epoch_part = best_path.split('epoch=')[1].split('-')[0]
                    epoch_str = epoch_part
                
                status = f" Complete (best: epoch {epoch_str}, val_loss={metrics['best_val_loss']:.4f})"
            else:
                status = " Training in progress or incomplete"
            
            print(f"\n{exp}")
            print(f"  Feature set: {feature_set}")
            print(f"  Status: {status}")
        else:
            print(f"\n{exp}")
            print(f"   No config.json (may not be valid experiment)")
    
    print("\n" + "="*70)
    print("\nTo see checkpoints for an experiment, run:")
    print("  python scripts/list_checkpoints.py <experiment_name>")
    print()


def list_checkpoints(experiment_name):
    """List checkpoints for a specific experiment."""
    exp_dir = f"experiments/{experiment_name}"
    
    if not os.path.exists(exp_dir):
        print(f"Experiment not found: {experiment_name}")
        print(f"Directory does not exist: {exp_dir}")
        print("\nRun without arguments to see available experiments.")
        sys.exit(1)
    
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    
    if not os.path.exists(ckpt_dir):
        print(f"No checkpoints directory found for: {experiment_name}")
        print(f"Directory does not exist: {ckpt_dir}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(f"CHECKPOINTS FOR: {experiment_name}")
    print("="*70)
    
    # Load final metrics to show best
    metrics_path = os.path.join(exp_dir, 'final_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"\nBest model (val_loss={metrics['best_val_loss']:.6f}):")
        print(f"  {metrics['best_model_path']}")
        
        if metrics.get('early_stopped', False):
            print(f"\n Training stopped early at epoch {metrics['total_epochs']}")
        else:
            print(f"\n Training completed {metrics['total_epochs']} epochs")
    else:
        print("\n No final_metrics.json found (training may be incomplete)")
    
    # List all checkpoints
    checkpoints = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')])
    
    if not checkpoints:
        print("\n No checkpoint files found")
        sys.exit(1)
    
    print(f"\nAll checkpoints ({len(checkpoints)} total):")
    for ckpt in checkpoints:
        path = os.path.join(ckpt_dir, ckpt)
        size_mb = os.path.getsize(path) / 1e6
        
        # Highlight if this is the best
        if os.path.exists(metrics_path):
            best_name = os.path.basename(metrics['best_model_path'])
            marker = " <-- BEST" if ckpt == best_name else ""
        else:
            marker = ""
        
        print(f"  {ckpt} ({size_mb:.1f} MB){marker}")
    
    print("\n" + "="*70)
    print("\nTo evaluate with best checkpoint:")
    print(f"  python train/evaluate_tft.py --experiment-name {experiment_name}")
    print("\nTo evaluate with specific checkpoint:")
    print(f"  python train/evaluate_tft.py --experiment-name {experiment_name} \\")
    print(f"      --checkpoint experiments/{experiment_name}/checkpoints/<checkpoint>.ckpt")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='List experiments and checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python scripts/list_checkpoints.py
  
  # List checkpoints for specific experiment
  python scripts/list_checkpoints.py exp004
        """
    )
    
    parser.add_argument('experiment_name', type=str, nargs='?', default=None,
                        help='Experiment name (if not provided, lists all experiments)')
    
    args = parser.parse_args()
    
    if args.experiment_name is None:
        list_experiments()
    else:
        list_checkpoints(args.experiment_name)


if __name__ == "__main__":
    main()
