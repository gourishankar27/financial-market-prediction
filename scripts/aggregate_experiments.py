#!/usr/bin/env python
"""
Aggregate all experiment results into a single comparison CSV.

Extracts key metrics from:
- config.json (architecture, hyperparameters)
- final_metrics.json (training outcome)
- evaluation/evaluation_metrics.json (test performance)
- collapse_monitoring/collapse_monitor_latest.json (training dynamics)
- evaluation/*.log (prediction statistics - parsed)

Usage:
    python aggregate_experiments.py --experiments-dir experiments/
    python aggregate_experiments.py --experiments-dir experiments/ --output results/comparison.csv
    python aggregate_experiments.py --experiments-dir experiments/06_regime_sweep/ --verbose
"""

import os
import sys
import json
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate experiment results into comparison CSV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Root experiments directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: experiments_comparison_TIMESTAMP.csv)')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Search subdirectories recursively')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed extraction info')
    parser.add_argument('--include-unevaluated', action='store_true',
                        help='Include experiments without evaluation results')
    return parser.parse_args()


def find_experiments(root_dir, recursive=True):
    """Find all experiment directories (those containing config.json)."""
    root = Path(root_dir)
    experiments = []
    
    if recursive:
        for config_path in root.rglob('config.json'):
            exp_dir = config_path.parent
            # Skip if this is a nested config (e.g., in checkpoints)
            if 'checkpoint' not in str(exp_dir).lower():
                experiments.append(exp_dir)
    else:
        for item in root.iterdir():
            if item.is_dir() and (item / 'config.json').exists():
                experiments.append(item)
    
    return sorted(experiments)


def safe_get(d, *keys, default=None):
    """Safely navigate nested dict."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def load_json(path):
    """Load JSON file, return empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def parse_eval_log(exp_dir):
    """Parse evaluation log for prediction statistics."""
    eval_dir = exp_dir / 'evaluation'
    if not eval_dir.exists():
        return {}
    
    # Find the most recent evaluation log
    logs = list(eval_dir.glob('evaluation_*.log'))
    if not logs:
        return {}
    
    latest_log = max(logs, key=lambda p: p.stat().st_mtime)
    
    stats = {}
    try:
        with open(latest_log) as f:
            content = f.read()
        
        # Parse prediction statistics block
        # Looking for:
        #   Min: -0.063527
        #   Max: 0.228986
        #   Mean: 0.040257
        #   Std: 0.037195
        #   Unique values: 1282
        #   Negative predictions: 359/1282
        
        patterns = {
            'pred_min': r'Min:\s*([-\d.]+)',
            'pred_max': r'Max:\s*([-\d.]+)',
            'pred_mean': r'Mean:\s*([-\d.]+)',
            'pred_std': r'Std:\s*([-\d.]+)',
            'pred_unique': r'Unique values:\s*(\d+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                val = match.group(1)
                stats[key] = float(val) if '.' in val else int(val)
        
        # Parse negative predictions: "Negative predictions: 359/1282"
        neg_match = re.search(r'Negative predictions:\s*(\d+)/(\d+)', content)
        if neg_match:
            neg_count = int(neg_match.group(1))
            total = int(neg_match.group(2))
            stats['pred_pct_negative'] = (neg_count / total) * 100
            stats['pred_pct_positive'] = ((total - neg_count) / total) * 100
        
        # Compute range from min/max if we have them
        if 'pred_min' in stats and 'pred_max' in stats:
            stats['pred_range'] = stats['pred_max'] - stats['pred_min']
        
    except Exception as e:
        pass
    
    return stats


def extract_experiment_data(exp_dir, verbose=False):
    """Extract all relevant data from a single experiment."""
    exp_dir = Path(exp_dir)
    exp_name = exp_dir.name
    
    # Determine phase from parent directory structure
    phase = ''
    if exp_dir.parent.name != 'experiments':
        phase = exp_dir.parent.name
    
    data = {
        'experiment_name': exp_name,
        'experiment_path': str(exp_dir),
        'phase': phase,
    }
    
    # Track missing files for verbose output
    missing = []
    
    # === config.json ===
    config = load_json(exp_dir / 'config.json')
    if not config:
        missing.append('config.json')
    else:
        # Architecture
        arch = config.get('architecture', {})
        data['hidden_size'] = arch.get('hidden_size')
        data['attention_heads'] = arch.get('attention_head_size')
        data['dropout'] = arch.get('dropout')
        data['max_encoder_length'] = arch.get('max_encoder_length')
        
        # Training
        training = config.get('training', {})
        data['learning_rate'] = training.get('learning_rate')
        data['batch_size'] = training.get('batch_size')
        data['max_epochs'] = training.get('max_epochs')
        
        # Features
        data['feature_set'] = config.get('feature_set')
        data['frequency'] = config.get('frequency')
        data['alignment'] = config.get('alignment')
        
        # Regime output
        regime = config.get('regime_output', {})
        data['regime_enabled'] = regime.get('enabled', False)
        data['num_regimes'] = regime.get('num_regimes')
        data['routing_strategy'] = regime.get('routing_strategy')
        data['routing_mode'] = regime.get('routing_mode')
        data['vix_threshold'] = regime.get('vix_threshold')
        data['vix_threshold_low'] = regime.get('vix_threshold_low')
        data['vix_threshold_high'] = regime.get('vix_threshold_high')
        data['expert_hidden_size'] = regime.get('expert_hidden_size')
        data['hard_routing_train'] = regime.get('hard_routing_train')
        data['load_balance_weight'] = regime.get('load_balance_weight')
        
        # Loss config
        loss = config.get('loss', {})
        data['dist_loss_std_weight'] = loss.get('dist_loss_std_weight')
        data['directional_weight'] = loss.get('directional_weight')
        
        # Transfer learning
        transfer = config.get('transfer_learning', {})
        data['freeze_backbone'] = transfer.get('freeze_backbone')
        data['load_checkpoint'] = transfer.get('load_checkpoint') is not None
    
    # === final_metrics.json ===
    final = load_json(exp_dir / 'final_metrics.json')
    if not final:
        missing.append('final_metrics.json')
    else:
        data['best_val_loss'] = final.get('best_val_loss')
        data['total_epochs'] = final.get('total_epochs')
        data['early_stopped'] = final.get('early_stopped')
    
    # === evaluation/evaluation_metrics.json ===
    eval_metrics = load_json(exp_dir / 'evaluation' / 'evaluation_metrics.json')
    if not eval_metrics:
        missing.append('evaluation/evaluation_metrics.json')
        data['evaluated'] = False
    else:
        data['evaluated'] = True
        
        # Prediction statistics (new)
        pred_stats = eval_metrics.get('prediction_stats', {})
        data['pred_min'] = pred_stats.get('min')
        data['pred_max'] = pred_stats.get('max')
        data['pred_mean'] = pred_stats.get('mean')
        data['pred_std'] = pred_stats.get('std')
        data['pred_range'] = pred_stats.get('range')
        data['pred_unique'] = pred_stats.get('num_unique')
        data['pred_pct_positive'] = pred_stats.get('pct_positive')
        data['pred_pct_negative'] = pred_stats.get('pct_negative')
        
        # Statistical metrics
        stat = eval_metrics.get('statistical_metrics', {})
        data['test_mse'] = stat.get('mse')
        data['test_rmse'] = stat.get('rmse')
        data['test_r2'] = stat.get('r2')
        data['test_mae'] = stat.get('mae')
        
        # Financial metrics
        fin = eval_metrics.get('financial_metrics', {})
        data['directional_accuracy'] = fin.get('directional_accuracy')
        data['sharpe_ratio'] = fin.get('sharpe_ratio')
        data['alpha'] = fin.get('alpha')
        data['auc_roc'] = fin.get('auc_roc')
        data['max_drawdown'] = fin.get('max_drawdown')
        data['total_return'] = fin.get('total_return')
        
        # Mode stats (collapse detection)
        mode = eval_metrics.get('mode_stats', {})
        data['healthy_pct'] = mode.get('healthy_pct')
        data['degraded_pct'] = mode.get('degraded_pct')
        data['unidirectional_pct'] = mode.get('unidirectional_pct')
        data['weak_collapse_pct'] = mode.get('weak_collapse_pct')
        data['strong_collapse_pct'] = mode.get('strong_collapse_pct')
        data['total_days'] = mode.get('total_days')
        
        # Derived
        if data.get('healthy_pct') is not None:
            data['problematic_pct'] = 100 - data['healthy_pct']
    
    # === Fallback: Parse evaluation log for prediction stats (for older experiments) ===
    if not data.get('pred_std'):
        log_stats = parse_eval_log(exp_dir)
        # Only fill in if not already present
        for key, val in log_stats.items():
            if key not in data or data[key] is None:
                data[key] = val
    
    # === collapse_monitoring/collapse_monitor_latest.json ===
    collapse = load_json(exp_dir / 'collapse_monitoring' / 'collapse_monitor_latest.json')
    if not collapse:
        missing.append('collapse_monitoring/collapse_monitor_latest.json')
    else:
        # Get final epoch values
        def get_last(key, default=None):
            vals = collapse.get(key, [])
            return vals[-1] if vals else default
        
        data['final_pred_std'] = get_last('prediction_std')
        data['final_pred_mean'] = get_last('prediction_mean')
        data['final_pct_positive'] = get_last('pct_positive')
        data['final_num_unique'] = get_last('num_unique_predictions')
        data['final_attention_entropy'] = get_last('attention_entropy')
        
        # Regime-specific (if available)
        data['final_routing_entropy'] = get_last('regime_entropy_normalized')
        data['final_dominant_regime_pct'] = get_last('dominant_regime_pct')
        data['final_vix_correlation'] = get_last('vix_correlation')
        
        # Expert stats
        expert_stds = collapse.get('expert_stds', {})
        for i in range(3):
            key = f'expert_{i}'
            if key in expert_stds:
                vals = expert_stds[key]
                data[f'final_expert_{i}_std'] = vals[-1] if vals else None
        
        expert_means = collapse.get('expert_means', {})
        for i in range(3):
            key = f'expert_{i}'
            if key in expert_means:
                vals = expert_means[key]
                data[f'final_expert_{i}_mean'] = vals[-1] if vals else None
        
        # Expert weight divergence
        data['final_expert_weight_diff'] = get_last('expert_weight_diff')
        data['final_expert_weight_cosine'] = get_last('expert_weight_cosine')
        
        # Gradient norms (summarize)
        grad_norms = collapse.get('gradient_norms', {})
        if 'output_layer' in grad_norms:
            vals = grad_norms['output_layer']
            data['final_grad_norm_output'] = vals[-1] if vals else None
    
    # === attention_analysis/attention_analysis_results.json ===
    attention = load_json(exp_dir / 'attention_analysis' / 'attention_analysis_results.json')
    if not attention:
        missing.append('attention_analysis/ (optional)')
    else:
        # Overall summary
        overall = attention.get('overall_summary', {})
        data['attention_entropy_mean'] = overall.get('mean_entropy')
        data['attention_top_feature'] = safe_get(overall, 'top_features', 0, 'feature') if overall.get('top_features') else None
    
    # Verbose output with missing file warnings
    if verbose:
        eval_status = "Y" if data.get('evaluated') else "N"
        print(f"  [{eval_status}] {exp_name}: val_loss={data.get('best_val_loss', 'N/A')}, "
              f"dir_acc={data.get('directional_accuracy', 'N/A')}")
        
        # Warn about missing files (excluding optional ones)
        critical_missing = [m for m in missing if 'optional' not in m]
        if critical_missing:
            print(f"       Missing: {', '.join(critical_missing)}")
    
    return data


def compute_derived_metrics(df):
    """Add derived/composite metrics."""
    # Composite score: balance directional accuracy and health
    if 'directional_accuracy' in df.columns and 'healthy_pct' in df.columns:
        df['composite_score'] = (
            df['directional_accuracy'].fillna(0.5) * 0.5 +
            df['healthy_pct'].fillna(0) / 100 * 0.3 +
            (1 - df['unidirectional_pct'].fillna(0) / 100) * 0.2
        )
    
    # Regime effectiveness: do experts have different predictions?
    expert_cols = [c for c in df.columns if 'final_expert_' in c and '_std' in c]
    if expert_cols:
        df['min_expert_std'] = df[expert_cols].min(axis=1)
        df['max_expert_std'] = df[expert_cols].max(axis=1)
        df['any_expert_collapsed'] = df['min_expert_std'] < 0.02
    
    return df


def main():
    args = parse_args()
    
    print("=" * 70)
    print("EXPERIMENT AGGREGATION")
    print("=" * 70)
    print(f"Scanning: {args.experiments_dir}")
    print(f"Recursive: {args.recursive}")
    print()
    
    # Find experiments
    experiments = find_experiments(args.experiments_dir, args.recursive)
    print(f"Found {len(experiments)} experiments")
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Extract data from each
    print("\nExtracting data...")
    all_data = []
    for exp_dir in experiments:
        data = extract_experiment_data(exp_dir, verbose=args.verbose)
        
        # Skip unevaluated unless requested
        if not args.include_unevaluated and not data.get('evaluated', False):
            continue
        
        all_data.append(data)
    
    if not all_data:
        print("No experiments with evaluation results found!")
        print("Use --include-unevaluated to include all experiments")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Compute derived metrics
    df = compute_derived_metrics(df)
    
    # Sort by composite score or directional accuracy
    sort_col = 'composite_score' if 'composite_score' in df.columns else 'directional_accuracy'
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position='last')
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'experiments_comparison_{timestamp}.csv'
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} experiments to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal experiments: {len(df)}")
    print(f"Evaluated: {df['evaluated'].sum()}")
    
    if 'regime_enabled' in df.columns:
        print(f"With regime output: {df['regime_enabled'].sum()}")
    
    if 'directional_accuracy' in df.columns:
        print(f"\nDirectional Accuracy:")
        print(f"  Mean: {df['directional_accuracy'].mean():.3f}")
        print(f"  Max:  {df['directional_accuracy'].max():.3f}")
        print(f"  Min:  {df['directional_accuracy'].min():.3f}")
    
    if 'healthy_pct' in df.columns:
        print(f"\nHealthy Percentage:")
        print(f"  Mean: {df['healthy_pct'].mean():.1f}%")
        print(f"  Max:  {df['healthy_pct'].max():.1f}%")
    
    # Top 5
    print("\n" + "-" * 70)
    print("TOP 5 EXPERIMENTS (by composite score or directional accuracy)")
    print("-" * 70)
    
    display_cols = ['experiment_name', 'phase', 'directional_accuracy', 'sharpe_ratio', 
                    'healthy_pct', 'pred_std', 'regime_enabled']
    display_cols = [c for c in display_cols if c in df.columns]
    
    print(df.head(5)[display_cols].to_string(index=False))
    
    # Bottom 5 (worst)
    print("\n" + "-" * 70)
    print("BOTTOM 5 EXPERIMENTS")
    print("-" * 70)
    print(df.tail(5)[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()