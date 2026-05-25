#!/usr/bin/env python3
"""
Summarize all experiments into CSV files for analysis.
Enhanced version incorporating collapse detection and ranking from compare_evaluations.py

Outputs:
    - experiments_summary.csv: Full details for all experiments
    - experiments_summary_key_metrics.csv: Condensed version with essential metrics
    - working_models.csv: Only non-collapsed evaluated models (with --split-working)
    - collapsed_models.csv: Only collapsed models (with --split-working)

Usage:
    python scripts/summarize_experiments.py --all  # All phases
    python scripts/summarize_experiments.py --phase 00_baseline_exploration
    python scripts/summarize_experiments.py --phase 01_staleness_features
    python scripts/summarize_experiments.py --group sweep2  # Across all phases
    python scripts/summarize_experiments.py --split-working  # Also output working/collapsed CSVs
    python scripts/summarize_experiments.py 00_baseline_exploration/exp001 01_staleness_features/exp002
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime


def infer_group(exp_name):
    """Infer experiment group from name."""
    if exp_name.startswith('sweep2'):
        return 'sweep2'
    elif exp_name.startswith('sweep'):
        return 'sweep1'
    elif exp_name.startswith('capacity'):
        return 'capacity'
    elif exp_name.startswith('exp'):
        return 'early_exp'
    else:
        return 'other'


def extract_experiment_data(exp_path):
    """Extract data from one experiment directory."""
    exp_name = exp_path.name
    
    # Determine phase from parent directory
    parent_name = exp_path.parent.name
    phase = parent_name if parent_name != 'experiments' else None
    
    data = {
        'experiment_name': exp_name,
        'phase': phase,
        'group': infer_group(exp_name),
    }
    
    # Read config.json
    config_path = exp_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Extract key hyperparameters
        data['hidden_size'] = config.get('architecture', {}).get('hidden_size')
        data['dropout'] = config.get('architecture', {}).get('dropout')
        data['learning_rate'] = config.get('training', {}).get('learning_rate')
        data['encoder_length'] = config.get('architecture', {}).get('max_encoder_length')
        data['attention_heads'] = config.get('architecture', {}).get('attention_head_size')
        data['batch_size'] = config.get('training', {}).get('batch_size')
        data['created_at'] = config.get('created_at')
    
    # Read final_metrics.json
    final_path = exp_path / 'final_metrics.json'
    if final_path.exists():
        with open(final_path) as f:
            final = json.load(f)
        
        data['best_val_loss'] = final.get('best_val_loss')
        data['total_epochs'] = final.get('total_epochs')
        data['early_stopped'] = final.get('early_stopped')
    
    # Read evaluation_metrics.json if exists
    eval_path = exp_path / 'evaluation' / 'evaluation_metrics.json'
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        # Statistical metrics
        stat = eval_data.get('statistical_metrics', {})
        data['test_mse'] = stat.get('mse')
        data['test_rmse'] = stat.get('rmse')
        data['test_mae'] = stat.get('mae')
        data['test_r2'] = stat.get('r2')
        
        # Financial metrics
        fin = eval_data.get('financial_metrics', {})
        data['dir_acc'] = fin.get('directional_accuracy')
        data['sharpe_ratio'] = fin.get('sharpe_ratio')
        data['max_drawdown'] = fin.get('max_drawdown')
        data['total_return'] = fin.get('total_return')
        data['num_trades'] = fin.get('num_trades')
        data['precision'] = fin.get('precision')
        data['recall'] = fin.get('recall')
        data['f1_score'] = fin.get('f1_score')
        data['auc_roc'] = fin.get('auc_roc')
        data['alpha'] = fin.get('alpha')
        
        # Mode statistics (from 4-mode quality detection system)
        if 'mode_stats' in eval_data:
            mode_stats = eval_data['mode_stats']
            data['healthy_pct'] = mode_stats.get('healthy_pct')
            data['degraded_pct'] = mode_stats.get('degraded_pct')
            data['unidirectional_pct'] = mode_stats.get('unidirectional_pct')
            data['weak_collapse_pct'] = mode_stats.get('weak_collapse_pct')
            data['strong_collapse_pct'] = mode_stats.get('strong_collapse_pct')
            data['healthy_days'] = mode_stats.get('healthy_days')
            data['degraded_days'] = mode_stats.get('degraded_days')
            data['unidirectional_days'] = mode_stats.get('unidirectional_days')
            data['weak_collapse_days'] = mode_stats.get('weak_collapse_days')
            data['strong_collapse_days'] = mode_stats.get('strong_collapse_days')
            
            # Compute boolean quality flags from mode percentages
            data['has_any_collapse'] = (mode_stats.get('weak_collapse_pct', 0) + mode_stats.get('strong_collapse_pct', 0)) > 0
            data['has_strong_collapse'] = mode_stats.get('strong_collapse_pct', 0) > 0
            data['has_degradation'] = mode_stats.get('degraded_pct', 0) > 0
            data['has_unidirectional'] = mode_stats.get('unidirectional_pct', 0) > 0
            data['problematic_pct'] = (mode_stats.get('degraded_pct', 0) + 
                                       mode_stats.get('unidirectional_pct', 0) +
                                       mode_stats.get('weak_collapse_pct', 0) + 
                                       mode_stats.get('strong_collapse_pct', 0))
        
        # Enhanced collapse detection from confusion matrix
        if 'confusion_matrix' in fin:
            conf = fin['confusion_matrix']
            # conf_matrix = [[TN, FP], [FN, TP]]
            predicted_down = conf[0][0] + conf[1][0]  # TN + FN
            predicted_up = conf[0][1] + conf[1][1]    # FP + TP
            total_predictions = predicted_down + predicted_up
            
            data['predicted_down'] = predicted_down
            data['predicted_up'] = predicted_up
            data['pct_down_predictions'] = predicted_down / total_predictions if total_predictions > 0 else 0
            data['pct_up_predictions'] = predicted_up / total_predictions if total_predictions > 0 else 0
        
        data['evaluated'] = True
    else:
        data['evaluated'] = False
    
    # Read collapse diagnosis if exists (from diagnose_existing_models.py)
    diagnosis_path = exp_path / 'collapse_diagnosis.json'
    pred_stats_loaded = False
    
    if diagnosis_path.exists():
        with open(diagnosis_path) as f:
            diag = json.load(f)
        
        # Use diagnosis collapse flag if available
        diag_collapsed = diag.get('collapsed')
        data['collapse_type'] = diag.get('collapse_type')
        
        # Extract prediction stats from diagnosis
        if 'predictions' in diag and diag['predictions']:
            pred = diag['predictions']
            data['pred_std'] = pred.get('std')
            data['pred_mean'] = pred.get('mean')
            data['pred_min'] = pred.get('min')
            data['pred_max'] = pred.get('max')
            data['num_unique'] = pred.get('num_unique')
            data['num_positive'] = pred.get('num_positive')
            data['num_negative'] = pred.get('num_negative')
            data['num_zero'] = pred.get('num_zero')
            data['pct_positive'] = pred.get('pct_positive')
            data['pct_negative'] = pred.get('pct_negative')
            pred_stats_loaded = True
    else:
        diag_collapsed = None
        data['collapse_type'] = None
    
    # Fallback: read from predictions.csv if collapse_diagnosis doesn't have stats
    if not pred_stats_loaded:
        pred_csv = exp_path / 'evaluation' / 'predictions.csv'
        if pred_csv.exists():
            import numpy as np
            pred_df = pd.read_csv(pred_csv)
            if 'Predicted' in pred_df.columns:
                preds = pred_df['Predicted'].values
                
                data['pred_std'] = float(preds.std())
                data['pred_mean'] = float(preds.mean())
                data['pred_min'] = float(preds.min())
                data['pred_max'] = float(preds.max())
                data['num_unique'] = int(len(np.unique(preds.round(6))))
                data['num_positive'] = int((preds > 0).sum())
                data['num_negative'] = int((preds < 0).sum())
                data['num_zero'] = int((preds == 0).sum())
                data['pct_positive'] = float((preds > 0).mean() * 100)
                data['pct_negative'] = float((preds < 0).mean() * 100)
    
    # Determine final collapse status using multiple indicators
    # Priority: confusion matrix > diagnosis > pred_std
    if data.get('evaluated') and 'pct_down_predictions' in data:
        # Use confusion matrix if available (most reliable)
        data['collapsed'] = (
            data['pct_down_predictions'] == 0 or 
            data['pct_up_predictions'] == 0 or
            data.get('pred_std', 1.0) < 0.05
        )
    elif diag_collapsed is not None:
        # Use diagnosis if available
        data['collapsed'] = diag_collapsed
    elif 'pred_std' in data:
        # Fallback to pred_std
        data['collapsed'] = data['pred_std'] < 0.05
    else:
        # Unknown
        data['collapsed'] = None
        
    return data


def compute_composite_score(df):
    """Compute composite score for ranking models."""
    if 'auc_roc' in df.columns and 'dir_acc' in df.columns:
        # Balance discriminative power (AUC) with accuracy
        # AUC already [0,1], dir_acc is [0,1], center at 0.5
        df['composite_score'] = df['auc_roc'] + (df['dir_acc'] - 0.5)
    return df


def print_working_vs_collapsed_summary(df):
    """Print analysis of working vs collapsed models."""
    if 'collapsed' not in df.columns or not df['collapsed'].notna().any():
        return
    
    evaluated = df[df['evaluated'] == True].copy()
    if len(evaluated) == 0:
        return
    
    working = evaluated[evaluated['collapsed'] == False].copy()
    collapsed = evaluated[evaluated['collapsed'] == True].copy()
    
    print(f"\n{'='*80}")
    print("WORKING vs COLLAPSED ANALYSIS")
    print(f"{'='*80}")
    print(f"Total evaluated: {len(evaluated)}")
    print(f"Working models (varied predictions): {len(working)}")
    print(f"Collapsed models (constant predictions): {len(collapsed)}")
    
    if len(working) > 0:
        print(f"\n{'='*80}")
        print("TOP WORKING MODELS")
        print(f"{'='*80}")
        
        # Add composite score
        working = compute_composite_score(working)
        
        display_cols = ['experiment_name', 'auc_roc', 'dir_acc', 'sharpe_ratio', 
                        'pct_down_predictions', 'num_trades']
        display_cols = [c for c in display_cols if c in working.columns]
        
        if 'composite_score' in working.columns:
            print("\nTop 5 by Composite Score (AUC + (dir_acc - 0.5)):")
            top = working.nlargest(5, 'composite_score')
            for col in ['auc_roc', 'dir_acc', 'sharpe_ratio', 'composite_score']:
                if col in top.columns:
                    print(f"\n{col}:")
                    for _, row in top.iterrows():
                        print(f"  {row['experiment_name']}: {row[col]:.4f}")
        
        if 'auc_roc' in working.columns:
            print("\nTop 5 by AUC-ROC (Discriminative Power):")
            top = working.nlargest(5, 'auc_roc')[display_cols]
            print(top.to_string(index=False))
        
        if 'dir_acc' in working.columns:
            print("\nTop 5 by Directional Accuracy:")
            top = working.nlargest(5, 'dir_acc')[display_cols]
            print(top.to_string(index=False))
        
        if 'sharpe_ratio' in working.columns:
            print("\nTop 5 by Sharpe Ratio:")
            top = working.nlargest(5, 'sharpe_ratio')[display_cols]
            print(top.to_string(index=False))
    
    if len(collapsed) > 0:
        print(f"\n{'='*80}")
        print("COLLAPSED MODELS (Reference)")
        print(f"{'='*80}")
        print(f"\nShowing first 5 collapsed models:")
        display_cols = ['experiment_name', 'dir_acc', 'sharpe_ratio', 'pct_down_predictions']
        display_cols = [c for c in display_cols if c in collapsed.columns]
        print(collapsed[display_cols].head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment results')
    parser.add_argument('experiments', nargs='*', help='Specific experiment names')
    parser.add_argument('--all', action='store_true', help='Process all experiments across all phases')
    parser.add_argument('--phase', type=str, 
                       help='Process specific phase directory (e.g., 00_baseline_exploration)')
    parser.add_argument('--group', type=str, help='Process specific group (sweep2, capacity, etc)')
    parser.add_argument('--output', type=str, default='experiments_summary.csv', 
                       help='Output CSV filename')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output files (default: current directory)')
    parser.add_argument('--evaluated-only', action='store_true',
                       help='Only include experiments with evaluation results')
    parser.add_argument('--split-working', action='store_true',
                       help='Also output working_models.csv and collapsed_models.csv')
    args = parser.parse_args()
    
    exp_dir = Path('experiments')
    if not exp_dir.exists():
        print(f"Error: {exp_dir} directory not found")
        return
    
    # Determine which experiments to process
    if args.all:
        # Search all phase directories (00_, 01_, 02_, etc.)
        exp_paths = []
        for phase_dir in sorted(exp_dir.iterdir()):
            if phase_dir.is_dir() and phase_dir.name[0].isdigit():  # Phase directories start with digit
                exp_paths.extend([p for p in phase_dir.iterdir() if p.is_dir()])
    elif args.phase:
        # Process specific phase directory
        phase_dir = exp_dir / args.phase
        if not phase_dir.exists():
            print(f"Error: Phase directory {phase_dir} not found")
            return
        exp_paths = [p for p in phase_dir.iterdir() if p.is_dir()]
    elif args.group:
        # Search across all phases for matching group
        exp_paths = []
        for phase_dir in sorted(exp_dir.iterdir()):
            if phase_dir.is_dir():
                exp_paths.extend([p for p in phase_dir.iterdir() 
                                 if p.is_dir() and infer_group(p.name) == args.group])
    elif args.experiments:
        exp_paths = [exp_dir / name for name in args.experiments]
    else:
        print("Error: Must specify --all, --phase, --group, or experiment names")
        return
    
    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path('.')
    
    # Extract data from each experiment
    print(f"\nScanning {len(exp_paths)} experiments...")
    results = []
    for exp_path in sorted(exp_paths):
        if not exp_path.exists():
            print(f"Warning: {exp_path.name} not found, skipping")
            continue
        
        try:
            data = extract_experiment_data(exp_path)
            
            # Skip if --evaluated-only and no evaluation
            if args.evaluated_only and not data.get('evaluated'):
                continue
            
            results.append(data)
            status = " evaluated" if data.get('evaluated') else "  not evaluated"
            collapsed = " (COLLAPSED)" if data.get('collapsed') else ""
            print(f"  {exp_path.name}: {status}{collapsed}")
        except Exception as e:
            print(f"  {exp_path.name}: ERROR - {e}")
    
    if not results:
        print("\nNo experiments processed")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add composite score for all evaluated models
    df = compute_composite_score(df)
    
    # Sort by phase, then group, then experiment name
    df = df.sort_values(['phase', 'group', 'experiment_name'])
    
    # Prepare output paths
    output_path = output_dir / args.output
    key_metrics_path = output_dir / args.output.replace('.csv', '_key_metrics.csv')
    
    # Save full results
    df.to_csv(output_path, index=False)
    print(f"\nâœ“ Saved {len(df)} experiments to {output_path}")
    
    # Create a condensed summary with key metrics
    key_metrics_cols = [
        'experiment_name', 'phase', 'group', 'hidden_size', 'dropout', 'learning_rate',
        'best_val_loss', 'total_epochs', 'early_stopped',
        'test_mse', 'test_rmse', 'test_r2',
        'dir_acc', 'sharpe_ratio', 'auc_roc', 'alpha',
        # 5-mode quality metrics (PRIMARY)
        'healthy_pct', 'degraded_pct', 'unidirectional_pct', 'weak_collapse_pct', 'strong_collapse_pct', 'problematic_pct',
        'has_any_collapse', 'has_strong_collapse', 'has_degradation', 'has_unidirectional',
        # Prediction characteristics
        'pred_std', 'num_unique', 'pct_positive', 'pct_negative',
        'composite_score', 'evaluated'
    ]
    # Only include columns that exist
    key_metrics_cols = [c for c in key_metrics_cols if c in df.columns]
    df_key_metrics = df[key_metrics_cols]
    
    df_key_metrics.to_csv(key_metrics_path, index=False)
    print(f" Saved key metrics summary to {key_metrics_path}")
    
    # Split working vs collapsed if requested
    if args.split_working and 'collapsed' in df.columns:
        evaluated = df[df['evaluated'] == True].copy()
        working = evaluated[evaluated['collapsed'] == False].copy()
        collapsed = evaluated[evaluated['collapsed'] == True].copy()
        
        if len(working) > 0:
            working_path = output_dir / 'working_models.csv'
            working.to_csv(working_path, index=False)
            print(f" Saved working_models.csv ({len(working)} models) to {working_path}")
        
        if len(collapsed) > 0:
            collapsed_path = output_dir / 'collapsed_models.csv'
            collapsed.to_csv(collapsed_path, index=False)
            print(f" Saved collapsed_models.csv ({len(collapsed)} models) to {collapsed_path}")
    
    # Print quick stats
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total experiments: {len(df)}")
    
    if 'phase' in df.columns and df['phase'].notna().any():
        print(f"\nBy phase:")
        for phase, count in df['phase'].value_counts().sort_index().items():
            print(f"  {phase}: {count}")
    
    print(f"\nBy group:")
    for group, count in df['group'].value_counts().items():
        print(f"  {group}: {count}")
    
    if 'evaluated' in df.columns:
        evaluated = df['evaluated'].sum()
        print(f"\nEvaluated: {evaluated}/{len(df)}")
    
    if 'collapsed' in df.columns and df['collapsed'].notna().any():
        collapsed_count = df[df['collapsed'] == True].shape[0]
        print(f"Collapsed: {collapsed_count}/{len(df)}")
    
    # Best models by different metrics
    evaluated_df = df[df['evaluated'] == True]
    if len(evaluated_df) > 0:
        print(f"\n{'='*80}")
        print("BEST MODELS BY METRIC")
        print(f"{'='*80}")
        
        if 'sharpe_ratio' in evaluated_df.columns:
            best = evaluated_df.nlargest(3, 'sharpe_ratio')
            print(f"\nTop 3 by Sharpe Ratio:")
            for _, row in best.iterrows():
                dir_acc = f"{row['dir_acc']:.2%}" if 'dir_acc' in row else 'N/A'
                print(f"  {row['experiment_name']}: Sharpe={row['sharpe_ratio']:.4f}, Acc={dir_acc}")
        
        if 'auc_roc' in evaluated_df.columns:
            best = evaluated_df.nlargest(3, 'auc_roc')
            print(f"\nTop 3 by AUC-ROC:")
            for _, row in best.iterrows():
                dir_acc = f"{row['dir_acc']:.2%}" if 'dir_acc' in row else 'N/A'
                print(f"  {row['experiment_name']}: AUC={row['auc_roc']:.4f}, Acc={dir_acc}")
        
        if 'composite_score' in evaluated_df.columns:
            best = evaluated_df.nlargest(3, 'composite_score')
            print(f"\nTop 3 by Composite Score:")
            for _, row in best.iterrows():
                print(f"  {row['experiment_name']}: Score={row['composite_score']:.4f}")
    
    # Print working vs collapsed analysis
    print_working_vs_collapsed_summary(df)
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
