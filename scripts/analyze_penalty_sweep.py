"""
Analyze results from penalty threshold sweep experiments.

Compares models with different regularization penalties on:
1. Collapse metrics (prediction diversity, directional bias)
2. Validation performance  
3. Financial metrics (Sharpe ratio, directional accuracy)

Usage:
    python analyze_penalty_sweep.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

def parse_penalty_params(exp_name):
    """Extract penalty parameters from experiment name."""
    params = {
        'collapse_weight': 0.0,
        'collapse_threshold': None,
        'directional_weight': 0.0,
        'directional_threshold': 0.90  # Hardcoded in run_penalty_sweep.sh
    }
    
    # Parse variance threshold (e.g., penalty_sweep_variance_thresh0.035)
    thresh_match = re.search(r'thresh([\d.]+)', exp_name)
    if thresh_match:
        params['collapse_threshold'] = float(thresh_match.group(1))
    
    # Parse directional weight (e.g., penalty_sweep_directional_w0.2)
    dir_match = re.search(r'_w([\d.]+)', exp_name)
    if dir_match:
        params['directional_weight'] = float(dir_match.group(1))
    
    # Parse combined experiments (e.g., penalty_sweep_combined_dirw0.1)
    combined_match = re.search(r'dirw([\d.]+)', exp_name)
    if combined_match:
        params['directional_weight'] = float(combined_match.group(1))
    
    return params

def load_experiment_results(exp_name):
    """Load metrics from an experiment."""
    exp_dir = Path('experiments/04_custom_losses') / exp_name
    
    # Load config
    config_path = exp_dir / 'config.json'
    if not config_path.exists():
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Load evaluation metrics
    eval_path = exp_dir / 'evaluation' / 'evaluation_metrics.json'
    if not eval_path.exists():
        return None
    
    with open(eval_path) as f:
        metrics = json.load(f)
    
    # Get penalty parameters from config (preferred) or experiment name (fallback)
    loss_config = config.get('loss', {})
    collapse_weight = loss_config.get('dist_loss_std_weight', 0.0)
    collapse_threshold = loss_config.get('collapse_threshold', None)
    directional_weight = loss_config.get('directional_weight', 0.0)
    directional_threshold = loss_config.get('directional_threshold', 0.90)
    temporal_weight = loss_config.get('temporal_consistency_weight', 0.0)
    
    # Fallback: parse from experiment name if not in config
    if collapse_threshold is None or directional_weight == 0.0:
        penalty_params = parse_penalty_params(exp_name)
        if collapse_threshold is None:
            collapse_threshold = penalty_params['collapse_threshold']
        if directional_weight == 0.0 and penalty_params['directional_weight'] > 0:
            directional_weight = penalty_params['directional_weight']
    
    # Build result dict
    result = {
        'experiment': exp_name,
        # Penalty parameters
        'collapse_weight': collapse_weight,
        'collapse_threshold': collapse_threshold,
        'directional_weight': directional_weight,
        'directional_threshold': directional_threshold,
        'temporal_weight': temporal_weight,
    }
    
    # Add statistical metrics
    if 'statistical_metrics' in metrics:
        for key, val in metrics['statistical_metrics'].items():
            result[f'stat_{key}'] = val
    
    # Add financial metrics
    if 'financial_metrics' in metrics:
        for key, val in metrics['financial_metrics'].items():
            if key != 'confusion_matrix':  # Skip confusion matrix (it's nested)
                result[f'fin_{key}'] = val
    
    # Add mode stats (collapse detection)
    if 'mode_stats' in metrics:
        for key, val in metrics['mode_stats'].items():
            result[key] = val
    
    # Add top-level flags
    result['collapse_detected'] = metrics.get('collapse_detected', False)
    result['degradation_detected'] = metrics.get('degradation_detected', False)
    
    return result


def main():
    print("="*80)
    print("PENALTY SWEEP ANALYSIS")
    print("="*80)
    print()
    
    # Find all Phase 4 experiments (new h16d* naming scheme + old penalty_* naming)
    exp_dir = Path('experiments/04_custom_losses')
    sweep_experiments = sorted([d.name for d in exp_dir.iterdir() 
                                if d.is_dir() and (d.name.startswith('h16d') or 
                                                   d.name.startswith('penalty_sweep_') or 
                                                   d.name.startswith('penalty_refined_') or
                                                   d.name.startswith('magnitude_test_'))])
    
    if not sweep_experiments:
        print("No Phase 4 experiments found in experiments/04_custom_losses/")
        return
    
    print(f"Found {len(sweep_experiments)} experiments:")
    for exp in sweep_experiments:
        print(f"  - {exp}")
    print()
    
    # Load all results
    results = []
    failed = []
    for exp_name in sweep_experiments:
        result = load_experiment_results(exp_name)
        if result:
            results.append(result)
        else:
            failed.append(exp_name)
    
    if failed:
        print(f"Warning: Failed to load {len(failed)} experiments:")
        for exp in failed:
            print(f"  - {exp}")
        print()
    
    if not results:
        print("No valid results found.")
        return
    
    df = pd.DataFrame(results)
    
    print(f"Successfully loaded {len(results)} experiments")
    print()
    
    # ========================================================================
    # Analysis 1: Impact on Collapse Metrics
    # ========================================================================
    print("="*80)
    print("COLLAPSE PREVENTION EFFECTIVENESS")
    print("="*80)
    print()
    
    collapse_cols = ['experiment', 'collapse_weight', 'collapse_threshold', 
                     'directional_weight', 'healthy_pct', 'degraded_pct', 
                     'unidirectional_pct', 'weak_collapse_pct', 'strong_collapse_pct']
    
    if all(col in df.columns for col in ['healthy_pct', 'unidirectional_pct']):
        collapse_df = df[collapse_cols].copy()
        collapse_df = collapse_df.sort_values('healthy_pct', ascending=False)
        
        print("Ranked by healthy prediction percentage:")
        print(collapse_df.to_string(index=False))
        print()
        
        # Show distribution of modes
        print("Collapse mode distribution:")
        mode_cols = ['healthy_pct', 'degraded_pct', 'unidirectional_pct', 
                     'weak_collapse_pct', 'strong_collapse_pct']
        print(df[['experiment'] + mode_cols].to_string(index=False))
        print()
    else:
        print("Collapse metrics not available in results.")
        print()
    
    # ========================================================================
    # Analysis 2: Validation Performance
    # ========================================================================
    print("="*80)
    print("STATISTICAL PERFORMANCE")
    print("="*80)
    print()
    
    val_cols = ['experiment', 'collapse_weight', 'directional_weight', 
                'stat_mse', 'stat_mae', 'stat_r2']
    
    if 'stat_mse' in df.columns:
        val_df = df[val_cols].copy()
        val_df = val_df.sort_values('stat_mse')
        
        print("Ranked by MSE (lower is better):")
        print(val_df.to_string(index=False))
        print()
    else:
        print("Statistical metrics not available.")
        print()
    
    # ========================================================================
    # Analysis 3: Financial Metrics
    # ========================================================================
    print("="*80)
    print("FINANCIAL PERFORMANCE")
    print("="*80)
    print()
    
    fin_cols = ['experiment', 'collapse_weight', 'directional_weight',
                'fin_sharpe_ratio', 'fin_directional_accuracy', 'fin_total_return', 'fin_max_drawdown']
    
    if 'fin_sharpe_ratio' in df.columns:
        fin_df = df[fin_cols].copy()
        fin_df = fin_df.sort_values('fin_sharpe_ratio', ascending=False)
        
        print("Ranked by Sharpe ratio (higher is better):")
        print(fin_df.to_string(index=False))
        print()
    else:
        print("Financial metrics not available.")
        print()
    
    # ========================================================================
    # Analysis 4: Trade-offs
    # ========================================================================
    print("="*80)
    print("PENALTY TRADE-OFFS")
    print("="*80)
    print()
    
    baseline = df[df['experiment'].str.endswith('_baseline') | (df['experiment'] == 'penalty_sweep_baseline')]
    if len(baseline) > 0:
        # If multiple baselines, use h16d015_baseline as reference (matches Phase 4 experiments)
        if len(baseline) > 1:
            h16d015_baseline = df[df['experiment'] == 'h16d015_baseline']
            if len(h16d015_baseline) > 0:
                baseline = h16d015_baseline.iloc[0]
            else:
                baseline = baseline.iloc[0]
        else:
            baseline = baseline.iloc[0]
        
        print("Baseline performance:")
        healthy_pct = baseline.get('healthy_pct')
        mse = baseline.get('stat_mse')
        sharpe = baseline.get('fin_sharpe_ratio')
        
        print(f"  Healthy %: {healthy_pct:.1f}%" if pd.notna(healthy_pct) else "  Healthy %: N/A")
        print(f"  MSE: {mse:.4f}" if pd.notna(mse) else "  MSE: N/A")
        print(f"  Sharpe: {sharpe:.4f}" if pd.notna(sharpe) else "  Sharpe: N/A")
        print()
        
        print("Improvements over baseline:")
        for _, row in df.iterrows():
            if row['experiment'] == 'penalty_sweep_baseline':
                continue
            
            # Calculate deltas, handling NaN values
            healthy_base = baseline.get('healthy_pct', 0) if pd.notna(baseline.get('healthy_pct')) else 0
            healthy_curr = row.get('healthy_pct', 0) if pd.notna(row.get('healthy_pct')) else 0
            healthy_delta = healthy_curr - healthy_base
            
            mse_base = baseline.get('stat_mse', 0) if pd.notna(baseline.get('stat_mse')) else 0
            mse_curr = row.get('stat_mse', 0) if pd.notna(row.get('stat_mse')) else 0
            mse_delta = mse_curr - mse_base
            
            sharpe_base = baseline.get('fin_sharpe_ratio', 0) if pd.notna(baseline.get('fin_sharpe_ratio')) else 0
            sharpe_curr = row.get('fin_sharpe_ratio', 0) if pd.notna(row.get('fin_sharpe_ratio')) else 0
            sharpe_delta = sharpe_curr - sharpe_base
            
            print(f"{row['experiment']}:")
            print(f"  Collapse weight: {row['collapse_weight']}, threshold: {row['collapse_threshold']}")
            print(f"  Directional weight: {row['directional_weight']}")
            print(f"  Healthy %: {healthy_delta:+.1f}pp")
            print(f"  MSE: {mse_delta:+.4f} ({'better' if mse_delta < 0 else 'worse'})")
            print(f"  Sharpe: {sharpe_delta:+.4f} ({'better' if sharpe_delta > 0 else 'worse'})")
            print()
    else:
        print("Baseline experiment not found.")
    
    # ========================================================================
    # Recommendations
    # ========================================================================
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    
    if 'healthy_pct' in df.columns and 'stat_mse' in df.columns:
        # Find best by different criteria
        best_healthy = df.loc[df['healthy_pct'].idxmax()]
        best_mse = df.loc[df['stat_mse'].idxmin()]
        
        print("Best configurations:")
        print(f"\n  Most healthy predictions: {best_healthy['experiment']}")
        print(f"    Collapse: weight={best_healthy['collapse_weight']:.3f}, threshold={best_healthy['collapse_threshold']}")
        print(f"    Directional: weight={best_healthy['directional_weight']:.3f}")
        print(f"    Healthy %: {best_healthy['healthy_pct']:.1f}%")
        
        print(f"\n  Best MSE: {best_mse['experiment']}")
        print(f"    MSE: {best_mse['stat_mse']:.4f}")
        print(f"    Healthy %: {best_mse['healthy_pct']:.1f}%")
        
        if 'fin_sharpe_ratio' in df.columns:
            best_sharpe = df.loc[df['fin_sharpe_ratio'].idxmax()]
            print(f"\n  Best Sharpe ratio: {best_sharpe['experiment']}")
            print(f"    Sharpe: {best_sharpe['fin_sharpe_ratio']:.4f}")
            print(f"    Healthy %: {best_sharpe['healthy_pct']:.1f}%")
        
        # Look for sweet spot (good on all metrics)
        # Normalize metrics (handling NaN)
        df_norm = df.copy()
        if 'healthy_pct' in df.columns and df['healthy_pct'].max() > 0:
            df_norm['healthy_norm'] = df['healthy_pct'] / df['healthy_pct'].max()
        else:
            df_norm['healthy_norm'] = 0
            
        if 'stat_mse' in df.columns and df['stat_mse'].max() > 0:
            df_norm['mse_norm'] = 1 - (df['stat_mse'] / df['stat_mse'].max())
        else:
            df_norm['mse_norm'] = 0
            
        if 'fin_sharpe_ratio' in df.columns and df['fin_sharpe_ratio'].max() > 0:
            df_norm['sharpe_norm'] = df['fin_sharpe_ratio'] / df['fin_sharpe_ratio'].max()
        else:
            df_norm['sharpe_norm'] = 0
        
        df_norm['composite_score'] = (
            df_norm['healthy_norm'] * 0.4 +
            df_norm['mse_norm'] * 0.3 +
            df_norm['sharpe_norm'] * 0.3
        )
        
        best_overall = df_norm.loc[df_norm['composite_score'].idxmax()]
        
        print(f"\n  Best overall (composite score): {best_overall['experiment']}")
        print(f"    Collapse: weight={best_overall['collapse_weight']:.3f}, threshold={best_overall['collapse_threshold']}")
        print(f"    Directional: weight={best_overall['directional_weight']:.3f}")
        print(f"    Healthy %: {best_overall['healthy_pct']:.1f}%")
        print(f"    MSE: {best_overall['stat_mse']:.4f}")
        if 'fin_sharpe_ratio' in df.columns:
            print(f"    Sharpe: {best_overall['fin_sharpe_ratio']:.4f}")
        print()
    
    print("="*80)
    print("Analysis complete. Results saved to experiments/04_custom_losses/penalty_sweep_*/")
    print("="*80)


if __name__ == "__main__":
    main()