"""
Loaders for experiment data.

Usage:
    from lib.loaders import load_experiments
    df = load_experiments('experiments_comparison.csv')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any


def load_experiments(
    csv_path: str,
    filter_evaluated: bool = True,
    filter_regime: Optional[bool] = None
) -> pd.DataFrame:
    """
    Load experiment comparison CSV with type cleanup.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV from aggregate_experiments.py
    filter_evaluated : bool, default=True
        Only include experiments with evaluation results
    filter_regime : bool, optional
        If True, only regime experiments. If False, only non-regime.
        If None, include all.
    
    Returns
    -------
    pd.DataFrame
        Cleaned experiment data
    """
    df = pd.read_csv(csv_path)
    
    # Type conversions
    bool_cols = ['evaluated', 'regime_enabled', 'hard_routing_train', 
                 'freeze_backbone', 'load_checkpoint', 'early_stopped']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    
    # Filters
    if filter_evaluated:
        df = df[df['evaluated'] == True].copy()
    
    if filter_regime is True:
        df = df[df['regime_enabled'] == True].copy()
    elif filter_regime is False:
        df = df[df['regime_enabled'] == False].copy()
    
    # Add derived columns
    df = _add_derived_columns(df)
    
    return df.reset_index(drop=True)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived columns."""
    
    # Expert type as string
    if 'expert_hidden_size' in df.columns:
        df['expert_type'] = df['expert_hidden_size'].apply(
            lambda x: 'linear' if pd.isna(x) or x == 0 else f'mlp_{int(x)}'
        )
    
    # Simplified routing label
    if 'routing_strategy' in df.columns and 'num_regimes' in df.columns:
        df['routing_label'] = df.apply(_make_routing_label, axis=1)
    
    # Collapse status - use evaluation metrics if available, else derive from pred_std
    if 'strong_collapse_pct' in df.columns:
        # Has collapse detected - flag experiments with significant collapse
        df['has_strong_collapse'] = df['strong_collapse_pct'] > 5  # >5% of days
        df['has_weak_collapse'] = df['weak_collapse_pct'] > 5
        df['has_any_collapse'] = (df['strong_collapse_pct'] + df['weak_collapse_pct']) > 5
        df['mostly_healthy'] = df['healthy_pct'] > 50
    elif 'pred_std' in df.columns:
        # Fallback to pred_std thresholds
        df['has_strong_collapse'] = df['pred_std'] < 0.01
        df['has_weak_collapse'] = (df['pred_std'] >= 0.01) & (df['pred_std'] < 0.02)
        df['has_any_collapse'] = df['pred_std'] < 0.02
        df['mostly_healthy'] = df['pred_std'] >= 0.05
    
    # Problematic percentage (inverse of healthy)
    if 'healthy_pct' in df.columns:
        df['problematic_pct'] = 100 - df['healthy_pct']
    
    return df


def _make_routing_label(row) -> str:
    """Create human-readable routing label."""
    if pd.isna(row.get('routing_strategy')):
        return 'baseline'
    
    strategy = row['routing_strategy']
    n_regimes = int(row['num_regimes']) if pd.notna(row.get('num_regimes')) else 2
    
    if strategy == 'learned':
        return f'learned_{n_regimes}r'
    elif strategy == 'vix_threshold':
        threshold = row.get('vix_threshold', '')
        hr = '_hr' if row.get('hard_routing_train', False) else ''
        return f'vix_{n_regimes}r_t{int(threshold) if pd.notna(threshold) else "?"}{hr}'
    else:
        return strategy


def get_regime_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only regime-enabled experiments."""
    return df[df['regime_enabled'] == True].copy()


def get_config_groups(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Group experiments by configuration dimensions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Experiment data
    group_by : list of str
        Column names to group by
        
    Returns
    -------
    pd.DataFrame
        Grouped summary with counts and mean metrics
    """
    # Filter to valid columns
    valid_cols = [c for c in group_by if c in df.columns]
    if not valid_cols:
        raise ValueError(f"No valid columns in {group_by}")
    
    # Metric columns to aggregate
    metric_cols = [
        'directional_accuracy', 'sharpe_ratio', 'healthy_pct', 
        'pred_std', 'final_expert_weight_cosine', 'best_val_loss'
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    
    # Group and aggregate
    agg_dict = {col: ['mean', 'std', 'count'] for col in metric_cols}
    grouped = df.groupby(valid_cols, dropna=False).agg(agg_dict)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
    
    return grouped.reset_index()


def parse_experiment_name(name: str) -> Dict[str, Any]:
    """
    Parse experiment name into components.
    
    This is a fallback - prefer config.json values from the CSV.
    
    Examples:
        'learn2r_d025_lb1_mlp' -> {
            'routing_strategy': 'learned',
            'num_regimes': 2,
            'dropout': 0.25,
            'load_balance_weight': 1.0,
            'expert_type': 'mlp'
        }
    """
    parts = name.split('_')
    result = {}
    
    for part in parts:
        # Routing strategy and regimes
        if part.startswith('learn') and 'r' in part:
            result['routing_strategy'] = 'learned'
            result['num_regimes'] = int(part.replace('learn', '').replace('r', ''))
        elif part.startswith('vix') and 'r' in part:
            result['routing_strategy'] = 'vix_threshold'
            result['num_regimes'] = int(part.replace('vix', '').replace('r', ''))
        
        # Dropout
        elif part.startswith('d') and part[1:].isdigit():
            result['dropout'] = int(part[1:]) / 100
        
        # Load balance weight
        elif part.startswith('lb') and part[2:].isdigit():
            result['load_balance_weight'] = float(part[2:])
        
        # VIX threshold (single value like t15, t20)
        elif part.startswith('t') and part[1:].isdigit():
            result['vix_threshold'] = int(part[1:])
        
        # Expert type
        elif part == 'mlp':
            result['expert_type'] = 'mlp'
        
        # Hard routing
        elif part == 'hr':
            result['hard_routing_train'] = True
    
    # Defaults
    result.setdefault('expert_type', 'linear')
    result.setdefault('hard_routing_train', False)
    
    return result