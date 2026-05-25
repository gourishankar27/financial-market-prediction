#!/usr/bin/env python3
"""
Analyze VIX distribution across train/val/test splits.
Shows raw VIX statistics and regime breakdown per split.

Usage:
    python analyze_vix_distribution.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_vix_split(split_name: str, data_dir: Path = Path('data/splits/vintage')):
    """Analyze VIX distribution for a single split."""
    filepath = data_dir / f"core_proposal_daily_vintage_{split_name}.csv"
    
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None
    
    df = pd.read_csv(filepath)
    vix = df['VIX'].values
    
    # Basic statistics
    stats = {
        'split': split_name,
        'count': len(vix),
        'mean': np.mean(vix),
        'median': np.median(vix),
        'std': np.std(vix),
        'min': np.min(vix),
        'max': np.max(vix),
        'p05': np.percentile(vix, 5),
        'p25': np.percentile(vix, 25),
        'p75': np.percentile(vix, 75),
        'p95': np.percentile(vix, 95),
    }
    
    # Regime breakdown
    low_vol = vix < 15
    med_vol = (vix >= 15) & (vix < 25)
    high_vol = vix >= 25
    extreme_vol = vix >= 40
    
    stats['low_vol_days'] = np.sum(low_vol)
    stats['low_vol_pct'] = np.mean(low_vol) * 100
    stats['med_vol_days'] = np.sum(med_vol)
    stats['med_vol_pct'] = np.mean(med_vol) * 100
    stats['high_vol_days'] = np.sum(high_vol)
    stats['high_vol_pct'] = np.mean(high_vol) * 100
    stats['extreme_vol_days'] = np.sum(extreme_vol)
    stats['extreme_vol_pct'] = np.mean(extreme_vol) * 100
    
    return stats


def print_vix_table(stats_list):
    """Print formatted table of VIX statistics."""
    print("\n" + "="*80)
    print("VIX DISTRIBUTION ACROSS SPLITS")
    print("="*80)
    
    # Header
    print(f"{'Metric':<25}", end='')
    for stats in stats_list:
        print(f"{stats['split'].upper():>15}", end='')
    print()
    print("-"*80)
    
    # Basic statistics
    metrics = [
        ('count', 'Count', '12.0f'),
        ('mean', 'Mean', '12.2f'),
        ('median', 'Median', '12.2f'),
        ('std', 'Std Dev', '12.2f'),
        ('min', 'Min', '12.2f'),
        ('max', 'Max', '12.2f'),
        ('p05', '5th percentile', '12.2f'),
        ('p25', '25th percentile', '12.2f'),
        ('p75', '75th percentile', '12.2f'),
        ('p95', '95th percentile', '12.2f'),
    ]
    
    for key, label, fmt in metrics:
        print(f"{label:<25}", end='')
        for stats in stats_list:
            print(f"{stats[key]:{fmt}}   ", end='')
        print()
    
    print()
    print("-"*80)
    print("REGIME BREAKDOWN")
    print("-"*80)
    
    # Regime statistics
    regime_metrics = [
        ('low_vol_days', 'Low vol days (VIX<15)', '12.0f'),
        ('low_vol_pct', 'Low vol %', '12.1f'),
        ('med_vol_days', 'Medium vol days (15-25)', '12.0f'),
        ('med_vol_pct', 'Medium vol %', '12.1f'),
        ('high_vol_days', 'High vol days (VIX≥25)', '12.0f'),
        ('high_vol_pct', 'High vol %', '12.1f'),
        ('extreme_vol_days', 'Extreme days (VIX≥40)', '12.0f'),
        ('extreme_vol_pct', 'Extreme %', '12.1f'),
    ]
    
    for key, label, fmt in regime_metrics:
        print(f"{label:<25}", end='')
        for stats in stats_list:
            print(f"{stats[key]:{fmt}}   ", end='')
        print()
    
    print("="*80 + "\n")


def main():
    splits = ['train', 'val', 'test']
    stats_list = []
    
    print("\nAnalyzing VIX distributions...")
    
    for split in splits:
        stats = analyze_vix_split(split)
        if stats:
            stats_list.append(stats)
    
    if not stats_list:
        print("Error: No data files found")
        return
    
    print_vix_table(stats_list)
    
    # Additional insights
    print("\nKEY INSIGHTS:")
    print("-" * 80)
    
    for stats in stats_list:
        split = stats['split']
        print(f"\n{split.upper()}:")
        print(f"  • VIX range: {stats['min']:.1f} - {stats['max']:.1f}")
        print(f"  • Most time in: ", end='')
        
        if stats['low_vol_pct'] > stats['med_vol_pct'] and stats['low_vol_pct'] > stats['high_vol_pct']:
            print(f"low volatility ({stats['low_vol_pct']:.1f}%)")
        elif stats['med_vol_pct'] > stats['low_vol_pct'] and stats['med_vol_pct'] > stats['high_vol_pct']:
            print(f"medium volatility ({stats['med_vol_pct']:.1f}%)")
        else:
            print(f"high volatility ({stats['high_vol_pct']:.1f}%)")
        
        if stats['extreme_vol_days'] > 0:
            print(f"  • Extreme volatility (VIX≥40): {stats['extreme_vol_days']} days ({stats['extreme_vol_pct']:.1f}%)")
        
        # Regime balance
        regime_entropy = -(
            (stats['low_vol_pct']/100) * np.log(stats['low_vol_pct']/100 + 1e-10) +
            (stats['med_vol_pct']/100) * np.log(stats['med_vol_pct']/100 + 1e-10) +
            (stats['high_vol_pct']/100) * np.log(stats['high_vol_pct']/100 + 1e-10)
        ) / np.log(3)  # Normalize by max entropy
        
        print(f"  • Regime balance: {regime_entropy:.2f} (1.0 = perfectly balanced)")
    
    print()


if __name__ == '__main__':
    main()
