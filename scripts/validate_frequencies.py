#!/usr/bin/env python3
"""
Validate and compare daily/weekly/monthly datasets.

Checks:
1. Basic integrity (shapes, date ranges, missing values)
2. Return distribution characteristics
3. Signal-to-noise ratio comparison
4. Feature correlation stability across frequencies

Usage:
    python validate_frequencies.py --data-dir data --version fixed
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats


def load_datasets(data_dir, version, enhanced=False):
    """Load all frequency datasets."""
    datasets = {}
    
    for freq in ['daily', 'weekly', 'monthly']:
        if enhanced:
            path = f"{data_dir}/financial_dataset_{freq}_{version}_enhanced.csv"
        else:
            path = f"{data_dir}/financial_dataset_{freq}_{version}.csv"
        
        try:
            df = pd.read_csv(path, index_col='Date', parse_dates=True)
            datasets[freq] = df
            print(f"✓ Loaded {freq}: {df.shape}")
        except FileNotFoundError:
            print(f"✗ Not found: {path}")
            
    return datasets


def basic_integrity(datasets):
    """Check basic dataset properties."""
    print("\n" + "="*70)
    print("BASIC INTEGRITY")
    print("="*70)
    
    for freq, df in datasets.items():
        print(f"\n{freq.upper()}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        
        # Check for duplicate indices
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            print(f"  ⚠ Duplicate dates: {dupes}")
        else:
            print(f"  ✓ No duplicate dates")
            
        # Check monotonic index
        if df.index.is_monotonic_increasing:
            print(f"  ✓ Dates monotonically increasing")
        else:
            print(f"  ⚠ Dates NOT monotonically increasing")


def return_characteristics(datasets):
    """Analyze return distributions across frequencies."""
    print("\n" + "="*70)
    print("RETURN CHARACTERISTICS")
    print("="*70)
    
    stats_table = []
    
    for freq, df in datasets.items():
        if 'SP500_Returns' not in df.columns:
            continue
            
        returns = df['SP500_Returns'].dropna()
        
        row = {
            'Frequency': freq,
            'N': len(returns),
            'Mean (%)': returns.mean(),
            'Std (%)': returns.std(),
            'Min (%)': returns.min(),
            'Max (%)': returns.max(),
            'Skew': returns.skew(),
            'Kurtosis': returns.kurtosis(),
        }
        stats_table.append(row)
    
    stats_df = pd.DataFrame(stats_table)
    print("\n" + stats_df.to_string(index=False))
    
    return stats_df


def signal_to_noise_analysis(datasets):
    """
    Compare signal-to-noise characteristics.
    
    SNR here = |mean| / std (absolute value of mean return over volatility)
    Also compute annualized versions for fair comparison.
    """
    print("\n" + "="*70)
    print("SIGNAL-TO-NOISE ANALYSIS")
    print("="*70)
    
    annualization = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
    }
    
    results = []
    
    for freq, df in datasets.items():
        if 'SP500_Returns' not in df.columns:
            continue
            
        returns = df['SP500_Returns'].dropna()
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # Raw SNR (per-period)
        snr_raw = abs(mean_ret) / std_ret if std_ret > 0 else 0
        
        # Annualized
        ann_factor = annualization[freq]
        ann_mean = mean_ret * ann_factor
        ann_std = std_ret * np.sqrt(ann_factor)
        snr_annualized = abs(ann_mean) / ann_std if ann_std > 0 else 0
        
        # Noise-to-signal (more intuitive for how hard prediction is)
        nsr = std_ret / abs(mean_ret) if mean_ret != 0 else float('inf')
        
        results.append({
            'Frequency': freq,
            'Mean/Period (%)': mean_ret,
            'Std/Period (%)': std_ret,
            'SNR (raw)': snr_raw,
            'Ann. Mean (%)': ann_mean,
            'Ann. Std (%)': ann_std,
            'SNR (ann.)': snr_annualized,
            'Noise:Signal': f"{nsr:.1f}:1" if nsr != float('inf') else "∞:1",
        })
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "-"*70)
    print("Interpretation:")
    print("-"*70)
    print("  SNR (raw): Higher = easier prediction target (for that period length)")
    print("  SNR (ann.): Normalized for comparison across frequencies")
    print("  Noise:Signal: How many units of noise per unit of signal")
    print("                Daily ~30:1 means noise is 30x larger than expected return")
    
    return results_df


def correlation_stability(datasets):
    """Check if feature correlations are stable across frequencies."""
    print("\n" + "="*70)
    print("FEATURE CORRELATION WITH TARGET")
    print("="*70)
    
    # Common features across all datasets
    common_features = None
    for df in datasets.values():
        if common_features is None:
            common_features = set(df.columns)
        else:
            common_features &= set(df.columns)
    
    common_features = sorted([f for f in common_features if f != 'SP500_Returns'])
    
    print(f"\nCommon features: {common_features}")
    
    corr_table = []
    
    for feature in common_features:
        row = {'Feature': feature}
        for freq, df in datasets.items():
            if feature in df.columns and 'SP500_Returns' in df.columns:
                corr = df[feature].corr(df['SP500_Returns'])
                row[freq] = corr
        corr_table.append(row)
    
    corr_df = pd.DataFrame(corr_table)
    print("\n" + corr_df.to_string(index=False))
    
    # Check for sign flips or major changes
    print("\n" + "-"*70)
    print("Correlation Stability Check:")
    print("-"*70)
    
    for _, row in corr_df.iterrows():
        feature = row['Feature']
        vals = [row.get(f) for f in ['daily', 'weekly', 'monthly'] if f in row and pd.notna(row.get(f))]
        
        if len(vals) >= 2:
            signs = [np.sign(v) for v in vals]
            if len(set(signs)) > 1 and 0 not in signs:
                print(f"  ⚠ {feature}: Sign flip across frequencies")
            
            spread = max(vals) - min(vals)
            if spread > 0.3:
                print(f"  ⚠ {feature}: Large correlation spread ({spread:.2f})")
    
    return corr_df


def temporal_coverage(datasets):
    """Verify temporal alignment across frequencies."""
    print("\n" + "="*70)
    print("TEMPORAL COVERAGE")
    print("="*70)
    
    for freq, df in datasets.items():
        print(f"\n{freq.upper()}:")
        print(f"  First date: {df.index[0]}")
        print(f"  Last date:  {df.index[-1]}")
        print(f"  Total periods: {len(df)}")
        
        # Expected vs actual
        date_range = (df.index[-1] - df.index[0]).days
        years = date_range / 365.25
        
        expected = {
            'daily': years * 252,
            'weekly': years * 52,
            'monthly': years * 12,
        }
        
        pct_coverage = len(df) / expected[freq] * 100 if expected[freq] > 0 else 0
        print(f"  Coverage: {pct_coverage:.1f}% of expected periods")


def validate_resampling_consistency(datasets):
    """
    Verify weekly/monthly returns are consistent with daily.
    Sum of daily returns should approximately equal period return.
    """
    print("\n" + "="*70)
    print("RESAMPLING CONSISTENCY CHECK")
    print("="*70)
    
    if 'daily' not in datasets:
        print("Need daily data for this check")
        return
    
    daily = datasets['daily']
    
    for freq in ['weekly', 'monthly']:
        if freq not in datasets:
            continue
            
        resampled = datasets[freq]
        
        # Resample daily returns by summing (approximate for small returns)
        freq_code = 'W-FRI' if freq == 'weekly' else 'M'
        daily_resampled = daily['SP500_Returns'].resample(freq_code).sum()
        
        # Compare with actual resampled returns
        # Align indices
        common_idx = resampled.index.intersection(daily_resampled.index)
        
        actual = resampled.loc[common_idx, 'SP500_Returns']
        summed = daily_resampled.loc[common_idx]
        
        # Correlation between the two methods
        corr = actual.corr(summed)
        
        # Mean absolute difference
        mad = (actual - summed).abs().mean()
        
        print(f"\n{freq.upper()} vs summed daily:")
        print(f"  Correlation: {corr:.4f}")
        print(f"  Mean abs diff: {mad:.4f}%")
        
        if corr < 0.99:
            print(f"  ⚠ Returns computed from close prices, not summed - this is expected")
        else:
            print(f"  ✓ High consistency")


def main():
    parser = argparse.ArgumentParser(description='Validate multi-frequency datasets')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing datasets')
    parser.add_argument('--version', type=str, default='fixed',
                        choices=['fixed', 'vintage'],
                        help='Data version to validate')
    parser.add_argument('--enhanced', action='store_true',
                        help='Analyze enhanced dataset (daily only, shows all features)')
    args = parser.parse_args()
    
    print("="*70)
    print("MULTI-FREQUENCY DATA VALIDATION")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Version: {args.version}")
    if args.enhanced:
        print(f"Mode: Enhanced dataset analysis (daily only)")
    
    # Load datasets
    datasets = load_datasets(args.data_dir, args.version, enhanced=args.enhanced)
    
    if not datasets:
        print("\nNo datasets found!")
        return
    
    # Run all checks
    basic_integrity(datasets)
    stats_df = return_characteristics(datasets)
    snr_df = signal_to_noise_analysis(datasets)
    corr_df = correlation_stability(datasets)
    temporal_coverage(datasets)
    
    if not args.enhanced:
        validate_resampling_consistency(datasets)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
