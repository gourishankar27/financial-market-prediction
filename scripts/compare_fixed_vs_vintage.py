#!/usr/bin/env python3
"""
Compare fixed-shift vs vintage-aligned datasets to validate vintage implementation.

Expected differences:
- CPI, Unemployment, Industrial_Production, Fed_Rate, Consumer_Sentiment values 
  should differ slightly due to vintage alignment
- All other columns should be identical (market data, derived features)
"""

import pandas as pd
import numpy as np
import sys

def compare_datasets(fixed_path, vintage_path):
    print("="*70)
    print("FIXED vs VINTAGE DATASET COMPARISON")
    print("="*70)
    
    # Load datasets
    df_fixed = pd.read_csv(fixed_path, index_col=0, parse_dates=True)
    df_vintage = pd.read_csv(vintage_path, index_col=0, parse_dates=True)
    
    print(f"\nFixed dataset:   {df_fixed.shape}")
    print(f"Vintage dataset: {df_vintage.shape}")
    
    # Check if shapes match
    if df_fixed.shape != df_vintage.shape:
        print("\n⚠ WARNING: Dataset shapes differ!")
        print("This could indicate different END_DATE usage or data availability")
        
        # Find date range differences
        fixed_dates = set(df_fixed.index)
        vintage_dates = set(df_vintage.index)
        only_fixed = fixed_dates - vintage_dates
        only_vintage = vintage_dates - fixed_dates
        
        if only_fixed:
            print(f"\nDates only in fixed: {len(only_fixed)}")
            print(f"  First: {min(only_fixed)}")
            print(f"  Last: {max(only_fixed)}")
        
        if only_vintage:
            print(f"\nDates only in vintage: {len(only_vintage)}")
            print(f"  First: {min(only_vintage)}")
            print(f"  Last: {max(only_vintage)}")
        
        # Use common date range for comparison
        common_dates = fixed_dates & vintage_dates
        print(f"\nComparing {len(common_dates)} common dates")
        df_fixed = df_fixed.loc[df_fixed.index.isin(common_dates)].sort_index()
        df_vintage = df_vintage.loc[df_vintage.index.isin(common_dates)].sort_index()
    
    # Columns that SHOULD differ (vintage-aligned macro features)
    vintage_aligned_cols = ['CPI', 'Unemployment', 'Industrial_Production', 
                           'Fed_Rate', 'Consumer_Sentiment']
    
    # Columns that should be IDENTICAL (market data, derived features)
    market_cols = ['SP500_Close', 'SP500_Volume', 'SP500_Returns', 'VIX', 
                   'Treasury_10Y', 'Yield_Spread', 'Wilshire5000']
    
    derived_cols = ['Inflation_YoY', 'SP500_Volatility']
    
    print("\n" + "="*70)
    print("COLUMN-BY-COLUMN ANALYSIS")
    print("="*70)
    
    all_cols = set(df_fixed.columns) | set(df_vintage.columns)
    
    for col in sorted(all_cols):
        if col not in df_fixed.columns:
            print(f"\n{col}:")
            print("  ⚠ Only in vintage dataset")
            continue
        
        if col not in df_vintage.columns:
            print(f"\n{col}:")
            print("  ⚠ Only in fixed dataset")
            continue
        
        # Compare values - align indices first
        fixed_vals = df_fixed[col]
        vintage_vals = df_vintage[col]
        
        # Get common indices
        common_idx = fixed_vals.index.intersection(vintage_vals.index)
        fixed_vals_aligned = fixed_vals.loc[common_idx].dropna()
        vintage_vals_aligned = vintage_vals.loc[common_idx].dropna()
        
        # Need to re-align after dropna
        final_common_idx = fixed_vals_aligned.index.intersection(vintage_vals_aligned.index)
        fixed_vals_aligned = fixed_vals_aligned.loc[final_common_idx]
        vintage_vals_aligned = vintage_vals_aligned.loc[final_common_idx]
        
        # Check if identical
        try:
            is_identical = np.allclose(fixed_vals_aligned, vintage_vals_aligned, rtol=1e-9, atol=1e-9, equal_nan=True)
        except:
            is_identical = (fixed_vals_aligned == vintage_vals_aligned).all()
        
        print(f"\n{col}:")
        
        if is_identical:
            if col in vintage_aligned_cols:
                print("  UNEXPECTED: Values are identical (should differ for vintage-aligned features)")
            else:
                print("  ✓ Values identical (expected)")
        else:
            if col in vintage_aligned_cols:
                print("  ✓ Values differ (expected for vintage-aligned features)")
            elif col in derived_cols:
                print("  ⚠ Values differ (may be expected if derived from vintage-aligned features)")
            else:
                print("  UNEXPECTED: Values differ (should be identical for market data)")
            
            # Show difference statistics
            diff = (fixed_vals_aligned - vintage_vals_aligned)
            if len(diff) > 0:
                non_zero_diff = diff[abs(diff) > 1e-9]
                print(f"     Differences found: {len(non_zero_diff)}/{len(diff)} rows")
                if len(non_zero_diff) > 0:
                    print(f"     Max absolute diff: {abs(diff).max():.6f}")
                    print(f"     Mean absolute diff: {abs(non_zero_diff).mean():.6f}")
                    
                    # Show example differences
                    print(f"     Example differences (first 3):")
                    for idx in non_zero_diff.head(3).index:
                        print(f"       {idx}: fixed={fixed_vals_aligned[idx]:.4f}, vintage={vintage_vals_aligned[idx]:.4f}, diff={diff[idx]:.4f}")
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    # Count issues
    unexpected_identical = []
    unexpected_different = []
    
    for col in vintage_aligned_cols:
        if col in df_fixed.columns and col in df_vintage.columns:
            fixed_vals = df_fixed[col]
            vintage_vals = df_vintage[col]
            common_idx = fixed_vals.index.intersection(vintage_vals.index)
            fixed_aligned = fixed_vals.loc[common_idx].dropna()
            vintage_aligned = vintage_vals.loc[common_idx].dropna()
            final_idx = fixed_aligned.index.intersection(vintage_aligned.index)
            fixed_aligned = fixed_aligned.loc[final_idx]
            vintage_aligned = vintage_aligned.loc[final_idx]
            
            try:
                is_identical = np.allclose(fixed_aligned, vintage_aligned, rtol=1e-9, atol=1e-9, equal_nan=True)
            except:
                is_identical = (fixed_aligned == vintage_aligned).all()
            
            if is_identical:
                unexpected_identical.append(col)
    
    for col in market_cols:
        if col in df_fixed.columns and col in df_vintage.columns:
            fixed_vals = df_fixed[col]
            vintage_vals = df_vintage[col]
            common_idx = fixed_vals.index.intersection(vintage_vals.index)
            fixed_aligned = fixed_vals.loc[common_idx].dropna()
            vintage_aligned = vintage_vals.loc[common_idx].dropna()
            final_idx = fixed_aligned.index.intersection(vintage_aligned.index)
            fixed_aligned = fixed_aligned.loc[final_idx]
            vintage_aligned = vintage_aligned.loc[final_idx]
            
            try:
                is_identical = np.allclose(fixed_aligned, vintage_aligned, rtol=1e-9, atol=1e-9, equal_nan=True)
            except:
                is_identical = (fixed_aligned == vintage_aligned).all()
            
            if not is_identical:
                unexpected_different.append(col)
    
    if unexpected_identical:
        print(f"\n ISSUES: {len(unexpected_identical)} vintage-aligned features are identical:")
        for col in unexpected_identical:
            print(f"   - {col}")
        print("   → Vintage alignment may not have been applied correctly")
    else:
        print(f"\n✓ All vintage-aligned features differ as expected")
    
    if unexpected_different:
        print(f"\n ISSUES: {len(unexpected_different)} market data columns differ:")
        for col in unexpected_different:
            print(f"   - {col}")
        print("   → These should be identical - may indicate data collection issues")
    else:
        print(f"\n✓ All market data columns identical as expected")
    
    if not unexpected_identical and not unexpected_different:
        print("\n VALIDATION PASSED: Datasets match expected behavior")
        return 0
    else:
        print("\n⚠ VALIDATION FAILED: See issues above")
        return 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compare_fixed_vs_vintage.py <fixed_csv> <vintage_csv>")
        sys.exit(1)
    
    fixed_path = sys.argv[1]
    vintage_path = sys.argv[2]
    
    exit_code = compare_datasets(fixed_path, vintage_path)
    sys.exit(exit_code)
