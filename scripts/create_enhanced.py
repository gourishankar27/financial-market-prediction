#!/usr/bin/env python3
"""
Create enhanced dataset with technical features for any frequency.

Loads baseline dataset, fetches gold data, applies frequency-appropriate
technical indicators, and saves enhanced version.

Usage:
    python create_enhanced.py --frequency daily --version fixed
    python create_enhanced.py --frequency weekly --version vintage
    python create_enhanced.py --frequency monthly --version fixed
    
    # All frequencies at once
    python create_enhanced.py --all --version fixed
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

from technical_features import create_technical_features


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


# ============================================================================
# GOLD DATA COLLECTION
# ============================================================================

def collect_gold_data(start_date, end_date, logger):
    """
    Collect gold price data with fallback sources.
    
    Returns:
        tuple: (gold_prices Series, source description)
    """
    logger.info("Collecting gold data...")
    
    # Add buffer for rolling calculations
    start_buffered = start_date - pd.Timedelta(days=90)
    end_buffered = end_date + pd.Timedelta(days=5)
    
    # Try 1: Gold futures
    try:
        logger.info("  Attempting gold futures (GC=F)...")
        gold = yf.download('GC=F', start=start_buffered, end=end_buffered, 
                          progress=False, auto_adjust=False)
        if hasattr(gold.columns, 'nlevels') and gold.columns.nlevels > 1:
            gold.columns = [col[0] for col in gold.columns]
        
        if len(gold) > 100:
            logger.info(f"  ✓ Gold futures: {len(gold)} observations")
            return gold['Close'], "GC=F (Gold Futures)"
    except Exception as e:
        logger.warning(f"  Gold futures failed: {e}")
    
    # Try 2: GLD ETF
    try:
        logger.info("  Attempting GLD ETF...")
        gld = yf.download('GLD', start=start_buffered, end=end_buffered,
                         progress=False, auto_adjust=False)
        if hasattr(gld.columns, 'nlevels') and gld.columns.nlevels > 1:
            gld.columns = [col[0] for col in gld.columns]
        
        if len(gld) > 100:
            logger.info(f"  ✓ GLD ETF: {len(gld)} observations (scaled 10x)")
            return gld['Close'] * 10, "GLD (SPDR Gold ETF, scaled 10x)"
    except Exception as e:
        logger.warning(f"  GLD failed: {e}")
    
    # Try 3: IAU ETF
    try:
        logger.info("  Attempting IAU ETF...")
        iau = yf.download('IAU', start=start_buffered, end=end_buffered,
                         progress=False, auto_adjust=False)
        if hasattr(iau.columns, 'nlevels') and iau.columns.nlevels > 1:
            iau.columns = [col[0] for col in iau.columns]
        
        if len(iau) > 100:
            logger.info(f"  ✓ IAU ETF: {len(iau)} observations (scaled 100x)")
            return iau['Close'] * 100, "IAU (iShares Gold ETF, scaled 100x)"
    except Exception as e:
        logger.warning(f"  IAU failed: {e}")
    
    logger.warning("  ⚠ All gold sources failed, proceeding without gold features")
    return None, None


# ============================================================================
# MAIN LOGIC
# ============================================================================

def create_enhanced_for_frequency(frequency, version, data_dir, output_dir, logger):
    """
    Create enhanced dataset for a specific frequency.
    
    Parameters:
    -----------
    frequency : str
        'daily', 'weekly', or 'monthly'
    version : str
        'fixed' or 'vintage'
    data_dir : Path
        Directory containing baseline datasets
    output_dir : Path
        Directory for output (defaults to data_dir)
    logger : logging.Logger
    """
    logger.info("="*70)
    logger.info(f"Creating Enhanced Dataset: {frequency} ({version})")
    logger.info("="*70)
    
    # Load baseline dataset
    input_path = data_dir / f"financial_dataset_{frequency}_{version}.csv"
    
    if not input_path.exists():
        logger.error(f"Baseline dataset not found: {input_path}")
        logger.error(f"Run collect_data.py first to generate baseline datasets.")
        return None
    
    logger.info(f"Loading: {input_path}")
    df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Collect gold data
    gold_prices, gold_source = collect_gold_data(df.index[0], df.index[-1], logger)
    
    # Create technical features
    df_enhanced = create_technical_features(
        df, 
        frequency=frequency,
        gold_prices=gold_prices,
        gold_source=gold_source,
        logger=logger
    )
    
    # Drop NaNs from rolling calculations
    original_len = len(df_enhanced)
    df_enhanced = df_enhanced.dropna()
    dropped = original_len - len(df_enhanced)
    
    logger.info(f"\nDropped {dropped} rows with NaN (from rolling window warmup)")
    logger.info(f"Final shape: {df_enhanced.shape}")
    
    # Save
    output_path = output_dir / f"financial_dataset_{frequency}_{version}_enhanced.csv"
    df_enhanced.to_csv(output_path)
    logger.info(f"Saved: {output_path}")
    
    return df_enhanced


def main():
    parser = argparse.ArgumentParser(
        description='Create enhanced datasets with technical features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_enhanced.py --frequency daily --version fixed
    python create_enhanced.py --frequency weekly --version vintage
    python create_enhanced.py --all --version fixed
        """
    )
    
    parser.add_argument(
        '--frequency', type=str,
        choices=['daily', 'weekly', 'monthly'],
        help='Frequency to process'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Process all frequencies'
    )
    parser.add_argument(
        '--version', type=str, default='fixed',
        choices=['fixed', 'vintage'],
        help='Data version (default: fixed)'
    )
    parser.add_argument(
        '--data-dir', type=Path, default=Path('data'),
        help='Directory containing baseline datasets (default: data)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help='Output directory (default: same as data-dir)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.frequency:
        parser.error("Must specify either --frequency or --all")
    
    # Setup
    logger = setup_logging()
    output_dir = args.output_dir or args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine frequencies to process
    if args.all:
        frequencies = ['daily', 'weekly', 'monthly']
    else:
        frequencies = [args.frequency]
    
    # Process each frequency
    results = {}
    for freq in frequencies:
        df = create_enhanced_for_frequency(
            frequency=freq,
            version=args.version,
            data_dir=args.data_dir,
            output_dir=output_dir,
            logger=logger
        )
        if df is not None:
            results[freq] = df
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    for freq, df in results.items():
        logger.info(f"  {freq}: {df.shape[0]} samples, {df.shape[1]} features")
    
    if not results:
        logger.error("No datasets created. Check that baseline files exist.")
        sys.exit(1)
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
