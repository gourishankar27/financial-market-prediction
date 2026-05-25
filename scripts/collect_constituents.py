#!/usr/bin/env python3
"""
Collect S&P 500 constituent daily returns for multi-task learning.

Standalone script - does not modify main data collection pipeline.
Outputs a separate CSV that can be joined with macro data on demand.

Usage:
    python collect_constituents.py --end-date 2025-11-18
    python collect_constituents.py --end-date 2025-11-18 --top-n 50
    python collect_constituents.py --end-date 2025-11-18 --output-dir data/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ============================================================================
# Configuration
# ============================================================================

# Top 100 S&P 500 constituents by market cap (as of Nov 2024)
# NOTE: Uses current constituents - accepts survivorship bias for simplicity.
SP500_CONSTITUENTS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'BRK-B', 'TSLA', 'UNH',
    'XOM', 'LLY', 'JPM', 'JNJ', 'V', 'AVGO', 'PG', 'MA', 'HD', 'CVX',
    'MRK', 'ABBV', 'COST', 'PEP', 'KO', 'ADBE', 'WMT', 'MCD', 'CRM', 'TMO',
    'CSCO', 'BAC', 'ACN', 'LIN', 'NFLX', 'AMD', 'ABT', 'NKE', 'DIS', 'TXN',
    'PM', 'WFC', 'ORCL', 'DHR', 'CMCSA', 'INTU', 'VZ', 'COP', 'NEE', 'QCOM',
    'UNP', 'IBM', 'AMGN', 'RTX', 'PFE', 'LOW', 'SPGI', 'HON', 'CAT', 'ELV',
    'UPS', 'MS', 'BA', 'AMAT', 'GE', 'BLK', 'PLD', 'DE', 'SYK', 'LMT',
    'BKNG', 'MDT', 'ADP', 'ADI', 'TJX', 'GILD', 'MDLZ', 'C', 'VRTX', 'MMC',
    'SBUX', 'AMT', 'AXP', 'ISRG', 'REGN', 'CI', 'PGR', 'MO', 'ZTS', 'BDX',
    'SCHW', 'CB', 'ETN', 'BMY', 'SO', 'DUK', 'CVS', 'LRCX', 'NOC', 'BSX'
]

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path):
    """Configure logging to both file and console."""
    log_file = output_dir / f"constituent_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# Data Collection
# ============================================================================

def collect_constituent_returns(tickers: list, start_date: str, end_date: str, 
                                logger: logging.Logger, min_coverage: float = 0.95) -> pd.DataFrame:
    """
    Collect daily returns for S&P 500 constituents using batch download.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    logger : logging.Logger
        Logger instance
    min_coverage : float
        Minimum data coverage required (0-1). Default 0.95.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns {ticker}_Returns, indexed by date.
    """
    logger.info("="*70)
    logger.info("Collecting S&P 500 Constituent Returns")
    logger.info("="*70)
    logger.info(f"Tickers requested: {len(tickers)}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Minimum coverage threshold: {min_coverage*100:.0f}%")
    
    # yfinance end date is exclusive, add 1 day
    end_date_adj = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        logger.info("Downloading data (this may take 30-60 seconds)...")
        raw_data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date_adj, 
            progress=False,
            auto_adjust=False,
            group_by='ticker',
            threads=True
        )
        
        if raw_data.empty:
            logger.error("No data returned from yfinance")
            return pd.DataFrame()
        
        # Handle DataFrame structure
        if len(tickers) == 1:
            ticker = tickers[0]
            close_prices = raw_data['Close']
            close_prices.name = ticker
            close_df = pd.DataFrame(close_prices)
        else:
            close_df = raw_data.xs('Close', axis=1, level=1)
        
        logger.info(f"Downloaded data for {len(close_df.columns)} tickers")
        logger.info(f"Date range in data: {close_df.index[0]} to {close_df.index[-1]}")
        
        # Screen for data quality
        expected_days = len(close_df)
        passed_tickers = []
        failed_tickers = []
        
        logger.info("")
        logger.info("Screening tickers for data coverage...")
        logger.info("-"*70)
        
        for ticker in close_df.columns:
            ticker_data = close_df[ticker]
            valid_days = ticker_data.notna().sum()
            coverage = valid_days / expected_days
            
            if coverage >= min_coverage:
                passed_tickers.append(ticker)
                if coverage < 0.99:
                    logger.info(f"✓ {ticker}: {coverage*100:.1f}% coverage (passed)")
            else:
                failed_tickers.append(ticker)
                logger.warning(f"✗ {ticker}: {coverage*100:.1f}% coverage (EXCLUDED)")
        
        close_df_filtered = close_df[passed_tickers].copy()
        
        logger.info("-"*70)
        logger.info(f"Screening results:")
        logger.info(f"  Passed: {len(passed_tickers)} tickers")
        logger.info(f"  Failed: {len(failed_tickers)} tickers")
        
        if 0 < len(failed_tickers) <= 10:
            logger.info(f"  Failed tickers: {', '.join(failed_tickers)}")
        
        # Calculate returns
        logger.info("")
        logger.info("Calculating daily returns...")
        returns_df = close_df_filtered.pct_change() * 100
        returns_df.columns = [f"{ticker}_Returns" for ticker in returns_df.columns]
        returns_df = returns_df.iloc[1:]  # Drop first row (NaN)
        
        # Summary statistics
        logger.info("")
        logger.info("="*70)
        logger.info("Constituent Returns Summary")
        logger.info("="*70)
        logger.info(f"Tickers included: {len(returns_df.columns)}")
        logger.info(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
        logger.info(f"Observations: {len(returns_df)}")
        
        mean_return = returns_df.mean(axis=1).mean()
        mean_vol = returns_df.std(axis=1).mean()
        logger.info(f"Cross-sectional mean daily return: {mean_return:.4f}%")
        logger.info(f"Cross-sectional mean daily volatility: {mean_vol:.2f}%")
        
        return returns_df, passed_tickers, failed_tickers
        
    except Exception as e:
        logger.error(f"Error collecting constituent returns: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return pd.DataFrame(), [], []

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect S&P 500 constituent returns for multi-task learning',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='1990-01-01',
        help='Start date (YYYY-MM-DD). Default: 1990-01-01'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        choices=[25, 50, 75, 100],
        help='Number of top constituents by market cap. Default: 100'
    )
    
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=0.95,
        help='Minimum data coverage required (0-1). Default: 0.95'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data'),
        help='Output directory. Default: data/'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default='vintage',
        choices=['fixed', 'vintage', 'vintage_1990', 'vintage_2005', 'vintage_2010'],
        help='Version suffix for output filename. Default: vintage'
    )
    
    parser.add_argument(
        '--resample-weekly',
        action='store_true',
        help='Also save weekly resampled returns (W-FRI)'
    )
    
    parser.add_argument(
        '--resample-monthly',
        action='store_true',
        help='Also save monthly resampled returns'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("="*70)
    logger.info("S&P 500 CONSTITUENT COLLECTION")
    logger.info("="*70)
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    logger.info(f"Top N constituents: {args.top_n}")
    logger.info(f"Min coverage: {args.min_coverage*100:.0f}%")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Version: {args.version}")
    logger.info("")
    
    # Select tickers
    tickers = SP500_CONSTITUENTS[:args.top_n]
    
    # Collect data
    returns_df, passed, failed = collect_constituent_returns(
        tickers, args.start_date, args.end_date, logger, args.min_coverage
    )
    
    if returns_df.empty:
        logger.error("No data collected. Exiting.")
        sys.exit(1)
    
    # Save daily data
    output_file = args.output_dir / f"constituents_daily_{args.version}.csv"
    returns_df.to_csv(output_file)
    logger.info(f"Saved to: {output_file}")
    
    # Optionally resample to weekly/monthly
    if args.resample_weekly:
        weekly_returns = returns_df.resample('W-FRI').apply(
            lambda x: ((1 + x/100).prod() - 1) * 100 if len(x) > 0 else float('nan')
        )
        weekly_file = args.output_dir / f"constituents_weekly_{args.version}.csv"
        weekly_returns.to_csv(weekly_file)
        logger.info(f"Saved weekly to: {weekly_file}")
    
    if args.resample_monthly:
        monthly_returns = returns_df.resample('ME').apply(
            lambda x: ((1 + x/100).prod() - 1) * 100 if len(x) > 0 else float('nan')
        )
        monthly_file = args.output_dir / f"constituents_monthly_{args.version}.csv"
        monthly_returns.to_csv(monthly_file)
        logger.info(f"Saved monthly to: {monthly_file}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'top_n': args.top_n,
            'min_coverage': args.min_coverage,
            'version': args.version,
        },
        'results': {
            'tickers_requested': len(tickers),
            'tickers_passed': len(passed),
            'tickers_failed': len(failed),
            'observations': len(returns_df),
            'date_range': {
                'start': str(returns_df.index[0]),
                'end': str(returns_df.index[-1]),
            },
        },
        'tickers': {
            'passed': passed,
            'failed': failed,
        },
    }
    
    metadata_file = args.output_dir / f"constituents_metadata_{args.version}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_file}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("COLLECTION COMPLETE")
    logger.info("="*70)
    logger.info(f"Output: {output_file}")
    logger.info(f"Tickers: {len(passed)}/{len(tickers)}")
    logger.info(f"Observations: {len(returns_df)}")
    logger.info("")


if __name__ == '__main__':
    main()