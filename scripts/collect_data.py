#!/usr/bin/env python3
"""
Financial Data Collection Pipeline
Collects S&P 500, FRED macro indicators, and Yahoo Finance data with proper release date alignment.

Usage:
    python scripts/collect_data.py --end-date 2025-10-06 --use-vintage
    python scripts/collect_data.py --start-date 1990-01-01 --end-date 2025-10-06 --output-dir ./data
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# ============================================================================
# Configuration
# ============================================================================

FRED_SERIES = {
    'VIX': 'VIXCLS',                    # VIX Volatility Index
    'Treasury_10Y': 'DGS10',            # 10-Year Treasury Rate
    'Yield_Spread': 'T10Y2Y',           # 10Y-2Y Yield Spread (daily, FRED-calculated, no lag adjustment needed)
    'CPI': 'CPIAUCSL',                  # Consumer Price Index
    'Unemployment': 'UNRATE',           # Unemployment Rate
    'Fed_Rate': 'FEDFUNDS',             # Federal Funds Rate
    'Consumer_Sentiment': 'UMCSENT',    # Consumer Sentiment Index
    # 'Credit_HY': 'BAMLH0A0HYM2',        # High yield credit spread
    # 'Credit_IG': 'BAMLC0A0CM'           # Investment grade credit spread
    # 'TB3MS': 'TB3MS',                   # for 10Y-3M yield spread - redundant with 10-2Y; removed
    # 'PCE': 'PCEPI',                     # alternative inflation measure - highly correlated with CPI; removed
    # 'GDP': 'GDP',                       # problematic. quarterly updates. vintage alignment complex; removed
    'Industrial_Production': 'INDPRO',  # Industrial Production Index
}

YAHOO_TICKERS = {
    # Volatility measures
    # 'VVIX': '^VVIX',                    # Volatility of VIX
    
    # Sector ETFs (Select Sector SPDRs)
    # 'Sector_Tech': 'XLK',               # Technology
    # 'Sector_Financials': 'XLF',         # Financials
    # 'Sector_Energy': 'XLE',             # Energy
    # 'Sector_Healthcare': 'XLV',         # Healthcare
    # 'Sector_ConsumerDiscretionary': 'XLY',  # Consumer Discretionary
    # 'Sector_ConsumerStaples': 'XLP',    # Consumer Staples
    # 'Sector_Industrials': 'XLI',        # Industrials
    # 'Sector_Materials': 'XLB',          # Materials
    # 'Sector_Utilities': 'XLU',          # Utilities
    # 'Sector_RealEstate': 'XLRE',        # Real Estate: began in 2015
    # 'Sector_Communications': 'XLC',     # Communication Services: began in 2018
    
    # Broad market measures
    'Wilshire5000': '^W5000',           # Wilshire 5000 Total Market Index
}

FRED_API_KEY = "c8d5b4c26407e7cbfcecca702e0e7aee"

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path):
    """Configure logging to both file and console"""
    log_file = output_dir / f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
# Data Collection Functions
# ============================================================================

def collect_market_data(start_date: str, end_date: str, logger: logging.Logger) -> pd.DataFrame:
    """Collect S&P 500 market data from Yahoo Finance"""
    logger.info("="*70)
    logger.info("Collecting S&P 500 Market Data")
    logger.info("="*70)
    
    # Add 1 day to end_date for yfinance API (end parameter is sometimes exclusive)
    end_date_adj = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    sp500 = yf.download("^GSPC", start=start_date, end=end_date_adj, progress=False, auto_adjust=False)
    
    # Handle MultiIndex columns if present
    if hasattr(sp500.columns, 'nlevels') and sp500.columns.nlevels > 1:
        sp500.columns = [col[0] for col in sp500.columns]
    
    # Calculate daily returns (percentage)
    sp500['Returns'] = sp500['Close'].pct_change() * 100
    
    # Keep only essential columns
    market_data = sp500[['Close', 'Volume', 'Returns']].copy()
    market_data.columns = ['SP500_Close', 'SP500_Volume', 'SP500_Returns']
    
    logger.info(f"Collected {len(market_data)} days of S&P 500 data")
    logger.info(f"Date range: {market_data.index[0]} to {market_data.index[-1]}")
    
    return market_data


def collect_fred_data(api_key: str, start_date: str, end_date: str, logger: logging.Logger) -> dict:
    """Collect macroeconomic data from FRED"""
    logger.info("="*70)
    logger.info("Collecting FRED Macroeconomic Data")
    logger.info("="*70)
    
    fred = Fred(api_key=api_key)
    fred_data = {}
    
    for name, series_id in FRED_SERIES.items():
        try:
            # FRED API uses observation_start/observation_end for date range
            data = fred.get_series(
                series_id, 
                observation_start=start_date,
                observation_end=end_date
            )
            fred_data[name] = data
            logger.info(f"✓ {name}: {len(data)} observations")
        except Exception as e:
            logger.error(f"✗ {name}: Error - {str(e)}")
            fred_data[name] = None
    
    success_count = sum(1 for v in fred_data.values() if v is not None)
    logger.info(f"Successfully collected {success_count}/{len(FRED_SERIES)} series")
    
    return fred_data


def collect_yahoo_data(tickers: dict, start_date: str, end_date: str, logger: logging.Logger) -> pd.DataFrame:
    """Collect data from Yahoo Finance for multiple tickers"""
    logger.info("="*70)
    logger.info("Collecting Yahoo Finance Data")
    logger.info("="*70)
    
    # Add 1 day to end_date for yfinance API (end parameter is sometimes exclusive)
    end_date_adj = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    yahoo_data = {}
    
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date_adj, progress=False, auto_adjust=False)
            
            # Handle MultiIndex columns if present
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = [col[0] for col in data.columns]
            
            # Keep only Close price
            yahoo_data[name] = data['Close']
            logger.info(f"✓ {name} ({ticker}): {len(data)} observations")
            
        except Exception as e:
            logger.error(f"✗ {name} ({ticker}): Error - {str(e)}")
            yahoo_data[name] = None
    
    # Combine into single DataFrame, remove None entries
    yahoo_data = {k: v for k, v in yahoo_data.items() if v is not None}
    combined = pd.DataFrame(yahoo_data) if yahoo_data else pd.DataFrame()
    
    logger.info(f"Successfully collected {len(yahoo_data)}/{len(tickers)} tickers")
    
    return combined

# ============================================================================
# Data Processing Functions
# ============================================================================

def align_and_combine_data(market_data: pd.DataFrame, 
                          fred_data: dict, 
                          yahoo_data: pd.DataFrame,
                          logger: logging.Logger) -> pd.DataFrame:
    """Align different frequency data and combine into master dataset"""
    logger.info("="*70)
    logger.info("Aligning and combining data")
    logger.info("="*70)
    
    # Start with market data (daily frequency)
    master_df = market_data.copy()
    
    # Add FRED data (forward-fill for non-trading days)
    for name, series in fred_data.items():
        if series is not None:
            # Reindex to match market data dates and forward-fill
            aligned_series = series.reindex(master_df.index, method='ffill')
            master_df[name] = aligned_series
    
    # Add yahoo data if provided
    if yahoo_data is not None and not yahoo_data.empty:
        for col in yahoo_data.columns:
            # Reindex to match market data dates and forward-fill
            aligned_series = yahoo_data[col].reindex(master_df.index, method='ffill')
            master_df[col] = aligned_series
    
    logger.info(f"Combined dataset shape: {master_df.shape}")
    logger.info(f"Columns: {list(master_df.columns)}")
    
    return master_df


def check_data_quality(df: pd.DataFrame, logger: logging.Logger):
    """Perform comprehensive data quality checks"""
    logger.info("-"*70)
    logger.info("Data Quality Check")
    logger.info("-"*70)
    
    # Check missing values
    total_missing = df.isnull().sum().sum()
    logger.info(f"Total missing values: {total_missing}")
    
    if total_missing > 0:
        logger.info("Missing by column:")
        missing_cols = df.isnull().sum()
        for col, count in missing_cols[missing_cols > 0].items():
            pct = (count / len(df)) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
    
    # Check for extreme returns
    if 'SP500_Returns' in df.columns:
        extreme_returns = df[abs(df['SP500_Returns']) > 10]
        logger.info(f"Extreme daily returns (>10%): {len(extreme_returns)}")
        
        if len(extreme_returns) > 0 and len(extreme_returns) < 10:
            logger.info("Dates with extreme returns:")
            for date, ret in extreme_returns['SP500_Returns'].items():
                logger.info(f"  {date.strftime('%Y-%m-%d')}: {ret:.2f}%")
    
    # Check data types
    logger.info("Data types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")


def apply_vintage_date_alignment(df: pd.DataFrame, api_key: str, logger: logging.Logger) -> pd.DataFrame:
    """Apply exact release date alignment using ALFRED vintage dates"""
    logger.info("-"*70)
    logger.info("Applying ALFRED vintage date alignment")
    logger.info("-"*70)
    
    fred = Fred(api_key=api_key)
    df_aligned = df.copy()
    
    series_mapping = {
        'CPI': 'CPIAUCSL',
        'Unemployment': 'UNRATE',
        'Industrial_Production': 'INDPRO',
        'Fed_Rate': 'FEDFUNDS',
        'Consumer_Sentiment': 'UMCSENT'
    }
    
    alignment_report = []
    
    for col_name, series_id in series_mapping.items():
        logger.info(f"Processing {col_name} ({series_id})...")
        
        try:
            # Get all vintage releases
            vintages = fred.get_series_all_releases(series_id)
            vintages['date'] = pd.to_datetime(vintages['date'])
            vintages['realtime_start'] = pd.to_datetime(vintages['realtime_start'])
            
            # Create aligned series
            aligned_series = pd.Series(index=df_aligned.index, dtype=float)
            
            for idx in df_aligned.index:
                available_vintages = vintages[vintages['realtime_start'] <= idx]
                
                if len(available_vintages) > 0:
                    past_obs = available_vintages[available_vintages['date'] < idx]
                    
                    if len(past_obs) > 0:
                        most_recent = past_obs.sort_values(['date', 'realtime_start'], ascending=False).iloc[0]
                        aligned_series.loc[idx] = most_recent['value']
                    else:
                        aligned_series.loc[idx] = np.nan
                else:
                    aligned_series.loc[idx] = np.nan
            
            # Count vintage coverage
            vintage_count = aligned_series.notna().sum()
            total_count = len(df_aligned)
            
            # If vintage has NaNs, fall back to original forward-filled values
            aligned_series = aligned_series.fillna(df_aligned[col_name])
            
            # Count how many were filled vs vintage
            filled_count = aligned_series.notna().sum() - vintage_count
            still_missing = aligned_series.isna().sum()
            
            # Replace column
            df_aligned[col_name] = aligned_series
            
            # Build report
            first_valid = aligned_series.first_valid_index()
            report = {
                'series': col_name,
                'vintage_values': vintage_count,
                'forward_filled': filled_count,
                'still_missing': still_missing,
                'first_valid_date': first_valid.strftime('%Y-%m-%d') if first_valid else None
            }
            alignment_report.append(report)
            
            # Log summary for this series
            if filled_count > 0 or still_missing > 0:
                logger.info(f"  ✓ Aligned {col_name}: {vintage_count} vintage, {filled_count} forward-filled, {still_missing} missing")
                if first_valid:
                    logger.info(f"    First valid date: {first_valid.strftime('%Y-%m-%d')}")
            else:
                logger.info(f"  ✓ Aligned {col_name} ({vintage_count}/{total_count} all from vintage)")
        
        except Exception as e:
            logger.error(f"  Error aligning {col_name}: {str(e)}")
            logger.info(f"  Keeping original values")
            alignment_report.append({
                'series': col_name,
                'vintage_values': 0,
                'forward_filled': 0,
                'still_missing': len(df_aligned),
                'error': str(e)
            })
    
    # Print final summary
    logger.info("")
    logger.info("VINTAGE ALIGNMENT SUMMARY:")
    logger.info("-"*70)
    for report in alignment_report:
        if 'error' in report:
            logger.info(f"  {report['series']}: ERROR - {report['error']}")
        else:
            total = len(df_aligned)
            vintage_pct = (report['vintage_values'] / total * 100) if total > 0 else 0
            filled_pct = (report['forward_filled'] / total * 100) if total > 0 else 0
            logger.info(f"  {report['series']}:")
            logger.info(f"    - Vintage dates: {report['vintage_values']:,} ({vintage_pct:.1f}%)")
            if report['forward_filled'] > 0:
                logger.info(f"    - Forward-filled: {report['forward_filled']:,} ({filled_pct:.1f}%)")
            if report['still_missing'] > 0:
                logger.info(f"    - Still missing: {report['still_missing']:,}")
    
    return df_aligned


def apply_fixed_shift_alignment(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Apply fixed-day shift approximation for release dates"""
    logger.info("-"*70)
    logger.info("Applying fixed-shift release date alignment")
    logger.info("-"*70)
    
    # Shift forward = make data available LATER (when actually released)
    df['CPI'] = df['CPI'].shift(14)  # released ~2 weeks after month end
    df['Unemployment'] = df['Unemployment'].shift(7)  # first fri of month
    df['Industrial_Production'] = df['Industrial_Production'].shift(14)  # ~2 weeks
    df['Fed_Rate'] = df['Fed_Rate'].shift(7)  # ~1 week after month end
    df['Consumer_Sentiment'] = df['Consumer_Sentiment'].shift(2)  # final release ~month end
    
    logger.info("✓ Release date adjustments applied")
    
    return df


def preprocess_data(df: pd.DataFrame, use_vintage: bool, api_key: str, logger: logging.Logger) -> pd.DataFrame:
    """Preprocess combined data with quality checks and alignment"""
    logger.info("="*70)
    logger.info("Preprocessing data")
    logger.info("="*70)
    
    # Initial quality check
    check_data_quality(df, logger)
    
    # Forward-fill macro variables
    logger.info("-"*70)
    logger.info("Forward-filling macro variables")
    logger.info("-"*70)
    
    macro_vars = ['CPI', 'Unemployment', 'Fed_Rate', 
                  'Consumer_Sentiment', 'Industrial_Production']
    df_clean = df.copy()
    df_clean[macro_vars] = df_clean[macro_vars].fillna(method='ffill')
    
    # Verify no remaining NaNs
    remaining_na = df_clean[macro_vars].isnull().sum()
    if remaining_na.sum() > 0:
        logger.warning("NaN values remain after forward-fill:")
        for col, count in remaining_na[remaining_na > 0].items():
            logger.warning(f"  {col}: {count}")
    else:
        logger.info("✓ All macro variables forward-filled successfully")
    
    # Apply release date alignment
    if use_vintage:
        df_clean = apply_vintage_date_alignment(df_clean, api_key, logger)
    else:
        df_clean = apply_fixed_shift_alignment(df_clean, logger)
    
    # Shifts create NaNs that need removal
    df_clean = df_clean.dropna()
    
    # Create derived features
    if 'CPI' in df_clean.columns:
        df_clean['Inflation_YoY'] = df_clean['CPI'].pct_change(252) * 100  # 252 trading days ≈ 1 year
    
    if 'SP500_Close' in df_clean.columns:
        df_clean['SP500_Volatility'] = df_clean['SP500_Returns'].rolling(20).std()
    
    logger.info(f"✓ Final dataset shape: {df_clean.shape}")
    logger.info(f"✓ Date range: {df_clean.index[0]} to {df_clean.index[-1]}")
    
    return df_clean


def resample_to_frequency(df: pd.DataFrame, freq: str, logger: logging.Logger) -> pd.DataFrame:
    """Resample daily data to weekly or monthly frequency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Daily frequency data with DatetimeIndex
    freq : str
        Target frequency: 'W', 'W-FRI', 'weekly', 'M', 'ME', 'monthly'
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    pd.DataFrame
        Resampled data at target frequency
    """
    logger.info("-"*70)
    logger.info(f"Resampling to {freq} frequency")
    logger.info("-"*70)
    
    # Normalize frequency codes
    if freq in ['W', 'W-FRI', 'weekly']:
        freq_code = 'W-FRI'  # Week ending Friday (aligns with trading week)
        freq_label = 'weekly'
    elif freq in ['M', 'ME', 'monthly']:
        freq_code = 'M'  # Month-end
        freq_label = 'monthly'
    else:
        freq_code = freq
        freq_label = freq
    
    # Define aggregation rules by feature type
    agg_rules = {}
    
    for col in df.columns:
        if 'Volume' in col:
            agg_rules[col] = 'sum'       # Total period volume
        elif col == 'SP500_Returns':
            agg_rules[col] = 'sum'       # Placeholder; will recalculate from close
        elif 'Volatility' in col:
            agg_rules[col] = 'mean'      # Average realized volatility
        elif col == 'VIX':
            agg_rules[col] = 'mean'      # Average fear over period
        else:
            agg_rules[col] = 'last'      # End-of-period value (prices, macro)
    
    # Resample
    df_resampled = df.resample(freq_code).agg(agg_rules)
    
    # Recalculate returns from resampled prices (more accurate than summing daily returns)
    if 'SP500_Close' in df_resampled.columns:
        df_resampled['SP500_Returns'] = df_resampled['SP500_Close'].pct_change() * 100
    
    # Drop incomplete first period
    df_resampled = df_resampled.iloc[1:]
    
    logger.info(f"✓ Resampled to {freq_label}: {df_resampled.shape}")
    logger.info(f"✓ Date range: {df_resampled.index[0]} to {df_resampled.index[-1]}")
    
    return df_resampled


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def collect_gold_data_optimized(start_date: pd.Timestamp, end_date: pd.Timestamp, logger: logging.Logger):
    """Collect gold data with explicit fallback logic and validation."""
    logger.info("Collecting gold price data (optimized approach)...")
    
    # Add 1 day to end_date for yfinance API (end parameter is sometimes exclusive)
    end_date_adj = end_date + pd.Timedelta(days=2)  # +2 because we already add +1 in the calls below
    
    gold_source_used = None
    
    # Try 1: Gold futures
    try:
        logger.info("   Attempting gold futures (GC=F)...")
        gold_data = yf.download('GC=F', 
                               start=start_date - pd.Timedelta(days=60), 
                               end=end_date_adj, 
                               progress=False,
                               auto_adjust=False)
        if hasattr(gold_data.columns, 'nlevels') and gold_data.columns.nlevels > 1:
            gold_data.columns = [col[0] for col in gold_data.columns]
        
        if len(gold_data) > 100:
            gold_source_used = "GC=F (Gold Futures)"
            logger.info(f"   ✓ Gold futures data: {len(gold_data)} observations")
            logger.info(f"     SOURCE USED: {gold_source_used}")
            return gold_data['Close'], gold_source_used
    except Exception as e:
        logger.warning(f"   Gold futures failed: {e}")

    # Try 2: SPDR Gold ETF
    try:
        logger.info("   Attempting Gold ETF (GLD)...")
        gld_data = yf.download('GLD', 
                              start=start_date - pd.Timedelta(days=60), 
                              end=end_date_adj,
                              progress=False,
                              auto_adjust=False)
        if hasattr(gld_data.columns, 'nlevels') and gld_data.columns.nlevels > 1:
            gld_data.columns = [col[0] for col in gld_data.columns]
        
        if len(gld_data) > 100:
            gold_source_used = "GLD (SPDR Gold ETF, scaled by 10x)"
            gold_proxy = gld_data['Close'] * 10
            logger.warning(f"   ⚠ Using GLD ETF (scaled): {len(gld_data)} observations")
            logger.warning(f"   ⚠ NOTE: GLD tracks gold but has tracking error")
            logger.info(f"     SOURCE USED: {gold_source_used}")
            return gold_proxy, gold_source_used
    except Exception as e:
        logger.warning(f"   Gold ETF failed: {e}")

    # Try 3: iShares Gold ETF
    try:
        logger.info("   Attempting Gold ETF (IAU)...")
        iau_data = yf.download('IAU', 
                              start=start_date - pd.Timedelta(days=60), 
                              end=end_date_adj,
                              progress=False,
                              auto_adjust=False)
        if hasattr(iau_data.columns, 'nlevels') and iau_data.columns.nlevels > 1:
            iau_data.columns = [col[0] for col in iau_data.columns]
        
        if len(iau_data) > 100:
            gold_source_used = "IAU (iShares Gold ETF, scaled by 100x)"
            gold_proxy = iau_data['Close'] * 100
            logger.warning(f"   ⚠ Using IAU ETF (scaled): {len(iau_data)} observations")
            logger.warning(f"   ⚠ NOTE: IAU tracks gold but has tracking error")
            logger.info(f"     SOURCE USED: {gold_source_used}")
            return gold_proxy, gold_source_used
    except Exception as e:
        logger.warning(f"   Alternative Gold ETF failed: {e}")

    raise ValueError("Failed to collect gold data from any source")


def add_optimized_gold_features(df: pd.DataFrame, logger: logging.Logger) -> tuple:
    """
    Add ONLY the most valuable gold features that don't duplicate existing signals
    
    Selection criteria:
    1. Unique signal not captured by VIX/technical indicators
    2. Proven predictive value in research
    3. Low correlation with existing features
    4. Captures different market dynamics (monetary policy, crisis, etc.)
    
    Returns:
        tuple: (enhanced_df, gold_source)
    """
    logger.info("Adding optimized gold features (quality over quantity)...")

    # Get gold data
    start_date = df.index[0]
    end_date = df.index[-1]
    
    gold_prices, gold_source = collect_gold_data_optimized(start_date, end_date, logger)

    # Align with stock data
    gold_prices_aligned = gold_prices.reindex(df.index, method='ffill')

    # Core gold features
    df['Gold_Price'] = gold_prices_aligned
    df['Gold_Returns'] = gold_prices_aligned.pct_change() * 100
    logger.info("   ✓ Basic gold price and returns")

    # 1. GOLD/SPX RATIO - Key risk-on/risk-off indicator (UNIQUE SIGNAL)
    if 'SP500_Close' in df.columns:
        df['Gold_SPX_Ratio'] = df['Gold_Price'] / df['SP500_Close']
        
        # FIXED: lag rolling calculations
        ratio_ma = df['Gold_SPX_Ratio'].rolling(60).mean().shift(1)
        ratio_std = df['Gold_SPX_Ratio'].rolling(60).std().shift(1)
        df['Gold_SPX_Ratio_Norm'] = (df['Gold_SPX_Ratio'] - ratio_ma) / ratio_std
        
        logger.info("   ✓ Gold/SPX ratio (lagged)")

    # 2. GOLD vs REAL INTEREST RATES - Monetary policy impact (UNIQUE SIGNAL)
    if 'Treasury_10Y' in df.columns:
        if 'CPI' in df.columns:
            # FIXED: Use lagged inflation
            inflation_proxy = df['CPI'].pct_change(252).shift(1) * 100  # YoY change
            df['Real_Interest_Rate'] = df['Treasury_10Y'] - inflation_proxy
            df['Gold_Real_Rate_Signal'] = (df['Real_Interest_Rate'] < 0).astype(int)
            logger.info("   ✓ Gold vs real interest rates (monetary policy signal)")
        else:
            # Simplified version without CPI
            df['Gold_Yield_Inverse'] = 1 / (df['Treasury_10Y'] + 0.01)  # Avoid division by zero
            logger.info("   ✓ Gold vs nominal rates (simplified monetary signal)")

    # 3. GOLD MOMENTUM - Different timeframe than stock momentum (COMPLEMENTARY SIGNAL)
    # FIXED: All momentum calculations lagged
    df['Gold_Momentum_10d'] = df['Gold_Price'].pct_change(10).shift(1)
    momentum_threshold = df['Gold_Momentum_10d'].rolling(60).quantile(0.7).shift(1)
    df['Gold_Momentum_Strength'] = (df['Gold_Momentum_10d'] > momentum_threshold).astype(int)
    logger.info("   ✓ Gold momentum (10-day, different from stock momentum)")

    # 4. GOLD VOLATILITY REGIME - Crisis detection (UNIQUE SIGNAL)
    # FIXED: Volatility regime lagged
    gold_vol = df['Gold_Returns'].rolling(20).std().shift(1)
    vol_threshold = gold_vol.rolling(120).quantile(0.8).shift(1)
    df['Gold_Vol_Regime'] = (gold_vol > vol_threshold).astype(int)
    logger.info("   ✓ Gold volatility regime (lagged)")

    # 5. GOLD TREND PERSISTENCE - Long-term trend strength (UNIQUE SIGNAL)
    # FIXED: Trend calculations lagged
    gold_ma_short = df['Gold_Price'].rolling(20).mean().shift(1)
    gold_ma_long = df['Gold_Price'].rolling(60).mean().shift(1)
    df['Gold_Trend_Direction'] = (gold_ma_short > gold_ma_long).astype(int)
    df['Gold_Trend_Strength'] = (df['Gold_Price'] - gold_ma_long) / gold_ma_long
    logger.info("   ✓ Gold trend (lagged)")

    # 6. GOLD SAFE HAVEN ACTIVATION - Crisis correlation (UNIQUE SIGNAL)
    if 'SP500_Returns' in df.columns:
        df['Gold_Safe_Haven'] = (
            (df['Gold_Returns'] > 0) & 
            (df['SP500_Returns'] < -1.0)  # Significant stock decline
        ).astype(int)
        logger.info("   ✓ Gold safe haven activation (crisis behavior)")

    # 7. SELECTIVE GOLD LAGS - Only the most predictive (MEMORY SIGNAL)
    df['Gold_Returns_lag5'] = df['Gold_Returns'].shift(5)
    logger.info("   ✓ Gold 5-day lag (momentum persistence)")

    # Count added gold features
    gold_features = [col for col in df.columns if 'Gold' in col or 'Real_Interest_Rate' in col]
    logger.info(f"   Total optimized gold features: {len(gold_features)}")
    logger.info(f"   Gold data source: {gold_source}")

    return df, gold_source


def create_core_technical_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Create technical features with FIXED look-ahead bias prevention."""
    logger.info("Creating core technical features...")

    df_enhanced = df.copy()

    # 1. VIX features (fear/volatility indicators)
    if 'VIX' in df_enhanced.columns:
        for lag in [1, 5, 10]:
            df_enhanced[f'VIX_lag_{lag}'] = df_enhanced['VIX'].shift(lag)

        # Lag MA before using
        df_enhanced['VIX_MA_10'] = df_enhanced['VIX'].rolling(10).mean().shift(1)
        df_enhanced['VIX_relative'] = df_enhanced['VIX'] / df_enhanced['VIX_MA_10']
        df_enhanced['VIX_spike'] = (df_enhanced['VIX'] > df_enhanced['VIX_MA_10'] * 1.5).astype(int)
        logger.info("   ✓ VIX features (6 features)")

    # 2. Rolling volatility features
    if 'SP500_Returns' in df_enhanced.columns:
        for window in [5, 10, 20]:
            df_enhanced[f'Volatility_{window}d'] = df_enhanced['SP500_Returns'].rolling(window).std().shift(1)
        logger.info("   ✓ Rolling volatility features (3 features)")

    # 3. Moving average features
    if 'SP500_Close' in df_enhanced.columns:
        for ma_period in [10, 20, 50]:
            df_enhanced[f'MA_{ma_period}'] = df_enhanced['SP500_Close'].rolling(ma_period).mean().shift(1)
            df_enhanced[f'Price_to_MA_{ma_period}'] = df_enhanced['SP500_Close'] / df_enhanced[f'MA_{ma_period}']

        # MA crossover signals
        df_enhanced['MA_10_vs_20'] = (df_enhanced['MA_10'] > df_enhanced['MA_20']).astype(int)
        df_enhanced['MA_20_vs_50'] = (df_enhanced['MA_20'] > df_enhanced['MA_50']).astype(int)
        logger.info("   ✓ Moving average features (8 features)")

    # 4. RSI and momentum oscillators
    if 'SP500_Close' in df_enhanced.columns:
        rsi_raw = calculate_rsi(df_enhanced['SP500_Close'])
        df_enhanced['RSI'] = rsi_raw.shift(1)
        df_enhanced['RSI_overbought'] = (df_enhanced['RSI'] > 70).astype(int)
        df_enhanced['RSI_oversold'] = (df_enhanced['RSI'] < 30).astype(int)
        logger.info("   ✓ RSI features (3 features)")

    # 5. Momentum features
    if 'SP500_Close' in df_enhanced.columns:
        for period in [5, 10, 20]:
            df_enhanced[f'Momentum_{period}d'] = df_enhanced['SP500_Close'].pct_change(period).shift(1)
            df_enhanced[f'Momentum_{period}d_positive'] = (df_enhanced[f'Momentum_{period}d'] > 0).astype(int)
        logger.info("   ✓ Momentum features (6 features)")

    # 6. Time-based features
    # EXPERIMENTAL: Require validation
    # NOTE: most calendar anomalies have disappeared post-publication
    # (Sullivan et al. 2001). Including these increases overfitting risk
    # with limited sample size. Set to False by default.
    INCLUDE_CALENDAR_FEATURES = False
    
    if INCLUDE_CALENDAR_FEATURES:
        df_enhanced['DayOfWeek'] = df_enhanced.index.dayofweek
        df_enhanced['Month'] = df_enhanced.index.month
        df_enhanced['Quarter'] = df_enhanced.index.quarter
        df_enhanced['IsMonthEnd'] = df_enhanced.index.is_month_end.astype(int)
        logger.info("   ✓ Time features (4 features) - EXPERIMENTAL")
    else:
        logger.info("   ⚠ Time features DISABLED (calendar anomalies likely spurious)")

    # 7. Yield curve features
    if 'Yield_Spread' in df_enhanced.columns:
        df_enhanced['Yield_Spread_MA'] = df_enhanced['Yield_Spread'].rolling(20).mean().shift(1)
        df_enhanced['Yield_Spread_relative'] = df_enhanced['Yield_Spread'] / df_enhanced['Yield_Spread_MA']
        df_enhanced['Yield_Curve_Inversion'] = (df_enhanced['Yield_Spread'] < 0).astype(int)
        logger.info("   ✓ Yield curve features (3 features)")

    # 8. Treasury features
    if 'Treasury_10Y' in df_enhanced.columns:
        df_enhanced['Treasury_10Y_change'] = df_enhanced['Treasury_10Y'].diff()
        df_enhanced['Treasury_10Y_MA'] = df_enhanced['Treasury_10Y'].rolling(20).mean().shift(1)
        df_enhanced['Treasury_Rising'] = (df_enhanced['Treasury_10Y_change'] > 0).astype(int)
        logger.info("   ✓ Treasury features (3 features)")

    # 9. CPI features
    if 'CPI' in df_enhanced.columns:
        df_enhanced['CPI_YoY'] = df_enhanced['CPI'].pct_change(252)
        df_enhanced['CPI_acceleration'] = df_enhanced['CPI_YoY'].diff()
        logger.info("   ✓ CPI features (2 features)")

    # 10. Unemployment features
    if 'Unemployment' in df_enhanced.columns:
        df_enhanced['Unemployment_change'] = df_enhanced['Unemployment'].diff()
        df_enhanced['Unemployment_Rising'] = (df_enhanced['Unemployment_change'] > 0).astype(int)
        logger.info("   ✓ Unemployment features (2 features)")

    # 11. Cross-asset correlations
    if 'Treasury_10Y' in df_enhanced.columns and 'SP500_Returns' in df_enhanced.columns:
        df_enhanced['Stock_Bond_Corr'] = df_enhanced['SP500_Returns'].rolling(60).corr(
            df_enhanced['Treasury_10Y'].diff().shift(1)
        )
        logger.info("   ✓ Cross-asset correlation (1 feature)")

    logger.info(f"Total technical features created: {df_enhanced.shape[1] - df.shape[1]}")
    
    return df_enhanced


def create_enhanced_dataset(df: pd.DataFrame, logger: logging.Logger) -> tuple:
    """
    Create enhanced dataset with all technical and gold features.
    
    Returns:
        tuple: (enhanced_df, gold_source, feature_counts)
    """
    logger.info("="*70)
    logger.info("Creating Enhanced Dataset")
    logger.info("="*70)
    
    original_features = df.shape[1]
    
    # Add technical features first
    df_technical = create_core_technical_features(df, logger)
    technical_features = df_technical.shape[1] - original_features
    technical_column_count = df_technical.shape[1]  # Save count before gold features
    
    # Add gold features
    df_enhanced, gold_source = add_optimized_gold_features(df_technical, logger)
    gold_features = df_enhanced.shape[1] - technical_column_count  # Use saved count
    
    # Clean data (remove NaNs created by rolling calculations)
    df_enhanced = df_enhanced.dropna()
    
    feature_counts = {
        'original': original_features,
        'technical': technical_features,
        'gold': gold_features,
        'total': df_enhanced.shape[1]
    }
    
    logger.info("")
    logger.info("="*70)
    logger.info("Enhanced Dataset Summary")
    logger.info("="*70)
    logger.info(f"Original features: {original_features}")
    logger.info(f"Technical features added: {technical_features}")
    logger.info(f"Gold features added: {gold_features}")
    logger.info(f"Total features: {feature_counts['total']}")
    logger.info(f"Final observations: {len(df_enhanced)}")
    logger.info(f"Gold data source: {gold_source}")
    
    return df_enhanced, gold_source, feature_counts


def save_data(df: pd.DataFrame, filename: str, output_dir: Path, logger: logging.Logger) -> Path:
    """Save dataset and print summary statistics"""
    filepath = output_dir / filename
    
    logger.info("="*70)
    logger.info(f"FINAL DATA SUMMARY - {filename}")
    logger.info("="*70)
    
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    years_covered = (df.index[-1] - df.index[0]).days / 365.25
    logger.info(f"Years covered: {years_covered:.1f}")
    
    logger.info(f"Columns: {list(df.columns)}")
    
    memory_usage = df.memory_usage(deep=True).sum() / 1024  # KB
    logger.info(f"Memory usage: {memory_usage:.1f} KB")
    
    # Target variable statistics
    if 'SP500_Returns' in df.columns:
        returns = df['SP500_Returns'].dropna()
        logger.info("Target variable (SP500_Returns) statistics:")
        logger.info(f"  Mean: {returns.mean():.4f}%")
        logger.info(f"  Std: {returns.std():.4f}%")
        logger.info(f"  Min: {returns.min():.4f}%")
        logger.info(f"  Max: {returns.max():.4f}%")
        logger.info(f"  Skewness: {returns.skew():.4f}")
        logger.info(f"  Kurtosis: {returns.kurtosis():.4f}")
    
    # Missing values
    total_missing = df.isnull().sum().sum()
    logger.info(f"Missing values: {total_missing}")
    
    if total_missing > 0:
        logger.warning("Missing values by column:")
        missing_cols = df.isnull().sum()
        for col, count in missing_cols[missing_cols > 0].items():
            logger.warning(f"  {col}: {count}")
    
    # Save
    df.to_csv(filepath)
    logger.info(f"Saved dataset to '{filepath}'")
    
    return filepath


def save_metadata(args: argparse.Namespace, output_dir: Path, daily_path: Path, 
                  weekly_path: Path, monthly_path: Path,
                  enhanced_daily_path: Path, gold_source: str, feature_counts: dict):
    """Save run metadata to JSON"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'use_vintage': args.use_vintage,
        },
        'outputs': {
            'daily_baseline': str(daily_path.name),
            'weekly_baseline': str(weekly_path.name),
            'monthly_baseline': str(monthly_path.name),
            'daily_enhanced': str(enhanced_daily_path.name),
        },
        'feature_counts': feature_counts,
        'gold_data_source': gold_source,
        'fred_series': FRED_SERIES,
        'yahoo_tickers': YAHOO_TICKERS,
    }
    
    metadata_path = output_dir / f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect and process financial data for S&P 500 forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='1990-01-01',
        help='Start date for data collection (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date for data collection (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--use-vintage',
        action='store_true',
        help='Use ALFRED vintage dates for release alignment (slower, more accurate)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory for datasets and logs'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("="*70)
    logger.info("FINANCIAL DATA COLLECTION PIPELINE")
    logger.info("="*70)
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    logger.info(f"Vintage alignment: {args.use_vintage}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    try:
        # Collect all data
        market_data = collect_market_data(args.start_date, args.end_date, logger)
        fred_data = collect_fred_data(FRED_API_KEY, args.start_date, args.end_date, logger)
        yahoo_data = collect_yahoo_data(YAHOO_TICKERS, args.start_date, args.end_date, logger)
        
        # Combine and process
        combined_data = align_and_combine_data(market_data, fred_data, yahoo_data, logger)
        final_data = preprocess_data(combined_data, args.use_vintage, FRED_API_KEY, logger)
        
        # Save baseline daily version
        daily_path = save_data(final_data, 'financial_dataset_daily.csv', args.output_dir, logger)
        
        # Create and save baseline weekly version
        weekly_data = resample_to_frequency(final_data, freq='W-FRI', logger=logger)
        weekly_path = save_data(weekly_data, 'financial_dataset_weekly.csv', args.output_dir, logger)
        
        # Create and save baseline monthly version
        monthly_data = resample_to_frequency(final_data, freq='M', logger=logger)
        monthly_path = save_data(monthly_data, 'financial_dataset_monthly.csv', args.output_dir, logger)
        
        # Create enhanced dataset with technical + gold features
        enhanced_data, gold_source, feature_counts = create_enhanced_dataset(final_data, logger)
        
        # Save enhanced daily version
        enhanced_daily_path = save_data(enhanced_data, 'financial_dataset_daily_enhanced.csv', args.output_dir, logger)
        
        # Save metadata with enhanced info
        metadata_path = save_metadata(args, args.output_dir, daily_path, weekly_path, monthly_path, 
                                     enhanced_daily_path, gold_source, feature_counts)
        logger.info(f"Saved metadata to '{metadata_path}'")
        
        logger.info("")
        logger.info("="*70)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info("="*70)
        logger.info("Generated files:")
        logger.info(f"  Baseline datasets:")
        logger.info(f"    - {daily_path}")
        logger.info(f"    - {weekly_path}")
        logger.info(f"    - {monthly_path}")
        logger.info(f"  Enhanced dataset:")
        logger.info(f"    - {enhanced_daily_path}")
        logger.info(f"  Metadata:")
        logger.info(f"    - {metadata_path}")
        logger.info("")
        logger.info("Dataset breakdown:")
        logger.info(f"  Baseline: {feature_counts['original']} features")
        logger.info(f"  Enhanced: {feature_counts['total']} features")
        logger.info(f"    - Technical indicators: {feature_counts['technical']}")
        logger.info(f"    - Gold features: {feature_counts['gold']}")
        logger.info("")
        
    except Exception as e:
        logger.error("="*70)
        logger.error("ERROR DURING DATA COLLECTION")
        logger.error("="*70)
        logger.error(f"{str(e)}", exc_info=True)
        logger.error("Please check your FRED API key and internet connection")
        sys.exit(1)


if __name__ == '__main__':
    main()
