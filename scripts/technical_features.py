"""
Frequency-aware technical feature creation.

This module provides functions to create technical indicators that are
appropriately scaled for daily, weekly, or monthly data frequencies.

The key insight: a 20-day moving average on daily data captures ~1 month.
For weekly data, you'd want a 4-week MA to capture the same timeframe.
For monthly, a 1-month MA (which is just the value itself, so use 3-month).

Usage:
    from technical_features import create_technical_features
    
    # For daily data
    df_enhanced = create_technical_features(df_daily, frequency='daily', logger=logger)
    
    # For weekly data  
    df_enhanced = create_technical_features(df_weekly, frequency='weekly', logger=logger)
"""

import pandas as pd
import numpy as np
import logging


# ============================================================================
# FREQUENCY-AWARE WINDOW CONFIGURATIONS
# ============================================================================

# Window sizes calibrated to capture similar timeframes across frequencies
# Daily: trading days, Weekly: weeks, Monthly: months
WINDOW_CONFIG = {
    'daily': {
        # Volatility windows
        'vol_short': 5,       # 1 week
        'vol_medium': 10,     # 2 weeks
        'vol_long': 20,       # 1 month
        
        # Moving average windows
        'ma_short': 10,       # 2 weeks
        'ma_medium': 20,      # 1 month
        'ma_long': 50,        # ~2.5 months
        
        # Momentum windows
        'mom_short': 5,       # 1 week
        'mom_medium': 10,     # 2 weeks
        'mom_long': 20,       # 1 month
        
        # RSI
        'rsi': 14,            # Standard
        
        # Correlation/regime windows
        'corr_window': 60,    # 3 months
        'regime_window': 120, # 6 months
        
        # Yield spread MA
        'yield_ma': 20,       # 1 month
        
        # YoY calculation (for CPI, etc.)
        'yoy_periods': 252,   # 1 year of trading days
        
        # VIX windows
        'vix_ma': 10,         # 2 weeks
        'vix_lags': [1, 5, 10],
        
        # Gold windows
        'gold_mom': 10,       # 2 weeks
        'gold_ma_short': 20,  # 1 month
        'gold_ma_long': 60,   # 3 months
        'gold_vol': 20,       # 1 month
    },
    'weekly': {
        'vol_short': 2,       # 2 weeks
        'vol_medium': 4,      # 1 month
        'vol_long': 8,        # 2 months
        
        'ma_short': 4,        # 1 month
        'ma_medium': 8,       # 2 months
        'ma_long': 13,        # 1 quarter
        
        'mom_short': 2,       # 2 weeks
        'mom_medium': 4,      # 1 month
        'mom_long': 8,        # 2 months
        
        'rsi': 14,            # Standard (14 weeks)
        
        'corr_window': 13,    # 1 quarter
        'regime_window': 26,  # 6 months
        
        'yield_ma': 4,        # 1 month
        
        'yoy_periods': 52,    # 1 year of weeks
        
        'vix_ma': 4,          # 1 month
        'vix_lags': [1, 2, 4],
        
        'gold_mom': 4,        # 1 month
        'gold_ma_short': 4,   # 1 month
        'gold_ma_long': 13,   # 1 quarter
        'gold_vol': 4,        # 1 month
    },
    'monthly': {
        'vol_short': 2,       # 2 months
        'vol_medium': 3,      # 1 quarter
        'vol_long': 6,        # 6 months
        
        'ma_short': 3,        # 1 quarter
        'ma_medium': 6,       # 6 months
        'ma_long': 12,        # 1 year
        
        'mom_short': 1,       # 1 month
        'mom_medium': 3,      # 1 quarter
        'mom_long': 6,        # 6 months
        
        'rsi': 14,            # Standard (14 months)
        
        'corr_window': 6,     # 6 months
        'regime_window': 12,  # 1 year
        
        'yield_ma': 3,        # 1 quarter
        
        'yoy_periods': 12,    # 1 year of months
        
        'vix_ma': 3,          # 1 quarter
        'vix_lags': [1, 2, 3],
        
        'gold_mom': 3,        # 1 quarter
        'gold_ma_short': 3,   # 1 quarter
        'gold_ma_long': 6,    # 6 months
        'gold_vol': 3,        # 1 quarter
    },
}


def get_windows(frequency):
    """Get window configuration for a frequency."""
    if frequency not in WINDOW_CONFIG:
        raise ValueError(f"Unknown frequency: {frequency}. Use 'daily', 'weekly', or 'monthly'")
    return WINDOW_CONFIG[frequency]


# ============================================================================
# RSI CALCULATION
# ============================================================================

def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ============================================================================
# CORE TECHNICAL FEATURES
# ============================================================================

def create_core_technical_features(df, frequency='daily', logger=None):
    """
    Create technical features with frequency-appropriate windows.
    
    All features are lagged to prevent look-ahead bias.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with at minimum SP500_Close, SP500_Returns
    frequency : str
        'daily', 'weekly', or 'monthly'
    logger : logging.Logger, optional
        Logger for progress messages
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical features
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    w = get_windows(frequency)
    df_enhanced = df.copy()
    
    logger.info(f"Creating technical features for {frequency} frequency...")
    
    # ========================================================================
    # 1. VIX FEATURES
    # ========================================================================
    if 'VIX' in df_enhanced.columns:
        # Lagged VIX values
        for lag in w['vix_lags']:
            df_enhanced[f'VIX_lag_{lag}'] = df_enhanced['VIX'].shift(lag)
        
        # VIX relative to its MA (lagged)
        df_enhanced['VIX_MA'] = df_enhanced['VIX'].rolling(w['vix_ma']).mean().shift(1)
        df_enhanced['VIX_relative'] = df_enhanced['VIX'] / df_enhanced['VIX_MA']
        df_enhanced['VIX_spike'] = (df_enhanced['VIX'] > df_enhanced['VIX_MA'] * 1.5).astype(int)
        
        logger.info(f"   ✓ VIX features ({3 + len(w['vix_lags'])} features)")
    
    # ========================================================================
    # 2. ROLLING VOLATILITY
    # ========================================================================
    if 'SP500_Returns' in df_enhanced.columns:
        for name, window in [('short', w['vol_short']), 
                             ('medium', w['vol_medium']), 
                             ('long', w['vol_long'])]:
            df_enhanced[f'Volatility_{name}'] = (
                df_enhanced['SP500_Returns'].rolling(window).std().shift(1)
            )
        logger.info(f"   ✓ Rolling volatility features (3 features)")
    
    # ========================================================================
    # 3. MOVING AVERAGES
    # ========================================================================
    if 'SP500_Close' in df_enhanced.columns:
        for name, window in [('short', w['ma_short']), 
                             ('medium', w['ma_medium']), 
                             ('long', w['ma_long'])]:
            df_enhanced[f'MA_{name}'] = (
                df_enhanced['SP500_Close'].rolling(window).mean().shift(1)
            )
            df_enhanced[f'Price_to_MA_{name}'] = (
                df_enhanced['SP500_Close'] / df_enhanced[f'MA_{name}']
            )
        
        # MA crossover signals
        df_enhanced['MA_short_vs_medium'] = (
            df_enhanced['MA_short'] > df_enhanced['MA_medium']
        ).astype(int)
        df_enhanced['MA_medium_vs_long'] = (
            df_enhanced['MA_medium'] > df_enhanced['MA_long']
        ).astype(int)
        
        logger.info(f"   ✓ Moving average features (8 features)")
    
    # ========================================================================
    # 4. RSI AND MOMENTUM OSCILLATORS
    # ========================================================================
    if 'SP500_Close' in df_enhanced.columns:
        rsi_raw = calculate_rsi(df_enhanced['SP500_Close'], window=w['rsi'])
        df_enhanced['RSI'] = rsi_raw.shift(1)
        df_enhanced['RSI_overbought'] = (df_enhanced['RSI'] > 70).astype(int)
        df_enhanced['RSI_oversold'] = (df_enhanced['RSI'] < 30).astype(int)
        
        logger.info(f"   ✓ RSI features (3 features)")
    
    # ========================================================================
    # 5. MOMENTUM FEATURES
    # ========================================================================
    if 'SP500_Close' in df_enhanced.columns:
        for name, window in [('short', w['mom_short']), 
                             ('medium', w['mom_medium']), 
                             ('long', w['mom_long'])]:
            df_enhanced[f'Momentum_{name}'] = (
                df_enhanced['SP500_Close'].pct_change(window).shift(1)
            )
            df_enhanced[f'Momentum_{name}_positive'] = (
                df_enhanced[f'Momentum_{name}'] > 0
            ).astype(int)
        
        logger.info(f"   ✓ Momentum features (6 features)")
    
    # ========================================================================
    # 6. YIELD CURVE FEATURES
    # ========================================================================
    if 'Yield_Spread' in df_enhanced.columns:
        df_enhanced['Yield_Spread_MA'] = (
            df_enhanced['Yield_Spread'].rolling(w['yield_ma']).mean().shift(1)
        )
        df_enhanced['Yield_Spread_relative'] = (
            df_enhanced['Yield_Spread'] / df_enhanced['Yield_Spread_MA']
        )
        df_enhanced['Yield_Curve_Inversion'] = (
            df_enhanced['Yield_Spread'] < 0
        ).astype(int)
        
        logger.info(f"   ✓ Yield curve features (3 features)")
    
    # ========================================================================
    # 7. TREASURY FEATURES
    # ========================================================================
    if 'Treasury_10Y' in df_enhanced.columns:
        df_enhanced['Treasury_10Y_change'] = df_enhanced['Treasury_10Y'].diff()
        df_enhanced['Treasury_10Y_MA'] = (
            df_enhanced['Treasury_10Y'].rolling(w['yield_ma']).mean().shift(1)
        )
        df_enhanced['Treasury_Rising'] = (
            df_enhanced['Treasury_10Y_change'] > 0
        ).astype(int)
        
        logger.info(f"   ✓ Treasury features (3 features)")
    
    # ========================================================================
    # 8. CPI/INFLATION FEATURES
    # ========================================================================
    if 'CPI' in df_enhanced.columns:
        df_enhanced['CPI_YoY'] = df_enhanced['CPI'].pct_change(w['yoy_periods'])
        df_enhanced['CPI_acceleration'] = df_enhanced['CPI_YoY'].diff()
        
        logger.info(f"   ✓ CPI features (2 features)")
    
    # ========================================================================
    # 9. UNEMPLOYMENT FEATURES
    # ========================================================================
    if 'Unemployment' in df_enhanced.columns:
        df_enhanced['Unemployment_change'] = df_enhanced['Unemployment'].diff()
        df_enhanced['Unemployment_Rising'] = (
            df_enhanced['Unemployment_change'] > 0
        ).astype(int)
        
        logger.info(f"   ✓ Unemployment features (2 features)")
    
    # ========================================================================
    # 10. CROSS-ASSET CORRELATION
    # ========================================================================
    if 'Treasury_10Y' in df_enhanced.columns and 'SP500_Returns' in df_enhanced.columns:
        df_enhanced['Stock_Bond_Corr'] = (
            df_enhanced['SP500_Returns'].rolling(w['corr_window']).corr(
                df_enhanced['Treasury_10Y'].diff().shift(1)
            )
        )
        logger.info(f"   ✓ Cross-asset correlation (1 feature)")
    
    total_added = df_enhanced.shape[1] - df.shape[1]
    logger.info(f"   Total technical features created: {total_added}")
    
    return df_enhanced


# ============================================================================
# GOLD FEATURES
# ============================================================================

def add_gold_features(df, gold_prices, gold_source, frequency='daily', logger=None):
    """
    Add gold-related features with frequency-appropriate windows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    gold_prices : pd.Series
        Gold price series aligned to df index
    gold_source : str
        Description of gold data source
    frequency : str
        'daily', 'weekly', or 'monthly'
    logger : logging.Logger, optional
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added gold features
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    w = get_windows(frequency)
    
    logger.info(f"Adding gold features for {frequency} frequency...")
    
    # Align gold prices
    gold_aligned = gold_prices.reindex(df.index, method='ffill')
    
    # Basic gold features
    df['Gold_Price'] = gold_aligned
    df['Gold_Returns'] = gold_aligned.pct_change() * 100
    logger.info("   ✓ Basic gold price and returns")
    
    # ========================================================================
    # 1. GOLD/SPX RATIO
    # ========================================================================
    if 'SP500_Close' in df.columns:
        df['Gold_SPX_Ratio'] = df['Gold_Price'] / df['SP500_Close']
        
        ratio_ma = df['Gold_SPX_Ratio'].rolling(w['corr_window']).mean().shift(1)
        ratio_std = df['Gold_SPX_Ratio'].rolling(w['corr_window']).std().shift(1)
        df['Gold_SPX_Ratio_Norm'] = (df['Gold_SPX_Ratio'] - ratio_ma) / ratio_std
        
        logger.info("   ✓ Gold/SPX ratio")
    
    # ========================================================================
    # 2. GOLD VS REAL INTEREST RATES
    # ========================================================================
    if 'Treasury_10Y' in df.columns and 'CPI' in df.columns:
        inflation_proxy = df['CPI'].pct_change(w['yoy_periods']).shift(1) * 100
        df['Real_Interest_Rate'] = df['Treasury_10Y'] - inflation_proxy
        df['Gold_Real_Rate_Signal'] = (df['Real_Interest_Rate'] < 0).astype(int)
        logger.info("   ✓ Gold vs real interest rates")
    
    # ========================================================================
    # 3. GOLD MOMENTUM
    # ========================================================================
    df['Gold_Momentum'] = df['Gold_Price'].pct_change(w['gold_mom']).shift(1)
    momentum_threshold = df['Gold_Momentum'].rolling(w['corr_window']).quantile(0.7).shift(1)
    df['Gold_Momentum_Strength'] = (df['Gold_Momentum'] > momentum_threshold).astype(int)
    logger.info("   ✓ Gold momentum")
    
    # ========================================================================
    # 4. GOLD VOLATILITY REGIME
    # ========================================================================
    gold_vol = df['Gold_Returns'].rolling(w['gold_vol']).std().shift(1)
    vol_threshold = gold_vol.rolling(w['regime_window']).quantile(0.8).shift(1)
    df['Gold_Vol_Regime'] = (gold_vol > vol_threshold).astype(int)
    logger.info("   ✓ Gold volatility regime")
    
    # ========================================================================
    # 5. GOLD TREND
    # ========================================================================
    gold_ma_short = df['Gold_Price'].rolling(w['gold_ma_short']).mean().shift(1)
    gold_ma_long = df['Gold_Price'].rolling(w['gold_ma_long']).mean().shift(1)
    df['Gold_Trend_Direction'] = (gold_ma_short > gold_ma_long).astype(int)
    df['Gold_Trend_Strength'] = (df['Gold_Price'] - gold_ma_long) / gold_ma_long
    logger.info("   ✓ Gold trend")
    
    # ========================================================================
    # 6. GOLD SAFE HAVEN
    # ========================================================================
    if 'SP500_Returns' in df.columns:
        # Threshold scales with frequency (daily -1%, weekly -2%, monthly -5%)
        threshold = {'daily': -1.0, 'weekly': -2.0, 'monthly': -5.0}[frequency]
        df['Gold_Safe_Haven'] = (
            (df['Gold_Returns'] > 0) & 
            (df['SP500_Returns'] < threshold)
        ).astype(int)
        logger.info("   ✓ Gold safe haven activation")
    
    # ========================================================================
    # 7. GOLD LAGGED RETURNS
    # ========================================================================
    lag = {'daily': 5, 'weekly': 2, 'monthly': 1}[frequency]
    df['Gold_Returns_lag'] = df['Gold_Returns'].shift(lag)
    logger.info(f"   ✓ Gold lagged returns (lag={lag})")
    
    gold_features = [col for col in df.columns if 'Gold' in col or 'Real_Interest_Rate' in col]
    logger.info(f"   Total gold features: {len(gold_features)}")
    logger.info(f"   Gold data source: {gold_source}")
    
    return df


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def create_technical_features(df, frequency='daily', gold_prices=None, gold_source=None, logger=None):
    """
    Create all technical features for a given frequency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe at target frequency
    frequency : str
        'daily', 'weekly', or 'monthly'
    gold_prices : pd.Series, optional
        Gold price series. If None, gold features are skipped.
    gold_source : str, optional
        Description of gold data source
    logger : logging.Logger, optional
        
    Returns:
    --------
    pd.DataFrame
        Enhanced dataframe with technical features
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
    
    logger.info("="*70)
    logger.info(f"Creating Technical Features ({frequency})")
    logger.info("="*70)
    
    original_cols = df.shape[1]
    
    # Core technical features
    df_enhanced = create_core_technical_features(df, frequency=frequency, logger=logger)
    
    # Gold features (optional)
    if gold_prices is not None:
        df_enhanced = add_gold_features(
            df_enhanced, gold_prices, gold_source or "Unknown", 
            frequency=frequency, logger=logger
        )
    else:
        logger.info("   ⚠ Gold features skipped (no gold data provided)")
    
    # Summary
    total_added = df_enhanced.shape[1] - original_cols
    logger.info("")
    logger.info(f"Total features added: {total_added}")
    logger.info(f"Final column count: {df_enhanced.shape[1]}")
    
    return df_enhanced


if __name__ == "__main__":
    # Quick test
    import yfinance as yf
    
    print("Testing frequency-aware technical features...")
    
    # Get some test data
    spy = yf.download("SPY", start="2020-01-01", end="2024-01-01", progress=False)
    
    df_daily = pd.DataFrame({
        'SP500_Close': spy['Close'],
        'SP500_Returns': spy['Close'].pct_change() * 100,
        'SP500_Volume': spy['Volume'],
    })
    
    # Test daily
    df_daily_enhanced = create_technical_features(df_daily, frequency='daily')
    print(f"\nDaily: {df_daily.shape} -> {df_daily_enhanced.shape}")
    
    # Test weekly
    df_weekly = df_daily.resample('W-FRI').agg({
        'SP500_Close': 'last',
        'SP500_Returns': 'sum',
        'SP500_Volume': 'sum',
    })
    df_weekly['SP500_Returns'] = df_weekly['SP500_Close'].pct_change() * 100
    
    df_weekly_enhanced = create_technical_features(df_weekly, frequency='weekly')
    print(f"Weekly: {df_weekly.shape} -> {df_weekly_enhanced.shape}")
    
    # Test monthly
    df_monthly = df_daily.resample('M').agg({
        'SP500_Close': 'last',
        'SP500_Returns': 'sum',
        'SP500_Volume': 'sum',
    })
    df_monthly['SP500_Returns'] = df_monthly['SP500_Close'].pct_change() * 100
    
    df_monthly_enhanced = create_technical_features(df_monthly, frequency='monthly')
    print(f"Monthly: {df_monthly.shape} -> {df_monthly_enhanced.shape}")
