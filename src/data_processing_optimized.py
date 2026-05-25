import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def collect_gold_data_optimized(start_date, end_date):
    """Collect gold data with explicit fallback logic and validation."""
    print("Collecting gold price data (optimized approach)...")

    gold_source_used = None

    # Try 1: Gold futures
    try:
        print("   Attempting gold futures (GC=F)...")
        gold_data = yf.download(
            'GC=F',
            start=start_date - pd.Timedelta(days=60),
            end=end_date + pd.Timedelta(days=1),
            progress=False
        )
        if hasattr(gold_data.columns, 'nlevels') and gold_data.columns.nlevels > 1:
            gold_data.columns = [col[0] for col in gold_data.columns]

        if len(gold_data) > 100:
            gold_source_used = "GC=F (Gold Futures)"
            print(f"   ✓ Gold futures data: {len(gold_data)} observations")
            print(f"     SOURCE USED: {gold_source_used}")
            return gold_data['Close'], gold_source_used
    except Exception as e:
        print(f"   Gold futures failed: {e}")

    # Try 2: SPDR Gold ETF
    try:
        print("   Attempting Gold ETF (GLD)...")
        gld_data = yf.download(
            'GLD',
            start=start_date - pd.Timedelta(days=60),
            end=end_date + pd.Timedelta(days=1),
            progress=False
        )
        if hasattr(gld_data.columns, 'nlevels') and gld_data.columns.nlevels > 1:
            gld_data.columns = [col[0] for col in gld_data.columns]

        if len(gld_data) > 100:
            gold_source_used = "GLD (SPDR Gold ETF, scaled by 10x)"
            gold_proxy = gld_data['Close'] * 10
            print(f"   ⚠ Using GLD ETF (scaled): {len(gld_data)} observations")
            print(f"   ⚠ NOTE: GLD tracks gold but has tracking error")
            print(f"     SOURCE USED: {gold_source_used}")
            return gold_proxy, gold_source_used
    except Exception as e:
        print(f"   Gold ETF failed: {e}")

    # Try 3: iShares Gold ETF
    try:
        print("   Attempting Gold ETF (IAU)...")
        iau_data = yf.download(
            'IAU',
            start=start_date - pd.Timedelta(days=60),
            end=end_date + pd.Timedelta(days=1),
            progress=False
        )
        if hasattr(iau_data.columns, 'nlevels') and iau_data.columns.nlevels > 1:
            iau_data.columns = [col[0] for col in iau_data.columns]

        if len(iau_data) > 100:
            gold_source_used = "IAU (iShares Gold ETF, scaled by 100x)"
            gold_proxy = iau_data['Close'] * 100
            print(f"   ⚠ Using IAU ETF (scaled): {len(iau_data)} observations")
            print(f"   ⚠ NOTE: IAU tracks gold but has tracking error")
            print(f"     SOURCE USED: {gold_source_used}")
            return gold_proxy, gold_source_used
    except Exception as e:
        print(f"   Alternative Gold ETF failed: {e}")

    raise ValueError("Failed to collect gold data from any source")


def add_optimized_gold_features(df):
    """
    Add ONLY the most valuable gold features that don't duplicate existing signals.
    """
    print(" Adding optimized gold features (quality over quantity)...")

    # Get gold data
    start_date = df.index[0]
    end_date = df.index[-1]

    gold_prices, gold_source = collect_gold_data_optimized(start_date, end_date)

    # Align with stock data
    gold_prices_aligned = gold_prices.reindex(df.index, method='ffill')

    # Core gold features
    df['Gold_Price'] = gold_prices_aligned
    df['Gold_Returns'] = gold_prices_aligned.pct_change() * 100
    print("   ✓ Basic gold price and returns")

    # 1. GOLD/SPX RATIO
    if 'SP500_Close' in df.columns:
        df['Gold_SPX_Ratio'] = df['Gold_Price'] / df['SP500_Close']

    # Rolling stats lagged to avoid peeking at current day
        ratio_ma = df['Gold_SPX_Ratio'].rolling(60).mean().shift(1)
        ratio_std = df['Gold_SPX_Ratio'].rolling(60).std().shift(1)
        df['Gold_SPX_Ratio_Norm'] = (df['Gold_SPX_Ratio'] - ratio_ma) / ratio_std

        print("   ✓ Gold/SPX ratio (lagged)")

    # 2. GOLD vs REAL INTEREST RATES
    if 'Treasury_10Y' in df.columns:
        if 'CPI' in df.columns:
            inflation_proxy = df['CPI'].pct_change(252).shift(1) * 100
            df['Real_Interest_Rate'] = df['Treasury_10Y'] - inflation_proxy
            df['Gold_Real_Rate_Signal'] = (df['Real_Interest_Rate'] < 0).astype(int)
            print("   ✓ Gold vs real interest rates (monetary policy signal)")
        else:
            df['Gold_Yield_Inverse'] = 1 / (df['Treasury_10Y'] + 0.01)
            print("   ✓ Gold vs nominal rates (simplified monetary signal)")

    # 3. GOLD MOMENTUM
    df['Gold_Momentum_10d'] = df['Gold_Price'].pct_change(10).shift(1)
    momentum_threshold = df['Gold_Momentum_10d'].rolling(60).quantile(0.7).shift(1)
    df['Gold_Momentum_Strength'] = (df['Gold_Momentum_10d'] > momentum_threshold).astype(int)
    print("   ✓ Gold momentum (10-day, lagged)")

    # 4. GOLD VOLATILITY REGIME
    gold_vol = df['Gold_Returns'].rolling(20).std().shift(1)
    vol_threshold = gold_vol.rolling(120).quantile(0.8).shift(1)
    df['Gold_Vol_Regime'] = (gold_vol > vol_threshold).astype(int)
    print("   ✓ Gold volatility regime (lagged)")

    # 5. GOLD TREND PERSISTENCE
    gold_ma_short = df['Gold_Price'].rolling(20).mean().shift(1)
    gold_ma_long = df['Gold_Price'].rolling(60).mean().shift(1)
    df['Gold_Trend_Direction'] = (gold_ma_short > gold_ma_long).astype(int)
    df['Gold_Trend_Strength'] = (df['Gold_Price'] - gold_ma_long) / gold_ma_long
    print("   ✓ Gold trend (lagged)")

    # 6. GOLD SAFE HAVEN ACTIVATION (lagged to avoid label leakage)
    if 'SP500_Returns' in df.columns:
        safe_haven_raw = (
            (df['Gold_Returns'] > 0) &
            (df['SP500_Returns'] < -1.0)
        ).astype(int)
        # Use previous day's safe-haven flag as the feature
        df['Gold_Safe_Haven'] = safe_haven_raw.shift(1)
        print("   ✓ Gold safe haven activation (lagged crisis behavior)")

    # 7. SELECTIVE GOLD LAGS
    df['Gold_Returns_lag5'] = df['Gold_Returns'].shift(5)
    print("   ✓ Gold 5-day lag (momentum persistence)")

    # Count added gold features
    gold_features = [col for col in df.columns if 'Gold' in col or 'Real_Interest_Rate' in col]
    print(f"   Total optimized gold features: {len(gold_features)}")

    print(f"\n  Gold data source: {gold_source}")

    print("\n Selected Gold Features:")
    for i, feature in enumerate(gold_features, 1):
        print(f"   {i:2d}. {feature}")

    return df


def create_core_technical_features(df):
    """Create technical features with look-ahead-safe construction."""
    print("Creating core technical features (53-feature winning set)...")

    df_enhanced = df.copy()

    # 1. VIX features
    if 'VIX' in df_enhanced.columns:
        for lag in [1, 5, 10]:
            df_enhanced[f'VIX_lag_{lag}'] = df_enhanced['VIX'].shift(lag)

        df_enhanced['VIX_MA_10'] = df_enhanced['VIX'].rolling(10).mean().shift(1)
        df_enhanced['VIX_relative'] = df_enhanced['VIX'] / df_enhanced['VIX_MA_10']
        df_enhanced['VIX_spike'] = (df_enhanced['VIX'] > df_enhanced['VIX_MA_10'] * 1.5).astype(int)
        print("   ✓ VIX features (6 features)")

    # 2. Rolling volatility features
    if 'SP500_Returns' in df_enhanced.columns:
        for window in [5, 10, 20]:
            df_enhanced[f'Volatility_{window}d'] = (
                df_enhanced['SP500_Returns'].rolling(window).std().shift(1)
            )
        print("   ✓ Rolling volatility features (3 features)")

    # 3. Moving average features
    if 'SP500_Close' in df_enhanced.columns:
        for ma_period in [10, 20, 50]:
            df_enhanced[f'MA_{ma_period}'] = (
                df_enhanced['SP500_Close'].rolling(ma_period).mean().shift(1)
            )
            df_enhanced[f'Price_to_MA_{ma_period}'] = (
                df_enhanced['SP500_Close'] / df_enhanced[f'MA_{ma_period}']
            )

        df_enhanced['MA_10_vs_20'] = (df_enhanced['MA_10'] > df_enhanced['MA_20']).astype(int)
        df_enhanced['MA_20_vs_50'] = (df_enhanced['MA_20'] > df_enhanced['MA_50']).astype(int)
        print("   ✓ Moving average features (8 features)")

    # 4. RSI and momentum oscillators
    if 'SP500_Close' in df_enhanced.columns:
        rsi_raw = calculate_rsi(df_enhanced['SP500_Close'])
        df_enhanced['RSI'] = rsi_raw.shift(1)
        df_enhanced['RSI_overbought'] = (df_enhanced['RSI'] > 70).astype(int)
        df_enhanced['RSI_oversold'] = (df_enhanced['RSI'] < 30).astype(int)
        print("   ✓ RSI features (3 features)")

    # 5. Momentum features
    if 'SP500_Close' in df_enhanced.columns:
        for period in [5, 10, 20]:
            df_enhanced[f'Momentum_{period}d'] = (
                df_enhanced['SP500_Close'].pct_change(period).shift(1)
            )
            df_enhanced[f'Momentum_{period}d_positive'] = (
                df_enhanced[f'Momentum_{period}d'] > 0
            ).astype(int)
        print("   ✓ Momentum features (6 features)")

    # 6. Time-based features (optional)
    INCLUDE_CALENDAR_FEATURES = True

    if INCLUDE_CALENDAR_FEATURES:
        df_enhanced['DayOfWeek'] = df_enhanced.index.dayofweek
        df_enhanced['Month'] = df_enhanced.index.month
        df_enhanced['Quarter'] = df_enhanced.index.quarter
        df_enhanced['IsMonthEnd'] = df_enhanced.index.is_month_end.astype(int)
        print("   ✓ Time features (4 features) - EXPERIMENTAL, validate before final use")
    else:
        print("     Time features DISABLED based on validation study")

    # 7. Yield curve features
    if 'Yield_Spread' in df_enhanced.columns:
        df_enhanced['Yield_Spread_MA'] = (
            df_enhanced['Yield_Spread'].rolling(20).mean().shift(1)
        )
        df_enhanced['Yield_Spread_relative'] = (
            df_enhanced['Yield_Spread'] / df_enhanced['Yield_Spread_MA']
        )
        df_enhanced['Yield_Curve_Inversion'] = (
            df_enhanced['Yield_Spread'] < 0
        ).astype(int)
        print("   ✓ Yield curve features (3 features)")

    # 8. Treasury features
    if 'Treasury_10Y' in df_enhanced.columns:
        df_enhanced['Treasury_10Y_change'] = df_enhanced['Treasury_10Y'].diff()
        df_enhanced['Treasury_10Y_MA'] = (
            df_enhanced['Treasury_10Y'].rolling(20).mean().shift(1)
        )
        df_enhanced['Treasury_Rising'] = (
            df_enhanced['Treasury_10Y_change'] > 0
        ).astype(int)
        print("   ✓ Treasury features (3 features)")

    # 9. CPI features
    if 'CPI' in df_enhanced.columns:
        df_enhanced['CPI_YoY'] = df_enhanced['CPI'].pct_change(252)
        df_enhanced['CPI_acceleration'] = df_enhanced['CPI_YoY'].diff()
        print("   ✓ CPI features (2 features)")

    # 10. Unemployment features
    if 'Unemployment' in df_enhanced.columns:
        df_enhanced['Unemployment_change'] = df_enhanced['Unemployment'].diff()
        df_enhanced['Unemployment_Rising'] = (
            df_enhanced['Unemployment_change'] > 0
        ).astype(int)
        print("   ✓ Unemployment features (2 features)")

    # 11. Cross-asset correlations (lagged to avoid peeking at the same day)
    if 'Treasury_10Y' in df_enhanced.columns and 'SP500_Returns' in df_enhanced.columns:
        stock_bond_corr = df_enhanced['SP500_Returns'].rolling(60).corr(
            df_enhanced['Treasury_10Y'].diff().shift(1)
        )
        df_enhanced['Stock_Bond_Corr'] = stock_bond_corr.shift(1)
        print("   ✓ Cross-asset correlation (lagged, 1 feature)")

    return df_enhanced


def calculate_vif(df, feature_cols):
    """Calculate VIF for all features to detect multicollinearity."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df_clean = df[feature_cols].dropna()

    vif_data = []
    for i, col in enumerate(feature_cols):
        vif = variance_inflation_factor(df_clean.values, i)
        vif_data.append({'feature': col, 'VIF': vif})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    return vif_df


def prepare_optimized_lstm_data():
    """Main data preparation with optimized gold integration."""
    print(" OPTIMIZED Data Preparation for LSTM")
    print("=" * 45)

    filename = 'data/financial_dataset_daily.csv'

    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Loaded {filename}: {df.shape}")
    except FileNotFoundError:
        print("prepare_optimized_lstm_data: FileNotFoundError")
        raise

    # Core technical features
    df_core = create_core_technical_features(df)

    # Optimized gold features
    df_optimized = add_optimized_gold_features(df_core)

    # Clean data
    df_optimized = df_optimized.dropna()

    original_features = df.shape[1]
    optimized_features = df_optimized.shape[1]
    gold_features = len(
        [col for col in df_optimized.columns
         if 'Gold' in col or 'Real_Interest_Rate' in col]
    )
    core_features = optimized_features - gold_features

    print(f"\n OPTIMIZED DATA PREPARATION COMPLETE!")
    print("=" * 45)
    print(f"Original features: {original_features}")
    print(f"Core features: {core_features}")
    print(f"Optimized gold features: {gold_features}")
    print(f"Total optimized features: {optimized_features}")
    print(f"Final observations: {len(df_optimized)}")

    print(f"\n FEATURE BREAKDOWN:")
    feature_categories = {
        'Market Data': [col for col in df_optimized.columns if 'SP500' in col],
        'VIX Features': [col for col in df_optimized.columns if 'VIX' in col],
        'Technical Indicators': [
            col for col in df_optimized.columns
            if any(x in col for x in ['MA_', 'RSI', 'Momentum']) and 'Gold' not in col
        ],
        'Volatility': [col for col in df_optimized.columns if 'Volatility' in col],
        'Time Features': [
            col for col in df_optimized.columns
            if any(x in col for x in ['DayOfWeek', 'Month', 'Quarter'])
        ],
        'Macro Economic': [
            col for col in df_optimized.columns
            if any(x in col for x in ['Treasury', 'Yield', 'CPI', 'Unemployment'])
            and 'Gold' not in col
        ],
        'Optimized Gold': [
            col for col in df_optimized.columns
            if 'Gold' in col or 'Real_Interest_Rate' in col
        ]
    }

    for category, features in feature_categories.items():
        if features:
            print(f"\n{category} ({len(features)} features):")
            for feature in features:
                print(f"   • {feature}")

    # Save optimized dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_ts = f'data/optimized_financial_data_{timestamp}.csv'
    df_optimized.to_csv(filename_ts)
    df_optimized.to_csv('data/optimized_financial_data.csv')

    print(f"\n Optimized data saved:")
    print(f"   Main file: data/optimized_financial_data.csv")
    print(f"   Timestamped: {filename_ts}")

    return df_optimized


def create_final_optimized_data():
    df_optimized = prepare_optimized_lstm_data()

    feature_cols = [col for col in df_optimized.columns if col != 'SP500_Returns']
    vif_results = calculate_vif(df_optimized, feature_cols)

    print("\n" + "=" * 70)
    print("MULTICOLLINEARITY CHECK (VIF Analysis)")
    print("=" * 70)
    print("VIF > 10: Severe multicollinearity (drop for ARIMAX)")
    print("VIF 5-10: Moderate multicollinearity")
    print("VIF < 5: Acceptable\n")

    print("Top 20 features by VIF:")
    print(vif_results.head(20).to_string(index=False))

    severe = vif_results[vif_results['VIF'] > 10]
    if len(severe) > 0:
        print(f"\n⚠ WARNING: {len(severe)} features have VIF > 10")
        print("These should be removed for econometric models (ARIMAX)")

    return df_optimized
