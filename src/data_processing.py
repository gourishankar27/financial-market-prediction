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


"""===================================================================================================================="""
# Configuration
# START_DATE = '2000-01-01'
START_DATE = '1990-01-01'  # earliest VIX data: gain ~2000 observations

# FRED series mapping
FRED_SERIES = {
    'VIX': 'VIXCLS',                    # VIX Volatility Index
    'Treasury_10Y': 'DGS10',            # 10-Year Treasury Rate
    'Yield_Spread': 'T10Y2Y',           # 10Y-2Y Yield Spread (daily, FRED-calculated, no lag adjustment needed)
    'CPI': 'CPIAUCSL',                  # Consumer Price Index
    'Unemployment': 'UNRATE',           # Unemployment Rate
    'Fed_Rate': 'FEDFUNDS',             # Federal Funds Rate
    'Consumer_Sentiment': 'UMCSENT',    # Consumer Sentiment Index
    'Industrial_Production': 'INDPRO',  # Industrial Production Index
}

# Yahoo Finance tickers mapping
YAHOO_TICKERS = {
    # Broad market measures
    'Wilshire5000': '^W5000',           # Wilshire 5000 Total Market Index
}

FRED_API_KEY = "c8d5b4c26407e7cbfcecca702e0e7aee"

# release date alignment method
USE_VINTAGE_DATES = False    # set True to use ALFRED, False for fixed shifts

"""===================================================================================================================="""


def print_section(title, char='='):
    """print a formatted section header"""
    print(f'\n{char * 70}')
    print(f'{title}')
    print(f'{char * 70}\n')


def print_subsection(title):
    """print a formatted subsection header"""
    print(f'\n{title}')
    print(f'{"-" * 70}')


def collect_market_data(start_date=START_DATE):
    """Collect S&P 500 market data from Yahoo Finance"""
    print_subsection('Collecting S&P 500 Market Data')

    # Download S&P 500 data
    sp500 = yf.download("^GSPC", start=start_date, progress=False)

    # Handle MultiIndex columns if present
    if hasattr(sp500.columns, 'nlevels') and sp500.columns.nlevels > 1:
        sp500.columns = [col[0] for col in sp500.columns]

    # Calculate daily returns (percentage)
    sp500['Returns'] = sp500['Close'].pct_change() * 100

    # Keep only essential columns
    market_data = sp500[['Close', 'Volume', 'Returns']].copy()
    market_data.columns = ['SP500_Close', 'SP500_Volume', 'SP500_Returns']

    print(f'   Collected {len(market_data)} days of S&P 500 data')
    print(f'   Date range: {market_data.index[0]} to {market_data.index[-1]}')
    return market_data


def collect_fred_data(api_key, start_date=START_DATE):
    """Collect macroeconomic data from FRED"""
    print_subsection('Collecting FRED Macroeconomic Data')

    fred = Fred(api_key=api_key)
    fred_data = {}

    for name, series_id in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, start=start_date)
            fred_data[name] = data
            print(f"   ✓ {name}: {len(data)} observations")
        except Exception as e:
            print(f"   ✗ {name}: Error - {str(e)}")
            fred_data[name] = None

    print(f'\n   Summary: Successfully collected {sum(1 for v in fred_data.values() if v is not None)}/{len(FRED_SERIES)} series')
    return fred_data


def collect_yahoo_data(tickers, start_date=START_DATE):
    """
    Collect data from Yahoo finance for multiple tickers.
    """
    print_subsection('Collecting Yahoo Finance Data')

    yahoo_data = {}

    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start_date, progress=False)

            # Handle MultiIndex columns if present
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = [col[0] for col in data.columns]

            yahoo_data[name] = data['Close']
            print(f"   ✓ {name} ({ticker}): {len(data)} observations")

        except Exception as e:
            print(f"   ✗ {name} ({ticker}): Error - {str(e)}")
            yahoo_data[name] = None

    # Combine into single DataFrame, remove None entries
    yahoo_data = {k: v for k, v in yahoo_data.items() if v is not None}
    combined = pd.DataFrame(yahoo_data) if yahoo_data else pd.DataFrame()

    print(f'\n   Summary: Successfully collected {len(yahoo_data)}/{len(tickers)} tickers')
    return combined


def align_and_combine_data(market_data, fred_data, yahoo_data):
    """Align different frequency data and combine into master dataset"""
    print("\nAligning and combining data...")

    # Start with market data (daily frequency)
    master_df = market_data.copy()

    # Add FRED data (forward-fill for non-trading days)
    for name, series in fred_data.items():
        if series is not None:
            aligned_series = series.reindex(master_df.index, method='ffill')
            master_df[name] = aligned_series

    # Add Yahoo data if provided
    if yahoo_data is not None and not yahoo_data.empty:
        for col in yahoo_data.columns:
            aligned_series = yahoo_data[col].reindex(master_df.index, method='ffill')
            master_df[col] = aligned_series

    return master_df


def check_data_quality(df):
    """Perform comprehensive data quality checks."""
    print_subsection('Data Quality Check')

    # Check missing values
    total_missing = df.isnull().sum().sum()
    print(f'Total missing values: {total_missing}')
    if total_missing > 0:
        print('\nMissing by column:')
        missing_cols = df.isnull().sum()
        for col, count in missing_cols[missing_cols > 0].items():
            pct = (count / len(df)) * 100
            print(f'  {col}: {count} ({pct:.1f}%)')

    # Check for extreme returns (potential data errors)
    if 'SP500_Returns' in df.columns:
        extreme_returns = df[abs(df['SP500_Returns']) > 10]
        print(f'\nExtreme daily returns (>10%): {len(extreme_returns)}')
        if 0 < len(extreme_returns) < 10:
            print('Dates with extreme returns:')
            for date, ret in extreme_returns['SP500_Returns'].items():
                print(f'  {date.strftime("%Y-%m-%d")}: {ret:.2f}%')

    # Check data types
    print('\nData types:')
    print(df.dtypes)


def apply_vintage_date_alignment(df, api_key):
    """
    Apply exact release date alignment using ALFRED vintage dates.

    Falls back to forward-fill for periods where vintage data unavailable.
    """
    print_subsection('Applying ALFRED vintage date alignment')

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
        print(f'  Processing {col_name} ({series_id})...')

        try:
            vintages = fred.get_series_all_releases(series_id)
            vintages['date'] = pd.to_datetime(vintages['date'])
            vintages['realtime_start'] = pd.to_datetime(vintages['realtime_start'])

            aligned_series = pd.Series(index=df_aligned.index, dtype=float)

            for idx in df_aligned.index:
                available_vintages = vintages[vintages['realtime_start'] <= idx]

                if len(available_vintages) > 0:
                    past_obs = available_vintages[available_vintages['date'] < idx]

                    if len(past_obs) > 0:
                        most_recent = past_obs.sort_values(['date', 'realtime_start'],
                                                           ascending=False).iloc[0]
                        aligned_series.loc[idx] = most_recent['value']
                    else:
                        aligned_series.loc[idx] = np.nan
                else:
                    aligned_series.loc[idx] = np.nan

            vintage_count = aligned_series.notna().sum()
            total_count = len(df_aligned)

            # Fill remaining gaps with original forward-filled values
            aligned_series = aligned_series.fillna(df_aligned[col_name])

            filled_count = aligned_series.notna().sum() - vintage_count
            still_missing = aligned_series.isna().sum()

            df_aligned[col_name] = aligned_series

            first_valid = aligned_series.first_valid_index()
            report = {
                'series': col_name,
                'vintage_values': vintage_count,
                'forward_filled': filled_count,
                'still_missing': still_missing,
                'first_valid_date': first_valid
            }
            alignment_report.append(report)

            if filled_count > 0 or still_missing > 0:
                print(f'    ✓ Aligned {col_name}: {vintage_count} vintage, '
                      f'{filled_count} forward-filled, {still_missing} missing')
                if first_valid:
                    print(f'      First valid date: {first_valid.strftime("%Y-%m-%d")}')
            else:
                print(f'    ✓ Aligned {col_name} ({vintage_count}/{total_count} all from vintage)')

        except Exception as e:
            print(f'    Error aligning {col_name}: {str(e)}')
            print('    Keeping original values')
            alignment_report.append({
                'series': col_name,
                'vintage_values': 0,
                'forward_filled': 0,
                'still_missing': len(df_aligned),
                'error': str(e)
            })

    print('\n VINTAGE ALIGNMENT SUMMARY:')
    print('─' * 70)
    for report in alignment_report:
        if 'error' in report:
            print(f"  {report['series']}: ERROR - {report['error']}")
        else:
            total = len(df_aligned)
            vintage_pct = (report['vintage_values'] / total * 100) if total > 0 else 0
            filled_pct = (report['forward_filled'] / total * 100) if total > 0 else 0
            print(f"  {report['series']}:")
            print(f"    - Vintage dates: {report['vintage_values']:,} ({vintage_pct:.1f}%)")
            if report['forward_filled'] > 0:
                print(f"    - Forward-filled: {report['forward_filled']:,} ({filled_pct:.1f}%)")
            if report['still_missing'] > 0:
                print(f"    - Still missing: {report['still_missing']:,}")

    return df_aligned


def apply_fixed_shift_alignment(df):
    """
    Apply fixed-day shift approximation for release dates.

    shift forward = make data available LATER (i.e. when actually released)
    """
    print_subsection('Applying fixed-shift release date alignment')

    df['CPI'] = df['CPI'].shift(14)                    # ~2 weeks after month end
    df['Unemployment'] = df['Unemployment'].shift(7)   # first Fri of month
    df['Industrial_Production'] = df['Industrial_Production'].shift(14)
    df['Fed_Rate'] = df['Fed_Rate'].shift(7)
    df['Consumer_Sentiment'] = df['Consumer_Sentiment'].shift(2)

    print('   ✓ Release date adjustments applied')
    return df


def preprocess_data(df):
    """Clean, align and engineer basic macro/volatility features."""
    print("\nPreprocessing data...")

    # Initial quality check
    check_data_quality(df)

    # Forward-fill macro variables (monthly → daily)
    print_subsection('Forward-filling macro variables...')
    macro_vars = [
        'CPI', 'Unemployment', 'Fed_Rate',
        'Consumer_Sentiment', 'Industrial_Production'
    ]

    df_clean = df.copy()
    df_clean[macro_vars] = df_clean[macro_vars].fillna(method='ffill')

    remaining_na = df_clean[macro_vars].isnull().sum()
    if remaining_na.sum() > 0:
        print('   Warning: NaN values remain after forward-fill:')
        print(remaining_na[remaining_na > 0])
    else:
        print('   ✓ All macro variables forward-filled successfully')

    # Apply release-date alignment
    if USE_VINTAGE_DATES:
        df_clean = apply_vintage_date_alignment(df_clean, FRED_API_KEY)
    else:
        df_clean = apply_fixed_shift_alignment(df_clean)

    # Derived macro/market features (no look-ahead)
    if 'CPI' in df_clean.columns:
        # 252 trading days ≈ 1 year; uses CPI known at t and t-252
        df_clean['Inflation_YoY'] = df_clean['CPI'].pct_change(252) * 100

    if 'SP500_Returns' in df_clean.columns:
        # 20-day rolling realized volatility
        df_clean['SP500_Volatility'] = df_clean['SP500_Returns'].rolling(20).std()

    # Drop rows with NaNs introduced by shifts / rolling
    df_clean = df_clean.dropna()

    print(f"   ✓ Final dataset shape: {df_clean.shape}")
    print(f"   ✓ Date range: {df_clean.index[0]} to {df_clean.index[-1]}")
    return df_clean


def resample_to_frequency(df, freq='ME'):
    """
    Resample daily data to weekly or monthly frequency.

    - takes last value for price levels
    - recalculates returns from resampled prices
    - for monthly: applies 1-month lag to macro variables
    """
    print_subsection(f'\nResampling to {freq} frequency...')

    agg_rules = {}
    for col in df.columns:
        if 'Volume' in col:
            agg_rules[col] = 'mean'
        else:
            agg_rules[col] = 'last'

    resampled = df.resample(freq).agg(agg_rules)

    if 'SP500_Close' in resampled.columns:
        resampled['SP500_Returns'] = resampled['SP500_Close'].pct_change() * 100
    if 'CPI' in resampled.columns:
        resampled['Inflation_YoY'] = resampled['CPI'].pct_change(12) * 100
    if 'SP500_Returns' in resampled.columns:
        resampled['SP500_Volatility'] = resampled['SP500_Returns'].rolling(12).std()

    if freq in ['M', 'ME', 'MS']:
        macro_vars = ['CPI', 'Unemployment', 'Industrial_Production',
                      'Fed_Rate', 'Consumer_Sentiment']
        for var in macro_vars:
            if var in resampled.columns:
                resampled[var] = resampled[var].shift(1)

    resampled = resampled.dropna()

    print(f'   ✓ Resampled shape: {resampled.shape}')
    print(f'   ✓ Date range: {resampled.index[0]} to {resampled.index[-1]}')
    return resampled


def create_train_val_test_split(df, train_pct=0.7, val_pct=0.15):
    """
    Create temporal train/validation/test splits.
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print('\n DATA SPLITS:')
    print('─' * 70)
    print(f'Train: {train.index[0].strftime("%Y-%m-%d")} to {train.index[-1].strftime("%Y-%m-%d")} '
          f'({len(train):,} obs, {train_pct*100:.0f}%)')
    print(f'Val:   {val.index[0].strftime("%Y-%m-%d")} to {val.index[-1].strftime("%Y-%m-%d")} '
          f'({len(val):,} obs, {val_pct*100:.0f}%)')
    print(f'Test:  {test.index[0].strftime("%Y-%m-%d")} to {test.index[-1].strftime("%Y-%m-%d")} '
          f'({len(test):,} obs, {(1-train_pct-val_pct)*100:.0f}%)')

    return train, val, test


def save_data(df, filename='data/financial_dataset.csv', label=''):
    """Save the processed dataset and print summary."""
    df.to_csv(filename)

    header = f' FINAL DATA SUMMARY - {label}' if label else ' FINAL DATA SUMMARY'
    print('\n' + '=' * 70)
    print(header)
    print('=' * 70)
    print(f'\nShape: {df.shape}')
    print(f'Date range: {df.index[0]} to {df.index[-1]}')
    print(f'Years covered: {(df.index[-1] - df.index[0]).days / 365.25:.1f}')
    print(f'\nColumns: {list(df.columns)}')
    print(f'\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB')

    if 'SP500_Returns' in df.columns:
        print(f'\nTarget variable (SP500_Returns) statistics:')
        print(f'  Mean: {df["SP500_Returns"].mean():.4f}%')
        print(f'  Std: {df["SP500_Returns"].std():.4f}%')
        print(f'  Min: {df["SP500_Returns"].min():.4f}%')
        print(f'  Max: {df["SP500_Returns"].max():.4f}%')
        print(f'  Skewness: {df["SP500_Returns"].skew():.4f}')
        print(f'  Kurtosis: {df["SP500_Returns"].kurtosis():.4f}')

    total_missing = df.isnull().sum().sum()
    print(f'\nMissing values: {total_missing}')
    if total_missing > 0:
        print('\nWARNING - Missing values by column:')
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                print(f'  {col}: {missing}')

    print(f"\nSaved dataset to '{filename}'")

    return filename


def create_financial_dataset():
    """End-to-end data collection and preprocessing pipeline."""
    print_section('FINANCIAL DATA COLLECTION PIPELINE')

    try:
        # Collect all data
        market_data = collect_market_data()
        fred_data = collect_fred_data(FRED_API_KEY)
        yahoo_data = collect_yahoo_data(YAHOO_TICKERS)

        # Combine and process
        combined_data = align_and_combine_data(market_data, fred_data, yahoo_data)
        final_data = preprocess_data(combined_data)

        # Save daily version (raw)
        daily_filename = save_data(final_data, 'data/financial_dataset_daily.csv', 'DAILY')

        # Create + save monthly version
        monthly_data = resample_to_frequency(final_data, freq='ME')
        monthly_filename = save_data(monthly_data, 'data/financial_dataset_monthly.csv', 'MONTHLY')

        print("\nData processing done and saved.")
        print(f"Daily file:   {daily_filename}")
        print(f"Monthly file: {monthly_filename}")

    except Exception as e:
        print(f"\nError during data collection: {str(e)}")
        print("Please check your FRED API key and internet connection.")
