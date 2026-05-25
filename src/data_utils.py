"""
Utilities for loading and preparing datasets for modeling.
"""

import pandas as pd
from src.feature_configs import FEATURE_SETS, FEATURE_METADATA, TARGET

def load_feature_set(config_name='core_proposal', 
                     frequency='daily',
                     version='fixed',
                     enhanced=False,
                     data_path='data',
                     verbose=True):
    """
    Load a specific feature set configuration.
    
    Parameters:
    -----------
    config_name : str
        Name of feature set from FEATURE_SETS
    frequency : str
        'daily', 'weekly', or 'monthly'
    version : str
        'fixed' or 'vintage'
    enhanced : bool
        If True, load enhanced dataset with technical features
    data_path : str
        Path to data directory
    verbose : bool
        Print loading information
        
    Returns:
    --------
    pd.DataFrame
        Dataset with selected features and date filtering applied
    """
    # Validate config
    if config_name not in FEATURE_SETS:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(FEATURE_SETS.keys())}")
    
    # Validate frequency
    valid_frequencies = ['daily', 'weekly', 'monthly']
    if frequency not in valid_frequencies:
        raise ValueError(f"Unknown frequency: {frequency}. "
                        f"Available: {valid_frequencies}")
    
    config = FEATURE_SETS[config_name]
    
    # Load full dataset
    if enhanced:
        filename = f"{data_path}/financial_dataset_{frequency}_{version}_enhanced.csv"
    else:
        filename = f"{data_path}/financial_dataset_{frequency}_{version}.csv"
    df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading Feature Set: {config_name}")
        print(f"{'='*70}")
        print(f"Description: {config['description']}")
    
    # Select features
    if config['features'] == 'all':
        # Kitchen sink: use everything except derived features we'll recalculate
        selected = df.copy()
    else:
        # Start with requested features
        features_to_load = config['features'].copy()
        
        # Add source columns needed for staleness computation
        from src.feature_configs import FEATURE_METADATA
        for feature in config['features']:
            if feature in FEATURE_METADATA:
                metadata = FEATURE_METADATA[feature]
                if metadata.get('needs_staleness', False):
                    source_col = metadata.get('source_column', feature)
                    if source_col != feature and source_col not in features_to_load:
                        features_to_load.append(source_col)
                        if verbose:
                            print(f"Including source column '{source_col}' for staleness computation of '{feature}'")
        
        selected = df[features_to_load].copy()
    
    # Apply date filtering if specified
    if config['min_date'] is not None:
        original_len = len(selected)
        selected = selected[selected.index >= config['min_date']]
        if verbose:
            print(f"Date filter: >={config['min_date']}")
            print(f"  Samples before: {original_len}")
            print(f"  Samples after: {len(selected)}")
    
    # Remove any remaining NaNs
    original_len = len(selected)
    selected = selected.dropna()
    
    if verbose:
        print(f"\nFinal dataset:")
        print(f"  Shape: {selected.shape}")
        print(f"  Date range: {selected.index[0].date()} to {selected.index[-1].date()}")
        print(f"  Years: {(selected.index[-1] - selected.index[0]).days / 365.25:.1f}")
        print(f"  Features: {list(selected.columns)}")
        if original_len != len(selected):
            print(f"  Rows dropped (NaN): {original_len - len(selected)}")
    
    return selected


def create_train_val_test_split(df, train_pct=0.7, val_pct=0.15, verbose=True):
    """
    Create temporal train/validation/test splits by percentage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with DatetimeIndex
    train_pct : float
        Proportion for training (default 0.7)
    val_pct : float
        Proportion for validation (default 0.15)
    verbose : bool
        Print split information
        
    Returns:
    --------
    train, val, test : tuple of pd.DataFrame
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    if verbose:
        print(f"\n{'='*70}")
        print("Train/Validation/Test Split")
        print(f"{'='*70}")
        print(f"Train: {train.index[0].date()} to {train.index[-1].date()} "
              f"({len(train):,} obs, {train_pct*100:.0f}%)")
        print(f"Val:   {val.index[0].date()} to {val.index[-1].date()} "
              f"({len(val):,} obs, {val_pct*100:.0f}%)")
        print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} "
              f"({len(test):,} obs, {(1-train_pct-val_pct)*100:.0f}%)")
    
    return train, val, test


def create_split_by_dates(df, train_start=None, train_end=None, val_end=None, 
                          test_start=None, test_end=None, verbose=True):
    """
    Create temporal train/validation/test splits by date boundaries.
    
    For rolling/walk-forward evaluation. Dates are inclusive.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with DatetimeIndex
    train_start : str or None
        Training start date (YYYY-MM-DD). None = start of data.
    train_end : str
        Training end date (YYYY-MM-DD). Required.
    val_end : str or None
        Validation end date. None = no validation split (val will be empty).
    test_start : str or None
        Test start date. None = day after val_end (or train_end if no val).
    test_end : str or None
        Test end date. None = end of data.
    verbose : bool
        Print split information
        
    Returns:
    --------
    train, val, test : tuple of pd.DataFrame
        Note: val may be empty DataFrame if val_end not specified
    """
    if train_end is None:
        raise ValueError("train_end is required for date-based splitting")
    
    # Convert to timestamps
    train_start = pd.Timestamp(train_start) if train_start else df.index[0]
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end) if val_end else None
    test_start = pd.Timestamp(test_start) if test_start else None
    test_end = pd.Timestamp(test_end) if test_end else df.index[-1]
    
    # Create splits
    train = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    
    if val_end:
        # Validation: from day after train_end to val_end
        val = df[(df.index > train_end) & (df.index <= val_end)].copy()
        # Test: from test_start (or day after val_end) to test_end
        if test_start is None:
            test_start = val_end + pd.Timedelta(days=1)
        test = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    else:
        # No validation split
        val = df.iloc[0:0].copy()  # Empty DataFrame with same columns
        # Test: from day after train_end to test_end
        if test_start is None:
            test_start = train_end + pd.Timedelta(days=1)
        test = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    
    if verbose:
        print(f"\n{'='*70}")
        print("Train/Validation/Test Split (Date-Based)")
        print(f"{'='*70}")
        if len(train) > 0:
            print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train):,} obs)")
        else:
            print(f"Train: EMPTY")
        if len(val) > 0:
            print(f"Val:   {val.index[0].date()} to {val.index[-1].date()} ({len(val):,} obs)")
        else:
            print(f"Val:   EMPTY (no validation split)")
        if len(test) > 0:
            print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test):,} obs)")
        else:
            print(f"Test:  EMPTY")
    
    return train, val, test


def add_staleness_features(df, use_vintage=False, verbose=True):
    """
    Add staleness indicators for low-frequency features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and financial features
    use_vintage : bool
        If True, compute staleness from actual release dates (vintage)
        If False, use typical lag patterns (fixed)
    verbose : bool
        Print information about staleness computation
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with additional staleness features
    """
    
    df = df.copy()
    
    if verbose:
        print(f"\n{'='*70}")
        print("Adding Staleness Features")
        print(f"{'='*70}")
        print(f"Mode: {'Vintage (ALFRED)' if use_vintage else 'Fixed lag'}")
    
    for feature, metadata in FEATURE_METADATA.items():
        if not metadata['needs_staleness']:
            continue
            
        if feature not in df.columns:
            continue
       
        # Use source_column if specified, otherwise use feature itself
        source_col = metadata.get('source_column', feature)
        
        if source_col not in df.columns:
            if verbose:
                print(f"\nSkipping {feature}: source column {source_col} not found")
            continue
        
        if verbose:
            print(f"\nProcessing: {feature} (detecting from {source_col})")
        
        # Detect updates from source column
        feature_values = df[source_col]
        
        # Detect updates: value changed from previous day
        # updates = (feature_values != feature_values.shift(1))
        # updates = (feature_values.diff().abs() > 1e-6)  # Only count changes > threshold

        # For CPI/inflation, changes should be substantial (at least 0.01%)
        updates = (feature_values.diff().abs() > 0.01)

        # For first observation, assume it's an update
        updates.at[updates.index[0]] = True
        
        # Days since last update
        days_since_update = pd.Series(0, index=df.index)
        last_update_idx = 0
        
        for i in range(len(df)):
            if updates.iloc[i]:
                last_update_idx = i
                days_since_update.iloc[i] = 0
            else:
                days_since_update.iloc[i] = i - last_update_idx
        
        # Binary freshness indicator
        is_fresh = updates.astype(int)
        
        # Add to dataframe with appropriate names
        staleness_col_name = metadata['staleness_features'][0]  # e.g., 'days_since_CPI_update'
        freshness_col_name = metadata['staleness_features'][1]  # e.g., 'CPI_is_fresh'
        
        df[staleness_col_name] = days_since_update
        df[freshness_col_name] = is_fresh
        
        if verbose:
            num_updates = is_fresh.sum()
            avg_staleness = days_since_update.mean()
            max_staleness = days_since_update.max()
            print(f"  Updates detected: {num_updates}")
            print(f"  Avg days stale: {avg_staleness:.1f}")
            print(f"  Max days stale: {max_staleness}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Total staleness features added: {len([c for c in df.columns if 'days_since' in c or 'is_fresh' in c])}")
        print(f"{'='*70}\n")
    
    return df



if __name__ == "__main__":
    print("Testing feature set loading...\n")
    
    # Test each config
    for config_name in FEATURE_SETS.keys():
        data = load_feature_set(config_name, frequency='monthly')
        train, val, test = create_train_val_test_split(data, verbose=False)
        print(f"\n{config_name}: {len(data)} total samples "
              f"(train={len(train)}, val={len(val)}, test={len(test)})")