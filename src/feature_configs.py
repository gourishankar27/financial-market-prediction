"""
Feature set configurations for different modeling experiments.

Each config specifies:
- features: list of column names to use
- description: what this feature set represents
- min_date: earliest date with all features available (None = use all)
- include_constituents: (optional) whether to join constituent returns for multi-task
- constituent_count: (optional) how many constituents to include (25, 50, 75, 100)
"""

FEATURE_SETS = {
    'core_proposal': {
        'features': [
            'SP500_Returns',      # Target (will be lagged for prediction)
            'VIX',                # Market volatility
            'Treasury_10Y',       # Long-term rate
            'Yield_Spread',       # Yield curve slope (10Y-2Y)
            'Inflation_YoY',      # CPI year-over-year
        ],
        'description': 'Original 5 features from proposal',
        'min_date': None,  # Available since 1990
    },
    
    'core_plus_credit': {
        'features': [
            'SP500_Returns',
            'VIX',
            'Treasury_10Y',
            'Yield_Spread',
            'Inflation_YoY',
            'Credit_HY',          # High yield spread
            'Credit_IG',          # Investment grade spread
        ],
        'description': 'Core + credit risk indicators',
        'min_date': '1997-01-01',  # Credit spreads start 1997
    },
    
    'macro_heavy': {
        'features': [
            'SP500_Returns',
            'VIX',
            'Inflation_YoY',
            'Unemployment',
            'Fed_Rate',
            'Consumer_Sentiment',
            'Industrial_Production',
        ],
        'description': 'Emphasis on macroeconomic fundamentals',
        'min_date': None,
    },
    
    'market_only': {
        'features': [
            'SP500_Returns',
            'VIX',
            'Treasury_10Y',
            'Yield_Spread',
            'SP500_Volatility',
        ],
        'description': 'Pure market-based features, no macro releases',
        'min_date': None,
    },
    
    'kitchen_sink': {
        'features': 'all',  # Special case: use everything
        'description': 'All available features',
        'min_date': '1997-01-01',  # Most restrictive constraint
    },
    
    # =========================================================================
    # Multi-task feature sets (include constituent returns as auxiliary targets)
    # =========================================================================
    
    'multitask_core': {
        'features': [
            'SP500_Returns',
            'VIX',
            'Treasury_10Y',
            'Yield_Spread',
            'Inflation_YoY',
        ],
        'include_constituents': True,
        'constituent_count': 50,  # Top 50 by market cap
        'constituent_version': 'vintage_2005',  # Which constituent file to use
        'description': 'Core features + top 50 constituent returns for multi-task learning',
        'min_date': None,
    },
    
    # =========================================================================
    # Single-stock prediction (transfer learning targets)
    # =========================================================================
    
    # as an example
    'predict_AAPL': {
        'features': [
            'AAPL_Returns',  # Target
            'SP500_Returns', # Index as feature
            'VIX',
            'Treasury_10Y',
            'Yield_Spread',
            'Inflation_YoY',
        ],
        'target': 'AAPL_Returns',  # Override default SP500_Returns
        'include_constituents': False,
        'description': 'Predict AAPL using macro features + index',
        'min_date': '2005-01-01',  # AAPL data reliable from here
    },
}

# Define default target variable (can be overridden per feature set)
TARGET = 'SP500_Returns'

# Define which features are lagged returns (for proper temporal splits)
AUTOREGRESSIVE_FEATURES = ['SP500_Returns']


def get_target(config_name: str) -> str:
    """
    Get target variable for a feature set.
    
    Parameters:
    -----------
    config_name : str
        Name of feature set from FEATURE_SETS
        
    Returns:
    --------
    str
        Target column name (default: SP500_Returns, or override if specified)
    """
    config = FEATURE_SETS.get(config_name, {})
    return config.get('target', TARGET)

# Constituent configuration
CONSTITUENT_CONFIG = {
    # Ordered by market cap (highest first)
    # Used to select top-N for multi-task learning
    'tickers': [
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
    ],
    'column_suffix': '_Returns',  # e.g., AAPL_Returns
}


def get_constituent_columns(count: int = 50) -> list:
    """
    Get column names for top-N constituents.
    
    Parameters:
    -----------
    count : int
        Number of top constituents (by market cap) to include.
        
    Returns:
    --------
    list
        Column names like ['AAPL_Returns', 'MSFT_Returns', ...]
    """
    tickers = CONSTITUENT_CONFIG['tickers'][:count]
    suffix = CONSTITUENT_CONFIG['column_suffix']
    return [f"{ticker}{suffix}" for ticker in tickers]


def get_all_targets(config_name: str) -> dict:
    """
    Get all target columns for a feature set (primary + auxiliary).
    
    Parameters:
    -----------
    config_name : str
        Name of feature set from FEATURE_SETS
        
    Returns:
    --------
    dict with keys:
        'primary': str - main target (SP500_Returns)
        'auxiliary': list - constituent return columns (if multi-task)
        'all': list - all targets combined
    """
    config = FEATURE_SETS.get(config_name, {})
    
    primary = TARGET
    auxiliary = []
    
    if config.get('include_constituents', False):
        count = config.get('constituent_count', 50)
        auxiliary = get_constituent_columns(count)
    
    return {
        'primary': primary,
        'auxiliary': auxiliary,
        'all': [primary] + auxiliary,
    }


# To create staleness features
FEATURE_METADATA = {
    # High-frequency features (daily updates, no staleness needed)
    'VIX': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    'Treasury_10Y': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    'Yield_Spread': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    'SP500_Returns': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    'SP500_Volatility': {
        'update_frequency': 'daily',
        'needs_staleness': False,
    },
    
    # Low-frequency features (monthly updates with lag, need staleness)
    'Inflation_YoY': {
        'update_frequency': 'monthly',
        'needs_staleness': True,
        'typical_lag_days': 14,  # CPI released ~2 weeks after month end
        'staleness_features': ['days_since_CPI_update', 'CPI_is_fresh'],
        'source_column': 'CPI',  # use CPI for staleness since Inflation_YoY updates daily from rolling window
    },
    'Unemployment': {
        'update_frequency': 'monthly',
        'needs_staleness': True,
        'typical_lag_days': 7,  # Jobs report released ~1 week after month end
        'staleness_features': ['days_since_unemployment_update', 'unemployment_is_fresh'],
    },
    'Fed_Rate': {
        'update_frequency': 'irregular',  # FOMC meetings ~8x per year
        'needs_staleness': True,
        'typical_lag_days': 0,  # Immediate release
        'staleness_features': ['days_since_fed_update', 'fed_is_fresh'],
    },
    'Consumer_Sentiment': {
        'update_frequency': 'monthly',
        'needs_staleness': True,
        'typical_lag_days': 0,  # Michigan survey, immediate release
        'staleness_features': ['days_since_sentiment_update', 'sentiment_is_fresh'],
    },
    'Industrial_Production': {
        'update_frequency': 'monthly',
        'needs_staleness': True,
        'typical_lag_days': 14,
        'staleness_features': ['days_since_indprod_update', 'indprod_is_fresh'],
    },
}


def get_staleness_features(feature_list):
    """
    Given a list of features, return staleness features that should be added.

    Parameters:
    -----------
    feature_list : list
        List of feature names (e.g., ['VIX', 'Inflation_YoY', ...])

    Returns:
    --------
    dict with keys:
        'staleness_features': list of staleness feature names to add
        'needs_staleness': dict mapping original features to bool
    """
    staleness_features = []
    needs_staleness = {}

    for feature in feature_list:
        if feature in FEATURE_METADATA:
            metadata = FEATURE_METADATA[feature]
            needs_staleness[feature] = metadata['needs_staleness']

            if metadata['needs_staleness']:
                staleness_features.extend(metadata['staleness_features'])

    return {
        'staleness_features': staleness_features,
        'needs_staleness': needs_staleness,
    }