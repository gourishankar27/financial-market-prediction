# Data pipeline

## Constituent data

Constituent returns collected separately from main macro data. Joined automatically when using `multitask_core` feature set.

### files
- `constituents_daily_vintage_2005.csv` - 43 S&P 500 stocks, 2005-2025
- `constituents_daily_vintage_1990.csv` - 32 stocks with full 1990-2025 history
- `splits/vintage/multitask_core_daily_vintage_*.csv` - pre-joined splits (49 columns)

### regenerating
```bash
# constituents
python scripts/collect_constituents.py --start-date 2005-01-01 --end-date 2025-10-06 --top-n 50 --output-dir data/ --version vintage_2005

# splits (auto-joins constituents)
python scripts/create_splits.py --feature-set multitask_core --frequency daily --data-version vintage
```

### Using in training

**Option A: constituents as features** (no architecture change)
- Load `multitask_core` splits
- Constituent columns are extra inputs
- Still predict `SP500_Returns` only

**Option B: multi-head prediction** (custom architecture)
- Predict `SP500_Returns` + constituent returns jointly
- Check `multitask_core_daily_vintage_metadata.json` for `auxiliary_targets` list
- Requires modifying TFT output layer

**Option C: single-stock prediction**
- `predict_AAPL` in `feature_configs.py` is an example config
- Copy and modify for other tickers (swap `AAPL_Returns` for `MSFT_Returns`, etc.)
- `train_tft.py` needs to use `get_target(feature_set)` instead of hardcoded target

```bash
# create single-stock splits (using AAPL example)
python scripts/create_splits.py --feature-set predict_AAPL --frequency daily --data-version vintage
```

### key ufnctions
```python
from src.feature_configs import get_target, get_all_targets

get_target('multitask_core')      # Returns 'SP500_Returns'
get_target('predict_AAPL')        # Returns 'AAPL_Returns'
get_all_targets('multitask_core') # Returns {'primary': ..., 'auxiliary': [...]}
```

### note
- `train_tft.py` currently hardcodes `SP500_Returns` - needs update to use `get_target()`
- weekly/monthly constituents not generated (daily-only for now, use `--resample-weekly` flag if needed)