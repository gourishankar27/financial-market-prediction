# S&P 500 Forecasting with Temporal Fusion Transformers

Research implementation examining whether attention-based Transformers can effectively model macro-market relationships for financial return prediction. This work focuses on the challenge of mixed-frequency data integration—combining daily market indicators (VIX, Treasury yields) with monthly macroeconomic releases (CPI, unemployment) that exhibit explicit staleness between updates.

## Key features

- **Temporal Fusion Transformer** implementation using pytorch-forecasting
- **Mixed-frequency data handling** with proper release date alignment to prevent look-ahead bias
- **Comprehensive evaluation framework** including 5-mode temporal quality classification
- **Collapse detection and monitoring** during training with automated diagnostics
- **Full experiment tracking** with reproducible configurations and checkpointing
- **Financial performance metrics** including Sharpe ratio, directional accuracy, and risk-adjusted returns

## Setup

```bash
# Clone repository
git clone https://github.com/samufrank/sp500-tft-forecasting.git
cd sp500-tft-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows (rip): venv\Scripts\activate

# Install dependencies (includes project package)
pip install -r requirements.txt
```

**Critical:** Project requires `pytorch-lightning==1.9.5` and `pytorch-forecasting==0.10.3` due to breaking API changes in Lightning 2.x. These versions are locked in `requirements.txt`.

## Quick start

```bash
# 1. Create train/val/test splits
python scripts/create_splits.py --feature-set core_proposal --frequency daily

# 2. Train a model
python train/train_tft.py --experiment-name my_experiment

# 3. Evaluate on test set (uses best checkpoint automatically)
python train/evaluate_tft.py --experiment-name my_experiment

# 4. Analyze existing experiments
python scripts/summarize_experiments.py
python scripts/list_checkpoints.py my_experiment
```

## Project structure

```
sp500-tft-forecasting/
├── src/
│   ├── feature_configs.py       # Feature set definitions and metadata
│   └── data_utils.py            # Data loading and staleness feature generation
├── train/
│   ├── train_tft.py             # Model training with collapse monitoring
│   ├── evaluate_tft.py          # Comprehensive evaluation pipeline
│   └── collapse_monitor.py      # Real-time collapse detection callback
├── scripts/
│   ├── collect_data.py          # Data collection pipeline (FRED + Yahoo Finance, supports fixed/vintage alignment)
│   ├── create_splits.py         # Generate train/val/test splits
│   ├── diagnose_existing_models.py  # Quick prediction statistics and basic collapse check
│   ├── summarize_experiments.py # Aggregate experiment results across phases
│   ├── analyze_attention_by_period.py  # Extract attention patterns
│   ├── summarize_attention_patterns.py # Attention analysis across experiments
│   └── list_checkpoints.py      # View available checkpoints
├── data/
│   ├── financial_dataset_daily.csv  # 8,913 daily observations (1990-2025)
│   └── splits/                  # Generated train/val/test splits
├── experiments/
│   ├── 00_baseline_exploration/     # Phase 0: Baseline TFT experiments
│   ├── 01_staleness_features_fixed/ # Phase 1: Staleness feature experiments
│   ├── 02_vintage_baseline/         # Phase 2: Vintage alignment validation
│   └── 03_distribution_loss/        # Phase 3: Distribution-aware loss (checkpoint failure)
└── results/
    ├── experiments_summary.csv      # Aggregated metrics across all experiments
    ├── experiments_summary_key_metrics.csv
    └── attention_analysis/          # Attention pattern analysis results
```

## Usage

### Training models

```bash
# Basic training with defaults
python train/train_tft.py --experiment-name baseline_h32

# Custom architecture
python train/train_tft.py --experiment-name large_model \
    --hidden-size 64 \
    --attention-heads 4 \
    --dropout 0.2 \
    --max-epochs 100

# Disable staleness features (for baseline comparison)
python train/train_tft.py --experiment-name no_staleness_h16 \
    --hidden-size 16 \
    --no-staleness

# Quick test run
python train/train_tft.py --experiment-name quick_test \
    --hidden-size 16 \
    --max-epochs 5
```

Common arguments:
- `--experiment-name` (required) - Unique identifier for experiment outputs
- `--hidden-size` - Model capacity (16/32/64, default: 32)
- `--attention-heads` - Number of attention heads (1/2/4, default: 2)
- `--dropout` - Dropout rate (default: 0.15)
- `--max-encoder-length` - Historical lookback window (default: 20)
- `--max-epochs` - Training epochs (default: 50)
- `--no-staleness` - Disable staleness features
- `--seed` - Random seed for reproducibility (default: 42)
- `--overwrite` - Allow overwriting existing experiment directory

See `python train/train_tft.py --help` for complete options.

Training outputs are saved to `experiments/{name}/`:
- `config.json` - Complete hyperparameter configuration
- `final_metrics.json` - Best validation loss and training metadata
- `checkpoints/` - Model checkpoints (best 3 + last)
- `collapse_monitor/` - Training dynamics tracking (prediction diversity, gradient flow, attention entropy per epoch)
- `evaluation/` - Test set evaluation outputs (if evaluated)
- `attention_analysis_year/` - Attention patterns (if analyzed)

### Evaluating models

```bash
# Automatic checkpoint selection (uses best validation loss)
python train/evaluate_tft.py --experiment-name baseline_h32

# Phase-organized experiment
python train/evaluate_tft.py --experiment-name 00_baseline_exploration/sweep2_h16_drop_0.2

# Manual checkpoint specification
python train/evaluate_tft.py \
    --experiment-name my_experiment \
    --checkpoint experiments/my_experiment/checkpoints/tft-epoch=05-val_loss=0.1234.ckpt

# Custom test split (for fixed vs vintage date comparison)
python train/evaluate_tft.py \
    --experiment-name my_experiment \
    --test-split data/splits/custom_test.csv
```

Evaluation outputs are saved to `experiments/{name}/evaluation/`:
- `evaluation_metrics.json` - Complete metrics in structured format
- `predictions.csv` - Actual vs predicted returns with quality classification
- `diagnostic_plots.png` - Statistical diagnostics (residuals, Q-Q plot, autocorrelation, distribution)
- `performance_plots.png` - Financial metrics (cumulative returns, rolling Sharpe)
- `quality_diagnostics.png` - Temporal quality classification and rolling accuracy

Metrics computed:

Statistical: MSE, RMSE, MAE, R²

Classification: Directional accuracy, AUC-ROC, precision/recall, F1

Financial: Sharpe ratio, total return, max drawdown, alpha

Quality: 5-mode temporal classification (HEALTHY/DEGRADED/UNIDIRECTIONAL/WEAK_COLLAPSE/STRONG_COLLAPSE)

### Analyzing results

```bash
# Generate summary of all experiments
python scripts/summarize_experiments.py
# Creates: results/experiments_summary.csv, results/experiments_summary_key_metrics.csv

# Quick diagnostics on existing experiments (prediction statistics, basic collapse detection)
python scripts/diagnose_existing_models.py sweep2_h16_drop_0.2
python scripts/diagnose_existing_models.py --compare sweep2_*

# Analyze attention patterns for specific experiment
python scripts/analyze_attention_by_period.py \
    --experiment 00_baseline_exploration/sweep2_h16_drop_0.2

# Summarize attention patterns across multiple experiments
python scripts/summarize_attention_patterns.py experiments/00_baseline_exploration/
python scripts/summarize_attention_patterns.py \
    experiments/00_baseline_exploration/ \
    experiments/01_staleness_features_fixed/
```

## Features and data

Core feature set (core_proposal):
1. **Lagged S&P 500 Returns** - Momentum and mean reversion
2. **VIX** - Market volatility (daily updates)
3. **10-year Treasury Yield** - Risk-free rate proxy (daily updates)
4. **Term Spread (10Y-2Y)** - Yield curve slope (daily updates)
5. **CPI Year-over-Year** - Inflation (monthly updates, ~14 day release lag)

Staleness features (automatically generated for low-frequency variables):
- `days_since_{feature}_update` - Continuous staleness indicator (0, 1, 2, ..., 30)
- `{feature}_is_fresh` - Binary indicator (1 if updated today, 0 otherwise)

Data characteristics:
- 8,913 daily observations from January 1990 to October 2025
- Mixed-frequency alignment with forward-filling between releases
- Vintage release dates from ALFRED database for temporal alignment accuracy
- Train/val/test split: 70% / 15% / 15% (temporal ordering preserved)

## Model Evaluation Framework

### 5-Mode quality classification

Models are evaluated with temporal quality classification across five modes:

- **HEALTHY**: Predictions vary appropriately with strong directional accuracy (>52%) and positive correlation
- **DEGRADED**: Predictions vary but show poor quality (directional accuracy <48% or negative correlation)
- **UNIDIRECTIONAL**: Predictions vary but show extreme directional bias (>98% same sign)
- **WEAK_COLLAPSE**: Reduced prediction variation (2/3 structural methods detect collapse)
- **STRONG_COLLAPSE**: Near-constant predictions (3/3 structural methods detect collapse)

Detection methods:
- Structural: Variance threshold, range check, consecutive-similarity analysis
- Quality: Rolling 60-day correlation and directional accuracy
- Unidirectional: Directional bias >98% threshold

This framework addresses the multi-dimensional nature of collapse in financial forecasting, where models can pass variance checks while still being fundamentally broken (anticorrelated predictions, unidirectional bias).

## Reproducibility

All experiments are fully reproducible:
- Fixed random seeds with deterministic CUDA operations
- Temporal (not random) data splits maintaining time-series structure
- Complete configuration logging with experiment metadata
- Version-locked dependencies (`pytorch-lightning==1.9.5`, `pytorch-forecasting==0.10.3`)

To reproduce an experiment:
```bash
# View configuration
cat experiments/{experiment_name}/config.json

# Recreate with same settings
python train/train_tft.py --experiment-name {new_name} [args from config]
```

## Current status

Completed:

Phase 0: Baseline TFT characterization (57 experiments, retrained with proper validation)
- Best baseline: sweep2_h16_drop_0.1, 53.6% directional accuracy, 0.529 AUC-ROC, 0.879 Sharpe
- Tight hyperparameter convergence: val_loss 0.392-0.403 across all configurations
- Small models (hidden_size=16) with moderate dropout (0.1-0.25) perform best

Phase 1: Staleness feature exploration (64 experiments, retrained)
- Comprehensive negative result: 60/64 experiments (94%) collapsed with unidirectional predictions
- Log-transformed staleness tested: collapse persists (100% positive predictions)
- Root cause: architectural incompatibility, not just scale mismatch

Phase 2: Vintage alignment validation (6 experiments)
- Vintage (realistic release dates) improves baseline quality vs fixed alignment
- Reduced unidirectional behavior: 23-43% (vintage) vs 36-72% (fixed)
- Increased healthy predictions: 50-62% (vintage) vs 21-48% (fixed)
- Staleness features still collapse on vintage (confirms architectural issue)

Phase 3: Distribution-aware loss (9 experiments trained, unevaluable)
- Attempted anti-collapse and anti-drift penalties via monkey-patching
- Training successful but checkpoints corrupted by pickle serialization incompatibility
- Demonstrated fundamental limitation of runtime patching with PyTorch checkpoints

Attention pattern analysis (121 experiments, post-retraining)
- 423 attention shifts detected, 114 at 2022->2023 Fed pivot
- Weak correlations (|r| < 0.15) confirm attention works but output layer unstable
- Models detect regime changes but cannot adapt predictions appropriately

### Next:
- Custom TFT implementation from scratch for full architectural control
- Native distribution-aware loss (mean/std penalties) without serialization issues
- Staleness-aware attention mechanisms to handle mixed-frequency data
- Regime-conditional attention heads for volatility-dependent processing


## Known issues / limitations

Version compatibility: Project requires old versions of PyTorch Lightning (1.9.5) due to breaking changes in 2.x. The `pytorch-forecasting` library is in maintenance mode but stable with this configuration. See troubleshooting documentation for details.

Model collapse: TFT models exhibit capacity-dependent collapse in financial forecasting tasks. Models with hidden_size 16-18 produce varied predictions, while larger models collapse to constant predictions. This is a novel finding not reported in the original TFT literature and represents a fundamental challenge for applying Transformers to extremely noisy financial data.

Financial forecasting context: R² near zero is expected and not indicative of model failure. Daily S&P 500 returns have a 30:1 noise-to-signal ratio. Directional accuracy of 52-54% represents strong performance (random walk baseline is 50%, top quantitative hedge funds operate at 52-55%). Lead with Sharpe ratio and cumulative returns when evaluating model quality.

Checkpoint serialization with monkey-patching: Phase 3 demonstrated that runtime method patching (monkey-patching) is fundamentally incompatible with PyTorch checkpoint serialization. While models train successfully, checkpoints cannot be loaded due to pickle attempting to restore runtime-added attributes on fresh class instances. This limitation necessitates custom TFT implementation for any non-trivial architectural modifications.

## References

- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." *International Journal of Forecasting*.
- Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*.
- Fischer, T., & Krauss, C. (2018). "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*.

## License

Research implementation for educational and academic purposes.
