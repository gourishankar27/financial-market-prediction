# VSN (Variable Selection Network) analysis

## Summary

Extracts and analyzes feature attention weights from TFT's Variable Selection Network across market regimes. TFT uses two complementary attention mechanisms: temporal attention (over timesteps) and feature attention via VSN (over input variables). This analysis covers the latter - which features the model selects at each timestep.

## Key findings

1. VIX dominates feature selection, especially during volatility
   - 2022 peak: 0.676 weight (Fed uncertainty, inflation shock)
   - 2020-2021: ~0.52-0.55 (elevated but not extreme)
   - 2024-2025: ~0.43 (normalizing as markets stabilize)

2. Feature importance shifts with macro regime
   - Yield_Spread rises dramatically post-2023: 0.07 → 0.28
   - Reflects model learning yield curve dynamics matter after rate hikes
   - Inflation_YoY stays consistently relevant (0.11-0.21)
   - Treasury_10Y weight drops as rates stabilize

3. Concentration spikes during crisis periods
   - 2022: Herfindahl index peaks at 0.56 (custom_losses) / 0.44 (vintage_sweep)
   - Model focuses hard on fewer features during uncertainty
   - Post-2023: concentration drops to 0.31-0.39 (more balanced feature use)

4. Feature selection shifts align with macro events
   - 2021->2022, 2022->2023, 2023->2024: 70+ experiments show shifts (min-signals=2)
   - 2020->2021: only 15 shifts (stable recovery period)
   - 2024->2025: only 9 shifts (markets stabilized)

## Method

### Feature attention weight extraction

VSN implements feature attention via softmax-weighted combination of input variables. Weights extracted via forward pass with `return_attention=True`:
```python
output = model(batch, return_attention=True)
encoder_weights = output.encoder_variables  # Shape: [batch, seq_len, 1, n_features]
```

Weights represent soft feature selection at each timestep - higher weight means feature contributes more to the processed representation.

### Metrics

Concentration (Herfindahl Index):
- Formula: `sum(w_i^2)` where w_i = normalized feature weights
- Range: 1/n (uniform) to 1.0 (single feature dominates)
- Higher = model focuses on fewer features

Feature Stability:
- Fraction of periods where same feature ranked #1
- Range: 0-1
- Higher = more consistent feature selection across regimes

### Shift detection signals

A regime shift is detected when comparing consecutive periods if:
- Cosine similarity < 0.95 (weight vector direction changed)
- L2 distance > 0.03 (weight magnitudes changed)
- Concentration change > 0.05 (focus breadth changed)
- Top feature changed (different feature became most important)

Default requires 1+ signals; use `--min-signals 2` for stricter detection.

## Note on relative_time_idx

`relative_time_idx` appears in VSN analysis with low weight (~0.03-0.06). This is a pytorch-forecasting implementation detail, not a domain feature. When `add_relative_time_idx=True`, the library adds an explicit position indicator to each sequence. The original TFT paper relies on LSTM dynamics for implicit position encoding; pytorch-forecasting provides this as an optional explicit signal. Low VSN weight confirms the model doesn't heavily rely on it.

## Note on target variable

pytorch-forecasting's TFT separates the target variable (SP500_Returns) into a privileged `encoder_target` pathway that bypasses VSN entirely. This differs from the original paper where lagged target received VSN weights of 0.30-0.70. Our analysis covers exogenous features only (VIX, Treasury_10Y, Yield_Spread, Inflation_YoY). The model still uses lagged returns - they're just not subject to variable selection.

## Scripts

### Single experiment analysis
```bash
python scripts/analyze_vsn_weights.py --experiment 02b_vintage_sweep/baseline_h16_drop0.10
```

Outputs to `experiments/{phase}/{exp}/vsn_analysis/`:
- `vsn_analysis_results.json` - raw weights and metrics
- `vsn_heatmap.png` - feature importance by period
- `vsn_comparison.png` - side-by-side period comparison
- `vsn_concentration.png` - concentration over time

### Batch phase analysis
```bash
python scripts/analyze_vsn_weights.py --phase 02b_vintage_sweep --continue-on-error
```

### Cross-Experiment summary
```bash
python scripts/summarize_vsn_patterns.py experiments/02b_vintage_sweep experiments/04_custom_losses
```

Outputs to `reports/`:
- `vsn_summary.csv` - per-experiment summary stats
- `vsn_feature_importance.csv` - cross-experiment feature rankings
- `vsn_regime_shifts.csv` - detected feature selection changes
- `vsn_summary_report.txt` - human-readable analysis
- `vsn_feature_importance_heatmap.png` - feature weights by period (averaged)
- `vsn_concentration_timeline.png` - concentration over time by phase
- `vsn_concentration_by_phase.png` - phase comparison (if multiple phases)
- `vsn_shift_frequency.png` - regime shifts by period transition

## Comparison: Temporal vs Feature Attention

TFT employs two attention mechanisms that serve complementary roles:

| Aspect | Temporal Attention | Feature Attention (VSN) |
|--------|-------------------|------------------------|
| Question answered | When to look? | What to use? |
| Attends over | Timesteps in encoder window | Input variables |
| Core metric | Entropy (temporal diffusion) | Concentration (feature focus) |
| Shift detection | Timestep weight changes | Feature weight changes |
| Regime sensitivity | 2022→2023 peak (102 shifts) | 2021→2024 sustained (70+ shifts) |

Feature attention stayed volatile longer than temporal attention. The model kept reweighting *which* features matter even after settling on *when* to look.

## Limitations

1. Target excluded from VSN - cannot compare lagged returns vs exogenous feature importance
2. Yearly aggregation may mask shorter-term dynamics
3. Weights are instance-level but averaged for analysis - individual sample variation lost
4. Cosine similarity can mask magnitude changes (L2 distance more sensitive)