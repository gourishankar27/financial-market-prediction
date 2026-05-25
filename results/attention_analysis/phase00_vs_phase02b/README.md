# Attention Analysis: Fixed vs Vintage release dates

## Summary
- Hypothesis: Vintage release date alignment changes which timesteps receive attention
- Result: Moderate architectural impact - vintage models shift from distant-past anchoring to recency-focused attention
- Significant differences: Temporal variance (p=0.025, d=0.46) and attention strategy restructuring

## Key Findings

1. Vintage models learn a fundamentally different attention strategy
   - Fixed Release: Front-loaded attention (peak at t-20: 0.076, most distant past)
   - Vintage Release: Recency-biased attention (peak at t-1 to t-5: ~0.08, recent data)
   - Pattern shift: -0.043 attention at t-20, +0.008 to +0.013 at recent timesteps
   - Cosine similarity 0.973 is deceptive - measures angle similarity, not strategic differences

2. The attention shift is theoretically sound
   - At t-20, macroeconomic data is heavily forward-filled (stale) in vintage dataset
   - Fixed-release data has consistent freshness across all timesteps
   - Vintage models rationally learn to de-weight stale distant information
   - Emphasize fresh recent market data (VIX, yields always up-to-date)
   - This is appropriate adaptation to data characteristics

3. Vintage models show more temporal adaptation
   - Temporal variance: 0.00004 (Fixed) vs 0.00010 (Vintage), p=0.025*
   - Effect size: d=0.46 (small but significant)
   - Interpretation: Vintage attention shifts more across different market regimes
   - Models detect regime changes and adjust attention dynamically

4. Attention adaptation doesn't prevent collapse
   - Despite learning appropriate attention strategy for vintage data
   - Collapse rates remain similar between Fixed and Vintage phases
   - Directional accuracy, Sharpe ratios, prediction diversity all comparable
   - Critical insight: Attention mechanism adapts correctly, but output layer still fails

## Interpretation

Vintage release date alignment has moderate architectural impact on attention mechanisms:
- Successfully changes attention strategy to account for data staleness
- Models learn to prioritize information based on freshness (recency for vintage)
- Attention patterns are more adaptive across time periods

However despite this appropriate attention adaptation, collapse rates don't improve.

Key insight for phase 4+:
The attention mechanism is working correctly - it successfully adapts to different data characteristics. The problem is **downstream in the output layers**. This strengthens the case for:
- Output-layer modifications (regime-conditional outputs, distribution-aware regularization)
- Loss function changes that encode domain knowledge
- NOT attention mechanism modifications (attention already adapts appropriately)

The vintage comparison serves as a validation that TFT's attention can detect and respond to data quality differences, but translation to stable predictions requires architectural changes beyond attention.

## Statistical comparisons

| Metric | Fixed Release | Vintage Release | Difference | p-value | Cohen's d | Significant |
|--------|---------------|-----------------|------------|---------|-----------|-------------|
| Avg Entropy | 2.90 ± 0.43 | 2.82 ± 0.14 | -0.08 (-2.7%) | 0.335 | -0.25 | No |
| Std Entropy | 0.11 ± 0.06 | 0.09 ± 0.07 | -0.02 (-16.3%) | 0.208 | -0.28 | No |
| Avg Concentration | 0.053 ± 0.022 | 0.060 ± 0.017 | +0.007 (+12.9%) | 0.143 | 0.35 | No |
| **Temporal Variance** | **0.00004 ± 0.0001** | **0.00010 ± 0.0002** | **+0.00006 (+142.5%)** | **0.025** | **0.46** | **Yes*** |
| Top Timestep Diversity | 0.22 ± 0.12 | 0.23 ± 0.15 | +0.01 (+4.2%) | 0.752 | 0.07 | No |

*p < 0.05, small effect size

## Attention Pattern Details

### Mean Attention by Timestep

| Timestep | Fixed Release | Vintage Release | Difference |
|----------|---------------|-----------------|------------|
| t-20 (most distant) | 0.076 | 0.034 | **-0.043** |
| t-19 | 0.050 | 0.036 | -0.014 |
| t-18 | 0.045 | 0.038 | -0.007 |
| t-10 (mid-range) | 0.043 | 0.045 | +0.002 |
| t-5 (recent) | 0.050 | 0.055 | +0.005 |
| t-3 | 0.053 | 0.064 | **+0.011** |
| t-2 | 0.056 | 0.069 | **+0.013** |
| t-1 (most recent) | 0.069 | 0.082 | **+0.013** |

Pattern interpretation:
- Fixed models anchor heavily on most distant past (t-20)
- Vintage models distribute attention more evenly with strong recency bias
- Peak attention shifts from t-20 → t-1 to t-5 window
- This reflects rational response to staleness in vintage macro data

### Cosine similarity: 0.973

Why high similarity despite different strategies?
- Cosine measures angle/direction, not magnitude distribution
- Both patterns show gradual increase toward recent timesteps (similar "shape")
- But the peak location and magnitude distribution changed substantially
- Cosine similarity can be misleading for attention pattern comparisons

Better metric: L2 distance = 0.053 (captures magnitude differences)

## Visualizations

See `reports/vintage_impact_v2/`:

### Primary Figures
- `effect_sizes.png` - Cohen's d for all metrics; temporal variance highlighted as significant
- `temporal_variance_comparison.png` - Detailed violin plot showing increased variance in vintage
- `attention_patterns.png` - Key result: Side-by-side comparison showing attention strategy shift
- `comprehensive_comparison.png` - Full dashboard with all metrics

### Individual Metric Plots
- `entropy_comparison.png` - Similar entropy distributions (not significant)
- `concentration_comparison.png` - Slightly higher concentration in vintage (not significant)

Use `attention_patterns.png` to show the strategic shift from distant-past anchoring to recency focus.

## Experiments Analyzed

- Phase 0 (Fixed Release): 57 experiments from `00_baseline_exploration/`
  - Various hidden sizes (8-64), dropout rates, learning rates
  - Fixed-shift alignment: all macro data aligned to month-end regardless of actual release
  
- Phase 2b (Vintage Release): 30 experiments from `02b_vintage_sweep/`
  - Hidden sizes 10-20, dropout 0.10-0.30
  - Vintage alignment: macro data aligned to actual FRED release dates (forward-filled until update)

### Pattern Comparison Details
- Experiments with encoder_length=20: 42 (Fixed) + 30 (Vintage)
- Only experiments with matching encoder length compared for pattern analysis
- All other metrics computed across all experiments in each phase

## Reproduction

### Generate Full Analysis
```bash
python scripts/compare_phase_attention.py \
    --baseline experiments/00_baseline_exploration/ \
    --treatment experiments/02b_vintage_sweep/ \
    --baseline-label "Fixed Release" \
    --treatment-label "Vintage Release" \
    --output reports/vintage_impact_v2/
```

### Generate Specific Plots

```bash
# Only temporal variance plot (for presentation)
python scripts/compare_phase_attention.py \
    --baseline experiments/00_baseline_exploration/ \
    --treatment experiments/02b_vintage_sweep/ \
    --baseline-label "Fixed Release" \
    --treatment-label "Vintage Release" \
    --output reports/vintage_temporal/ \
    --plot-only temporal_variance

# Only attention patterns plot (key finding)
python scripts/compare_phase_attention.py \
    --baseline experiments/00_baseline_exploration/ \
    --treatment experiments/02b_vintage_sweep/ \
    --baseline-label "Fixed Release" \
    --treatment-label "Vintage Release" \
    --output reports/vintage_patterns/ \
    --plot-only patterns

# Effect sizes overview
python scripts/compare_phase_attention.py \
    --baseline experiments/00_baseline_exploration/ \
    --treatment experiments/02b_vintage_sweep/ \
    --baseline-label "Fixed Release" \
    --treatment-label "Vintage Release" \
    --output reports/vintage_effects/ \
    --plot-only effect_sizes
```

### Skip comprehensive dashboard
```bash
# Generate all individual plots but skip large comprehensive plot
python scripts/compare_phase_attention.py \
    --baseline experiments/00_baseline_exploration/ \
    --treatment experiments/02b_vintage_sweep/ \
    --baseline-label "Fixed Release" \
    --treatment-label "Vintage Release" \
    --output reports/vintage_impact_v2/ \
    --skip-comprehensive
```

## Technical Notes

### Attention extraction
- Uses `model.interpret_output()` from pytorch-forecasting
- Extracts encoder attention weights (temporal fusion layer)
- Averages across batch dimension and attention heads
- Shape: `[batch, encoder_length]` → `[encoder_length]` (mean attention per timestep)

### Statistical tests
- Independent samples t-test for group comparisons
- Cohen's d for effect size (standardized mean difference)
  - d < 0.2: negligible
  - 0.2 ≤ d < 0.5: small
  - 0.5 ≤ d < 0.8: medium
  - d ≥ 0.8: large
- Significance threshold: p < 0.05

### Metrics computed

Entropy (attention diffusion):
- Formula: `-sum(p * log(p))` where p = attention weights (normalized)
- Range: 0 (completely focused) to log(encoder_length) (uniform)
- Typical values: 2.5-3.0 for encoder_length=20
- Higher = more diffuse, lower = more focused

Concentration (peakedness):
- Formula: `sum(top_3_weights)` 
- Range: 0-1 (1 = all attention on top 3 timesteps)
- Typical values: 0.04-0.08 for diffuse attention

Temporal Variance:
- Variance of attention patterns across different time periods
- Measures how much attention strategy changes over time
- Higher = more adaptive to regime changes

Recency Bias:
- Mean attention weight on most recent 5 timesteps (t-1 to t-5)
- Range: 0-1
- Higher = stronger focus on recent data

### Pattern comparison

Cosine Similarity:
- Measures angle between attention vectors
- Range: -1 (opposite) to 1 (identical)
- Values > 0.95 typically considered "very similar"
- Limitation: Doesn't capture magnitude distribution changes

L2 Distance (Euclidean):
- Measures magnitude difference between patterns
- Range: 0 (identical) to ~1.4 (maximum for normalized vectors)
- More sensitive to distribution changes than cosine
- Values < 0.1 suggest similar patterns

## Implications for phase 4+ architecture

### What this analysis tells us

1. Attention mechanism is not the bottlenec
   - TFT attention successfully adapts to data characteristics
   - Responds appropriately to staleness cues in vintage data
   - Detects and shifts with market regime changes
   - Conclusion: Don't modify attention mechanism

2. Focus on output layer modifications
   - Attention detects regimes but predictions still collapse
   - Output layer cannot translate attention insights into stable forecasts
   - Priority targets:
     - Regime-conditional output transformations
     - Distribution-aware loss functions
     - Output regularization that encodes financial domain knowledge

3. Staleness information is being used
   - Vintage models implicitly detect staleness (de-weight t-20)
   - Don't need explicit staleness features as inputs
   - Can encode staleness through data alignment choices
   - Recommendation: Use vintage alignment + output modifications, not staleness features

4. Temporal adaptation is preserved
   - Both model types show regime-aware attention shifts
   - Vintage models are more adaptive (higher temporal variance)
   - This adaptability should be maintained in Phase 3+ modifications

### Recommended Phase 3+ Approaches

Based on this analysis:

Promising directions:
- Regime-conditional output layers that adapt to attention entropy
- Mixed-frequency attention with explicit staleness penalties
- Loss functions with temporal consistency regularization
- Multi-task learning (predict constituents + index)

Less promising:
- Modifying base attention mechanism (already works well)
- Adding raw staleness features as inputs (causes collapse in Phase 1)
- Removing temporal adaptation (models benefit from it)

## Files Reference

### comparison_report.txt
Human-readable statistical summary with:
- All metric comparisons (means, std devs, differences, p-values, effect sizes)
- Significance indicators
- Attention pattern similarity metrics
- Top timesteps with largest attention shifts

### detailed_metrics.csv
Per-experiment metrics for all experiments in both phases.

Columns:
- `experiment`: Experiment directory name
- `avg_entropy`: Mean attention entropy across time periods
- `std_entropy`: Std dev of entropy (variability over time)
- `min_entropy`, `max_entropy`: Entropy range
- `entropy_range`: max - min
- `avg_concentration`: Mean attention concentration (top-3 timesteps)
- `std_concentration`: Std dev of concentration
- `temporal_variance`: Variance of attention patterns across periods
- `top_timestep_diversity`: Fraction of unique top-attended timesteps
- `mean_attention_pattern`: Array of mean attention weights per timestep
- `phase`: "Fixed Release" or "Vintage Release"

Use for per-experiment analysis and identifying outliers.

## Caveats and Limitations

1. Encoder length filtering
   - Pattern comparison only includes experiments with encoder_length=20
   - Phase 0 had experiments with varying encoder lengths (10-60)
   - May not capture full diversity of Phase 0 attention strategies

2. Temporal aggregation
   - Analysis uses yearly groupings (2020-2025)
   - May mask shorter-term attention dynamics
   - Models might shift attention more frequently than captured

3. Statistical power
   - Phase 2b has fewer experiments (30) than Phase 0 (57)
   - Some true differences may not reach significance
   - Effect sizes should be considered alongside p-values

4. Cosine similarity interpretation
   - High cosine similarity (0.973) is somewhat misleading
   - Masks important changes in magnitude distribution
   - L2 distance and visual inspection more informative

5. Causality
   - Analysis shows association, not causation
   - Cannot prove vintage alignment *causes* attention shift
   - Could be confounded by hyperparameter differences

6. Generalization
   - Findings specific to TFT architecture
   - May not apply to other attention mechanisms
   - Results depend on S&P 500 data characteristics