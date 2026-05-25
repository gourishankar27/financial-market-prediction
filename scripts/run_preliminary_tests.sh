#!/bin/bash
# run_preliminary_tests.sh
# Quick 30-epoch tests to determine which factors matter before full overnight sweep
#
# Tests:
# 1. Quantile count: 7q vs 3q vs median (h=1)
# 2. Horizon length: h1 vs h3 vs h5 (q=3q)
#
# Runtime estimate: ~40-50 min total (weekly only, 30 epochs each)

set -e

# === WEEKLY BASELINE (from 06a_weekly_sweep/h16_enc12_d015_bs16) ===
WEEKLY_ARGS="
    --frequency weekly
    --feature-set core_proposal
    --alignment vintage
    --hidden-size 16
    --max-encoder-length 12
    --dropout 0.15
    --batch-size 16
    --learning-rate 0.0005
    --max-epochs 30
    --early-stop-patience 15
    --dist-loss-std-weight 0.0
    --directional-weight 0.0
    --temporal-consistency-weight 0.0
    --seed 42
    --overwrite
"

# === DAILY BASELINE (from 02b_vintage_sweep/baseline_h16_drop0.10) ===
DAILY_ARGS="
    --frequency daily
    --feature-set core_proposal
    --alignment vintage
    --hidden-size 16
    --max-encoder-length 20
    --dropout 0.1
    --batch-size 64
    --learning-rate 0.0005
    --max-epochs 30
    --early-stop-patience 10
    --dist-loss-std-weight 0.0
    --directional-weight 0.0
    --temporal-consistency-weight 0.0
    --seed 42
    --overwrite
"

# Output directory
PHASE="09_preliminary"

# Function to run train + eval
run_experiment() {
    local exp_name=$1
    local base_args=$2
    shift 2
    local extra_args="$@"
    
    echo "=============================================="
    echo "RUNNING: $exp_name"
    echo "=============================================="
    
    # Train
    python train/train_tft.py \
        --experiment-name "${PHASE}/${exp_name}" \
        $base_args \
        $extra_args
    
    # Evaluate top 2 checkpoints per metric
    echo ""
    echo "Evaluating checkpoints for ${exp_name}..."
    python train/evaluate_checkpoints.py \
        "experiments/${PHASE}/${exp_name}" \
        --top-per-metric 2 
    
    echo ""
    echo "Completed: $exp_name"
    echo ""
}

# Print plan
echo "=============================================="
echo "PRELIMINARY TESTS - Quick 30-epoch runs"
echo "=============================================="
echo ""
echo "WEEKLY (baseline: 06a_weekly_sweep/h16_enc12_d015_bs16 = q7_h1)"
echo "  - q3_h1: 3 quantiles, horizon 1"
echo "  - q1_h1: 1 quantile, horizon 1"
echo "  - q3_h3: 3 quantiles, horizon 3"
echo "  - q3_h5: 3 quantiles, horizon 5"
echo ""
echo "DAILY (baseline: 02b_vintage_sweep/baseline_h16_drop0.10 = q7_h1)"
echo "  - q3_h1_daily: 3 quantiles, horizon 1"
echo "  - q7_h3_daily: 7 quantiles, horizon 3"
echo ""
echo "Total: 6 experiments (~50-60 min)"
echo "=============================================="
echo ""

# Parse arguments
if [[ "$1" == "--dry-run" ]]; then
    echo "[DRY RUN] Would run:"
    echo "  WEEKLY:"
    echo "    1. ${PHASE}/q3_h1 (3 quantiles, horizon 1)"
    echo "    2. ${PHASE}/q1_h1 (1 quantile, horizon 1)"
    echo "    3. ${PHASE}/q3_h3 (3 quantiles, horizon 3)"
    echo "    4. ${PHASE}/q3_h5 (3 quantiles, horizon 5)"
    echo "  DAILY:"
    echo "    5. ${PHASE}/q3_h1_daily (3 quantiles, horizon 1)"
    echo "    6. ${PHASE}/q7_h3_daily (7 quantiles, horizon 3)"
    echo ""
    echo "Compare against existing baselines:"
    echo "  Weekly: 06a_weekly_sweep/h16_enc12_d015_bs16 (q7, h1)"
    echo "  Daily:  02b_vintage_sweep/baseline_h16_drop0.10 (q7, h1)"
    exit 0
fi

# ============================================
# TEST 1: Quantile count effect (weekly)
# ============================================
echo ">>> WEEKLY: Quantile count effect <<<"
echo ""

run_experiment "q3_h1" "$WEEKLY_ARGS" --quantiles 3q --max-prediction-length 1
run_experiment "q1_h1" "$WEEKLY_ARGS" --quantiles median --max-prediction-length 1

# ============================================
# TEST 2: Horizon effect (weekly, using 3q)
# ============================================
echo ">>> WEEKLY: Horizon effect <<<"
echo ""

# q3_h1 already run above, skip
run_experiment "q3_h3" "$WEEKLY_ARGS" --quantiles 3q --max-prediction-length 3
run_experiment "q3_h5" "$WEEKLY_ARGS" --quantiles 3q --max-prediction-length 5

# ============================================
# TEST 3: Daily - quantile and horizon
# ============================================
echo ">>> DAILY: Quantile + Horizon tests <<<"
echo ""

run_experiment "q3_h1_daily" "$DAILY_ARGS" --quantiles 3q --max-prediction-length 1
run_experiment "q7_h3_daily" "$DAILY_ARGS" --quantiles 7q --max-prediction-length 3

# ============================================
# SUMMARY
# ============================================
echo "=============================================="
echo "PRELIMINARY TESTS COMPLETE"
echo "=============================================="
echo ""
echo "Results in: experiments/${PHASE}/"
echo ""
echo "Quick comparison - collapse_monitor_latest.json:"
echo ""

echo "WEEKLY:"
# Show existing weekly baseline first
baseline_json="experiments/06a_weekly_sweep/h16_enc12_d015_bs16/collapse_monitoring/collapse_monitor_latest.json"
if [[ -f "$baseline_json" ]]; then
    pred_std=$(python -c "import json; d=json.load(open('$baseline_json')); print(f\"{d['prediction_std'][-1]:.4f}\")" 2>/dev/null || echo "N/A")
    n_unique=$(python -c "import json; d=json.load(open('$baseline_json')); print(d['num_unique_predictions'][-1])" 2>/dev/null || echo "N/A")
    pct_pos=$(python -c "import json; d=json.load(open('$baseline_json')); print(f\"{d['pct_positive'][-1]:.1f}\")" 2>/dev/null || echo "N/A")
    dir_acc=$(python -c "import json; d=json.load(open('$baseline_json')); print(f\"{d['directional_accuracy'][-1]*100:.1f}\")" 2>/dev/null || echo "N/A")
    echo "  [BASELINE] q7_h1: pred_std=$pred_std, unique=$n_unique, pct_pos=${pct_pos}%, dir_acc=${dir_acc}%"
else
    echo "  [BASELINE] q7_h1: [06a_weekly_sweep/h16_enc12_d015_bs16 not found]"
fi

for exp in q3_h1 q1_h1 q3_h3 q3_h5; do
    json_file="experiments/${PHASE}/${exp}/collapse_monitoring/collapse_monitor_latest.json"
    if [[ -f "$json_file" ]]; then
        pred_std=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['prediction_std'][-1]:.4f}\")" 2>/dev/null || echo "N/A")
        n_unique=$(python -c "import json; d=json.load(open('$json_file')); print(d['num_unique_predictions'][-1])" 2>/dev/null || echo "N/A")
        pct_pos=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['pct_positive'][-1]:.1f}\")" 2>/dev/null || echo "N/A")
        dir_acc=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['directional_accuracy'][-1]*100:.1f}\")" 2>/dev/null || echo "N/A")
        echo "  $exp: pred_std=$pred_std, unique=$n_unique, pct_pos=${pct_pos}%, dir_acc=${dir_acc}%"
    else
        echo "  $exp: [results not found]"
    fi
done

echo ""
echo "DAILY:"
# Show existing daily baseline
baseline_json="experiments/02b_vintage_sweep/baseline_h16_drop0.10/collapse_monitoring/collapse_monitor_latest.json"
if [[ -f "$baseline_json" ]]; then
    pred_std=$(python -c "import json; d=json.load(open('$baseline_json')); print(f\"{d['prediction_std'][-1]:.4f}\")" 2>/dev/null || echo "N/A")
    n_unique=$(python -c "import json; d=json.load(open('$baseline_json')); print(d['num_unique_predictions'][-1])" 2>/dev/null || echo "N/A")
    pct_pos=$(python -c "import json; d=json.load(open('$baseline_json')); print(f\"{d['pct_positive'][-1]:.1f}\")" 2>/dev/null || echo "N/A")
    dir_acc=$(python -c "import json; d=json.load(open('$baseline_json')); print(f\"{d['directional_accuracy'][-1]*100:.1f}\")" 2>/dev/null || echo "N/A")
    echo "  [BASELINE] q7_h1_daily: pred_std=$pred_std, unique=$n_unique, pct_pos=${pct_pos}%, dir_acc=${dir_acc}%"
else
    echo "  [BASELINE] q7_h1_daily: [02b_vintage_sweep/baseline_h16_drop0.10 not found]"
fi

for exp in q3_h1_daily q7_h3_daily; do
    json_file="experiments/${PHASE}/${exp}/collapse_monitoring/collapse_monitor_latest.json"
    if [[ -f "$json_file" ]]; then
        pred_std=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['prediction_std'][-1]:.4f}\")" 2>/dev/null || echo "N/A")
        n_unique=$(python -c "import json; d=json.load(open('$json_file')); print(d['num_unique_predictions'][-1])" 2>/dev/null || echo "N/A")
        pct_pos=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['pct_positive'][-1]:.1f}\")" 2>/dev/null || echo "N/A")
        dir_acc=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['directional_accuracy'][-1]*100:.1f}\")" 2>/dev/null || echo "N/A")
        echo "  $exp: pred_std=$pred_std, unique=$n_unique, pct_pos=${pct_pos}%, dir_acc=${dir_acc}%"
    else
        echo "  $exp: [results not found]"
    fi
done

echo ""
echo "Decision guide:"
echo "  - Higher pred_std = less collapsed"
echo "  - unique near 252 (weekly) or 1282 (daily) = healthy"
echo "  - pct_pos near 50% = balanced predictions"
echo "  - dir_acc > 50% = predictive skill"
echo ""
echo "Checkpoint comparisons in:"
for exp in q3_h1 q1_h1 q3_h3 q3_h5 q3_h1_daily q7_h3_daily; do
    echo "  experiments/${PHASE}/${exp}/checkpoint_comparison.csv"
done