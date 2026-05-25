
# overnight_sweep.sh
# Comprehensive quantile × horizon × frequency × seed sweep
#
# Design:
#   Frequencies: weekly, daily
#   Quantiles: 3q, 7q (skip 1q - proven to collapse)
#   Horizons: h1, h3, h5
#   Seeds: 42, 123, 456
#
# Total: 2 × 2 × 3 × 3 = 36 experiments
#
# Runtime estimate: ~15-18 hours

PHASE="10_overnight_sweep"
SEEDS=(42 123 456)
QUANTILES=(3q 7q)
HORIZONS=(1 3 5)

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
    --max-epochs 100
    --early-stop-patience 15
    --dist-loss-std-weight 0.0
    --directional-weight 0.0
    --temporal-consistency-weight 0.0
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
    --max-epochs 100
    --early-stop-patience 10
    --dist-loss-std-weight 0.0
    --directional-weight 0.0
    --temporal-consistency-weight 0.0
"

# Track progress
TOTAL_EXPERIMENTS=36
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Log file for overall progress
MASTER_LOG="experiments/${PHASE}/sweep_progress.log"
mkdir -p "experiments/${PHASE}"

log_progress() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

# Function to run single experiment with evaluation
run_experiment() {
    local freq=$1
    local quant=$2
    local horizon=$3
    local seed=$4
    
    local exp_name="${freq}_${quant}_h${horizon}_s${seed}"
    local exp_path="experiments/${PHASE}/${exp_name}"
    
    log_progress "Starting: $exp_name ($((COMPLETED + 1))/$TOTAL_EXPERIMENTS)"
    
    # Select base args
    if [[ "$freq" == "weekly" ]]; then
        BASE_ARGS="$WEEKLY_ARGS"
    else
        BASE_ARGS="$DAILY_ARGS"
    fi
    
    # Train
    local train_start=$(date +%s)
    if python train/train_tft.py \
        --experiment-name "${PHASE}/${exp_name}" \
        --quantiles "$quant" \
        --max-prediction-length "$horizon" \
        --seed "$seed" \
        --overwrite \
        $BASE_ARGS; then
        
        local train_end=$(date +%s)
        local train_duration=$((train_end - train_start))
        log_progress "  Training completed in ${train_duration}s"
        
        # Evaluate checkpoints
        local eval_log="${exp_path}/evaluation/evaluate_checkpoints.log"
        mkdir -p "${exp_path}/evaluation"
        
        log_progress "  Evaluating checkpoints..."
        if python train/evaluate_checkpoints.py \
            "$exp_path" \
            --top-per-metric 3 \
            > "$eval_log" 2>&1; then
            log_progress "  Evaluation completed"
        else
            log_progress "  WARNING: Evaluation failed (see $eval_log)"
        fi
        
        ((COMPLETED++))
        
        # Print running summary
        local elapsed=$(($(date +%s) - START_TIME))
        local avg_time=$((elapsed / COMPLETED))
        local remaining=$(((TOTAL_EXPERIMENTS - COMPLETED) * avg_time))
        log_progress "  Progress: $COMPLETED/$TOTAL_EXPERIMENTS, ~$((remaining / 60))min remaining"
        
    else
        log_progress "  FAILED: Training error for $exp_name"
        ((FAILED++))
    fi
    
    echo ""
}

# Print plan
echo "=============================================="
echo "OVERNIGHT SWEEP - Phase 10"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Frequencies: weekly, daily"
echo "  Quantiles: 3q, 7q"
echo "  Horizons: h1, h3, h5"
echo "  Seeds: ${SEEDS[*]}"
echo ""
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Estimated runtime: 15-18 hours"
echo ""
echo "Results will be in: experiments/${PHASE}/"
echo "Progress log: $MASTER_LOG"
echo "=============================================="
echo ""

# Parse arguments
if [[ "$1" == "--dry-run" ]]; then
    echo "[DRY RUN] Would run:"
    for freq in weekly daily; do
        for quant in "${QUANTILES[@]}"; do
            for horizon in "${HORIZONS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    echo "  ${PHASE}/${freq}_${quant}_h${horizon}_s${seed}"
                done
            done
        done
    done
    echo ""
    echo "Total: $TOTAL_EXPERIMENTS experiments"
    exit 0
fi

if [[ "$1" == "--weekly-only" ]]; then
    echo "Running weekly experiments only..."
    FREQS=(weekly)
    TOTAL_EXPERIMENTS=18
elif [[ "$1" == "--daily-only" ]]; then
    echo "Running daily experiments only..."
    FREQS=(daily)
    TOTAL_EXPERIMENTS=18
else
    FREQS=(weekly daily)
fi

# Initialize log
log_progress "Sweep started"
log_progress "Total experiments: $TOTAL_EXPERIMENTS"

# Run all experiments
# Order: all weekly first (faster), then daily
for freq in "${FREQS[@]}"; do
    log_progress "=== Starting $freq experiments ==="
    for quant in "${QUANTILES[@]}"; do
        for horizon in "${HORIZONS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_experiment "$freq" "$quant" "$horizon" "$seed"
            done
        done
    done
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "SWEEP COMPLETE"
echo "=============================================="
echo ""
log_progress "Completed: $COMPLETED/$TOTAL_EXPERIMENTS"
log_progress "Failed: $FAILED"
log_progress "Total time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo ""

# Generate comparison table
echo "Generating comparison table..."
python scripts/compare_experiments.py "experiments/${PHASE}" \
    --baseline experiments/06a_weekly_sweep/h16_enc12_d015_bs16 \
    --baseline experiments/02b_vintage_sweep/baseline_h16_drop0.10 \
    --min-epoch 20 \
    --csv "experiments/${PHASE}/final_comparison.csv"

echo ""
echo "Results saved to:"
echo "  experiments/${PHASE}/final_comparison.csv"
echo "  experiments/${PHASE}/sweep_progress.log"
echo ""
echo "To re-run comparison:"
echo "  python scripts/compare_experiments.py experiments/${PHASE} --baseline experiments/06a_weekly_sweep/h16_enc12_d015_bs16 --baseline experiments/02b_vintage_sweep/baseline_h16_drop0.10"
