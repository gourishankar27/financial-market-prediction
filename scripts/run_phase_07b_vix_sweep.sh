#Phase 07b: VIX Threshold Sweep for Regime Attention
# Tests VIX thresholds (20, 25, 30) on weekly enc12 configuration
#
# Hypothesis: Different thresholds affect regime balance and gate learning
# - VIX 20: ~30% high-vol samples (more balanced)
# - VIX 25: ~18% high-vol samples (current default)
# - VIX 30: ~10% high-vol samples (fewer high-vol)
#
# Config: weekly enc12 (best performing from Phase 07 fixed-split)
# Baseline comparison: 06a_weekly_sweep/h16_enc12_d015_bs16

echo "========================================================================"
echo "PHASE 07b: VIX THRESHOLD SWEEP (Weekly enc12)"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

LOG_DIR="logs/phase_07b"
PHASE_NAME="07b_regime_attention_sweep"
EVAL_LOG_DIR="experiments/$PHASE_NAME/evaluation_logs"

mkdir -p "$LOG_DIR"
mkdir -p "$EVAL_LOG_DIR"

echo "VIX thresholds to test: 20, 25, 30"
echo "Configuration: weekly, h16, enc12, d0.15, bs16"
echo "Gradient scale: 100 (default)"
echo ""

# Fixed parameters (weekly enc12 best config)
HIDDEN_SIZE=16
ENCODER_LEN=12
DROPOUT=0.15
BATCH_SIZE=16
GRAD_SCALE=100

# Function for fixed-split training
train_fixed_split() {
    local vix_thresh=$1
    local exp_name="vix${vix_thresh}_enc12_gs100"
    local full_exp_name="$PHASE_NAME/$exp_name"
    
    echo "[$exp_name] Training fixed-split..."
    
    python train/train_tft.py \
        --experiment-name "$full_exp_name" \
        --frequency weekly \
        --alignment vintage \
        --feature-set core_proposal \
        --hidden-size $HIDDEN_SIZE \
        --max-encoder-length $ENCODER_LEN \
        --dropout $DROPOUT \
        --batch-size $BATCH_SIZE \
        --learning-rate 0.0005 \
        --max-epochs 100 \
        --early-stop-patience 100 \
        --attention-heads 2 \
        --hidden-continuous-size $HIDDEN_SIZE \
        --gradient-clip 0.1 \
        --regime-attention \
        --regime-attention-vix-threshold $vix_thresh \
        --regime-attention-grad-scale $GRAD_SCALE \
        --overwrite \
        > "$LOG_DIR/train_${exp_name}.log" 2>&1
    
    local train_status=$?
    
    if [ $train_status -ne 0 ]; then
        echo "[$exp_name] Training FAILED (exit code: $train_status)"
        return 1
    fi
    
    echo "[$exp_name] Training complete, evaluating top 2 checkpoints per metric..."
    
    python train/evaluate_checkpoints.py \
        "experiments/$full_exp_name" \
        --top-per-metric 2 \
        > "$EVAL_LOG_DIR/eval_${exp_name}.log" 2>&1
    
    local eval_status=$?
    
    if [ $eval_status -ne 0 ]; then
        echo "[$exp_name] Evaluation FAILED (exit code: $eval_status)"
        return 1
    fi
    
    echo "[$exp_name] Fixed-split complete"
    return 0
}

# Function for rolling evaluation
run_rolling() {
    local vix_thresh=$1
    local exp_name="vix${vix_thresh}_enc12_gs100"
    local rolling_prefix="$PHASE_NAME/rolling_${exp_name}"
    
    echo "[$exp_name] Running rolling evaluation (9 folds)..."
    
    python train/rolling_evaluation.py \
        --experiment-prefix "$rolling_prefix" \
        --frequency weekly \
        --mode rolling \
        --train-years 10 \
        --val-years 1 \
        --test-years 1 \
        --step-years 1 \
        --start-test-year 2016 \
        --end-test-year 2024 \
        --hidden-size $HIDDEN_SIZE \
        --max-encoder-length $ENCODER_LEN \
        --dropout $DROPOUT \
        --batch-size $BATCH_SIZE \
        --max-epochs 100 \
        --early-stop-patience 100 \
        --regime-attention \
        --regime-attention-vix-threshold $vix_thresh \
        --regime-attention-grad-scale $GRAD_SCALE \
        --overwrite \
        > "$LOG_DIR/rolling_${exp_name}.log" 2>&1
    
    local status=$?
    
    if [ $status -ne 0 ]; then
        echo "[$exp_name] Rolling eval FAILED (exit code: $status)"
        return 1
    fi
    
    echo "[$exp_name] Rolling complete"
    return 0
}

echo "========================================================================"
echo "FIXED-SPLIT TRAINING"
echo "========================================================================"
echo ""

# Run fixed-split for each threshold
for vix in 20 25 30; do
    train_fixed_split $vix
    echo ""
done

echo "========================================================================"
echo "ROLLING EVALUATION"
echo "========================================================================"
echo ""

# Run rolling for each threshold
for vix in 20 25 30; do
    run_rolling $vix
    echo ""
done

echo "========================================================================"
echo "ANALYSIS"
echo "========================================================================"
echo ""

# Analyze rolling results
python scripts/analyze_rolling.py \
    experiments/$PHASE_NAME/rolling_vix20_enc12_gs100 \
    experiments/$PHASE_NAME/rolling_vix25_enc12_gs100 \
    experiments/$PHASE_NAME/rolling_vix30_enc12_gs100 \
    --compare experiments/06a_weekly_sweep/h16_enc12_d015_bs16 \
    --output experiments/$PHASE_NAME/vix_sweep_comparison.csv \
    --plot-dir experiments/$PHASE_NAME \
    > "$LOG_DIR/analysis_vix_sweep.log" 2>&1

echo ""
echo "========================================================================"
echo "PHASE 07b VIX SWEEP COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results:"
echo "  Fixed-split: experiments/$PHASE_NAME/vix{20,25,30}_enc12_gs100/"
echo "  Rolling: experiments/$PHASE_NAME/rolling_vix{20,25,30}_enc12_gs100/"
echo "  Comparison: experiments/$PHASE_NAME/vix_sweep_comparison.csv"
echo ""
echo "Logs: $LOG_DIR/"
echo ""

