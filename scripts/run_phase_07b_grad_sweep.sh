#!/bin/bash
# Phase 07b: Gradient Scaling Sweep for Regime Attention
# Tests gradient scaling (100, 200, 500) on weekly enc8 configuration
#
# Hypothesis: Weekly has fewer samples → weaker gradient signal to gates
# - Gate spread on daily (100x): 0.38
# - Gate spread on weekly enc12 (100x): 0.10
# - Gate spread on weekly enc8 (100x): 0.06
# Higher scaling may help gates learn more aggressively on weekly
#
# Config: weekly enc8 (weaker config, more room for improvement)
# Baseline comparison: existing 07_regime_attention enc8 results

echo "========================================================================"
echo "PHASE 07b: GRADIENT SCALING SWEEP (Weekly enc8)"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

LOG_DIR="logs/phase_07b"
PHASE_NAME="07b_regime_attention_sweep"
EVAL_LOG_DIR="experiments/$PHASE_NAME/evaluation_logs"

mkdir -p "$LOG_DIR"
mkdir -p "$EVAL_LOG_DIR"

echo "Gradient scales to test: 100, 200, 500"
echo "Configuration: weekly, h16, enc8, d0.25, bs32"
echo "VIX threshold: 25 (default)"
echo ""

# Fixed parameters (weekly enc8 config)
HIDDEN_SIZE=16
ENCODER_LEN=8
DROPOUT=0.25
BATCH_SIZE=32
VIX_THRESH=25

# Function for fixed-split training
train_fixed_split() {
    local grad_scale=$1
    local exp_name="enc8_gs${grad_scale}_vix25"
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
        --regime-attention-vix-threshold $VIX_THRESH \
        --regime-attention-grad-scale $grad_scale \
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
    local grad_scale=$1
    local exp_name="enc8_gs${grad_scale}_vix25"
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
        --regime-attention-vix-threshold $VIX_THRESH \
        --regime-attention-grad-scale $grad_scale \
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

# Run fixed-split for each gradient scale
for gs in 100 200 500; do
    train_fixed_split $gs
    echo ""
done

echo "========================================================================"
echo "ROLLING EVALUATION"
echo "========================================================================"
echo ""

# Run rolling for each gradient scale
for gs in 100 200 500; do
    run_rolling $gs
    echo ""
done

echo "========================================================================"
echo "ANALYSIS"
echo "========================================================================"
echo ""

# Analyze rolling results
python scripts/analyze_rolling.py \
    experiments/$PHASE_NAME/rolling_enc8_gs100_vix25 \
    experiments/$PHASE_NAME/rolling_enc8_gs200_vix25 \
    experiments/$PHASE_NAME/rolling_enc8_gs500_vix25 \
    --compare experiments/06a_weekly_sweep/h16_enc8_d025_bs32 \
    --output experiments/$PHASE_NAME/grad_sweep_comparison.csv \
    --plot-dir experiments/$PHASE_NAME \
    > "$LOG_DIR/analysis_grad_sweep.log" 2>&1

echo ""
echo "========================================================================"
echo "PHASE 07b GRADIENT SWEEP COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results:"
echo "  Fixed-split: experiments/$PHASE_NAME/enc8_gs{100,200,500}_vix25/"
echo "  Rolling: experiments/$PHASE_NAME/rolling_enc8_gs{100,200,500}_vix25/"
echo "  Comparison: experiments/$PHASE_NAME/grad_sweep_comparison.csv"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
