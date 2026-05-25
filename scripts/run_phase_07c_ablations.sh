#!/bin/bash
# Phase 07c: Regime Attention Ablations
# Quick tests for potential improvements to regime attention
#
# Tests:
# 1. Pre-initialized gates (separated) - enc12
# 2. 4 attention heads instead of 2 - enc12
# 3. Gate separation loss (auxiliary loss rewarding gate differentiation)
# 4. Combined: pre-init + 4 heads + gate separation
#
# Rolling eval only (fixed-split already shown to not generalize)

echo "========================================================================"
echo "PHASE 07c: REGIME ATTENTION ABLATIONS"
echo "========================================================================"
echo "Started at: $(date)"
echo ""

LOG_DIR="logs/phase_07c"
PHASE_NAME="07c_regime_attention_ablations"

mkdir -p "$LOG_DIR"

# Fixed parameters (weekly enc12 config)
HIDDEN_SIZE=16
ENCODER_LEN=12
DROPOUT=0.15
BATCH_SIZE=16
VIX_THRESH=25
GRAD_SCALE=100

echo "Tests to run:"
echo "  1. Pre-initialized gates (--regime-gate-init separated)"
echo "  2. 4 attention heads (--attention-heads 4)"
echo "  3. Gate separation loss (--gate-separation-weight 0.2)"
echo "  4. Combined (pre-init + 4 heads + gate separation)"
echo ""

# Test 1: Pre-initialized gates
echo "========================================================================"
echo "[1/4] Pre-initialized gates"
echo "========================================================================"

python train/rolling_evaluation.py \
    --experiment-prefix "$PHASE_NAME/rolling_preinit" \
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
    --early-stop-patience 25 \
    --attention-heads 2 \
    --regime-attention \
    --regime-attention-vix-threshold $VIX_THRESH \
    --regime-attention-grad-scale $GRAD_SCALE \
    --regime-gate-init separated \
    --overwrite \
    > "$LOG_DIR/rolling_preinit.log" 2>&1

echo "[1/4] Complete. Status: $?"

# Test 2: 4 attention heads
echo "========================================================================"
echo "[2/4] 4 attention heads"
echo "========================================================================"

python train/rolling_evaluation.py \
    --experiment-prefix "$PHASE_NAME/rolling_4heads" \
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
    --early-stop-patience 25 \
    --attention-heads 4 \
    --regime-attention \
    --regime-attention-vix-threshold $VIX_THRESH \
    --regime-attention-grad-scale $GRAD_SCALE \
    --overwrite \
    > "$LOG_DIR/rolling_4heads.log" 2>&1

echo "[2/4] Complete. Status: $?"

# Test 3: Gate separation loss
echo "========================================================================"
echo "[3/4] Gate separation loss"
echo "========================================================================"

python train/rolling_evaluation.py \
    --experiment-prefix "$PHASE_NAME/rolling_gate_sep" \
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
    --early-stop-patience 25 \
    --attention-heads 2 \
    --regime-attention \
    --regime-attention-vix-threshold $VIX_THRESH \
    --regime-attention-grad-scale $GRAD_SCALE \
    --gate-separation-weight 0.2 \
    --overwrite \
    > "$LOG_DIR/rolling_gate_sep.log" 2>&1

echo "[3/4] Complete. Status: $?"

# Test 4: Combined (pre-init + 4 heads + gate separation)
echo "========================================================================"
echo "[4/4] Combined (pre-init + 4 heads + gate separation)"
echo "========================================================================"

python train/rolling_evaluation.py \
    --experiment-prefix "$PHASE_NAME/rolling_combined" \
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
    --early-stop-patience 25 \
    --attention-heads 4 \
    --regime-attention \
    --regime-attention-vix-threshold $VIX_THRESH \
    --regime-attention-grad-scale $GRAD_SCALE \
    --regime-gate-init separated \
    --gate-separation-weight 0.2 \
    --overwrite \
    > "$LOG_DIR/rolling_combined.log" 2>&1

echo "[4/4] Complete. Status: $?"

# Analysis
echo "========================================================================"
echo "ANALYSIS"
echo "========================================================================"

python scripts/analyze_rolling.py \
    experiments/$PHASE_NAME/rolling_preinit \
    experiments/$PHASE_NAME/rolling_4heads \
    experiments/$PHASE_NAME/rolling_gate_sep \
    experiments/$PHASE_NAME/rolling_combined \
    --output experiments/$PHASE_NAME/ablation_comparison.csv \
    --plot-dir experiments/$PHASE_NAME \
    > "$LOG_DIR/analysis.log" 2>&1

echo ""
echo "========================================================================"
echo "PHASE 07c COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results: experiments/$PHASE_NAME/"
echo "Logs: $LOG_DIR/"
echo ""
echo "To check gate values learned:"
echo "  grep 'Final gate values' $LOG_DIR/*.log"
echo ""