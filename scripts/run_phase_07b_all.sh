#!/bin/bash
# Phase 07b: Combined Regime Attention Sweeps
# Runs VIX threshold sweep (enc12) then gradient scaling sweep (enc8)
#
# Total experiments:
# - VIX sweep: 3 fixed-split + 3×9 rolling = 30 training runs
# - Grad sweep: 3 fixed-split + 3×9 rolling = 30 training runs
# - Total: 60 training runs
#
# Estimated time: ~6-8 hours (weekly trains fast, ~40min per rolling config)

echo "========================================================================"
echo "PHASE 07b: COMBINED REGIME ATTENTION SWEEPS"
echo "========================================================================"
echo "Started at: $(date)"
echo ""
echo "This will run:"
echo "  1. VIX threshold sweep (20, 25, 30) on weekly enc12"
echo "  2. Gradient scaling sweep (100, 200, 500) on weekly enc8"
echo ""
echo "Estimated runtime: 6-8 hours"
echo ""

# Run VIX sweep first
echo "========================================================================"
echo "STARTING VIX THRESHOLD SWEEP"
echo "========================================================================"
bash scripts/run_phase_07b_vix_sweep.sh

VIX_STATUS=$?
if [ $VIX_STATUS -ne 0 ]; then
    echo "WARNING: VIX sweep exited with status $VIX_STATUS"
fi

echo ""
echo "========================================================================"
echo "STARTING GRADIENT SCALING SWEEP"
echo "========================================================================"
bash scripts/run_phase_07b_grad_sweep.sh

GRAD_STATUS=$?
if [ $GRAD_STATUS -ne 0 ]; then
    echo "WARNING: Gradient sweep exited with status $GRAD_STATUS"
fi

echo ""
echo "========================================================================"
echo "ALL SWEEPS COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results summary:"
echo "  VIX sweep: experiments/07b_regime_attention_sweep/vix_sweep_comparison.csv"
echo "  Grad sweep: experiments/07b_regime_attention_sweep/grad_sweep_comparison.csv"
echo ""
echo "To view gate learning across configs, check final gate values in training logs:"
echo "  grep 'Final gate values' logs/phase_07b/*.log"
echo ""
