#!/usr/bin/env python3
"""
Rolling Window / Walk-Forward Evaluation for TFT Experiments.

Orchestrates multiple train/evaluate cycles across time folds to assess
model robustness and out-of-sample performance stability.

Supports two modes:
1. Rolling window: Fixed-size training window that slides forward
2. Expanding window: Training window grows with each fold

Supports two granularities:
1. Year-based (original): --step-years, --val-years, --test-years
2. Month-based (walk-forward): --step-months, --val-months, --test-months

Usage:
    # Year-based rolling window (backward compatible)
    python rolling_evaluation.py \
        --mode rolling \
        --train-years 10 \
        --val-years 1 \
        --test-years 1 \
        --step-years 1 \
        --start-test-year 2016 \
        --end-test-year 2024 \
        --feature-set core_proposal \
        --frequency daily \
        --experiment-prefix rolling_baseline

    # Month-based walk-forward (new)
    python rolling_evaluation.py \
        --mode expanding \
        --val-months 3 \
        --test-months 1 \
        --step-months 1 \
        --start-test-date 2020-01-01 \
        --end-test-date 2023-12-01 \
        --feature-set core_proposal \
        --frequency daily \
        --experiment-prefix wf_monthly

    # Dry run (show folds without executing)
    python rolling_evaluation.py --dry-run ...
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir, experiment_prefix):
    """Configure logging to both console and file."""
    log_file = output_dir / f"{experiment_prefix}_rolling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    logger = logging.getLogger('rolling_eval')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


# ============================================================================
# FOLD GENERATION - YEAR-BASED (Original)
# ============================================================================

def generate_folds_yearly(mode, train_years, val_years, test_years, step_years,
                          start_test_year, end_test_year, data_start_year=1990):
    """
    Generate train/val/test date boundaries for each fold (year granularity).
    
    Parameters:
    -----------
    mode : str
        'rolling' (fixed train window) or 'expanding' (growing train window)
    train_years : int
        Training window size in years (for rolling mode)
    val_years : int
        Validation window size in years
    test_years : int
        Test window size in years
    step_years : int
        Step size between folds in years
    start_test_year : int
        First test period start year
    end_test_year : int
        Last test period start year (inclusive)
    data_start_year : int
        Earliest year available in data (for expanding mode)
    
    Returns:
    --------
    list of dict
        Each dict contains fold boundaries and metadata
    """
    folds = []
    
    for test_start_year in range(start_test_year, end_test_year + 1, step_years):
        # Test period
        test_start = f"{test_start_year}-01-01"
        test_end = f"{test_start_year + test_years - 1}-12-31"
        
        # Validation period (immediately before test)
        val_end_year = test_start_year - 1
        val_start_year = val_end_year - val_years + 1
        val_start = f"{val_start_year}-01-01"
        val_end = f"{val_end_year}-12-31"
        
        # Training period
        if mode == 'rolling':
            # Fixed-size window ending before validation
            train_end_year = val_start_year - 1
            train_start_year = train_end_year - train_years + 1
        else:  # expanding
            # Start from beginning of data, end before validation
            train_start_year = data_start_year
            train_end_year = val_start_year - 1
        
        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_end_year}-12-31"
        
        # Validate we have enough data
        if train_start_year < data_start_year:
            continue  # Skip folds that require data before available
        
        fold = {
            'fold_id': f"fold_{test_start_year}",
            'test_year': test_start_year,
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_years': train_end_year - train_start_year + 1,
        }
        folds.append(fold)
    
    return folds


# ============================================================================
# FOLD GENERATION - MONTH-BASED (Walk-Forward)
# ============================================================================

def generate_folds_monthly(mode, train_months, val_months, test_months, step_months,
                           start_test_date, end_test_date, data_start_date='1990-01-01'):
    """
    Generate train/val/test date boundaries for each fold (month granularity).
    
    Parameters:
    -----------
    mode : str
        'rolling' (fixed train window) or 'expanding' (growing train window)
    train_months : int
        Training window size in months (for rolling mode). None for expanding.
    val_months : int
        Validation window size in months
    test_months : int
        Test window size in months
    step_months : int
        Step size between folds in months
    start_test_date : str
        First test period start date (YYYY-MM-DD)
    end_test_date : str
        Last test period start date (YYYY-MM-DD)
    data_start_date : str
        Earliest date available in data (for expanding mode)
    
    Returns:
    --------
    list of dict
        Each dict contains fold boundaries and metadata
    """
    folds = []
    
    # Parse dates
    current_test_start = datetime.strptime(start_test_date, '%Y-%m-%d')
    end_test = datetime.strptime(end_test_date, '%Y-%m-%d')
    data_start = datetime.strptime(data_start_date, '%Y-%m-%d')
    
    fold_idx = 0
    while current_test_start <= end_test:
        # Test period
        test_start = current_test_start
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)
        
        # Validation period (immediately before test)
        val_end = test_start - relativedelta(days=1)
        val_start = val_end - relativedelta(months=val_months) + relativedelta(days=1)
        
        # Training period
        if mode == 'rolling':
            train_end = val_start - relativedelta(days=1)
            train_start = train_end - relativedelta(months=train_months) + relativedelta(days=1)
        else:  # expanding
            train_start = data_start
            train_end = val_start - relativedelta(days=1)
        
        # Validate we have enough data
        if train_start < data_start:
            current_test_start += relativedelta(months=step_months)
            continue
        
        # Calculate training period in months for logging
        train_duration_months = (train_end.year - train_start.year) * 12 + (train_end.month - train_start.month) + 1
        
        # Create fold ID from test start date
        fold_id = f"fold_{test_start.strftime('%Y%m')}"
        
        fold = {
            'fold_id': fold_id,
            'fold_idx': fold_idx,
            'test_year': test_start.year,  # For compatibility with filtering
            'test_month': test_start.month,
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'val_start': val_start.strftime('%Y-%m-%d'),
            'val_end': val_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'train_months': train_duration_months,
            'train_years': train_duration_months / 12,  # For compatibility
        }
        folds.append(fold)
        fold_idx += 1
        
        # Step forward
        current_test_start += relativedelta(months=step_months)
    
    return folds


def print_fold_summary(folds, logger, monthly=False):
    """Print summary of all folds."""
    logger.info("="*100)
    logger.info("FOLD SUMMARY")
    logger.info("="*100)
    
    if monthly:
        logger.info(f"{'Fold':<15} {'Train':<25} {'Val':<25} {'Test':<25} {'Train Mo':<10}")
    else:
        logger.info(f"{'Fold':<12} {'Train':<25} {'Val':<25} {'Test':<25} {'Train Yrs':<10}")
    logger.info("-"*100)
    
    for fold in folds:
        train_range = f"{fold['train_start']} to {fold['train_end']}"
        val_range = f"{fold['val_start']} to {fold['val_end']}"
        test_range = f"{fold['test_start']} to {fold['test_end']}"
        
        if monthly:
            logger.info(f"{fold['fold_id']:<15} {train_range:<25} {val_range:<25} {test_range:<25} {fold['train_months']:<10}")
        else:
            logger.info(f"{fold['fold_id']:<12} {train_range:<25} {val_range:<25} {test_range:<25} {fold['train_years']:<10}")
    
    logger.info("="*100)
    logger.info(f"Total folds: {len(folds)}")
    logger.info("="*100 + "\n")


# ============================================================================
# SUBPROCESS EXECUTION
# ============================================================================

def run_command(cmd, logger, description, dry_run=False):
    """
    Run a subprocess command with logging.
    
    Returns:
    --------
    tuple (success: bool, return_code: int)
    """
    cmd_str = ' '.join(cmd)
    logger.info(f"[CMD] {description}")
    logger.debug(f"  {cmd_str}")
    
    if dry_run:
        logger.info("  [DRY RUN] Skipping execution")
        return True, 0
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per command
        )
        
        if result.returncode != 0:
            logger.error(f"  Command failed with return code {result.returncode}")
            logger.error(f"  STDERR: {result.stderr[:1000] if result.stderr else 'None'}")
            return False, result.returncode
        
        logger.info(f"  ✓ Completed successfully")
        return True, 0
        
    except subprocess.TimeoutExpired:
        logger.error(f"  Command timed out after 2 hours")
        return False, -1
    except Exception as e:
        logger.error(f"  Exception: {e}")
        return False, -1


# ============================================================================
# FOLD EXECUTION
# ============================================================================

def create_fold_splits(fold, args, splits_base_dir, logger, dry_run=False):
    """Create train/val/test splits for a specific fold."""
    
    # Output directory for this fold's splits
    fold_splits_dir = splits_base_dir / fold['fold_id'] / args.alignment
    fold_splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if splits already exist
    expected_files = [
        f"{args.feature_set}_{args.frequency}_{args.alignment}_{split}.csv"
        for split in ['train', 'val', 'test']
    ]
    splits_exist = all((fold_splits_dir / f).exists() for f in expected_files)
    
    if splits_exist:
        logger.info(f"  Splits already exist for {fold['fold_id']}, skipping creation")
        return True
    
    cmd = [
        'python', 'scripts/create_splits.py',
        '--feature-set', args.feature_set,
        '--frequency', args.frequency,
        '--data-version', args.alignment,
        '--train-start', fold['train_start'],
        '--train-end', fold['train_end'],
        '--val-end', fold['val_end'],
        '--test-end', fold['test_end'],
        '--output-dir', str(splits_base_dir / fold['fold_id']),
    ]
    
    if args.enhanced:
        cmd.append('--enhanced')
    
    cmd.extend(['--lookback-buffer', str(args.max_encoder_length)])
    
    success, _ = run_command(cmd, logger, f"Creating splits for {fold['fold_id']}", dry_run)
    return success


def train_fold(fold, args, splits_base_dir, logger, dry_run=False):
    """Train model for a specific fold."""
    
    experiment_name = f"{args.experiment_prefix}/{fold['fold_id']}"
    fold_splits_dir = splits_base_dir / fold['fold_id']
    
    cmd = [
        'python', 'train/train_tft.py',
        '--experiment-name', experiment_name,
        '--feature-set', args.feature_set,
        '--frequency', args.frequency,
        '--alignment', args.alignment,
        '--splits-dir', str(fold_splits_dir),
        '--seed', str(args.seed),
        # Architecture
        '--hidden-size', str(args.hidden_size),
        '--attention-heads', str(args.attention_heads),
        '--dropout', str(args.dropout),
        '--max-encoder-length', str(args.max_encoder_length),
        # Training
        '--learning-rate', str(args.learning_rate),
        '--batch-size', str(args.batch_size),
        '--max-epochs', str(args.max_epochs),
        '--early-stop-patience', str(args.early_stop_patience),
        '--gradient-clip', str(args.gradient_clip),
    ]
    
    # Custom loss modifications (only add if non-default)
    if args.directional_weight > 0:
        cmd.extend(['--directional-weight', str(args.directional_weight)])
        cmd.extend(['--directional-threshold', str(args.directional_threshold)])
        cmd.extend(['--directional-window', str(args.directional_window)])
    
    if args.collapse_threshold != 0.005:
        cmd.extend(['--collapse-threshold', str(args.collapse_threshold)])
    
    if args.dist_loss_mean_weight > 0:
        cmd.extend(['--dist-loss-mean-weight', str(args.dist_loss_mean_weight)])
    
    if args.dist_loss_std_weight > 0:
        cmd.extend(['--dist-loss-std-weight', str(args.dist_loss_std_weight)])
    
    if args.temporal_consistency_weight > 0:
        cmd.extend(['--temporal-consistency-weight', str(args.temporal_consistency_weight)])
    
    if args.magnitude_weight_alpha > 0:
        cmd.extend(['--magnitude-weight-alpha', str(args.magnitude_weight_alpha)])
    
    if args.extreme_move_weight != 1.0:
        cmd.extend(['--extreme-move-weight', str(args.extreme_move_weight)])
        cmd.extend(['--extreme-move-percentile', str(args.extreme_move_percentile)])
    
    # Regime output
    if args.regime_output:
        cmd.append('--regime-output')
        cmd.extend(['--num-regimes', str(args.num_regimes)])
        cmd.extend(['--routing-strategy', args.routing_strategy])
        cmd.extend(['--load-balance-weight', str(args.load_balance_weight)])
        cmd.extend(['--vix-threshold', str(args.vix_threshold)])
        if args.vix_threshold_low is not None:
            cmd.extend(['--vix-threshold-low', str(args.vix_threshold_low)])
        if args.vix_threshold_high is not None:
            cmd.extend(['--vix-threshold-high', str(args.vix_threshold_high)])
        if args.expert_hidden_size > 0:
            cmd.extend(['--expert-hidden-size', str(args.expert_hidden_size)])
        if args.hard_routing_train:
            cmd.append('--hard-routing-train')
    
    # Regime-aware attention
    if args.regime_attention:
        cmd.append('--regime-attention')
        cmd.extend(['--regime-attention-vix-threshold', str(args.regime_attention_vix_threshold)])
        cmd.extend(['--regime-attention-grad-scale', str(args.regime_attention_grad_scale)])
        cmd.extend(['--regime-gate-init', args.regime_gate_init])
        
    if args.gate_separation_weight > 0:
        cmd.extend(['--gate-separation-weight', str(args.gate_separation_weight)])

    # Classification head
    if args.classification:
        cmd.append('--classification')
        cmd.extend(['--classification-mode', args.classification_mode])
        cmd.extend(['--classification-weight', str(args.classification_weight)])
        cmd.extend(['--regression-weight', str(args.regression_weight)])
    
    # Other options
    if args.staleness:
        cmd.append('--staleness')
    
    if args.overwrite:
        cmd.append('--overwrite')
    
    success, _ = run_command(cmd, logger, f"Training {fold['fold_id']}", dry_run)
    return success


def evaluate_fold(fold, args, logger, dry_run=False):
    """Evaluate trained model for a specific fold."""
    
    experiment_name = f"{args.experiment_prefix}/{fold['fold_id']}"
    
    cmd = [
        'python', 'train/evaluate_tft.py',
        '--experiment-name', experiment_name,
        '--batch-size', str(args.eval_batch_size),
        '--checkpoint-type', args.checkpoint_type,
    ]
    success, _ = run_command(cmd, logger, f"Evaluating {fold['fold_id']}", dry_run)
    return success


def run_fold(fold, args, splits_base_dir, logger, dry_run=False):
    """Run complete pipeline for a single fold: split -> train -> evaluate."""
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"PROCESSING {fold['fold_id'].upper()}")
    logger.info(f"  Test period: {fold['test_start']} to {fold['test_end']}")
    if 'train_months' in fold:
        logger.info(f"  Training samples: ~{fold['train_months']} months ({fold['train_years']:.1f} years)")
    else:
        logger.info(f"  Training samples: ~{fold['train_years']} years")
    logger.info("="*80)
    
    # Step 1: Create splits
    if not create_fold_splits(fold, args, splits_base_dir, logger, dry_run):
        logger.error(f"  ✗ Failed to create splits for {fold['fold_id']}")
        return {'fold_id': fold['fold_id'], 'status': 'split_failed'}
    
    # Step 2: Train
    if not train_fold(fold, args, splits_base_dir, logger, dry_run):
        logger.error(f"  ✗ Failed to train {fold['fold_id']}")
        return {'fold_id': fold['fold_id'], 'status': 'train_failed'}
    
    # Step 3: Evaluate
    if not evaluate_fold(fold, args, logger, dry_run):
        logger.error(f"  ✗ Failed to evaluate {fold['fold_id']}")
        return {'fold_id': fold['fold_id'], 'status': 'eval_failed'}
    
    logger.info(f"  ✓ {fold['fold_id']} completed successfully")
    return {'fold_id': fold['fold_id'], 'status': 'success', **fold}


# ============================================================================
# RESULTS AGGREGATION
# ============================================================================

def aggregate_results(folds, args, logger):
    """Aggregate evaluation results across all folds."""
    
    logger.info("")
    logger.info("="*80)
    logger.info("AGGREGATING RESULTS")
    logger.info("="*80)
    
    all_metrics = []
    
    for fold in folds:
        experiment_dir = Path('experiments') / args.experiment_prefix / fold['fold_id']
        metrics_path = experiment_dir / 'evaluation' / 'evaluation_metrics.json'
        
        if not metrics_path.exists():
            logger.warning(f"  No metrics found for {fold['fold_id']}")
            continue
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Add fold metadata
        metrics['fold_id'] = fold['fold_id']
        metrics['test_year'] = fold['test_year']
        if 'test_month' in fold:
            metrics['test_month'] = fold['test_month']
        metrics['test_start'] = fold['test_start']
        metrics['test_end'] = fold['test_end']
        metrics['train_years'] = fold['train_years']
        if 'train_months' in fold:
            metrics['train_months'] = fold['train_months']
        
        all_metrics.append(metrics)
        logger.info(f"  Loaded metrics for {fold['fold_id']}")
    
    if not all_metrics:
        logger.error("  No metrics found to aggregate!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Save full results
    output_dir = Path('experiments') / args.experiment_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_path = output_dir / 'rolling_results_full.csv'
    df.to_csv(full_path, index=False)
    logger.info(f"  Saved full results to: {full_path}")
    
    # Create summary statistics
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Exclude fold metadata from summary
    exclude_cols = ['test_year', 'test_month', 'train_years', 'train_months', 'fold_idx']
    metric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if len(df) == 1:
        # Single fold - just transpose the metrics
        logger.info("  Single fold - skipping aggregate statistics")
        summary = df[metric_cols].T
        summary.columns = ['value']
    elif len(metric_cols) > 0:
        summary = df[metric_cols].agg(['mean', 'std', 'min', 'max']).T
        summary.columns = ['mean', 'std', 'min', 'max']
        summary['cv'] = summary['std'] / summary['mean'].abs()  # Coefficient of variation
    else:
        logger.warning("  No numeric metric columns found for summary")
        summary = pd.DataFrame()
    
    if not summary.empty:
        summary_path = output_dir / 'rolling_results_summary.csv'
        summary.to_csv(summary_path)
        logger.info(f"  Saved summary to: {summary_path}")
    
    # Print key metrics
    logger.info("")
    logger.info("KEY METRICS ACROSS FOLDS:")
    logger.info("-"*60)
    
    key_metrics = ['directional_accuracy', 'sharpe_ratio', 'total_return', 
                   'healthy_pct', 'strong_collapse_pct']
    
    for metric in key_metrics:
        if metric in df.columns:
            mean = df[metric].mean()
            std = df[metric].std()
            logger.info(f"  {metric:<25}: {mean:>8.4f} ± {std:.4f}")
    
    logger.info("-"*60)
    
    return df


# ============================================================================
# CLI ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Rolling window / walk-forward evaluation for TFT experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode and fold configuration
    parser.add_argument('--mode', type=str, default='rolling',
                        choices=['rolling', 'expanding'],
                        help='Evaluation mode: rolling (fixed window) or expanding (growing window)')
    
    # Year-based parameters (original)
    parser.add_argument('--train-years', type=int, default=10,
                        help='Training window size in years (rolling mode only)')
    parser.add_argument('--val-years', type=int, default=1,
                        help='Validation window size in years')
    parser.add_argument('--test-years', type=int, default=1,
                        help='Test window size in years')
    parser.add_argument('--step-years', type=int, default=1,
                        help='Step size between folds in years')
    parser.add_argument('--start-test-year', type=int, default=None,
                        help='First test period start year (for year-based mode)')
    parser.add_argument('--end-test-year', type=int, default=None,
                        help='Last test period start year (for year-based mode)')
    parser.add_argument('--data-start-year', type=int, default=1990,
                        help='Earliest year in dataset (for expanding mode)')
    
    # Month-based parameters (walk-forward)
    parser.add_argument('--train-months', type=int, default=None,
                        help='Training window size in months (rolling mode only, for month-based)')
    parser.add_argument('--val-months', type=int, default=None,
                        help='Validation window size in months (enables month-based mode)')
    parser.add_argument('--test-months', type=int, default=None,
                        help='Test window size in months')
    parser.add_argument('--step-months', type=int, default=None,
                        help='Step size between folds in months')
    parser.add_argument('--start-test-date', type=str, default=None,
                        help='First test period start date YYYY-MM-DD (for month-based mode)')
    parser.add_argument('--end-test-date', type=str, default=None,
                        help='Last test period start date YYYY-MM-DD (for month-based mode)')
    parser.add_argument('--data-start-date', type=str, default='1990-01-01',
                        help='Earliest date in dataset (for expanding mode, month-based)')
    
    # Experiment configuration
    parser.add_argument('--experiment-prefix', type=str, required=True,
                        help='Prefix for experiment names (e.g., "06_rolling/baseline")')
    parser.add_argument('--feature-set', type=str, default='core_proposal',
                        choices=['core_proposal', 'core_plus_credit', 'macro_heavy',
                                 'market_only', 'kitchen_sink', 'core_dynamics'],
                        help='Feature set configuration')
    parser.add_argument('--frequency', type=str, default='daily',
                        choices=['daily', 'weekly', 'monthly'],
                        help='Data frequency')
    parser.add_argument('--alignment', type=str, default='vintage',
                        choices=['fixed', 'vintage'],
                        help='Release date alignment mode')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced dataset with technical features')
    
    # Model architecture (pass through to train_tft.py)
    parser.add_argument('--hidden-size', type=int, default=16,
                        help='Hidden layer size')
    parser.add_argument('--attention-heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.10,
                        help='Dropout rate')
    parser.add_argument('--max-encoder-length', type=int, default=20,
                        help='Lookback window length')
    
    # Training (pass through to train_tft.py)
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--gradient-clip', type=float, default=0.1,
                        help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Evaluation
    parser.add_argument('--eval-batch-size', type=int, default=128,
                        help='Evaluation batch size')
    
    # Custom loss modifications (pass through to train_tft.py)
    parser.add_argument('--directional-weight', type=float, default=0.0,
                        help='Weight for directional diversity penalty')
    parser.add_argument('--directional-threshold', type=float, default=0.90,
                        help='Threshold for directional penalty activation')
    parser.add_argument('--directional-window', type=int, default=30,
                        help='Window size for directional penalty calculation')
    parser.add_argument('--collapse-threshold', type=float, default=0.005,
                        help='Std threshold for variance collapse penalty')
    parser.add_argument('--dist-loss-mean-weight', type=float, default=0.0,
                        help='Weight for distribution mean constraint')
    parser.add_argument('--dist-loss-std-weight', type=float, default=0.0,
                        help='Weight for distribution std constraint')
    parser.add_argument('--temporal-consistency-weight', type=float, default=0.0,
                        help='Weight for temporal consistency penalty')
    parser.add_argument('--magnitude-weight-alpha', type=float, default=0.0,
                        help='Alpha for linear magnitude weighting')
    parser.add_argument('--extreme-move-weight', type=float, default=1.0,
                        help='Weight multiplier for extreme moves')
    parser.add_argument('--extreme-move-percentile', type=int, default=95,
                        help='Percentile threshold for extreme moves')
    
    # Regime output (pass through to train_tft.py)
    parser.add_argument('--regime-output', action='store_true',
                        help='Enable regime-conditional output layer')
    parser.add_argument('--num-regimes', type=int, default=2,
                        help='Number of regime experts')
    parser.add_argument('--routing-strategy', type=str, default='learned',
                        choices=['learned', 'vix_threshold'],
                        help='Routing strategy for regime selection')
    parser.add_argument('--load-balance-weight', type=float, default=0.5,
                        help='Weight for load balancing loss')
    parser.add_argument('--vix-threshold', type=float, default=25.0,
                        help='VIX threshold for 2-regime routing')
    parser.add_argument('--vix-threshold-low', type=float, default=None,
                        help='Low VIX threshold for 3-regime routing')
    parser.add_argument('--vix-threshold-high', type=float, default=None,
                        help='High VIX threshold for 3-regime routing')
    parser.add_argument('--expert-hidden-size', type=int, default=0,
                        help='Hidden size for MLP experts (0 for linear)')
    parser.add_argument('--hard-routing-train', action='store_true',
                        help='Use hard routing during training')
    
    # Regime-aware attention (pass through to train_tft.py)
    parser.add_argument('--regime-attention', action='store_true',
                        help='Enable regime-aware attention gating')
    parser.add_argument('--regime-attention-vix-threshold', type=float, default=25.0,
                        help='VIX threshold for regime switching')
    parser.add_argument('--regime-attention-grad-scale', type=float, default=100.0,
                        help='Gradient scaling factor for regime gates')
    parser.add_argument('--regime-gate-init', type=str, default='neutral',
                        choices=['neutral', 'separated'],
                        help='Gate initialization: neutral (0.5) or separated (0.38/0.62)')
    parser.add_argument('--gate-separation-weight', type=float, default=0.0,
                    help='Weight for regime gate separation reward')

    # Classification head (pass through to train_tft.py)
    parser.add_argument('--classification', action='store_true',
                        help='Enable parallel classification head')
    parser.add_argument('--classification-mode', type=str, default='direction',
                        choices=['direction', 'direction_3class', 'regime_volatility'],
                        help='Classification target mode')
    parser.add_argument('--classification-weight', type=float, default=1.0,
                        help='Weight for classification loss')
    parser.add_argument('--regression-weight', type=float, default=1.0,
                        help='Weight for regression loss')
    
    # Other model options
    parser.add_argument('--staleness', action='store_true',
                        help='Include staleness features')
    
    # Execution control
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing experiment directories')
    parser.add_argument('--splits-dir', type=str, default='data/splits/rolling',
                        help='Base directory for fold splits')
    parser.add_argument('--start-from-fold', type=str, default=None,
                        help='Start from specific fold (fold_id or year/YYYYMM)')
    parser.add_argument('--only-fold', type=str, default=None,
                        help='Run only a specific fold (fold_id or year/YYYYMM)')
                        
    # ckpt selection
    parser.add_argument('--checkpoint-type', type=str, default='best_pred_std_path',
                    choices=['best_val_loss_path', 'best_pred_std_path', 'best_unique_path'],
                    help='Checkpoint selection metric for evaluation')
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Determine if using month-based or year-based mode
    use_monthly = (args.val_months is not None or 
                   args.step_months is not None or
                   args.start_test_date is not None)
    
    # Validate arguments based on mode
    if use_monthly:
        # Month-based mode - validate required args
        if args.step_months is None:
            args.step_months = 1  # Default to 1 month step
        if args.val_months is None:
            args.val_months = 3  # Default to 3 month validation
        if args.test_months is None:
            args.test_months = 1  # Default to 1 month test
        if args.start_test_date is None or args.end_test_date is None:
            print("ERROR: --start-test-date and --end-test-date required for month-based mode")
            sys.exit(1)
        if args.mode == 'rolling' and args.train_months is None:
            print("ERROR: --train-months required for rolling mode with month-based folds")
            sys.exit(1)
    else:
        # Year-based mode - use defaults if not specified
        if args.start_test_year is None:
            args.start_test_year = 2016
        if args.end_test_year is None:
            args.end_test_year = 2024
    
    # Setup output directory and logging
    output_dir = Path('experiments') / args.experiment_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger, log_file = setup_logging(output_dir, args.experiment_prefix.replace('/', '_'))
    
    logger.info("="*80)
    logger.info("ROLLING WINDOW / WALK-FORWARD EVALUATION")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Granularity: {'monthly' if use_monthly else 'yearly'}")
    
    if use_monthly:
        if args.mode == 'rolling':
            logger.info(f"  Train months: {args.train_months}")
        logger.info(f"  Val months: {args.val_months}")
        logger.info(f"  Test months: {args.test_months}")
        logger.info(f"  Step months: {args.step_months}")
        logger.info(f"  Test range: {args.start_test_date} - {args.end_test_date}")
    else:
        logger.info(f"  Train years: {args.train_years}")
        logger.info(f"  Val years: {args.val_years}")
        logger.info(f"  Test years: {args.test_years}")
        logger.info(f"  Step years: {args.step_years}")
        logger.info(f"  Test range: {args.start_test_year} - {args.end_test_year}")
    
    logger.info(f"  Feature set: {args.feature_set}")
    logger.info(f"  Frequency: {args.frequency}")
    logger.info(f"  Alignment: {args.alignment}")
    logger.info(f"  Enhanced: {args.enhanced}")
    logger.info(f"  Experiment prefix: {args.experiment_prefix}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("")
    
    # Generate folds
    if use_monthly:
        folds = generate_folds_monthly(
            mode=args.mode,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            step_months=args.step_months,
            start_test_date=args.start_test_date,
            end_test_date=args.end_test_date,
            data_start_date=args.data_start_date
        )
    else:
        folds = generate_folds_yearly(
            mode=args.mode,
            train_years=args.train_years,
            val_years=args.val_years,
            test_years=args.test_years,
            step_years=args.step_years,
            start_test_year=args.start_test_year,
            end_test_year=args.end_test_year,
            data_start_year=args.data_start_year
        )
    
    if not folds:
        logger.error("No valid folds generated! Check date ranges and data availability.")
        return
    
    # Filter folds if requested
    if args.only_fold:
        # Support both fold_id format (fold_2020 or fold_202001) and raw values
        target = args.only_fold
        folds = [f for f in folds if (f['fold_id'] == target or 
                                       f['fold_id'] == f"fold_{target}" or
                                       str(f.get('test_year')) == target)]
        if not folds:
            logger.error(f"No fold found matching '{args.only_fold}'")
            return
    elif args.start_from_fold:
        target = args.start_from_fold
        # Find index of starting fold
        start_idx = None
        for i, f in enumerate(folds):
            if (f['fold_id'] == target or 
                f['fold_id'] == f"fold_{target}" or
                str(f.get('test_year')) == target):
                start_idx = i
                break
        if start_idx is None:
            logger.error(f"No fold found matching '{args.start_from_fold}'")
            return
        folds = folds[start_idx:]
    
    print_fold_summary(folds, logger, monthly=use_monthly)
    
    # Estimate runtime
    if use_monthly:
        est_time_per_fold = 8 if args.frequency == 'daily' else 4  # minutes
    else:
        est_time_per_fold = 10 if args.frequency == 'daily' else 5
    est_total = len(folds) * est_time_per_fold
    logger.info(f"Estimated runtime: ~{est_total} minutes ({est_total/60:.1f} hours)")
    logger.info("")
    
    # Save fold configuration
    folds_config_path = output_dir / 'folds_config.json'
    with open(folds_config_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'folds': folds,
            'granularity': 'monthly' if use_monthly else 'yearly',
            'created_at': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"Saved folds configuration to: {folds_config_path}")
    
    # Setup splits directory
    splits_base_dir = Path(args.splits_dir)
    splits_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Run each fold
    results = []
    for i, fold in enumerate(folds):
        logger.info(f"\n[FOLD {i+1}/{len(folds)}]")
        result = run_fold(fold, args, splits_base_dir, logger, args.dry_run)
        results.append(result)
        
        # Save intermediate results
        results_path = output_dir / 'fold_status.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Aggregate results (skip if dry run)
    if not args.dry_run:
        aggregate_results(folds, args, logger)
    
    # Final summary
    logger.info("")
    logger.info("="*80)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*80)
    
    success_count = sum(1 for r in results if r.get('status') == 'success')
    logger.info(f"  Completed: {success_count}/{len(folds)} folds")
    
    failed = [r for r in results if r.get('status') != 'success']
    if failed:
        logger.info(f"  Failed folds:")
        for r in failed:
            logger.info(f"    - {r['fold_id']}: {r['status']}")
    
    logger.info(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()