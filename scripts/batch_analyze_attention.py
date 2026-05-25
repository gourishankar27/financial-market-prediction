"""
Batch run attention analysis across multiple experiments.

Usage:
    # Run on all experiments (skips existing results)
    python scripts/batch_analyze_attention.py
    
    # Run on all experiments in a phase
    python scripts/batch_analyze_attention.py --phase 02_vintage_baseline
    
    # Run on specific experiment
    python scripts/batch_analyze_attention.py --experiment 00_baseline_exploration/sweep2_h16_drop_0.25
    
    # Force re-run (overwrite existing)
    python scripts/batch_analyze_attention.py --force
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json


def find_all_experiments(base_dir='experiments'):
    """
    Find all experiment directories with checkpoints.
    
    Returns list of experiment paths relative to base_dir.
    Example: ['00_baseline_exploration/sweep2_h16_drop_0.25', ...]
    """
    experiments = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: {base_dir} does not exist")
        return experiments
    
    # Look for phase subdirectories (00_*, 01_*, etc.)
    for phase_dir in sorted(base_path.iterdir()):
        if not phase_dir.is_dir():
            continue
        
        # Look for experiment subdirectories
        for exp_dir in sorted(phase_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            # Check if it has a checkpoints directory
            checkpoints_dir = exp_dir / 'checkpoints'
            if checkpoints_dir.exists() and list(checkpoints_dir.glob('*.ckpt')):
                # Store relative path
                rel_path = exp_dir.relative_to(base_path)
                experiments.append(str(rel_path))
    
    return experiments


def has_existing_results(experiment_path, output_subdir='attention_analysis_year'):
    """Check if attention analysis results already exist for this experiment."""
    results_dir = Path('experiments') / experiment_path / output_subdir
    
    if not results_dir.exists():
        return False
    
    # Check for key output files
    required_files = [
        'attention_analysis_results.json',
        'attention_heatmap.png',
        'attention_trends.png',
    ]
    
    return all((results_dir / f).exists() for f in required_files)


def run_attention_analysis(experiment_path, output_subdir='attention_analysis_year'):
    """
    Run attention analysis for a single experiment.
    
    Returns (success, error_message)
    """
    output_dir = Path('experiments') / experiment_path / output_subdir
    
    # Build command
    cmd = [
        'python', 'scripts/analyze_attention_by_period.py',
        '--experiment', experiment_path,
        '--output-dir', str(output_dir),
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f" Success: {experiment_path}")
        return True, None
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"  Failed: {experiment_path}")
        print(f"  Error: {error_msg[:200]}")  # First 200 chars
        return False, error_msg
    
    except Exception as e:
        print(f"  Failed: {experiment_path}")
        print(f"  Error: {str(e)}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Batch run attention analysis on experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--experiment', type=str, default=None,
                        help='Specific experiment to analyze (e.g., 00_baseline_exploration/sweep2_h16_drop_0.25)')
    parser.add_argument('--phase', type=str, default=None,
                        help='Process all experiments in a phase (e.g., 02_vintage_baseline)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if results exist')
    parser.add_argument('--output-subdir', type=str, default='attention_analysis_year',
                        help='Subdirectory name for outputs within each experiment')
    parser.add_argument('--base-dir', type=str, default='experiments',
                        help='Base experiments directory')
    
    args = parser.parse_args()
    
    # Determine which experiments to process
    if args.experiment:
        experiments = [args.experiment]
        print(f"Processing single experiment: {args.experiment}")
    elif args.phase:
        # Process all experiment subdirs in specified phase
        phase_path = Path(args.base_dir) / args.phase
        if not phase_path.exists():
            print(f"Error: Phase directory does not exist: {phase_path}")
            return
        
        experiments = []
        for exp_dir in sorted(phase_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            # Check if it has checkpoints
            checkpoints_dir = exp_dir / 'checkpoints'
            if checkpoints_dir.exists() and list(checkpoints_dir.glob('*.ckpt')):
                rel_path = exp_dir.relative_to(args.base_dir)
                experiments.append(str(rel_path))
        
        print(f"Processing phase '{args.phase}': found {len(experiments)} experiments with checkpoints")
    else:
        experiments = find_all_experiments(args.base_dir)
        print(f"Found {len(experiments)} experiments with checkpoints")
    
    if not experiments:
        print("No experiments found to process")
        return
    
    # Process each experiment
    results = {
        'success': [],
        'skipped': [],
        'failed': [],
    }
    
    for i, exp_path in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(experiments)}] Processing: {exp_path}")
        print('='*70)
        
        # Check if results already exist
        if not args.force and has_existing_results(exp_path, args.output_subdir):
            print(f"Skipping (results exist): {exp_path}")
            results['skipped'].append(exp_path)
            continue
        
        # Run analysis
        success, error = run_attention_analysis(exp_path, args.output_subdir)
        
        if success:
            results['success'].append(exp_path)
        else:
            results['failed'].append({
                'experiment': exp_path,
                'error': error
            })
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(experiments)}")
    print(f"  Successful: {len(results['success'])}")
    print(f"  Skipped (existing): {len(results['skipped'])}")
    print(f"  Failed: {len(results['failed'])}")
    
    if results['failed']:
        print("\nFailed experiments:")
        for item in results['failed']:
            print(f"  - {item['experiment']}")
            if item['error']:
                print(f"    Error: {item['error'][:100]}")
    
    # Save summary
    summary_path = Path(args.base_dir) / f'batch_attention_analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()