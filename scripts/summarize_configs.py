#!/usr/bin/env python3
"""
Summarize config.json values across all experiments in a phase directory.

Usage:
    python summarize_configs.py experiments/04_custom_losses
    python summarize_configs.py experiments/04_custom_losses --output summary.csv
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Set
import pandas as pd


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary with dot notation for keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to comma-separated strings for display
            items.append((new_key, ', '.join(map(str, v)) if v else '[]'))
        else:
            items.append((new_key, v))
    return dict(items)


def load_experiment_configs(phase_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all config.json files from experiment subdirectories."""
    configs = {}
    
    for exp_dir in sorted(phase_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
            
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            print(f"Warning: No config.json in {exp_dir.name}")
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                configs[exp_dir.name] = flatten_dict(config)
        except Exception as e:
            print(f"Error loading {config_path}: {e}")
            
    return configs


def get_all_keys(configs: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Get union of all keys across all configs."""
    all_keys = set()
    for config in configs.values():
        all_keys.update(config.keys())
    return all_keys


def summarize_key_values(configs: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Summarize values for a specific key across all experiments.
    
    Returns:
        dict with 'unique_values', 'counts', 'all_same'
    """
    values = []
    for exp_name, config in configs.items():
        if key in config:
            values.append(config[key])
    
    if not values:
        return {
            'unique_values': [],
            'counts': {},
            'all_same': True,
            'present_in': 0
        }
    
    # Count occurrences
    value_counts = defaultdict(int)
    for v in values:
        # Convert to string for hashability
        v_str = str(v)
        value_counts[v_str] += 1
    
    unique_values = sorted(value_counts.keys())
    all_same = len(unique_values) == 1
    
    return {
        'unique_values': unique_values,
        'counts': dict(value_counts),
        'all_same': all_same,
        'present_in': len(values)
    }


def print_summary(configs: Dict[str, Dict[str, Any]], verbose: bool = False):
    """Print human-readable summary of config variations."""
    print(f"\nFound {len(configs)} experiments")
    print("=" * 80)
    
    all_keys = sorted(get_all_keys(configs))
    
    # Group keys by prefix for better organization
    key_groups = defaultdict(list)
    for key in all_keys:
        prefix = key.split('.')[0]
        key_groups[prefix].append(key)
    
    for group_name in sorted(key_groups.keys()):
        print(f"\n{group_name.upper()}:")
        print("-" * 80)
        
        for key in sorted(key_groups[group_name]):
            summary = summarize_key_values(configs, key)
            
            if summary['present_in'] == 0:
                continue
                
            # Show key name
            short_key = key.split('.')[-1] if '.' in key else key
            coverage = f"({summary['present_in']}/{len(configs)})" if summary['present_in'] < len(configs) else ""
            
            if summary['all_same']:
                # All experiments have same value - show concisely
                value = summary['unique_values'][0]
                print(f"  {key:40s} = {value} {coverage}")
            else:
                # Values differ - show distribution
                print(f"  {key:40s} VARIES {coverage}")
                if verbose or len(summary['unique_values']) <= 10:
                    for value in summary['unique_values']:
                        count = summary['counts'][value]
                        print(f"    {value:35s} : {count:3d} experiments")
                else:
                    print(f"    ({len(summary['unique_values'])} unique values - use --verbose to see all)")


def create_variation_table(configs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a table showing only keys that vary across experiments."""
    all_keys = sorted(get_all_keys(configs))
    
    varying_keys = []
    for key in all_keys:
        summary = summarize_key_values(configs, key)
        if not summary['all_same'] and summary['present_in'] > 0:
            varying_keys.append(key)
    
    if not varying_keys:
        return pd.DataFrame()
    
    # Build table
    data = []
    for exp_name, config in sorted(configs.items()):
        row = {'experiment': exp_name}
        for key in varying_keys:
            row[key] = config.get(key, 'N/A')
        data.append(row)
    
    return pd.DataFrame(data)


def create_full_table(configs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a complete table with all keys."""
    all_keys = sorted(get_all_keys(configs))
    
    data = []
    for exp_name, config in sorted(configs.items()):
        row = {'experiment': exp_name}
        for key in all_keys:
            row[key] = config.get(key, 'N/A')
        data.append(row)
    
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize config.json values across experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Print summary to console
    python summarize_configs.py experiments/04_custom_losses
    
    # Show all unique values
    python summarize_configs.py experiments/04_custom_losses --verbose
    
    # Export variations to CSV
    python summarize_configs.py experiments/04_custom_losses --output variations.csv
    
    # Export full table
    python summarize_configs.py experiments/04_custom_losses --output full.csv --full
        """
    )
    
    parser.add_argument('phase_dir', type=Path,
                       help='Path to phase directory containing experiment subdirs')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output CSV file (defaults to console output)')
    parser.add_argument('--full', action='store_true',
                       help='Export all keys, not just varying ones')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all unique values even for keys with many variations')
    
    args = parser.parse_args()
    
    if not args.phase_dir.exists():
        print(f"Error: Directory not found: {args.phase_dir}")
        return 1
    
    # Load all configs
    configs = load_experiment_configs(args.phase_dir)
    
    if not configs:
        print(f"No config.json files found in {args.phase_dir}")
        return 1
    
    # Print or export
    if args.output:
        if args.full:
            df = create_full_table(configs)
        else:
            df = create_variation_table(configs)
        
        if df.empty:
            print("No varying keys found across experiments")
            return 0
            
        df.to_csv(args.output, index=False)
        print(f"Exported {len(df)} experiments × {len(df.columns)-1} keys to {args.output}")
    else:
        print_summary(configs, verbose=args.verbose)
    
    return 0


if __name__ == '__main__':
    exit(main())
