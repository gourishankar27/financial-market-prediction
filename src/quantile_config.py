"""
Quantile configuration presets and utilities.

Provides standardized quantile configurations for TFT experiments,
with helper functions to find median index dynamically.

Presets:
- median (1q): [0.5] - Single point prediction
- 3q: [0.1, 0.5, 0.9] - Minimal uncertainty quantification  
- 7q: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98] - Full distribution (default)

Reference: Original TFT paper (Lim et al., 2021) uses 3 quantiles for 
most experiments, with 7 quantiles for detailed uncertainty analysis.
"""

from typing import List, Tuple

# Quantile presets
QUANTILE_PRESETS = {
    'median': [0.5],
    '3q': [0.1, 0.5, 0.9],
    '7q': [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
}

# Aliases for CLI convenience
QUANTILE_ALIASES = {
    '1q': 'median',
    '1': 'median',
    '3': '3q',
    '7': '7q',
}


def get_quantiles(preset: str) -> List[float]:
    """
    Get quantile list from preset name.
    
    Args:
        preset: Preset name ('median', '3q', '7q') or alias ('1q', '1', '3', '7')
    
    Returns:
        List of quantile values
    
    Raises:
        ValueError: If preset not recognized
    """
    # Resolve aliases
    preset = QUANTILE_ALIASES.get(preset, preset)
    
    if preset not in QUANTILE_PRESETS:
        valid = list(QUANTILE_PRESETS.keys()) + list(QUANTILE_ALIASES.keys())
        raise ValueError(f"Unknown quantile preset '{preset}'. Valid options: {valid}")
    
    return QUANTILE_PRESETS[preset]


def get_median_index(quantiles: List[float]) -> int:
    """
    Find index of median (0.5) quantile in list.
    
    Args:
        quantiles: List of quantile values
    
    Returns:
        Index of 0.5 quantile
    
    Raises:
        ValueError: If 0.5 not in quantiles list
    """
    try:
        return quantiles.index(0.5)
    except ValueError:
        raise ValueError(
            f"Median quantile (0.5) not found in quantiles: {quantiles}. "
            "All presets must include 0.5 for point prediction extraction."
        )


def get_output_size(quantiles: List[float]) -> int:
    """Get output size (number of quantiles)."""
    return len(quantiles)


def validate_quantiles(quantiles: List[float]) -> Tuple[bool, str]:
    """
    Validate quantile configuration.
    
    Checks:
    - All values in (0, 1)
    - 0.5 (median) is present
    - Values are sorted ascending
    - No duplicates
    
    Returns:
        (is_valid, error_message)
    """
    if not quantiles:
        return False, "Quantiles list cannot be empty"
    
    if any(q <= 0 or q >= 1 for q in quantiles):
        return False, f"All quantiles must be in (0, 1), got: {quantiles}"
    
    if 0.5 not in quantiles:
        return False, "Quantiles must include 0.5 (median) for point prediction"
    
    if quantiles != sorted(quantiles):
        return False, f"Quantiles must be sorted ascending, got: {quantiles}"
    
    if len(quantiles) != len(set(quantiles)):
        return False, f"Quantiles cannot contain duplicates: {quantiles}"
    
    return True, ""


# CLI argument configuration
def add_quantile_args(parser):
    """Add quantile-related arguments to argparse parser."""
    parser.add_argument(
        '--quantiles', 
        type=str, 
        default='7q',
        choices=['median', '1q', '3q', '7q'],
        help='Quantile preset: median/1q (single), 3q (standard), 7q (full distribution)'
    )
    return parser
