"""
Test script to verify regime diagnostics caching.

Verifies:
1. Non-persistent buffers don't affect checkpoint serialization
2. Diagnostics are cached correctly during forward pass
3. Collapse monitor can read cached values
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path.cwd()))

from src.regime_output import RegimeConditionalOutput


def test_caching():
    """Test that diagnostics are cached correctly."""
    print("=== Testing Regime Diagnostics Caching ===\n")
    
    # Create regime output layer
    regime_layer = RegimeConditionalOutput(
        hidden_size=16,
        output_size=7,
        num_regimes=2,
        routing_mode='learned'
    )
    
    # Create dummy input
    batch_size = 32
    seq_len = 1
    hidden_state = torch.randn(batch_size, seq_len, 16)
    
    print("1. Testing forward pass with caching...")
    regime_layer.eval()  # Must be in eval mode for caching
    
    # Forward pass
    output = regime_layer(hidden_state)
    
    # Check that caches are populated
    assert regime_layer._cached_routing_weights is not None, "Routing weights not cached!"
    assert regime_layer._cached_expert_preds_0 is not None, "Expert 0 predictions not cached!"
    assert regime_layer._cached_expert_preds_1 is not None, "Expert 1 predictions not cached!"
    
    print("   Caches populated during eval mode")
    print(f"   - Routing weights shape: {regime_layer._cached_routing_weights.shape}")
    print(f"   - Expert 0 preds shape: {regime_layer._cached_expert_preds_0.shape}")
    print(f"   - Expert 1 preds shape: {regime_layer._cached_expert_preds_1.shape}")
    
    # Verify data on CPU
    assert regime_layer._cached_routing_weights.device.type == 'cpu', "Cache not on CPU!"
    print("   Caches stored on CPU\n")
    
    print("2. Testing training mode (no caching)...")
    regime_layer.train()
    
    # Clear caches manually
    regime_layer._cached_routing_weights = None
    regime_layer._cached_expert_preds_0 = None
    regime_layer._cached_expert_preds_1 = None
    
    # Forward in training mode
    output = regime_layer(hidden_state)
    
    # Should NOT cache in training mode
    assert regime_layer._cached_routing_weights is None, "Cached in training mode!"
    print("   No caching in training mode\n")
    
    print("3. Testing checkpoint serialization...")
    
    # Save checkpoint
    checkpoint = {
        'model_state_dict': regime_layer.state_dict(),
        'config': {
            'hidden_size': 16,
            'output_size': 7,
            'num_regimes': 2,
            'routing_mode': 'learned'
        }
    }
    
    torch.save(checkpoint, '/tmp/test_regime_checkpoint.pt')
    
    # Load checkpoint
    loaded_checkpoint = torch.load('/tmp/test_regime_checkpoint.pt')
    
    new_regime_layer = RegimeConditionalOutput(**loaded_checkpoint['config'])
    new_regime_layer.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    print("   Checkpoint saved and loaded successfully")
    print("   Non-persistent buffers did not break serialization\n")
    
    print("4. Testing return_diagnostics flag...")
    
    regime_layer.eval()
    output, diagnostics = regime_layer(hidden_state, return_diagnostics=True)
    
    assert 'routing_weights' in diagnostics, "Routing weights not in diagnostics!"
    assert 'expert_preds' in diagnostics, "Expert predictions not in diagnostics!"
    assert len(diagnostics['expert_preds']) == 2, "Wrong number of expert predictions!"
    
    print("   Diagnostics returned correctly")
    print(f"   - Routing weights shape: {diagnostics['routing_weights'].shape}")
    print(f"   - Number of experts: {len(diagnostics['expert_preds'])}\n")
    
    print("=== All Tests Passed ===")


def test_disabled_mode():
    """Test that disabled mode (single expert) works."""
    print("\n=== Testing Disabled Mode ===\n")
    
    regime_layer = RegimeConditionalOutput(
        hidden_size=16,
        output_size=7,
        num_regimes=1,
        routing_mode='disabled'
    )
    
    hidden_state = torch.randn(32, 1, 16)
    regime_layer.eval()
    
    output = regime_layer(hidden_state)
    
    # Should not have caches in disabled mode
    assert regime_layer._cached_routing_weights is None, "Routing weights cached in disabled mode!"
    
    print("   Disabled mode works correctly")
    print("   No diagnostics cached for single expert\n")


if __name__ == '__main__':
    test_caching()
    test_disabled_mode()
    print("\nAll regime diagnostics tests passed!")
