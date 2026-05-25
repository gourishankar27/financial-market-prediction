"""
Test script for EnhancedQuantileLoss implementation.

Validates:
1. Loss computes correctly with penalties
2. Checkpoint serialization works (Phase 3 issue)
3. Temporal consistency state tracking
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.custom_losses import EnhancedQuantileLoss

def test_basic_loss():
    """Test that loss computes without errors."""
    print("TEST 1: Basic loss computation...")
    
    loss_fn = EnhancedQuantileLoss(
        dist_mean_weight=0.1,
        dist_std_weight=0.1,
        temporal_consistency_weight=0.05
    )
    
    # Simulate batch of predictions
    batch_size, time_steps, n_quantiles = 32, 1, 7
    y_pred = torch.randn(batch_size, time_steps, n_quantiles)
    y_true = torch.randn(batch_size, time_steps)
    
    # Compute loss
    loss = loss_fn.loss(y_pred, y_true)
    
    assert loss.shape == (batch_size, time_steps, n_quantiles), \
        f"Expected shape {(batch_size, time_steps, n_quantiles)}, got {loss.shape}"
    
    assert not torch.isnan(loss).any(), "Loss contains NaN values"
    assert not torch.isinf(loss).any(), "Loss contains Inf values"
    
    print(f"  Loss shape: {loss.shape} - OK")
    print(f"  Loss mean: {loss.mean().item():.4f}")
    print(f"  Pred mean: {loss_fn.last_pred_mean:.6f}")
    print(f"  Pred std: {loss_fn.last_pred_std:.6f}")
    if loss_fn.last_mean_penalty:
        print(f"  Mean penalty: {loss_fn.last_mean_penalty:.6f}")
    if loss_fn.last_std_penalty:
        print(f"  Std penalty: {loss_fn.last_std_penalty:.6f}")
    print("  PASSED\n")

def test_temporal_consistency():
    """Test temporal consistency penalty across batches."""
    print("TEST 2: Temporal consistency tracking...")
    
    loss_fn = EnhancedQuantileLoss(temporal_consistency_weight=0.1)
    loss_fn.train(False)  # Set to eval mode
    
    batch_size, time_steps, n_quantiles = 4, 1, 7
    
    # First batch
    y_pred_1 = torch.randn(batch_size, time_steps, n_quantiles)
    y_true_1 = torch.randn(batch_size, time_steps)
    loss_1 = loss_fn.loss(y_pred_1, y_true_1)
    
    print(f"  First batch - temporal penalty: {loss_fn.last_temporal_penalty}")
    assert loss_fn.last_temporal_penalty == 0.0, "First batch should have zero penalty"
    
    # Second batch - should have penalty
    y_pred_2 = torch.randn(batch_size, time_steps, n_quantiles)
    y_true_2 = torch.randn(batch_size, time_steps)
    loss_2 = loss_fn.loss(y_pred_2, y_true_2)
    
    print(f"  Second batch - temporal penalty: {loss_fn.last_temporal_penalty:.6f}")
    assert loss_fn.last_temporal_penalty > 0, "Second batch should have non-zero penalty"
    
    # Reset and check
    loss_fn.reset_temporal_state()
    y_pred_3 = torch.randn(batch_size, time_steps, n_quantiles)
    y_true_3 = torch.randn(batch_size, time_steps)
    loss_3 = loss_fn.loss(y_pred_3, y_true_3)
    
    print(f"  After reset - temporal penalty: {loss_fn.last_temporal_penalty}")
    assert loss_fn.last_temporal_penalty == 0.0, "Reset should clear penalty"
    
    print("  PASSED\n")

def test_serialization():
    """Test that loss can be saved and loaded (Phase 3 critical issue)."""
    print("TEST 3: Checkpoint serialization...")
    
    import tempfile
    
    loss_fn = EnhancedQuantileLoss(
        dist_mean_weight=0.1,
        dist_std_weight=0.1,
        temporal_consistency_weight=0.05
    )
    
    # Create a dummy model dict
    checkpoint = {
        'loss': loss_fn,
        'epoch': 10,
        'val_loss': 0.42
    }
    
    # Try to save
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    
    try:
        torch.save(checkpoint, temp_path)
        print(f"  Saved checkpoint to {temp_path} - OK")
        
        # Try to load
        loaded = torch.load(temp_path)
        loaded_loss = loaded['loss']
        
        assert isinstance(loaded_loss, EnhancedQuantileLoss), \
            f"Expected EnhancedQuantileLoss, got {type(loaded_loss)}"
        assert loaded_loss.dist_mean_weight == 0.1, "Weight not preserved"
        assert loaded_loss.temporal_consistency_weight == 0.05, "Weight not preserved"
        
        print(f"  Loaded checkpoint successfully - OK")
        print(f"  Loss type: {type(loaded_loss).__name__}")
        print(f"  Weights preserved: dist_mean={loaded_loss.dist_mean_weight}, temporal={loaded_loss.temporal_consistency_weight}")
        print("  PASSED\n")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_penalty_toggles():
    """Test that penalties can be individually disabled."""
    print("TEST 4: Penalty toggles...")
    
    batch_size, time_steps, n_quantiles = 8, 1, 7
    y_pred = torch.randn(batch_size, time_steps, n_quantiles)
    y_true = torch.randn(batch_size, time_steps)
    
    # All disabled
    loss_fn_none = EnhancedQuantileLoss()
    loss_none = loss_fn_none.loss(y_pred, y_true)
    print(f"  All disabled - mean: {loss_none.mean().item():.4f}")
    
    # Only distribution
    loss_fn_dist = EnhancedQuantileLoss(dist_mean_weight=0.1, dist_std_weight=0.1)
    loss_dist = loss_fn_dist.loss(y_pred, y_true)
    print(f"  Distribution only - mean: {loss_dist.mean().item():.4f}")
    assert loss_dist.mean() > loss_none.mean(), "Distribution penalty should increase loss"
    
    # Only temporal (eval mode)
    loss_fn_temp = EnhancedQuantileLoss(temporal_consistency_weight=0.1)
    loss_fn_temp.train(False)
    loss_temp = loss_fn_temp.loss(y_pred, y_true)
    print(f"  Temporal only - mean: {loss_temp.mean().item():.4f}")
    
    # All enabled
    loss_fn_all = EnhancedQuantileLoss(
        dist_mean_weight=0.1,
        dist_std_weight=0.1,
        temporal_consistency_weight=0.1
    )
    loss_fn_all.train(False)
    loss_all = loss_fn_all.loss(y_pred, y_true)
    print(f"  All enabled - mean: {loss_all.mean().item():.4f}")
    
    print("  PASSED\n")

if __name__ == '__main__':
    print("="*70)
    print("EnhancedQuantileLoss Validation Tests")
    print("="*70 + "\n")
    
    test_basic_loss()
    test_temporal_consistency()
    test_serialization()
    test_penalty_toggles()
    
    print("="*70)
    print("ALL TESTS PASSED")
    print("="*70)
    print("\nReady for training experiments.")
