"""
Validate that EnhancedQuantileLoss with all penalties disabled (weights=0.0)
is numerically equivalent to pytorch-forecasting's QuantileLoss.

Tests:
1. Forward pass equivalence (single batch)
2. Gradient equivalence (backprop)
3. Multi-step training equivalence (5 epochs)
4. Loss trajectory comparison

Run on CPU only to eliminate GPU non-determinism.
"""

import torch
import torch.nn as nn
import numpy as np
from pytorch_forecasting.metrics import QuantileLoss
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_losses import EnhancedQuantileLoss


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_forward_equivalence():
    """Test 1: Forward pass should be identical."""
    print("="*80)
    print("TEST 1: Forward Pass Equivalence")
    print("="*80)
    
    set_seed(42)
    
    # Create both losses
    loss_original = QuantileLoss()
    loss_enhanced = EnhancedQuantileLoss(
        collapse_weight=0.0,
        temporal_consistency_weight=0.0
    )
    
    # Generate test data
    batch_size, time_steps, n_quantiles = 32, 1, 7
    y_pred = torch.randn(batch_size, time_steps, n_quantiles, requires_grad=True)
    y_true = torch.randn(batch_size, time_steps)
    
    # Compute losses
    out_original = loss_original.loss(y_pred, y_true)
    out_enhanced = loss_enhanced.loss(y_pred, y_true)
    
    # Check shapes
    print(f"Original shape: {out_original.shape}")
    print(f"Enhanced shape: {out_enhanced.shape}")
    assert out_original.shape == out_enhanced.shape, "Shape mismatch!"
    
    # Check values
    diff = (out_original - out_enhanced).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✓ PASS: Forward passes are numerically equivalent")
        return True
    else:
        print("✗ FAIL: Forward passes differ!")
        return False


def test_gradient_equivalence():
    """Test 2: Gradients should be identical."""
    print("\n" + "="*80)
    print("TEST 2: Gradient Equivalence")
    print("="*80)
    
    set_seed(42)
    
    # Create losses
    loss_original = QuantileLoss()
    loss_enhanced = EnhancedQuantileLoss(
        collapse_weight=0.0,
        temporal_consistency_weight=0.0
    )
    
    # Create identical input tensors for both
    batch_size, time_steps, n_quantiles = 32, 1, 7
    y_pred_orig = torch.randn(batch_size, time_steps, n_quantiles, requires_grad=True)
    y_pred_enh = y_pred_orig.clone().detach().requires_grad_(True)
    y_true = torch.randn(batch_size, time_steps)
    
    # Forward pass
    out_original = loss_original.loss(y_pred_orig, y_true).mean()
    out_enhanced = loss_enhanced.loss(y_pred_enh, y_true).mean()
    
    # Backward pass
    out_original.backward()
    out_enhanced.backward()
    
    # Compare gradients
    grad_diff = (y_pred_orig.grad - y_pred_enh.grad).abs()
    max_grad_diff = grad_diff.max().item()
    mean_grad_diff = grad_diff.mean().item()
    
    print(f"Max gradient difference: {max_grad_diff:.2e}")
    print(f"Mean gradient difference: {mean_grad_diff:.2e}")
    
    if max_grad_diff < 1e-6:
        print("✓ PASS: Gradients are numerically equivalent")
        return True
    else:
        print("✗ FAIL: Gradients differ!")
        return False


def test_training_equivalence():
    """Test 3: Multi-step training should produce identical trajectories."""
    print("\n" + "="*80)
    print("TEST 3: Training Trajectory Equivalence (5 epochs)")
    print("="*80)
    
    # Simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 7)  # Output 7 quantiles
        
        def forward(self, x):
            return self.linear(x).unsqueeze(1)  # [batch, 1, 7]
    
    # Training function
    def train_model(loss_fn, n_epochs=5, n_batches=10):
        set_seed(42)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            for batch_idx in range(n_batches):
                # Generate deterministic data based on batch_idx
                set_seed(42 + batch_idx)
                x = torch.randn(32, 10)
                y = torch.randn(32, 1)
                
                # Forward
                y_pred = model(x)
                loss = loss_fn.loss(y_pred, y).mean()
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")
        
        return losses
    
    # Train with both losses
    print("\nTraining with QuantileLoss:")
    loss_original = QuantileLoss()
    losses_original = train_model(loss_original)
    
    print("\nTraining with EnhancedQuantileLoss (weights=0.0):")
    loss_enhanced = EnhancedQuantileLoss(
        collapse_weight=0.0,
        temporal_consistency_weight=0.0
    )
    losses_enhanced = train_model(loss_enhanced)
    
    # Compare trajectories
    print("\nLoss Comparison:")
    print(f"{'Epoch':<10} {'Original':<15} {'Enhanced':<15} {'Diff':<15}")
    print("-" * 55)
    
    all_close = True
    for i, (l_orig, l_enh) in enumerate(zip(losses_original, losses_enhanced)):
        diff = abs(l_orig - l_enh)
        print(f"{i+1:<10} {l_orig:<15.6f} {l_enh:<15.6f} {diff:<15.2e}")
        if diff > 1e-6:
            all_close = False
    
    if all_close:
        print("\n✓ PASS: Training trajectories are equivalent")
        return True
    else:
        print("\n✗ FAIL: Training trajectories diverge!")
        return False


def test_loss_properties():
    """Test 4: Verify loss properties match."""
    print("\n" + "="*80)
    print("TEST 4: Loss Properties")
    print("="*80)
    
    loss_original = QuantileLoss()
    loss_enhanced = EnhancedQuantileLoss(
        collapse_weight=0.0,
        temporal_consistency_weight=0.0
    )
    
    # Check quantiles
    print(f"Original quantiles: {loss_original.quantiles}")
    print(f"Enhanced quantiles: {loss_enhanced.quantiles}")
    assert loss_original.quantiles == loss_enhanced.quantiles, "Quantiles mismatch!"
    
    # Check inheritance
    print(f"\nOriginal base classes: {[c.__name__ for c in type(loss_original).__mro__]}")
    print(f"Enhanced base classes: {[c.__name__ for c in type(loss_enhanced).__mro__]}")
    
    print("\n✓ PASS: Loss properties match")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("VALIDATING EnhancedQuantileLoss EQUIVALENCE TO QuantileLoss")
    print("="*80)
    print("\nThis validates that our custom loss with penalties=0.0 is identical")
    print("to pytorch-forecasting's QuantileLoss.\n")
    
    # Force CPU
    if torch.cuda.is_available():
        print("WARNING: CUDA detected. Tests should run on CPU for reproducibility.")
        print("Results may differ slightly on GPU due to floating point precision.\n")
    
    results = {
        "Forward Pass": test_forward_equivalence(),
        "Gradients": test_gradient_equivalence(),
        "Training Trajectory": test_training_equivalence(),
        "Loss Properties": test_loss_properties(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("RESULT: All tests passed. EnhancedQuantileLoss is equivalent to QuantileLoss.")
        print("The phase4_baseline difference is likely due to CPU vs GPU training.")
    else:
        print("RESULT: Tests failed. There is a bug in EnhancedQuantileLoss implementation.")
        print("The phase4_baseline difference may be due to our loss wrapper.")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
