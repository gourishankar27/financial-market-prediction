"""
Test if routing implementation correctly propagates gradients to both experts.

This isolates the routing mechanism to verify:
1. Soft routing computes correct weighted combination
2. Gradients flow to both experts
3. Router learns from expert performance differences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path.cwd()))

from src.regime_output import RegimeConditionalOutput


def test_gradient_flow():
    """Test that gradients flow to all experts through soft routing."""
    print("=== Testing Gradient Flow Through Soft Routing ===\n")
    
    torch.manual_seed(42)
    
    # Create regime output layer
    regime_layer = RegimeConditionalOutput(
        hidden_size=16,
        output_size=7,
        num_regimes=2,
        routing_mode='learned'
    )
    
    # Training mode (important for gradient tracking)
    regime_layer.train()
    
    # Create dummy input and target
    batch_size = 32
    hidden_state = torch.randn(batch_size, 1, 16, requires_grad=True)
    target = torch.randn(batch_size, 1, 7)
    
    print("1. Forward pass...")
    output = regime_layer(hidden_state)
    
    # Simple MSE loss
    loss = F.mse_loss(output, target)
    print(f"   Loss: {loss.item():.6f}\n")
    
    print("2. Backward pass...")
    loss.backward()
    
    # Check gradients on all components
    print("3. Checking gradients:\n")
    
    # Router gradients
    if regime_layer.router.weight.grad is not None:
        router_grad_norm = regime_layer.router.weight.grad.norm().item()
        print(f"   Router weight grad norm: {router_grad_norm:.6f}")
    else:
        print("   ❌ Router has NO gradients!")
    
    # Expert 0 gradients
    expert0_grad = regime_layer.experts[0].weight.grad
    if expert0_grad is not None:
        expert0_grad_norm = expert0_grad.norm().item()
        print(f"   Expert 0 weight grad norm: {expert0_grad_norm:.6f}")
    else:
        print("   ❌ Expert 0 has NO gradients!")
    
    # Expert 1 gradients
    expert1_grad = regime_layer.experts[1].weight.grad
    if expert1_grad is not None:
        expert1_grad_norm = expert1_grad.norm().item()
        print(f"   Expert 1 weight grad norm: {expert1_grad_norm:.6f}")
    else:
        print("   ❌ Expert 1 has NO gradients!")
    
    print()
    
    # Check routing weights
    with torch.no_grad():
        regime_layer.eval()
        output, diagnostics = regime_layer(hidden_state, return_diagnostics=True)
        routing_weights = diagnostics['routing_weights']
        
        # Average routing weights across batch
        avg_routing = routing_weights.mean(dim=0).squeeze()  # [num_regimes]
        print(f"4. Average routing weights: {avg_routing.numpy()}")
        print(f"   (Should be roughly balanced, not [1.0, 0.0])\n")
    
    # Verify both experts received gradients
    if expert0_grad is not None and expert1_grad is not None:
        ratio = expert1_grad_norm / expert0_grad_norm
        print(f"5. Expert gradient ratio (E1/E0): {ratio:.6f}")
        
        if ratio < 0.01:
            print("   ⚠️  Expert 1 gradients are 100x smaller - may not learn!")
        elif ratio > 100:
            print("   ⚠️  Expert 0 gradients are 100x smaller - may not learn!")
        else:
            print("   ✓ Both experts receiving comparable gradients")
    
    print("\n" + "="*60)


def test_routing_symmetry():
    """Test if routing is symmetric (no inherent bias toward one expert)."""
    print("\n=== Testing Routing Symmetry ===\n")
    
    torch.manual_seed(42)
    
    regime_layer = RegimeConditionalOutput(
        hidden_size=16,
        output_size=7,
        num_regimes=2,
        routing_mode='learned'
    )
    
    regime_layer.eval()
    
    # Test with multiple random inputs
    routing_weights_list = []
    
    for _ in range(100):
        hidden_state = torch.randn(32, 1, 16)
        output, diagnostics = regime_layer(hidden_state, return_diagnostics=True)
        routing_weights = diagnostics['routing_weights']
        
        # Average across batch and time
        avg_routing = routing_weights.mean(dim=(0, 1))  # [num_regimes]
        routing_weights_list.append(avg_routing)
    
    # Stack and average across all samples
    all_routing = torch.stack(routing_weights_list)
    final_avg = all_routing.mean(dim=0)
    final_std = all_routing.std(dim=0)
    
    print(f"Average routing over 100 random batches:")
    print(f"  Regime 0: {final_avg[0]:.4f} ± {final_std[0]:.4f}")
    print(f"  Regime 1: {final_avg[1]:.4f} ± {final_std[1]:.4f}")
    print()
    
    if abs(final_avg[0] - 0.5) > 0.1:
        print(f"⚠️  Routing is biased! Expected ~0.5 for each regime at init")
        print(f"   Router may have bad initialization")
    else:
        print(f"✓ Routing is approximately balanced at initialization")
    
    print("\n" + "="*60)


def test_router_logits():
    """Check router logit magnitudes."""
    print("\n=== Testing Router Logit Magnitudes ===\n")
    
    torch.manual_seed(42)
    
    regime_layer = RegimeConditionalOutput(
        hidden_size=16,
        output_size=7,
        num_regimes=2,
        routing_mode='learned'
    )
    
    regime_layer.eval()
    
    hidden_state = torch.randn(32, 1, 16)
    
    # Get router logits directly
    router_logits = regime_layer.router(hidden_state)  # [batch, seq_len, num_regimes]
    
    # Statistics
    logit_mean = router_logits.mean(dim=(0, 1))  # [num_regimes]
    logit_std = router_logits.std(dim=(0, 1))
    logit_diff = router_logits[:, :, 0] - router_logits[:, :, 1]  # Difference
    
    print(f"Router logit statistics (before softmax):")
    print(f"  Regime 0: mean={logit_mean[0]:.4f}, std={logit_std[0]:.4f}")
    print(f"  Regime 1: mean={logit_mean[1]:.4f}, std={logit_std[1]:.4f}")
    print(f"  Difference (R0-R1): mean={logit_diff.mean():.4f}, std={logit_diff.std():.4f}")
    print()
    
    # After softmax
    routing_weights = F.softmax(router_logits, dim=-1)
    weight_mean = routing_weights.mean(dim=(0, 1))
    
    print(f"After softmax:")
    print(f"  Regime 0: {weight_mean[0]:.4f}")
    print(f"  Regime 1: {weight_mean[1]:.4f}")
    print()
    
    # Check for strong bias
    if abs(logit_mean[0] - logit_mean[1]) > 1.0:
        print(f"⚠️  Large logit difference ({abs(logit_mean[0] - logit_mean[1]):.2f})")
        print(f"   This creates strong routing bias before any training!")
        print(f"   Router initialization may be problematic")
    else:
        print(f"✓ Logit difference is small (<1.0)")
    
    print("\n" + "="*60)


def test_expert_outputs_differ():
    """Verify experts can produce different outputs."""
    print("\n=== Testing Expert Output Diversity ===\n")
    
    torch.manual_seed(42)
    
    regime_layer = RegimeConditionalOutput(
        hidden_size=16,
        output_size=7,
        num_regimes=2,
        routing_mode='learned'
    )
    
    regime_layer.eval()
    
    hidden_state = torch.randn(32, 1, 16)
    
    output, diagnostics = regime_layer(hidden_state, return_diagnostics=True)
    expert_preds = diagnostics['expert_preds']
    
    # Compare expert outputs
    expert0_preds = expert_preds[0]  # [batch, seq_len, output_size]
    expert1_preds = expert_preds[1]
    
    # Compute difference
    pred_diff = (expert0_preds - expert1_preds).abs().mean().item()
    
    print(f"Expert prediction statistics:")
    print(f"  Expert 0 mean: {expert0_preds.mean():.6f}")
    print(f"  Expert 1 mean: {expert1_preds.mean():.6f}")
    print(f"  Absolute difference: {pred_diff:.6f}")
    print()
    
    if pred_diff < 0.01:
        print(f"⚠️  Experts produce nearly identical outputs!")
        print(f"   They may be initialized too similarly")
    else:
        print(f"✓ Experts produce different outputs (good)")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    test_gradient_flow()
    test_routing_symmetry()
    test_router_logits()
    test_expert_outputs_differ()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nIf you see any ❌ or ⚠️  above, the implementation has issues.")
    print("Otherwise, the routing mechanism is working correctly in isolation.")
    print("\nNext step: Test with actual TFT training to see if collapse occurs")
    print("due to training dynamics rather than implementation bugs.")
