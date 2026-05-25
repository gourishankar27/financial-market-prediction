"""
Unit tests for RegimeConditionalOutput module.

Tests basic functionality before integration with TFT training.

Run:
    python test_regime_output.py
"""

import torch
import torch.nn as nn
from src.regime_output import RegimeConditionalOutput, replace_output_layer


def test_shape_correctness():
    """Test output shapes match expected TFT dimensions."""
    print("\n" + "="*80)
    print("TEST 1: Shape Correctness")
    print("="*80)
    
    hidden_size = 16
    output_size = 7
    batch_size = 32
    seq_len = 1
    
    # Test learned routing
    module = RegimeConditionalOutput(
        hidden_size=hidden_size,
        output_size=output_size,
        num_regimes=2,
        routing_mode='learned'
    )
    
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    output = module(hidden_state)
    
    expected_shape = (batch_size, seq_len, output_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape: {output.shape}")
    
    # Test with diagnostics
    output, diagnostics = module(hidden_state, return_diagnostics=True)
    assert output.shape == expected_shape
    assert diagnostics['routing_weights'].shape == (batch_size, seq_len, 2)
    assert len(diagnostics['expert_preds']) == 2
    print(f"✓ Diagnostics shape: routing_weights={diagnostics['routing_weights'].shape}")
    
    # Test disabled mode
    module_disabled = RegimeConditionalOutput(
        hidden_size=hidden_size,
        output_size=output_size,
        num_regimes=1,
        routing_mode='disabled'
    )
    
    output_disabled = module_disabled(hidden_state)
    assert output_disabled.shape == expected_shape
    print(f"✓ Disabled mode shape: {output_disabled.shape}")
    
    print("\n✓ All shape tests passed")


def test_routing_behavior():
    """Test routing weights sum to 1 and are in valid range."""
    print("\n" + "="*80)
    print("TEST 2: Routing Behavior")
    print("="*80)
    
    hidden_size = 16
    output_size = 7
    batch_size = 32
    seq_len = 5  # Multi-horizon
    
    module = RegimeConditionalOutput(
        hidden_size=hidden_size,
        output_size=output_size,
        num_regimes=2,
        routing_mode='learned'
    )
    
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    _, diagnostics = module(hidden_state, return_diagnostics=True)
    
    routing_weights = diagnostics['routing_weights']
    
    # Check weights sum to 1 across regimes
    weight_sums = routing_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6), \
        f"Routing weights don't sum to 1: {weight_sums}"
    print(f"✓ Routing weights sum to 1.0 (mean: {weight_sums.mean():.6f})")
    
    # Check weights are in [0, 1]
    assert (routing_weights >= 0).all() and (routing_weights <= 1).all(), \
        "Routing weights outside [0, 1] range"
    print(f"✓ Routing weights in [0, 1] range")
    
    # Print routing statistics
    print(f"\nRouting statistics (across batch):")
    print(f"  Regime 0 (normal): mean={routing_weights[:, :, 0].mean():.3f}, "
          f"std={routing_weights[:, :, 0].std():.3f}")
    print(f"  Regime 1 (volatile): mean={routing_weights[:, :, 1].mean():.3f}, "
          f"std={routing_weights[:, :, 1].std():.3f}")
    
    print("\n✓ All routing behavior tests passed")


def test_gradient_flow():
    """Test gradients flow through all experts and router."""
    print("\n" + "="*80)
    print("TEST 3: Gradient Flow")
    print("="*80)
    
    hidden_size = 16
    output_size = 7
    batch_size = 8
    seq_len = 1
    
    module = RegimeConditionalOutput(
        hidden_size=hidden_size,
        output_size=output_size,
        num_regimes=2,
        routing_mode='learned'
    )
    
    hidden_state = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    target = torch.randn(batch_size, seq_len, output_size)
    
    # Forward pass
    output = module(hidden_state)
    loss = ((output - target) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Check all parameters have gradients
    for name, param in module.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        print(f"✓ Gradient for {name}: mean={param.grad.mean():.6f}, "
              f"std={param.grad.std():.6f}")
    
    # Check hidden_state received gradients
    assert hidden_state.grad is not None, "No gradient for input"
    print(f"✓ Input gradient: mean={hidden_state.grad.mean():.6f}")
    
    print("\n✓ All gradient flow tests passed")


def test_parameter_count():
    """Test parameter count matches expected."""
    print("\n" + "="*80)
    print("TEST 4: Parameter Count")
    print("="*80)
    
    hidden_size = 16
    output_size = 7
    num_regimes = 2
    
    module = RegimeConditionalOutput(
        hidden_size=hidden_size,
        output_size=output_size,
        num_regimes=num_regimes,
        routing_mode='learned'
    )
    
    param_info = module.get_expert_parameters()
    
    # Expected counts
    expert_params_per_head = (hidden_size + 1) * output_size  # Weight + bias
    expected_expert_params = num_regimes * expert_params_per_head
    expected_router_params = (hidden_size + 1) * num_regimes  # Weight + bias
    expected_total = expected_expert_params + expected_router_params
    
    assert param_info['experts'] == expected_expert_params, \
        f"Expected {expected_expert_params} expert params, got {param_info['experts']}"
    assert param_info['router'] == expected_router_params, \
        f"Expected {expected_router_params} router params, got {param_info['router']}"
    assert param_info['total'] == expected_total, \
        f"Expected {expected_total} total params, got {param_info['total']}"
    
    print(f"Parameter counts:")
    print(f"  Experts: {param_info['experts']} (expected: {expected_expert_params})")
    print(f"  Router: {param_info['router']} (expected: {expected_router_params})")
    print(f"  Total: {param_info['total']} (expected: {expected_total})")
    
    # Compare to baseline
    baseline_params = (hidden_size + 1) * output_size
    overhead = param_info['total'] - baseline_params
    overhead_pct = (overhead / baseline_params) * 100
    
    print(f"\nComparison to baseline:")
    print(f"  Baseline (single Linear): {baseline_params} params")
    print(f"  Regime-conditional: {param_info['total']} params")
    print(f"  Overhead: {overhead} params (+{overhead_pct:.1f}%)")
    
    print("\n✓ All parameter count tests passed")


def test_disabled_mode_equivalence():
    """Test disabled mode behaves like baseline."""
    print("\n" + "="*80)
    print("TEST 5: Disabled Mode Equivalence")
    print("="*80)
    
    hidden_size = 16
    output_size = 7
    batch_size = 32
    seq_len = 1
    
    # Disabled mode
    module_disabled = RegimeConditionalOutput(
        hidden_size=hidden_size,
        output_size=output_size,
        num_regimes=1,
        routing_mode='disabled'
    )
    
    # Baseline
    baseline = nn.Linear(hidden_size, output_size)
    
    # Copy weights to make them identical
    with torch.no_grad():
        baseline.weight.copy_(module_disabled.experts[0].weight)
        baseline.bias.copy_(module_disabled.experts[0].bias)
    
    # Test identical output
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    output_disabled = module_disabled(hidden_state)
    output_baseline = baseline(hidden_state)
    
    assert torch.allclose(output_disabled, output_baseline, atol=1e-6), \
        "Disabled mode output doesn't match baseline"
    
    print(f"✓ Disabled mode output matches baseline (max diff: {(output_disabled - output_baseline).abs().max():.2e})")
    
    # Test parameter count
    disabled_params = module_disabled.get_expert_parameters()['total']
    baseline_params = sum(p.numel() for p in baseline.parameters())
    
    assert disabled_params == baseline_params, \
        f"Disabled mode has {disabled_params} params, baseline has {baseline_params}"
    
    print(f"✓ Disabled mode parameter count matches baseline ({disabled_params} params)")
    
    print("\n✓ All disabled mode tests passed")


def test_multi_horizon():
    """Test with realistic multi-horizon sequences."""
    print("\n" + "="*80)
    print("TEST 6: Multi-Horizon Prediction")
    print("="*80)
    
    hidden_size = 16
    output_size = 7
    batch_size = 16
    seq_len = 5  # Predict 5 steps ahead
    
    module = RegimeConditionalOutput(
        hidden_size=hidden_size,
        output_size=output_size,
        num_regimes=2,
        routing_mode='learned'
    )
    
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    output, diagnostics = module(hidden_state, return_diagnostics=True)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, output_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Multi-horizon output shape: {output.shape}")
    
    # Check routing per timestep
    routing_weights = diagnostics['routing_weights']
    print(f"✓ Per-timestep routing shape: {routing_weights.shape}")
    
    # Check routing varies across time
    routing_variance = routing_weights.var(dim=1).mean()
    print(f"  Routing variance across time: {routing_variance:.6f}")
    
    # Check expert predictions
    for i, expert_pred in enumerate(diagnostics['expert_preds']):
        assert expert_pred.shape == expected_shape
        print(f"✓ Expert {i} predictions shape: {expert_pred.shape}")
    
    print("\n✓ All multi-horizon tests passed")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*80)
    print("REGIME-CONDITIONAL OUTPUT - UNIT TEST SUITE")
    print("="*80)
    
    tests = [
        test_shape_correctness,
        test_routing_behavior,
        test_gradient_flow,
        test_parameter_count,
        test_disabled_mode_equivalence,
        test_multi_horizon,
    ]
    
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_fn.__name__}")
            print(f"Error: {e}")
            raise
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    print("\nRegime-conditional output module is ready for integration.")
    print("\nNext steps:")
    print("  1. Apply patches to train_tft.py (see train_tft_patches.py)")
    print("  2. Test with single epoch: python train_tft.py --experiment-name test_regime --regime-output --max-epochs 1")
    print("  3. Verify checkpoint saving/loading")
    print("  4. Run full training experiments")


if __name__ == '__main__':
    torch.manual_seed(42)
    run_all_tests()
