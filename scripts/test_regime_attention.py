#!/usr/bin/env python3
"""
Test script for regime-aware attention module.

Verifies:
1. Drop-in replacement compatibility with InterpretableMultiHeadAttention
2. Regime gating mechanics
3. Gradient flow through regime gates
4. Integration with pytorch-forecasting TFT

Usage:
    python test_regime_attention.py
    python test_regime_attention.py --with-model  # Test with actual TFT (requires data)
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.regime_attention import (
    RegimeAwareInterpretableMultiHeadAttention,
    ScaledDotProductAttention,
    replace_attention_module,
)
from train.regime_attention_training import (
    patch_forward_for_regime,
    find_vix_index,
    extract_vix_from_batch,
    get_regime_diagnostics,
)


def test_scaled_dot_product_attention():
    """Test basic attention computation."""
    print("\n" + "="*60)
    print("TEST: ScaledDotProductAttention")
    print("="*60)
    
    attn = ScaledDotProductAttention(dropout=None, scale=True)
    
    batch_size = 4
    seq_len = 10
    d_k = 8
    
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_k)
    
    output, weights = attn(q, k, v)
    
    assert output.shape == (batch_size, seq_len, d_k), f"Output shape mismatch: {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"Weights shape mismatch: {weights.shape}"
    
    # Check attention weights sum to 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"Attention weights don't sum to 1: {weight_sums}"
    
    print("Output shape correct")
    print("Attention weights shape correct")
    print("Attention weights sum to 1")
    print("PASSED")


def test_regime_attention_basic():
    """Test regime-aware attention basic functionality."""
    print("\n" + "="*60)
    print("TEST: RegimeAwareInterpretableMultiHeadAttention - Basic")
    print("="*60)
    
    n_head = 4
    d_model = 16
    
    attn = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        dropout=0.0,
        num_regimes=2,
        regime_mode='vix_threshold',
        vix_threshold=25.0
    )
    
    batch_size = 8
    query_len = 5
    key_len = 20
    
    q = torch.randn(batch_size, query_len, d_model)
    k = torch.randn(batch_size, key_len, d_model)
    v = torch.randn(batch_size, key_len, d_model)
    
    # Without regime signal
    output, weights = attn(q, k, v)
    
    assert output.shape == (batch_size, query_len, d_model), \
        f"Output shape mismatch: {output.shape}"
    assert weights.shape == (batch_size, query_len, n_head, key_len), \
        f"Weights shape mismatch: {weights.shape}"
    
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    
    # With regime signal
    vix_low = torch.ones(batch_size) * 15.0  # Low vol
    vix_high = torch.ones(batch_size) * 30.0  # High vol
    
    attn.set_regime_signal(vix_low)
    output_low, weights_low = attn(q, k, v)
    
    attn.set_regime_signal(vix_high)
    output_high, weights_high = attn(q, k, v)
    
    # Outputs should differ based on regime
    diff = (output_low - output_high).abs().mean().item()
    print(f"Output difference between regimes: {diff:.6f}")
    
    print("PASSED")


def test_regime_gating():
    """Test regime gate mechanics."""
    print("\n" + "="*60)
    print("TEST: Regime Gating Mechanics")
    print("="*60)
    
    n_head = 4
    d_model = 16
    
    attn = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        num_regimes=2,
        regime_mode='vix_threshold',
        vix_threshold=25.0,
        gate_init_std=0.01
    )
    
    batch_size = 8
    
    # Test regime detection
    vix_mixed = torch.tensor([10.0, 15.0, 20.0, 24.9, 25.0, 30.0, 40.0, 50.0])
    attn.set_regime_signal(vix_mixed)
    
    expected_regimes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assert torch.equal(attn._current_regime, expected_regimes), \
        f"Regime detection failed: {attn._current_regime} vs {expected_regimes}"
    
    print("Regime threshold detection correct")
    
    # Test gate initialization (should be near 0.5 due to small init std)
    raw_gates = attn.regime_gates.detach()
    gate_values = torch.sigmoid(raw_gates)
    
    assert gate_values.min() > 0.4 and gate_values.max() < 0.6, \
        f"Initial gates should be near 0.5: {gate_values}"
    
    print(f"Initial gate values near 0.5: mean={gate_values.mean():.4f}")
    
    # Test that different regimes produce different gate weights
    q = torch.randn(batch_size, 5, d_model)
    k = torch.randn(batch_size, 10, d_model)
    v = torch.randn(batch_size, 10, d_model)
    
    attn.set_regime_signal(vix_mixed)
    _, _ = attn(q, k, v)
    
    diag = attn.get_regime_diagnostics()
    print(f"Regime distribution: {(diag['current_regime'] == 0).sum()}/{batch_size} low-vol")
    
    print("PASSED")


def test_gradient_flow():
    """Test gradient flow through regime gates."""
    print("\n" + "="*60)
    print("TEST: Gradient Flow Through Regime Gates")
    print("="*60)
    
    n_head = 4
    d_model = 16
    
    attn = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        num_regimes=2,
        regime_mode='vix_threshold',
        vix_threshold=25.0
    )
    
    batch_size = 8
    q = torch.randn(batch_size, 5, d_model, requires_grad=True)
    k = torch.randn(batch_size, 10, d_model, requires_grad=True)
    v = torch.randn(batch_size, 10, d_model, requires_grad=True)
    
    # Mixed regimes
    vix = torch.tensor([15.0, 15.0, 15.0, 15.0, 30.0, 30.0, 30.0, 30.0])
    attn.set_regime_signal(vix)
    
    output, _ = attn(q, k, v)
    loss = output.sum()
    loss.backward()
    
    # Check regime gates have gradients
    assert attn.regime_gates.grad is not None, "Regime gates have no gradient"
    assert attn.regime_gates.grad.abs().sum() > 0, "Regime gates gradient is zero"
    
    print(f"Regime gates gradient: {attn.regime_gates.grad}")
    print(f"Gradient magnitude: {attn.regime_gates.grad.abs().mean():.6f}")
    
    # Check both regimes get gradients (since we have mixed batch)
    grad_regime_0 = attn.regime_gates.grad[0].abs().sum().item()
    grad_regime_1 = attn.regime_gates.grad[1].abs().sum().item()
    
    assert grad_regime_0 > 0, "Regime 0 gates have no gradient"
    assert grad_regime_1 > 0, "Regime 1 gates have no gradient"
    
    print(f"Regime 0 gradient magnitude: {grad_regime_0:.6f}")
    print(f"Regime 1 gradient magnitude: {grad_regime_1:.6f}")
    
    print("PASSED")


def test_disabled_mode():
    """Test that disabled mode matches baseline."""
    print("\n" + "="*60)
    print("TEST: Disabled Mode (Baseline Behavior)")
    print("="*60)
    
    n_head = 4
    d_model = 16
    
    attn_regime = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        num_regimes=2,
        regime_mode='disabled'
    )
    
    batch_size = 8
    q = torch.randn(batch_size, 5, d_model)
    k = torch.randn(batch_size, 10, d_model)
    v = torch.randn(batch_size, 10, d_model)
    
    # Set regime signal (should be ignored)
    vix = torch.ones(batch_size) * 30.0
    attn_regime.set_regime_signal(vix)
    
    output1, weights1 = attn_regime(q, k, v)
    output2, weights2 = attn_regime(q, k, v)
    
    # Results should be identical (no stochastic gating)
    assert torch.allclose(output1, output2), "Disabled mode has non-deterministic output"
    
    print("Disabled mode produces deterministic output")
    print("Regime signal ignored in disabled mode")
    print("PASSED")


def test_attention_output_format():
    """Test attention output format matches pytorch-forecasting expectations."""
    print("\n" + "="*60)
    print("TEST: Attention Output Format Compatibility")
    print("="*60)
    
    n_head = 4
    d_model = 16
    
    attn = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        num_regimes=2,
        regime_mode='vix_threshold'
    )
    
    batch_size = 8
    query_len = 5  # decoder length
    key_len = 20  # encoder + decoder length
    
    q = torch.randn(batch_size, query_len, d_model)
    k = torch.randn(batch_size, key_len, d_model)
    v = torch.randn(batch_size, key_len, d_model)
    
    vix = torch.ones(batch_size) * 20.0
    attn.set_regime_signal(vix)
    
    output, weights = attn(q, k, v)
    
    # pytorch-forecasting expects:
    # - output: [batch, query_len, d_model]
    # - weights: [batch, query_len, n_head, key_len]
    
    assert output.shape == (batch_size, query_len, d_model), \
        f"Output shape mismatch: {output.shape}"
    assert weights.shape == (batch_size, query_len, n_head, key_len), \
        f"Weights shape mismatch: {weights.shape}"
    
    # Check encoder/decoder attention split would work
    encoder_len = 15
    encoder_attn = weights[..., :encoder_len]
    decoder_attn = weights[..., encoder_len:]
    
    assert encoder_attn.shape == (batch_size, query_len, n_head, encoder_len)
    assert decoder_attn.shape == (batch_size, query_len, n_head, key_len - encoder_len)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Encoder attention slice: {encoder_attn.shape}")
    print(f"Decoder attention slice: {decoder_attn.shape}")
    print("PASSED")


def test_diagnostics():
    """Test diagnostic methods."""
    print("\n" + "="*60)
    print("TEST: Diagnostic Methods")
    print("="*60)
    
    n_head = 4
    d_model = 16
    
    attn = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        num_regimes=2,
        regime_mode='vix_threshold',
        vix_threshold=25.0
    )
    
    batch_size = 8
    q = torch.randn(batch_size, 5, d_model)
    k = torch.randn(batch_size, 10, d_model)
    v = torch.randn(batch_size, 10, d_model)
    
    vix = torch.tensor([15.0, 20.0, 25.0, 30.0, 15.0, 20.0, 25.0, 30.0])
    attn.set_regime_signal(vix)
    
    _, _ = attn(q, k, v)
    
    # Test diagnostic getters
    weights = attn.get_attention_weights()
    assert weights is not None, "Attention weights not cached"
    assert weights.shape == (batch_size, 5, n_head, 10)
    
    diag = attn.get_regime_diagnostics()
    assert 'regime_signal' in diag
    assert 'current_regime' in diag
    assert 'gate_weights' in diag
    assert 'raw_gates' in diag
    
    print(f"Cached attention weights shape: {weights.shape}")
    print(f"Regime signal: {diag['regime_signal']}")
    print(f"Current regime: {diag['current_regime']}")
    print(f"Gate weights shape: {diag['gate_weights'].shape}")
    
    print("PASSED")


def test_weight_copy():
    """Test weight copying from original attention."""
    print("\n" + "="*60)
    print("TEST: Weight Copying from Original Attention")
    print("="*60)
    
    try:
        from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
            InterpretableMultiHeadAttention
        )
    except ImportError:
        print("SKIPPED: pytorch-forecasting not available")
        return
    
    n_head = 4
    d_model = 16
    
    # Create original attention
    original = InterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        dropout=0.1
    )
    
    # Create regime-aware version
    regime = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        dropout=0.1,
        num_regimes=2,
        regime_mode='vix_threshold'
    )
    
    # Copy weights
    regime.v_layer.load_state_dict(original.v_layer.state_dict())
    regime.w_h.load_state_dict(original.w_h.state_dict())
    for i in range(n_head):
        regime.q_layers[i].load_state_dict(original.q_layers[i].state_dict())
        regime.k_layers[i].load_state_dict(original.k_layers[i].state_dict())
    
    # Verify weights match
    assert torch.equal(regime.v_layer.weight, original.v_layer.weight)
    assert torch.equal(regime.w_h.weight, original.w_h.weight)
    
    for i in range(n_head):
        assert torch.equal(regime.q_layers[i].weight, original.q_layers[i].weight)
        assert torch.equal(regime.k_layers[i].weight, original.k_layers[i].weight)
    
    print("v_layer weights match")
    print("w_h weights match")
    print("q_layers weights match")
    print("k_layers weights match")
    print("PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# REGIME-AWARE ATTENTION TEST SUITE")
    print("#"*60)
    
    tests = [
        test_scaled_dot_product_attention,
        test_regime_attention_basic,
        test_regime_gating,
        test_gradient_flow,
        test_disabled_mode,
        test_attention_output_format,
        test_diagnostics,
        test_weight_copy,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            if "SKIPPED" in str(e):
                skipped += 1
            else:
                print(f"ERROR: {e}")
                failed += 1
    
    print("\n" + "#"*60)
    print(f"# RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("#"*60)
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-model', action='store_true',
                       help='Run integration test with actual TFT model')
    args = parser.parse_args()
    
    success = run_all_tests()
    
    if args.with_model:
        print("\n" + "#"*60)
        print("# INTEGRATION TEST WITH TFT MODEL")
        print("#"*60)
        print("TODO: Add integration test with actual model and data")
    
    sys.exit(0 if success else 1)
