"""
Regime-Aware Interpretable Multi-Head Attention for TFT.

Drop-in replacement for pytorch-forecasting's InterpretableMultiHeadAttention
that modulates attention based on market regime (detected via VIX threshold).

Architecture:
    Standard InterpretableMultiHeadAttention with regime-conditional scaling:
    1. Compute attention logits as normal: attn = softmax(QK^T / sqrt(d_k))
    2. Detect regime from VIX signal (threshold-based, deterministic)
    3. Apply learned per-head regime gates: attn_scaled = attn * gate[regime, head]
    
    The regime gates are small scalar multipliers per head that learn to
    amplify/dampen specific heads based on current market conditions.

Design Rationale:
    - Your attention analysis shows the encoder already adapts behavior by regime
      (lower entropy in 2022, concentration on recent timesteps during volatility)
    - VIX is already the most-attended feature
    - Hypothesis: explicit regime modulation provides stronger gradient signal
      to the output layer, not that attention isn't adapting at all
    - Minimal parameter addition to avoid capacity-collapse (hidden_size > 16-18)

Usage:
    # Replace attention module after model creation
    from regime_attention import replace_attention_module
    
    model = TemporalFusionTransformer.from_dataset(dataset, ...)
    model = replace_attention_module(
        model,
        regime_mode='vix_threshold',
        vix_threshold=25.0,
        num_regimes=2
    )

References:
    - Original TFT: Lim et al. (2021) "Temporal Fusion Transformers for 
      Interpretable Multi-horizon Time Series Forecasting"
    - Regime-switching: Hamilton (1989) regime-switching models
    - Mixture of Experts gating: Shazeer et al. (2017) "Outrageously Large
      Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
"""

import math
from typing import Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (copied from pytorch-forecasting for standalone use).
    
    Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout: float = None, scale: bool = True, mask_bias: float = -1e9):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale
        self.mask_bias = mask_bias
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor [batch * n_heads, query_len, d_k]
            k: Key tensor [batch * n_heads, key_len, d_k]  
            v: Value tensor [batch * n_heads, key_len, d_v]
            mask: Optional attention mask [batch, query_len, key_len]
            
        Returns:
            output: Attended values [batch * n_heads, query_len, d_v]
            attn: Attention weights [batch * n_heads, query_len, key_len]
        """
        # Compute attention scores
        attn = torch.bmm(q, k.permute(0, 2, 1))  # [batch*heads, q_len, k_len]
        
        if self.scale:
            dimension = torch.as_tensor(
                k.size(-1), dtype=attn.dtype, device=attn.device
            ).sqrt()
            attn = attn / dimension
        
        if mask is not None:
            attn = attn.masked_fill(mask, self.mask_bias)
        
        attn = self.softmax(attn)
        
        if self.dropout is not None:
            attn = self.dropout(attn)
        
        output = torch.bmm(attn, v)
        return output, attn


class RegimeAwareInterpretableMultiHeadAttention(nn.Module):
    """
    Regime-Aware Interpretable Multi-Head Attention.
    
    Drop-in replacement for pytorch-forecasting's InterpretableMultiHeadAttention
    with regime-conditional head gating.
    
    Key difference from standard attention:
        After computing attention weights, applies learned per-head scaling
        factors conditioned on detected market regime:
        
        attn_final[h] = attn[h] * sigmoid(regime_gate[regime, h])
        
    Parameters
    ----------
    n_head : int
        Number of attention heads
    d_model : int  
        Model dimension (hidden_size)
    dropout : float, default=0.0
        Dropout rate
    mask_bias : float, default=-1e9
        Bias for masked positions
    num_regimes : int, default=2
        Number of market regimes (2 = low/high volatility)
    regime_mode : str, default='vix_threshold'
        How to detect regime:
        - 'vix_threshold': Deterministic threshold on VIX feature
        - 'disabled': No regime modulation (baseline behavior)
    vix_threshold : float, default=25.0
        VIX level above which regime=1 (high volatility)
    gate_init_std : float, default=0.01
        Std for regime gate initialization (small = near-identity start)
    gate_grad_scale : float, default=100.0
        Gradient scaling factor for regime gates (they're downstream, get weak signal)
    gate_init : str, default='neutral'
        Gate initialization strategy:
        - 'neutral': All gates start near 0.5 (default behavior)
        - 'separated': Low-vol starts dampened (~0.38), high-vol amplified (~0.62)
        
    Notes
    -----
    - Maintains exact same interface as InterpretableMultiHeadAttention
    - VIX must be passed through set_regime_signal() before forward()
    - Regime gates add only n_heads * num_regimes parameters (~8 for typical config)
    - Designed to avoid capacity-collapse by minimal parameter addition
    """
    
    def __init__(
        self,
        n_head: int,
        d_model: int,
        dropout: float = 0.0,
        mask_bias: float = -1e9,
        num_regimes: int = 2,
        regime_mode: str = 'vix_threshold',
        vix_threshold: float = 25.0,
        gate_init_std: float = 0.01,
        gate_grad_scale: float = 100.0,
        gate_init: str = 'neutral'
    ):
        super().__init__()
        
        if regime_mode not in ['vix_threshold', 'disabled']:
            raise ValueError(f"regime_mode must be 'vix_threshold' or 'disabled', got '{regime_mode}'")
        
        if gate_init not in ['neutral', 'separated']:
            raise ValueError(f"gate_init must be 'neutral' or 'separated', got '{gate_init}'")
        
        self.n_head = n_head
        self.d_model = d_model
        self.mask_bias = mask_bias
        self.num_regimes = num_regimes
        self.regime_mode = regime_mode
        self.vix_threshold = vix_threshold
        
        # Head dimensions (same as original InterpretableMultiHeadAttention)
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        
        # Shared value projection (interpretability constraint from original TFT)
        self.v_layer = nn.Linear(d_model, self.d_v)
        
        # Per-head Q and K projections
        self.q_layers = nn.ModuleList([
            nn.Linear(d_model, self.d_q) for _ in range(n_head)
        ])
        self.k_layers = nn.ModuleList([
            nn.Linear(d_model, self.d_k) for _ in range(n_head)
        ])
        
        # Attention computation
        self.attention = ScaledDotProductAttention(mask_bias=mask_bias)
        
        # Output projection
        self.w_h = nn.Linear(self.d_v, d_model, bias=False)
        
        # === REGIME-AWARE COMPONENTS ===
        if regime_mode != 'disabled':
            # Learnable per-head gates for each regime
            # Shape: [num_regimes, n_head]
            if gate_init == 'separated':
                # Pre-separated initialization:
                # - Regime 0 (low-vol): sigmoid(-0.5) ≈ 0.38 (dampen)
                # - Regime 1 (high-vol): sigmoid(0.5) ≈ 0.62 (amplify)
                init_gates = torch.zeros(num_regimes, n_head)
                init_gates[0, :] = -0.5  # Low-vol regime dampens
                init_gates[1, :] = 0.5   # High-vol regime amplifies
                self.regime_gates = nn.Parameter(init_gates)
            else:
                # Neutral: initialized near zero so sigmoid(gate) ≈ 0.5
                self.regime_gates = nn.Parameter(
                    torch.randn(num_regimes, n_head) * gate_init_std
                )
            
            # Store for logging
            self.gate_grad_scale = gate_grad_scale
            self.gate_init = gate_init

            # Scale up gradients for gates (they're far downstream, get weak signal)
            self.regime_gates.register_hook(lambda grad: grad * self.gate_grad_scale)
            
            # Optional: per-regime bias for attention logits (more expressive)
            # Disabled by default to keep minimal
            self.use_logit_bias = False
            if self.use_logit_bias:
                self.regime_logit_bias = nn.Parameter(
                    torch.zeros(num_regimes, n_head)
                )
        
        # Buffer for regime signal (set externally before forward)
        self.register_buffer('_regime_signal', None, persistent=False)
        self.register_buffer('_current_regime', None, persistent=False)
        
        # Diagnostic caches
        self.register_buffer('_cached_attention_weights', None, persistent=False)
        self.register_buffer('_cached_regime_gates', None, persistent=False)
        self.register_buffer('_cached_head_contributions', None, persistent=False)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier uniform (matches original TFT)."""
        for layer in self.q_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.k_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.v_layer.weight)
        nn.init.zeros_(self.v_layer.bias)
        nn.init.xavier_uniform_(self.w_h.weight)
    
    def set_regime_signal(self, vix_values: Optional[torch.Tensor]):
        """
        Set the VIX values for regime detection.
        
        Must be called before forward() with the current batch's VIX values.
        
        Args:
            vix_values: Raw VIX values [batch_size] or None to disable
        """
        if vix_values is None:
            self._regime_signal = None
            self._current_regime = None
            return
        
        self._regime_signal = vix_values
        
        # Detect regime based on VIX threshold
        if self.regime_mode == 'vix_threshold':
            # Simple binary: above threshold = high vol regime (1), else low vol (0)
            self._current_regime = (vix_values >= self.vix_threshold).long()
        else:
            self._current_regime = None
    
    def _get_regime_gates(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Get per-sample regime gate values.
        
        Returns:
            gate_weights: [batch_size, n_head] sigmoid-scaled gate values, or None
        """
        if self.regime_mode == 'disabled' or self._current_regime is None:
            return None
        
        # Get gate values for each sample's regime
        # regime_gates: [num_regimes, n_head]
        # current_regime: [batch_size]
        regime_indices = self._current_regime.to(device)
        
        # Index into gates: [batch_size, n_head]
        raw_gates = self.regime_gates[regime_indices]
        
        # Apply sigmoid to get scaling factors in (0, 1)
        gate_weights = torch.sigmoid(raw_gates)
        
        # Cache for diagnostics
        self._cached_regime_gates = gate_weights.detach()
        
        return gate_weights
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with regime-conditional attention gating.
        
        Args:
            q: Query tensor [batch, query_len, d_model]
            k: Key tensor [batch, key_len, d_model]
            v: Value tensor [batch, key_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Attended output [batch, query_len, d_model]
            attn_weights: Attention weights [batch, n_head, query_len, key_len]
        """
        batch_size = q.size(0)
        
        # Shared value projection (interpretability constraint)
        v_proj = self.v_layer(v)  # [batch, key_len, d_v]
        
        # Get regime gates if enabled
        regime_gates = self._get_regime_gates(batch_size, q.device)
        
        # Per-head attention computation
        head_outputs = []
        head_attns = []
        
        for h in range(self.n_head):
            # Project Q and K for this head
            q_h = self.q_layers[h](q)  # [batch, query_len, d_q]
            k_h = self.k_layers[h](k)  # [batch, key_len, d_k]
            
            # Compute attention
            # Reshape for batch matrix multiply
            q_h = q_h.view(batch_size, -1, self.d_q)
            k_h = k_h.view(batch_size, -1, self.d_k)
            v_h = v_proj.view(batch_size, -1, self.d_v)
            
            # Attention scores
            attn_scores = torch.bmm(q_h, k_h.transpose(1, 2))  # [batch, q_len, k_len]
            attn_scores = attn_scores / math.sqrt(self.d_k)
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask, self.mask_bias)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply regime gating to attention weights
            if regime_gates is not None:
                # gate_h: [batch_size, 1, 1] for broadcasting
                gate_h = regime_gates[:, h].unsqueeze(-1).unsqueeze(-1)
                attn_weights = attn_weights * gate_h
            
            # Compute attended values
            head_out = torch.bmm(attn_weights, v_h)  # [batch, q_len, d_v]
            
            head_outputs.append(head_out)
            head_attns.append(attn_weights)
        
        # Average head outputs (interpretability constraint from TFT)
        stacked = torch.stack(head_outputs, dim=0)  # [n_head, batch, q_len, d_v]
        averaged = stacked.mean(dim=0)  # [batch, q_len, d_v]
        
        # Cache head contributions for analysis
        self._cached_head_contributions = stacked.detach()
        
        # Final output projection
        output = self.w_h(averaged)  # [batch, q_len, d_model]
        
        # Stack attention weights for return
        attn_weights_stacked = torch.stack(head_attns, dim=1)  # [batch, n_head, q_len, k_len]
        self._cached_attention_weights = attn_weights_stacked.detach()
        
        return output, attn_weights_stacked
    
    def get_regime_diagnostics(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get diagnostic information about regime gating.
        
        Returns:
            dict with:
                - 'regime_signal': VIX values used for regime detection
                - 'current_regime': Detected regime indices
                - 'gate_weights': Per-head gate values applied
                - 'raw_gates': Pre-sigmoid gate parameters
        """
        diagnostics = {
            'regime_signal': self._regime_signal,
            'current_regime': self._current_regime,
        }
        
        if self.regime_mode != 'disabled':
            diagnostics['gate_weights'] = self._cached_regime_gates
            diagnostics['raw_gates'] = self.regime_gates.detach().cpu()
        
        return diagnostics
    
    def extra_repr(self) -> str:
        """String representation for print/debug."""
        return (
            f"n_head={self.n_head}, d_model={self.d_model}, "
            f"regime_mode='{self.regime_mode}', "
            f"num_regimes={self.num_regimes}, "
            f"vix_threshold={self.vix_threshold}"
        )


class RegimeAttentionTFT(nn.Module):
    """
    Wrapper that threads regime signals through TFT forward pass.
    
    This is a thin wrapper around an existing TFT model that:
    1. Extracts VIX values from input batch
    2. Sets regime signal on the attention module
    3. Calls original TFT forward
    
    Usage:
        base_model = TemporalFusionTransformer.from_dataset(...)
        base_model = replace_attention_module(base_model, ...)
        model = RegimeAttentionTFT(base_model, vix_feature_idx=5)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        vix_feature_name: str = 'VIX',
        vix_feature_idx: Optional[int] = None
    ):
        """
        Args:
            base_model: TFT model with RegimeAwareInterpretableMultiHeadAttention
            vix_feature_name: Name of VIX feature in x_reals
            vix_feature_idx: Index of VIX in continuous features (if known)
        """
        super().__init__()
        self.model = base_model
        self.vix_feature_name = vix_feature_name
        self._vix_feature_idx = vix_feature_idx
        
        # Try to find VIX index from model hparams
        if self._vix_feature_idx is None and hasattr(base_model, 'hparams'):
            x_reals = getattr(base_model.hparams, 'x_reals', [])
            if vix_feature_name in x_reals:
                self._vix_feature_idx = x_reals.index(vix_feature_name)
    
    def _extract_vix(self, x: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Extract VIX values from input batch."""
        if self._vix_feature_idx is None:
            return None
        
        # x['encoder_cont'] shape: [batch, encoder_len, n_cont_features]
        encoder_cont = x.get('encoder_cont')
        if encoder_cont is None:
            return None
        
        # Get VIX from last encoder timestep
        vix = encoder_cont[:, -1, self._vix_feature_idx]
        return vix
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with regime signal injection."""
        # Extract VIX and set regime signal
        vix = self._extract_vix(x)
        
        if hasattr(self.model, 'multihead_attn') and \
           hasattr(self.model.multihead_attn, 'set_regime_signal'):
            self.model.multihead_attn.set_regime_signal(vix)
        
        # Standard forward
        return self.model(x)
    
    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def replace_attention_module(
    model: nn.Module,
    regime_mode: str = 'vix_threshold',
    vix_threshold: float = 25.0,
    num_regimes: int = 2,
    gate_init_std: float = 0.01,
    gate_grad_scale: float = 100.0,
    gate_init: str = 'neutral'
) -> nn.Module:
    """
    Replace TFT's attention module with regime-aware version.
    
    Parameters
    ----------
    model : TemporalFusionTransformer
        TFT model instance
    regime_mode : str, default='vix_threshold'
        'vix_threshold' or 'disabled'
    vix_threshold : float, default=25.0
        VIX level for regime switching
    num_regimes : int, default=2
        Number of regimes
    gate_init_std : float, default=0.01
        Initialization std for regime gates (only used if gate_init='neutral')
    gate_grad_scale : float, default=100.0
        Gradient scaling factor for regime gates
    gate_init : str, default='neutral'
        Gate initialization: 'neutral' (all ~0.5) or 'separated' (0.38/0.62)
        
    Returns
    -------
    model : TemporalFusionTransformer
        Modified model (in-place)
    """
    if not hasattr(model, 'multihead_attn'):
        raise ValueError("Model must have 'multihead_attn' attribute")
    
    old_attn = model.multihead_attn
    
    # Extract parameters from existing attention
    n_head = old_attn.n_head
    d_model = old_attn.d_model
    dropout = old_attn.dropout.p if hasattr(old_attn.dropout, 'p') else 0.0
    mask_bias = getattr(old_attn, 'mask_bias', -1e9)
    
    # Create regime-aware replacement
    new_attn = RegimeAwareInterpretableMultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        dropout=dropout,
        mask_bias=mask_bias,
        num_regimes=num_regimes,
        regime_mode=regime_mode,
        vix_threshold=vix_threshold,
        gate_init_std=gate_init_std,
        gate_grad_scale=gate_grad_scale,
        gate_init=gate_init
    )
    
    # Copy weights from original attention
    new_attn.v_layer.load_state_dict(old_attn.v_layer.state_dict())
    new_attn.w_h.load_state_dict(old_attn.w_h.state_dict())
    
    for i in range(n_head):
        new_attn.q_layers[i].load_state_dict(old_attn.q_layers[i].state_dict())
        new_attn.k_layers[i].load_state_dict(old_attn.k_layers[i].state_dict())
    
    # Replace in model
    model.multihead_attn = new_attn
    
    # Calculate parameter overhead
    regime_params = num_regimes * n_head  # Just the gate parameters
    total_attn_params = sum(p.numel() for p in new_attn.parameters())
    
    print(f"\n[REGIME ATTENTION] Replaced attention module:")
    print(f"  Mode: {regime_mode}")
    print(f"  Regimes: {num_regimes}")
    print(f"  VIX threshold: {vix_threshold}")
    print(f"  Gate grad scale: {gate_grad_scale}")
    print(f"  Gate init: {gate_init}")
    print(f"  Heads: {n_head}")
    print(f"  New parameters: {regime_params} (regime gates only)")
    print(f"  Total attention parameters: {total_attn_params}")
    
    return model


def create_regime_attention_model(
    base_model: nn.Module,
    vix_feature_name: str = 'VIX',
    regime_mode: str = 'vix_threshold',
    vix_threshold: float = 25.0,
    num_regimes: int = 2
) -> RegimeAttentionTFT:
    """
    Create complete regime-aware TFT model.
    
    Convenience function that:
    1. Replaces attention module
    2. Wraps in RegimeAttentionTFT for automatic VIX extraction
    
    Parameters
    ----------
    base_model : TemporalFusionTransformer
        Base TFT model
    vix_feature_name : str
        Name of VIX feature in x_reals
    regime_mode : str
        'vix_threshold' or 'disabled'
    vix_threshold : float
        VIX threshold for regime detection
    num_regimes : int
        Number of regimes
        
    Returns
    -------
    RegimeAttentionTFT
        Wrapped model with regime-aware attention
    """
    # Replace attention module
    model = replace_attention_module(
        base_model,
        regime_mode=regime_mode,
        vix_threshold=vix_threshold,
        num_regimes=num_regimes
    )
    
    # Wrap for automatic VIX extraction
    wrapped = RegimeAttentionTFT(
        model,
        vix_feature_name=vix_feature_name
    )
    
    return wrapped