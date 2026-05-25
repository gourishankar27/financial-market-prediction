"""
Regime-conditional output layer for Temporal Fusion Transformer.

Implements Mixture-of-Experts architecture for regime-adaptive predictions.
Drop-in replacement for TFT's standard output layer.

Usage:
    # After model creation
    from src.regime_output import RegimeConditionalOutput
    
    model.output_layer = RegimeConditionalOutput(
        hidden_size=model.hparams.hidden_size,
        output_size=model.hparams.output_size,
        num_regimes=2,
        routing_mode='learned'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class RegimeConditionalOutput(nn.Module):
    """
    Mixture-of-Experts output layer for regime-conditional predictions.
    
    Replaces TFT's standard nn.Linear output layer with multiple expert heads
    that specialize in different market regimes (e.g., high/low volatility).
    A learned router dynamically weights expert predictions.
    
    Parameters
    ----------
    hidden_size : int
        TFT hidden dimension (input size)
    output_size : int
        Number of quantiles (typically 7 for QuantileLoss)
    num_regimes : int, default=2
        Number of expert heads (e.g., 2 for normal/volatile regimes, 3 for low/med/high)
    routing_mode : str, default='learned'
        Routing strategy:
        - 'learned': Learn regime detection from hidden state
        - 'disabled': Single expert (equivalent to baseline)
    routing_strategy : str, default='learned'
        - 'learned': MLP router learns routing from hidden state
        - 'vix_threshold': Deterministic routing based on VIX level
    vix_threshold : float, default=25.0
        VIX threshold for 2-regime routing (VIX > threshold -> regime 1)
    vix_threshold_low : float, optional
        Lower VIX threshold for 3-regime routing (VIX <= low -> regime 0)
    vix_threshold_high : float, optional  
        Upper VIX threshold for 3-regime routing (VIX > high -> regime 2)
    load_balance_weight : float, default=0.5
        Weight for load balancing auxiliary loss (prevents winner-takes-all)
    expert_hidden_size : int, default=0
        Hidden layer size for MLP experts. 0 = linear experts (recommended default),
        >0 = 2-layer MLP with ReLU activation
    hard_routing_train : bool, default=False
        If True, use hard routing during training: each sample's loss only backprops
        through its assigned expert based on VIX threshold. This encourages expert
        specialization by preventing gradient mixing. Requires routing_strategy='vix_threshold'.
        Validation/test always uses soft routing regardless of this setting.
    
    Notes
    -----
    - Input shape: [batch, seq_len, hidden_size]
    - Output shape: [batch, seq_len, output_size]
    - Routing happens per timestep in sequence (supports multi-horizon)
    - For 3 regimes with VIX routing:
        - regime 0: VIX <= vix_threshold_low (low volatility)
        - regime 1: vix_threshold_low < VIX <= vix_threshold_high (medium)
        - regime 2: VIX > vix_threshold_high (high volatility)
    - Hard routing during training prevents gradient flow to non-assigned experts,
      encouraging each expert to specialize on its regime's samples.
    
    Examples
    --------
    >>> # 2-regime linear experts (recommended baseline)
    >>> model.output_layer = RegimeConditionalOutput(
    ...     hidden_size=16, output_size=7, num_regimes=2, expert_hidden_size=0
    ... )
    
    >>> # 3-regime with VIX thresholds and hard routing
    >>> model.output_layer = RegimeConditionalOutput(
    ...     hidden_size=16, output_size=7, num_regimes=3,
    ...     routing_strategy='vix_threshold',
    ...     vix_threshold_low=15.0, vix_threshold_high=23.0,
    ...     hard_routing_train=True
    ... )
    """
    
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_regimes: int = 2,
        routing_mode: str = 'learned',
        routing_strategy: str = 'learned',
        vix_threshold: float = 25.0,
        vix_threshold_low: Optional[float] = None,
        vix_threshold_high: Optional[float] = None,
        load_balance_weight: float = 0.5,
        expert_hidden_size: int = 0,
        hard_routing_train: bool = False
    ):
        super().__init__()
        
        if routing_mode not in ['learned', 'disabled']:
            raise ValueError(
                f"routing_mode must be 'learned' or 'disabled', got '{routing_mode}'"
            )
        
        if routing_strategy not in ['learned', 'vix_threshold']:
            raise ValueError(
                f"routing_strategy must be 'learned' or 'vix_threshold', got '{routing_strategy}'"
            )
        
        if num_regimes < 1:
            raise ValueError(f"num_regimes must be >= 1, got {num_regimes}")
        
        if routing_mode == 'disabled' and num_regimes != 1:
            raise ValueError(
                f"routing_mode='disabled' requires num_regimes=1, got {num_regimes}"
            )
        
        # Validate 3-regime VIX threshold configuration
        if num_regimes == 3 and routing_strategy == 'vix_threshold':
            if vix_threshold_low is None or vix_threshold_high is None:
                raise ValueError(
                    "3-regime VIX routing requires both --vix-threshold-low and "
                    "--vix-threshold-high to be specified"
                )
            if vix_threshold_low >= vix_threshold_high:
                raise ValueError(
                    f"vix_threshold_low ({vix_threshold_low}) must be < "
                    f"vix_threshold_high ({vix_threshold_high})"
                )
        
        # Validate hard routing configuration
        if hard_routing_train and routing_strategy != 'vix_threshold':
            raise ValueError(
                "hard_routing_train requires routing_strategy='vix_threshold'"
            )
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_regimes = num_regimes
        self.routing_mode = routing_mode
        self.routing_strategy = routing_strategy
        self.vix_threshold = vix_threshold
        self.vix_threshold_low = vix_threshold_low
        self.vix_threshold_high = vix_threshold_high
        self.load_balance_weight = load_balance_weight
        self.expert_hidden_size = expert_hidden_size
        self.hard_routing_train = hard_routing_train
        
        self.register_buffer('_routing_history', None, persistent=False)  # For loss computation
        
        # Create expert heads
        if routing_mode == 'disabled':
            # Single expert (baseline equivalent)
            self.experts = nn.ModuleList([
                nn.Linear(hidden_size, output_size)
            ])
        else:
            # Multiple experts - architecture depends on expert_hidden_size
            if expert_hidden_size > 0:
                # MLP experts with hidden layer
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, expert_hidden_size),
                        nn.ReLU(),
                        nn.Linear(expert_hidden_size, output_size)
                    )
                    for _ in range(num_regimes)
                ])
            else:
                # Linear experts (default, recommended)
                self.experts = nn.ModuleList([
                    nn.Linear(hidden_size, output_size)
                    for _ in range(num_regimes)
                ])
            
            # Router: learns regime from hidden state
            if routing_mode == 'learned':
                self.router = nn.Linear(hidden_size, num_regimes)

                # Initialize to zero for balanced routing at start
                # This prevents random initialization bias that causes winner-takes-all
                nn.init.zeros_(self.router.weight)
                nn.init.zeros_(self.router.bias)
        
        # Diagnostic caches (non-persistent - not saved to checkpoint)
        # These store detached tensors on CPU for monitoring without memory buildup
        self.register_buffer('_cached_routing_weights', None, persistent=False)
        for i in range(num_regimes):
            self.register_buffer(f'_cached_expert_preds_{i}', None, persistent=False)
        
        # Cache for load balancing loss (computed during forward pass)
        self.register_buffer('_cached_lb_loss', None, persistent=False)
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        vix_values: torch.Tensor = None,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional diagnostics.
        
        Parameters
        ----------
        hidden_state : torch.Tensor
            TFT's final hidden state, shape [batch, seq_len, hidden_size]
        vix_values : torch.Tensor, optional
            VIX values for VIX-based routing, shape [batch]
        return_diagnostics : bool, default=False
            If True, return (prediction, diagnostics_dict)
        
        Returns
        -------
        prediction : torch.Tensor
            Quantile predictions, shape [batch, seq_len, output_size]
        diagnostics : dict, optional
            Dictionary with keys:
            - 'routing_weights': [batch, seq_len, num_regimes] - soft routing weights
            - 'expert_preds': List of [batch, seq_len, output_size] - individual expert predictions
        """
        if self.routing_mode == 'disabled':
            # Baseline mode: single expert
            pred = self.experts[0](hidden_state)
            
            if return_diagnostics:
                return pred, {
                    'routing_weights': None,
                    'expert_preds': [pred]
                }
            return pred
        
        # Get predictions from all experts
        # Each: [batch, seq_len, hidden_size] -> [batch, seq_len, output_size]
        expert_preds = [expert(hidden_state) for expert in self.experts]
        
        # Compute routing weights based on strategy
        if self.routing_strategy == 'vix_threshold':
            vix = vix_values or getattr(self, '_vix_for_forward', None)
            if vix is not None and not getattr(self, '_vix_logged_this_epoch', False):
                print(f"[DEBUG] VIX routing - batch vix: min={vix.min().item():.1f}, max={vix.max().item():.1f}, mean={vix.mean().item():.1f}")
                self._vix_logged_this_epoch = True

            if vix is None and hasattr(self, '_vix_for_forward'):
                vix = self._vix_for_forward
            
            if vix is not None:
                # Deterministic VIX-based routing
                batch_size = hidden_state.size(0)
                seq_len = hidden_state.size(1)
                
                if self.num_regimes == 2:
                    # 2-regime routing: high VIX -> regime 1, low VIX -> regime 0
                    is_high_vol = (vix > self.vix_threshold).float()  # [batch]
                    routing_weights_2d = torch.stack([1 - is_high_vol, is_high_vol], dim=1)
                
                elif self.num_regimes == 3:
                    # 3-regime routing: low/medium/high volatility
                    # regime 0: VIX <= threshold_low
                    # regime 1: threshold_low < VIX <= threshold_high
                    # regime 2: VIX > threshold_high
                    is_low = (vix <= self.vix_threshold_low).float()
                    is_high = (vix > self.vix_threshold_high).float()
                    is_medium = 1.0 - is_low - is_high  # Everything else
                    
                    routing_weights_2d = torch.stack([is_low, is_medium, is_high], dim=1)
                
                else:
                    raise ValueError(
                        f"VIX threshold routing only supports 2 or 3 regimes, got {self.num_regimes}"
                    )
                
                # Expand to [batch, seq_len, num_regimes] for multi-step predictions
                routing_weights = routing_weights_2d.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # Fallback to learned routing if VIX not available
                router_logits = self.router(hidden_state)
                routing_weights = F.softmax(router_logits, dim=-1)
        else:
            # Learned routing (default)
            router_logits = self.router(hidden_state)  # [batch, seq_len, num_regimes]
            routing_weights = F.softmax(router_logits, dim=-1)  # Soft routing over regimes
        
        # Cache diagnostics for monitoring (detached, on CPU to prevent memory buildup)
        # Only cache during validation (not training) to avoid overhead
        if not self.training:
            self._cached_routing_weights = routing_weights.detach().cpu()
            for i, expert_pred in enumerate(expert_preds):
                setattr(self, f'_cached_expert_preds_{i}', expert_pred.detach().cpu())
        else:
            # During training, compute and cache load balancing loss
            avg_routing = routing_weights.mean(dim=(0, 1))  # [num_regimes]
            target = torch.ones_like(avg_routing) / self.num_regimes
            lb_loss = F.mse_loss(avg_routing, target) * self.load_balance_weight
            self._cached_lb_loss = lb_loss  # Store for loss function to add
        
        # Compute final predictions - hard vs soft routing
        if self.training and self.hard_routing_train and self.routing_strategy == 'vix_threshold':
            # HARD ROUTING: Each sample's loss only backprops through assigned expert
            # This encourages expert specialization by preventing gradient mixing
            vix = vix_values or getattr(self, '_vix_for_forward', None)
            
            if vix is not None:
                batch_size = hidden_state.size(0)
                seq_len = hidden_state.size(1)
                output_size = expert_preds[0].size(-1)
                
                # Log hard routing activation once per epoch
                if not getattr(self, '_hard_routing_logged_this_epoch', False):
                    if self.num_regimes == 2:
                        n_expert0 = (vix <= self.vix_threshold).sum().item()
                        n_expert1 = (vix > self.vix_threshold).sum().item()
                        print(f"[DEBUG] HARD routing APPLIED: {n_expert0} samples -> expert0, {n_expert1} samples -> expert1")
                    elif self.num_regimes == 3:
                        n_expert0 = (vix <= self.vix_threshold_low).sum().item()
                        n_expert2 = (vix > self.vix_threshold_high).sum().item()
                        n_expert1 = batch_size - n_expert0 - n_expert2
                        print(f"[DEBUG] HARD routing APPLIED: {n_expert0} -> exp0, {n_expert1} -> exp1, {n_expert2} -> exp2")
                    self._hard_routing_logged_this_epoch = True
                
                if self.num_regimes == 2:
                    # 2-regime hard routing: select expert based on VIX threshold
                    is_high_vol = (vix > self.vix_threshold)  # [batch] bool
                    
                    # Expand for broadcasting: [batch, 1, 1] -> broadcast to [batch, seq_len, output_size]
                    mask = is_high_vol.view(-1, 1, 1).expand(-1, seq_len, output_size)
                    
                    # Select: high vol -> expert 1, low vol -> expert 0
                    final_pred = torch.where(mask, expert_preds[1], expert_preds[0])
                
                elif self.num_regimes == 3:
                    # 3-regime hard routing using gather
                    # Compute regime index: 0 (low), 1 (medium), 2 (high)
                    regime_idx = (vix > self.vix_threshold_low).long() + (vix > self.vix_threshold_high).long()
                    # regime_idx: [batch], values in {0, 1, 2}
                    
                    # Stack experts: [batch, seq_len, num_regimes, output_size]
                    stacked_preds = torch.stack(expert_preds, dim=2)
                    
                    # Expand regime_idx for gather: [batch, seq_len, 1, output_size]
                    idx_expanded = regime_idx.view(-1, 1, 1, 1).expand(-1, seq_len, 1, output_size)
                    
                    # Gather along regime dimension: [batch, seq_len, 1, output_size] -> squeeze -> [batch, seq_len, output_size]
                    final_pred = stacked_preds.gather(dim=2, index=idx_expanded).squeeze(2)
                
                else:
                    raise ValueError(f"Hard routing only supports 2 or 3 regimes, got {self.num_regimes}")
            else:
                # Fallback to soft routing if VIX not available
                if not getattr(self, '_soft_fallback_logged', False):
                    print(f"[DEBUG] HARD routing FALLBACK to soft (VIX is None)")
                    self._soft_fallback_logged = True
                stacked_preds = torch.stack(expert_preds, dim=2)
                weights_expanded = routing_weights.unsqueeze(-1)
                final_pred = torch.sum(stacked_preds * weights_expanded, dim=2)
        else:
            # SOFT ROUTING: Weighted combination of expert predictions (default)
            # Stack: [batch, seq_len, num_regimes, output_size]
            stacked_preds = torch.stack(expert_preds, dim=2)
            
            # Expand routing weights: [batch, seq_len, num_regimes, 1]
            weights_expanded = routing_weights.unsqueeze(-1)
            
            # Weighted sum: [batch, seq_len, output_size]
            final_pred = torch.sum(stacked_preds * weights_expanded, dim=2)
        
        if return_diagnostics:
            return final_pred, {
                'routing_weights': routing_weights,
                'expert_preds': expert_preds
            }
        
        return final_pred
    
    def get_load_balancing_loss(self) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Penalizes deviation from uniform expert usage to prevent
        winner-takes-all routing collapse.
        
        Returns
        -------
        torch.Tensor
            Scalar loss term (0.0 if no routing or disabled mode)
        
        Notes
        -----
        Based on Switch Transformer load balancing (Fedus et al. 2021).
        Loss = alpha * sum((f_i - 1/K)^2) where:
        - f_i = fraction of samples routed to expert i
        - K = number of experts
        - alpha = load_balance_weight hyperparameter
        """
        if self.routing_mode == 'disabled' or self._routing_history is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Compute average routing weights across batch and time
        # routing_history shape: [batch, seq_len, num_regimes]
        avg_routing = self._routing_history.mean(dim=(0, 1))  # [num_regimes]
        
        # Target: uniform distribution [1/K, 1/K, ...]
        target = torch.ones_like(avg_routing) / self.num_regimes
        
        # L2 penalty on deviation from uniform
        loss = F.mse_loss(avg_routing, target)
        
        return self.load_balance_weight * loss
    
    def get_expert_parameters(self) -> Dict[str, int]:
        """
        Get parameter counts for analysis.
        
        Returns
        -------
        dict
            Parameter counts: {'total', 'experts', 'router', 'expert_type'}
        """
        expert_params = sum(
            p.numel() for expert in self.experts for p in expert.parameters()
        )
        
        router_params = 0
        if hasattr(self, 'router'):
            router_params = sum(p.numel() for p in self.router.parameters())
        
        expert_type = 'mlp' if self.expert_hidden_size > 0 else 'linear'
        
        return {
            'total': expert_params + router_params,
            'experts': expert_params,
            'router': router_params,
            'expert_type': expert_type,
            'expert_hidden_size': self.expert_hidden_size
        }
    
    def extra_repr(self) -> str:
        """String representation for print/debug."""
        expert_type = f"mlp({self.expert_hidden_size})" if self.expert_hidden_size > 0 else "linear"
        return (
            f"hidden_size={self.hidden_size}, "
            f"output_size={self.output_size}, "
            f"num_regimes={self.num_regimes}, "
            f"routing_mode='{self.routing_mode}', "
            f"expert_type={expert_type}"
        )


def replace_output_layer(
    model: nn.Module,
    num_regimes: int = 2,
    routing_mode: str = 'learned',
    routing_strategy: str = 'learned',
    vix_threshold: float = 25.0,
    vix_threshold_low: Optional[float] = None,
    vix_threshold_high: Optional[float] = None,
    load_balance_weight: float = 0.5,
    expert_hidden_size: int = 0,
    hard_routing_train: bool = False
) -> nn.Module:
    """
    Replace TFT's output layer with RegimeConditionalOutput.
    
    Parameters
    ----------
    model : TemporalFusionTransformer
        TFT model instance
    num_regimes : int, default=2
        Number of expert heads
    routing_mode : str, default='learned'
        'learned' or 'disabled'
    routing_strategy : str, default='learned'
        'learned' (MLP router) or 'vix_threshold' (deterministic VIX routing)
    vix_threshold : float, default=25.0
        VIX threshold for 2-regime deterministic routing
    vix_threshold_low : float, optional
        Lower VIX threshold for 3-regime routing
    vix_threshold_high : float, optional
        Upper VIX threshold for 3-regime routing
    load_balance_weight : float, default=0.5
        Weight for load balancing auxiliary loss
    expert_hidden_size : int, default=0
        Expert hidden layer size (0=linear, >0=MLP)
    hard_routing_train : bool, default=False
        If True, use hard routing during training (each sample only trains its assigned expert).
        Requires routing_strategy='vix_threshold'.
    
    Returns
    -------
    model : TemporalFusionTransformer
        Modified model (in-place)
    
    Examples
    --------
    >>> # 2-regime learned routing with linear experts
    >>> model = replace_output_layer(model, num_regimes=2, expert_hidden_size=0)
    >>> 
    >>> # 3-regime VIX routing with MLP experts and hard routing
    >>> model = replace_output_layer(
    ...     model, num_regimes=3, routing_strategy='vix_threshold',
    ...     vix_threshold_low=15.0, vix_threshold_high=23.0,
    ...     expert_hidden_size=16, hard_routing_train=True
    ... )
    """
    if not hasattr(model, 'output_layer'):
        raise ValueError("Model must have 'output_layer' attribute")
    
    if not hasattr(model.hparams, 'hidden_size') or not hasattr(model.hparams, 'output_size'):
        raise ValueError("Model must have hparams.hidden_size and hparams.output_size")
    
    # Handle multi-target case (not yet implemented - for future)
    if isinstance(model.output_layer, nn.ModuleList):
        raise NotImplementedError(
            "Multi-target regime output not yet implemented. "
            "Currently only supports single-target TFT."
        )
    
    # Replace with regime-conditional output
    model.output_layer = RegimeConditionalOutput(
        hidden_size=model.hparams.hidden_size,
        output_size=model.hparams.output_size,
        num_regimes=num_regimes,
        routing_mode=routing_mode,
        routing_strategy=routing_strategy,
        vix_threshold=vix_threshold,
        vix_threshold_low=vix_threshold_low,
        vix_threshold_high=vix_threshold_high,
        load_balance_weight=load_balance_weight,
        expert_hidden_size=expert_hidden_size,
        hard_routing_train=hard_routing_train
    )
    
    expert_type = f"MLP({expert_hidden_size})" if expert_hidden_size > 0 else "Linear"
    routing_type = "HARD" if hard_routing_train else "soft"
    
    print(f"\n[REGIME OUTPUT] Replaced output layer:")
    print(f"  Mode: {routing_mode}")
    print(f"  Strategy: {routing_strategy}")
    if routing_strategy == 'vix_threshold':
        if num_regimes == 2:
            print(f"  VIX threshold: {vix_threshold}")
        elif num_regimes == 3:
            print(f"  VIX thresholds: low={vix_threshold_low}, high={vix_threshold_high}")
    print(f"  Experts: {num_regimes} x {expert_type}")
    print(f"  Training routing: {routing_type}")
    print(f"  Load balance weight: {load_balance_weight}")
    print(f"  Parameters: {model.output_layer.get_expert_parameters()}")
    
    return model