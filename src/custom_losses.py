"""
Custom loss functions for TFT with modular penalty components.

Implements EnhancedQuantileLoss that extends pytorch-forecasting's QuantileLoss
with toggleable penalties for:
- Anti-collapse penalty (prevent uniform predictions via minimum variance threshold)
- Temporal consistency (penalize erratic prediction changes)

All penalties are optional via weight parameters (weight=0.0 disables).
"""

import torch
from pytorch_forecasting.metrics import QuantileLoss


class EnhancedQuantileLoss(QuantileLoss):
    """
    QuantileLoss with modular regularization penalties.
    
    Extends base QuantileLoss to prevent common failure modes in financial forecasting:
    1. Collapse: uniform/constant predictions (minimum variance enforcement)
    2. Temporal instability: erratic prediction changes between timesteps
    
    Each penalty is independently toggleable via weight parameters.
    
    NOTE: Does NOT enforce fixed distribution targets (mean/std) as these vary by regime.
    Rolling window analysis shows data period  mean varies -0.38% to +0.19%, std varies 0.59% to 2.59%.
    Fixed targets would fight legitimate regime-appropriate behavior.
    """
    
    def __init__(
        self,
        quantiles=None,
        # Anti-collapse penalty (minimum variance threshold)
        collapse_weight=0.0,
        collapse_threshold=0.005,  # 0.5% in decimal form - only penalize if pred_std < 0.5%
        # Directional diversity penalty (prevent unidirectional predictions)
        directional_weight=0.0,
        directional_threshold=0.90,  # 90% in decimal form - penalize if >90% same sign
        # Temporal consistency
        temporal_consistency_weight=0.0,
        # Magnitude-aware loss weighting (encourage larger predictions)
        magnitude_weight_alpha=0.0,  # Linear magnitude weighting (0.0 = disabled)
        extreme_move_weight=1.0,     # Weight for extreme moves (1.0 = disabled, >1.0 = active)
        extreme_move_percentile=95,  # Percentile threshold for extreme moves
        **kwargs
    ):
        """
        Initialize enhanced quantile loss.
        
        Args:
            quantiles: List of quantiles for loss computation (default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
            collapse_weight: Weight for variance-based anti-collapse penalty (0.0 = disabled, typical: 0.1-0.2)
            collapse_threshold: Minimum prediction std (0.005 = 0.5% in decimal form)
                               Only penalizes when pred_std < threshold (collapse detection)
                               No penalty when pred_std >= threshold (normal behavior)
            directional_weight: Weight for directional diversity penalty (0.0 = disabled, typical: 0.1-0.2)
            directional_threshold: Maximum directional bias (0.90 = 90% in decimal form)
                                  Penalizes when >90% predictions have same sign in a batch
                                  Based on empirical analysis: real market never exceeds 90% over 30-day windows
            temporal_consistency_weight: Weight for smoothness penalty (0.0 = disabled, typical: 0.05-0.1)
            magnitude_weight_alpha: Linear magnitude weighting coefficient (0.0 = disabled)
                                   loss_weight = 1.0 + alpha * |target|
                                   Upweights larger moves. Typical: 0.5-2.0
            extreme_move_weight: Weight multiplier for extreme moves (1.0 = disabled, >1.0 = active)
                                Applies to moves beyond extreme_move_percentile. Typical: 2.0-5.0
            extreme_move_percentile: Percentile threshold for extreme moves (default: 95 = top 5%)
        
        Note: magnitude_weight_alpha and extreme_move_weight are mutually exclusive.
        """
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        
        # Validate mutual exclusivity
        if magnitude_weight_alpha > 0 and extreme_move_weight > 1.0:
            raise ValueError(
                f"Cannot use both magnitude_weight_alpha ({magnitude_weight_alpha}) and "
                f"extreme_move_weight ({extreme_move_weight}). Choose one magnitude weighting scheme."
            )
        
        super().__init__(quantiles=quantiles, **kwargs)
        
        # Anti-collapse parameters
        self.collapse_weight = collapse_weight
        self.collapse_threshold = collapse_threshold
        
        # Directional diversity parameters
        self.directional_weight = directional_weight
        self.directional_threshold = directional_threshold
        # No longer need directional_window or prediction_buffer
        
        # Temporal consistency parameters
        self.temporal_consistency_weight = temporal_consistency_weight
        self.prev_prediction = None  # Track last prediction for cross-batch consistency
        
        # Magnitude-aware loss weighting
        self.magnitude_weight_alpha = magnitude_weight_alpha
        self.extreme_move_weight = extreme_move_weight
        self.extreme_move_percentile = extreme_move_percentile
        self.extreme_threshold = None  # Will be computed on first batch if needed
        
        # Logging attributes (for monitoring)
        self.last_pred_mean = None
        self.last_pred_std = None
        self.last_collapse_penalty = None
        self.last_directional_penalty = None
        self.last_temporal_penalty = None
    
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with optional penalties.
        
        Args:
            y_pred: Model predictions, shape [batch, time, quantiles]
            target: Ground truth returns, shape [batch, time]
        
        Returns:
            Loss tensor, shape [batch, time, quantiles] (framework handles reduction)
        """
        # Compute base quantile loss
        base_losses = super().loss(y_pred, target)  # [batch, time, quantiles]
        
        # Apply magnitude-aware weighting if enabled
        if self.magnitude_weight_alpha > 0:
            # Option 2: Linear magnitude weighting
            # weight = 1.0 + alpha * |target|
            target_magnitude = torch.abs(target).unsqueeze(-1)  # [batch, time, 1]
            magnitude_weights = 1.0 + self.magnitude_weight_alpha * target_magnitude
            base_losses = base_losses * magnitude_weights
        
        elif self.extreme_move_weight > 1.0:
            # Option 3: Extreme move weighting
            # Compute threshold on first batch if needed
            if self.extreme_threshold is None:
                target_flat = torch.abs(target).flatten()
                self.extreme_threshold = torch.quantile(target_flat, self.extreme_move_percentile / 100.0)
            
            # Apply higher weight to extreme moves
            target_magnitude = torch.abs(target).unsqueeze(-1)  # [batch, time, 1]
            is_extreme = (target_magnitude > self.extreme_threshold).float()
            extreme_weights = torch.where(
                is_extreme.bool(),
                torch.ones_like(is_extreme) * self.extreme_move_weight,
                torch.ones_like(is_extreme)
            )
            base_losses = base_losses * extreme_weights
        
        # Extract median predictions for penalties
        median_idx = self.quantiles.index(0.5)
        median_pred = y_pred[..., median_idx]  # [batch, time]
        
        # Initialize total loss with (potentially weighted) base
        total_losses = base_losses.clone()
        
        # Add anti-collapse penalty (always active during training/validation)
        if self.collapse_weight > 0:
            collapse_penalty = self._anti_collapse_penalty(median_pred)
            # Broadcast penalty across all quantiles
            collapse_penalty_expanded = collapse_penalty.unsqueeze(-1).expand_as(base_losses)
            total_losses = total_losses + collapse_penalty_expanded
        
        # Add directional diversity penalty (only during validation/test with sequential data)
        if self.directional_weight > 0 and not self.training:
            directional_penalty = self._directional_diversity_penalty(median_pred)
            # Broadcast penalty across all quantiles
            directional_penalty_expanded = directional_penalty.unsqueeze(-1).expand_as(base_losses)
            total_losses = total_losses + directional_penalty_expanded
        
        # Add temporal consistency penalty (only during sequential evaluation)
        if self.temporal_consistency_weight > 0 and not self.training:
            temp_penalty = self._temporal_consistency_penalty(median_pred)
            # Apply only to first timestep of first batch sample
            if temp_penalty > 0:
                # Broadcast to match loss shape, but only affect first position
                temp_penalty_expanded = torch.zeros_like(base_losses)
                temp_penalty_expanded[0, 0, :] = temp_penalty
                total_losses = total_losses + temp_penalty_expanded
        
        return total_losses
    
    def _anti_collapse_penalty(self, median_pred: torch.Tensor) -> torch.Tensor:
        """
        Penalize predictions with insufficient variance (collapse detection).
        
        Only fires when pred_std < threshold (e.g., 0.5%). Does NOT penalize when
        variance is normal or high - this preserves regime-appropriate behavior.
        
        Args:
            median_pred: Median predictions, shape [batch, time]
        
        Returns:
            Penalty tensor, shape [batch, time]
        """
        # Flatten predictions for statistics
        pred_flat = median_pred.reshape(-1)
        pred_mean = pred_flat.mean()
        pred_std = pred_flat.std(unbiased=False)  # Population std, not sample estimate
        
        # DEBUG: Track std computation
        #print(f"[DEBUG] pred_flat shape: {pred_flat.shape}, pred_std raw: {pred_std.item():.6f}, contains nan: {torch.isnan(pred_std).item()}")
        
        # Store for logging
        self.last_pred_mean = pred_mean.detach().item()
        self.last_pred_std = pred_std.detach().item()
        
        # Compute penalty - only when variance too low (collapse)
        penalty = torch.zeros_like(median_pred)
        
        if pred_std < self.collapse_threshold:
            # Penalize distance below threshold
            # pred_std=0.001 (0.1%), threshold=0.005 (0.5%) -> penalty = (0.005-0.001)^2 = 0.000016
            collapse_penalty_value = (self.collapse_threshold - pred_std) ** 2
            self.last_collapse_penalty = collapse_penalty_value.detach().item()
            penalty = penalty + self.collapse_weight * collapse_penalty_value
        else:
            # No penalty when variance is healthy
            self.last_collapse_penalty = 0.0
        
        return penalty
    
    def _directional_diversity_penalty(self, median_pred: torch.Tensor) -> torch.Tensor:
        """
        Penalize extreme directional bias in predictions (prediction diversity regularization).
        
        Prevents unidirectional collapse where model predicts >90% positive (or negative) returns.
        Based on empirical analysis showing real S&P 500 never exhibits >90% directional bias 
        over 30-day windows, but collapsed models often show 95-100%.
        
        Computes directional bias on the current batch of predictions.
        
        Args:
            median_pred: Median predictions, shape [batch, time]
        
        Returns:
            Penalty tensor, shape [batch, time]
        """
        # Flatten predictions
        pred_flat = median_pred.reshape(-1)
        
        # Compute directional bias for this batch
        pct_positive = (pred_flat > 0).float().mean()
        pct_negative = (pred_flat < 0).float().mean()
        
        # Check if directional bias exceeds threshold (in either direction)
        max_directional_bias = max(pct_positive.item(), pct_negative.item())
        
        penalty = torch.zeros_like(median_pred)
        
        if max_directional_bias > self.directional_threshold:
            # Penalize excess bias
            # e.g., 95% positive, threshold 90% -> penalty = (0.95 - 0.90)^2 = 0.0025
            penalty_value = (max_directional_bias - self.directional_threshold) ** 2
            self.last_directional_penalty = float(penalty_value)
            penalty = penalty + self.directional_weight * penalty_value
        else:
            self.last_directional_penalty = 0.0
        
        return penalty
    
    def _temporal_consistency_penalty(self, median_pred: torch.Tensor) -> torch.Tensor:
        """
        Penalize discontinuity between consecutive predictions in sequential data.
        
        Only applies during validation/test (not training) since training batches are shuffled.
        Penalizes difference between last prediction of previous batch and first of current batch.
        
        Args:
            median_pred: Median predictions, shape [batch, time]
        
        Returns:
            Scalar penalty (only for first prediction of batch)
        """
        # Get first prediction of current batch
        current_first = median_pred[0, 0]
        
        # Penalize discontinuity with previous batch's last prediction
        if self.prev_prediction is not None:
            discontinuity = torch.abs(current_first - self.prev_prediction)
            penalty = self.temporal_consistency_weight * discontinuity
            self.last_temporal_penalty = penalty.detach().item()
        else:
            # First batch has no previous context
            penalty = torch.tensor(0.0, device=median_pred.device)
            self.last_temporal_penalty = 0.0
        
        # Store last prediction for next batch (detach to avoid gradient accumulation)
        self.prev_prediction = median_pred[-1, -1].detach()
        
        return penalty
    
    def reset_temporal_state(self):
        """
        Reset temporal consistency state.
        
        Call this at the start of each validation epoch to avoid carrying over 
        state from previous epochs.
        """
        self.prev_prediction = None
        self.last_temporal_penalty = None
        self.last_directional_penalty = None


# Convenience function for backward compatibility with existing CLI flags
def create_loss_from_args(args, quantiles=None):
    """
    Create EnhancedQuantileLoss from command-line arguments.
    
    Maps CLI flags to loss parameters:
    - dist_loss_std_weight -> collapse_weight (variance-based anti-collapse penalty)
    - dist_loss_mean_weight -> IGNORED (dropped - fixed targets don't work with regime variation)
    - collapse_threshold -> minimum variance threshold
    - directional_weight -> directional diversity penalty weight
    - directional_threshold -> maximum directional bias threshold
    - magnitude_weight_alpha -> linear magnitude weighting coefficient
    - extreme_move_weight -> weight multiplier for extreme moves
    - extreme_move_percentile -> percentile threshold for extreme moves
    
    Args:
        args: argparse.Namespace with loss configuration
        quantiles: List of quantile values (default: 7q preset)
    
    Returns:
        EnhancedQuantileLoss instance configured from args
    """
    return EnhancedQuantileLoss(
        quantiles=quantiles,
        collapse_weight=args.dist_loss_std_weight,
        collapse_threshold=getattr(args, 'collapse_threshold', 0.005),  # Default 0.5%
        directional_weight=getattr(args, 'directional_weight', 0.0),  # Default disabled
        directional_threshold=getattr(args, 'directional_threshold', 0.90),  # Default 90%
        temporal_consistency_weight=getattr(args, 'temporal_consistency_weight', 0.0),
        magnitude_weight_alpha=getattr(args, 'magnitude_weight_alpha', 0.0),  # Default disabled
        extreme_move_weight=getattr(args, 'extreme_move_weight', 1.0),  # Default disabled
        extreme_move_percentile=getattr(args, 'extreme_move_percentile', 95),  # Default 95th percentile
    )