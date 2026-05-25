"""
Training integration for Regime-Aware Attention TFT.

Provides hooks to thread VIX regime signals through the TFT forward pass
during training with pytorch-lightning.

Two integration approaches:
1. Monkey-patch: Modify model.forward() to extract VIX and set regime signal
2. Callback: Use Lightning callback to set regime before each batch

Usage (monkey-patch approach - recommended):
    from regime_attention import replace_attention_module
    from regime_attention_training import patch_forward_for_regime
    
    model = TemporalFusionTransformer.from_dataset(dataset, ...)
    model = replace_attention_module(model, regime_mode='vix_threshold', ...)
    model = patch_forward_for_regime(model, vix_feature_name='VIX_close')
    
    # Train normally with pytorch-lightning
    trainer.fit(model, train_dataloader, val_dataloader)

Usage (callback approach):
    from regime_attention_training import RegimeSignalCallback
    
    callback = RegimeSignalCallback(model, vix_feature_name='VIX_close')
    trainer = Trainer(callbacks=[callback, ...])
"""

from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
import numpy as np
from functools import wraps


def find_vix_index(model: nn.Module, vix_feature_name: str = 'VIX') -> Optional[int]:
    """
    Find the index of VIX feature in model's continuous features.
    
    Args:
        model: TFT model with hparams
        vix_feature_name: Name of VIX feature
        
    Returns:
        Index of VIX in x_reals, or None if not found
    """
    if not hasattr(model, 'hparams'):
        return None
    
    x_reals = getattr(model.hparams, 'x_reals', [])
    
    if vix_feature_name in x_reals:
        return x_reals.index(vix_feature_name)
    
    # Try common variations
    variations = [
        vix_feature_name,
        'VIX_close',
        'vix_close', 
        'VIX',
        'vix',
        'VIXCLS',
    ]
    
    for var in variations:
        if var in x_reals:
            return x_reals.index(var)
    
    return None


def extract_vix_from_batch(
    x: Dict[str, torch.Tensor],
    vix_idx: int,
    use_encoder: bool = True
) -> Optional[torch.Tensor]:
    """
    Extract VIX values from a batch.
    
    Args:
        x: Input batch dictionary
        vix_idx: Index of VIX in continuous features
        use_encoder: If True, use encoder_cont; else use decoder_cont
        
    Returns:
        VIX values [batch] from last timestep, or None
    """
    key = 'encoder_cont' if use_encoder else 'decoder_cont'
    
    cont = x.get(key)
    if cont is None:
        return None
    
    # Shape: [batch, seq_len, n_features]
    # Get last timestep
    if cont.dim() == 3:
        vix = cont[:, -1, vix_idx]
    else:
        vix = cont[:, vix_idx]
    
    return vix


def patch_forward_for_regime(
    model: nn.Module,
    vix_feature_name: str = 'VIX',
    raw_vix_train: Optional[np.ndarray] = None,
    raw_vix_val: Optional[np.ndarray] = None
) -> nn.Module:
    """
    Monkey-patch model forward to automatically set regime signal.
    
    Args:
        model: TFT model with regime-aware attention
        vix_feature_name: Name of VIX feature (for logging)
        raw_vix_train: Raw (unnormalized) VIX values for training set
        raw_vix_val: Raw (unnormalized) VIX values for validation set
        
    Returns:
        model: Same model with patched forward
    """
    if raw_vix_train is None or raw_vix_val is None:
        print(f"[REGIME ATTENTION] WARNING: raw_vix not provided, regime gating disabled")
        return model
    
    # Store raw VIX on model as tensors
    model._raw_vix_train = torch.tensor(raw_vix_train, dtype=torch.float32)
    model._raw_vix_val = torch.tensor(raw_vix_val, dtype=torch.float32)
    model._train_len = len(raw_vix_train)
    
    print(f"[REGIME ATTENTION] Using raw VIX values (train: {len(raw_vix_train)}, val: {len(raw_vix_val)})")
    print(f"[REGIME ATTENTION] VIX range - train: {raw_vix_train.min():.1f}-{raw_vix_train.max():.1f}, val: {raw_vix_val.min():.1f}-{raw_vix_val.max():.1f}")
    
    # Store original forward
    original_forward = model.forward
    
    @wraps(original_forward)
    def forward_with_regime(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Extract time index to look up raw VIX
        vix_values = None
        if 'decoder_time_idx' in x:
            time_idx = x['decoder_time_idx'][:, 0]  # First prediction step
            
            if model.training:
                vix_values = model._raw_vix_train[time_idx.cpu()].to(x['encoder_cont'].device)
                #regime = (vix_values >= 25.0).int()
                #print(f"[DEBUG TRAIN] regime dist: 0={( regime==0).sum()}/{len(regime)}, 1={(regime==1).sum()}/{len(regime)}")
            else:
                # Validation indices are offset by train length
                offset_idx = time_idx.cpu() - model._train_len
                # Clamp to valid range
                offset_idx = offset_idx.clamp(0, len(model._raw_vix_val) - 1)
                vix_values = model._raw_vix_val[offset_idx].to(x['encoder_cont'].device)
                
                """
                if model.training:
                    print(f"[DEBUG TRAIN] time_idx: {time_idx[:5]}, vix: {vix_values[:5]}")
                else:
                    raw_idx = time_idx.cpu() - model._train_len
                    print(f"[DEBUG VAL] time_idx: {time_idx[:5]}, offset: {raw_idx[:5]}, vix: {vix_values[:5]}")
                """
        
        # Set regime signal on attention module
        if vix_values is not None and hasattr(model, 'multihead_attn'):
            attn = model.multihead_attn
            if hasattr(attn, 'set_regime_signal'):
                attn.set_regime_signal(vix_values)
        
        # Call original forward
        return original_forward(x)
    
    # Replace forward method
    model.forward = forward_with_regime

    return model


class RegimeSignalCallback:
    """
    PyTorch Lightning callback to set regime signal before each batch.
    
    Alternative to monkey-patching forward(). Use this if you need more
    control over when regime signals are set.
    
    Usage:
        callback = RegimeSignalCallback(model, vix_feature_name='VIX_close')
        trainer = Trainer(callbacks=[callback])
    """
    
    def __init__(
        self,
        model: nn.Module,
        vix_feature_name: str = 'VIX',
        vix_idx: Optional[int] = None
    ):
        self.model = model
        self.vix_feature_name = vix_feature_name
        self.vix_idx = vix_idx or find_vix_index(model, vix_feature_name)
        
        if self.vix_idx is None:
            print(f"[REGIME CALLBACK] WARNING: VIX feature not found")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Set regime signal before training batch."""
        self._set_regime_signal(batch)
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Set regime signal before validation batch."""
        self._set_regime_signal(batch)
    
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Set regime signal before test batch."""
        self._set_regime_signal(batch)
    
    def _set_regime_signal(self, batch):
        """Extract VIX and set regime signal on attention module."""
        if self.vix_idx is None:
            return
        
        x, _ = batch  # Unpack (x, y) tuple
        vix = extract_vix_from_batch(x, self.vix_idx)
        
        if vix is not None and hasattr(self.model, 'multihead_attn'):
            attn = self.model.multihead_attn
            if hasattr(attn, 'set_regime_signal'):
                attn.set_regime_signal(vix)


def get_regime_diagnostics(model: nn.Module) -> Dict[str, Any]:
    """
    Get regime attention diagnostics from model.
    
    Args:
        model: TFT model with regime-aware attention
        
    Returns:
        Dictionary with regime gating statistics
    """
    if not hasattr(model, 'multihead_attn'):
        return {}
    
    attn = model.multihead_attn
    if not hasattr(attn, 'get_regime_diagnostics'):
        return {}
    
    diag = attn.get_regime_diagnostics()
    
    # Add summary statistics
    result = {
        'regime_mode': getattr(attn, 'regime_mode', 'unknown'),
        'vix_threshold': getattr(attn, 'vix_threshold', None),
        'num_regimes': getattr(attn, 'num_regimes', 0),
    }
    
    if diag.get('current_regime') is not None:
        regime = diag['current_regime']
        result['regime_distribution'] = {
            'regime_0_fraction': (regime == 0).float().mean().item(),
            'regime_1_fraction': (regime == 1).float().mean().item() if attn.num_regimes > 1 else 0.0,
        }
    
    if diag.get('gate_weights') is not None:
        gates = diag['gate_weights']
        result['gate_statistics'] = {
            'mean': gates.mean().item(),
            'std': gates.std().item(),
            'min': gates.min().item(),
            'max': gates.max().item(),
        }
    
    if diag.get('raw_gates') is not None:
        raw = diag['raw_gates']
        result['raw_gate_parameters'] = raw.tolist()
    
    return result


def log_regime_metrics(model: nn.Module, logger, step: int, prefix: str = 'regime'):
    """
    Log regime attention metrics to tensorboard/wandb.
    
    Args:
        model: TFT model with regime-aware attention
        logger: Lightning logger
        step: Current global step
        prefix: Metric prefix
    """
    diag = get_regime_diagnostics(model)
    
    if not diag:
        return
    
    # Log gate statistics
    if 'gate_statistics' in diag:
        for key, val in diag['gate_statistics'].items():
            logger.log_metrics({f'{prefix}/gate_{key}': val}, step=step)
    
    # Log regime distribution
    if 'regime_distribution' in diag:
        for key, val in diag['regime_distribution'].items():
            logger.log_metrics({f'{prefix}/{key}': val}, step=step)
    
    # Log raw gate parameters (if small enough)
    if 'raw_gate_parameters' in diag:
        raw = diag['raw_gate_parameters']
        for regime_idx, head_gates in enumerate(raw):
            for head_idx, gate_val in enumerate(head_gates):
                logger.log_metrics({
                    f'{prefix}/gate_r{regime_idx}_h{head_idx}': gate_val
                }, step=step)
