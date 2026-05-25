"""
Callback for monitoring collapse behavior during TFT training.

Tracks prediction diversity, gradient flow, and weight statistics to understand
when and how models collapse to constant predictions.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import json
import os
from pathlib import Path

# Quantile configuration utilities
try:
    from src.quantile_config import get_median_index
except ImportError:
    # Fallback for standalone usage
    def get_median_index(quantiles):
        return quantiles.index(0.5)


class CollapseMonitor(Callback):
    """
    Monitor training dynamics to detect and analyze prediction collapse.
    
    Tracks:
    - Prediction diversity (std, range, sign distribution)
    - Gradient magnitudes by layer
    - Weight statistics
    - Variable selection network outputs
    - Attention weight entropy
    """
    
    def __init__(self, val_dataloader, log_dir, log_every_n_epochs=1, quantiles=None, max_prediction_length=1):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_epochs = log_every_n_epochs
        
        # Quantile configuration for median index lookup
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]  # Default 7q
        self.quantiles = quantiles
        self.median_idx = get_median_index(quantiles)
        
        # Multi-horizon configuration
        self.max_prediction_length = max_prediction_length
        
        # Buffer to store gradients captured during training
        self._current_epoch_gradients = {}
        
        self.history = {
            'epoch': [],
            'prediction_std': [],
            'prediction_range': [],
            'prediction_mean': [],
            'pct_positive': [],
            'pct_negative': [],
            'num_unique_predictions': [],
            'gradient_norms': {},
            'weight_norms': {},
            'weight_stds': {},
            'vsn_output_std': {},
            'attention_entropy': [],
        }
        
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        """Capture gradients before they're cleared by optimizer."""
        # Store gradient norms for this batch
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if name not in self._current_epoch_gradients:
                    self._current_epoch_gradients[name] = []
                    
                self._current_epoch_gradients[name].append(grad_norm)
            
            # Log regime gate gradients
            if 'regime_gates' in name:
                if not hasattr(self, '_regime_gate_grads_epoch'):
                    self._regime_gate_grads_epoch = []
                self._regime_gate_grads_epoch.append(param.grad.clone().cpu())
            
            """
            if 'regime_gates' in name and param.grad is not None:
                param.grad *= 100  # Amplify gradient
                print(f"[DEBUG] regime_gates grad (amplified): {param.grad}")
    
            if 'regime_gates' in name:
                print(f"[DEBUG] regime_gates grad in optimizer step: {param.grad}")
            """

    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset temporal state in loss function before validation epoch."""
        # Reset directional penalty buffer and temporal consistency state
        # This ensures validation statistics are computed on fresh, sequential data
        print(f"[DEBUG CollapseMonitor] on_validation_epoch_start called for epoch {trainer.current_epoch}")
        if hasattr(pl_module, 'loss') and hasattr(pl_module.loss, 'reset_temporal_state'):
            pl_module.loss.reset_temporal_state()
            print(f"[DEBUG CollapseMonitor] reset_temporal_state() called on loss function")
        else:
            print(f"[DEBUG CollapseMonitor] WARNING: Could not find reset_temporal_state on loss")
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Log metrics at end of each epoch."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
            
        print(f"\n[CollapseMonitor] Epoch {trainer.current_epoch}")
        
        # 1. Prediction diversity metrics
        self._log_prediction_diversity(trainer, pl_module)
        
        # 2. Gradient flow
        self._log_gradient_flow(pl_module)
        
        # 3. Weight statistics
        self._log_weight_statistics(pl_module)
        
        # 4. Variable selection network activity
        self._log_vsn_activity(trainer, pl_module)
        
        # 5. Attention patterns
        self._log_attention_patterns(trainer, pl_module)
        
        # 6. Regime diagnostics (if regime output enabled)
        self._log_regime_diagnostics(trainer, pl_module)
        
        # 7. Expert weight divergence (if regime output enabled)
        self._log_expert_weight_divergence(pl_module)

        # 8. Regime attention diagnostics (if regime attention enabled)
        self._log_regime_attention_diagnostics(pl_module)
        
        # Save to disk
        self._save_history(trainer.current_epoch)
        
        # Reset model state after monitoring
        pl_module.train()
        if hasattr(pl_module, '_current_fx_name'):
            pl_module._current_fx_name = None

        # Reset debug flags for next epoch
        if hasattr(pl_module, 'output_layer'):
            if hasattr(pl_module.output_layer, '_vix_logged_this_epoch'):
                pl_module.output_layer._vix_logged_this_epoch = False
            if hasattr(pl_module.output_layer, '_hard_routing_logged_this_epoch'):
                pl_module.output_layer._hard_routing_logged_this_epoch = False
        
    def _log_prediction_diversity(self, trainer, pl_module):
        """
        Measure diversity of predictions on validation set.
        
        Metrics (per horizon if multi-horizon):
        - std/range/mean: Basic statistics over all validation predictions
        - Pos/Neg %: Percentage of predictions above/below zero
        - Unique: Number of distinct prediction values across entire val set
          (1 = collapsed to constant, hundreds = healthy diversity)
        - Directional accuracy: % of times sign(pred) == sign(actual)
        
        For multi-horizon (max_prediction_length > 1):
        - Per-horizon metrics stored with suffix (e.g., prediction_std_h1, prediction_std_h2)
        - Aggregate metrics computed as mean across horizons
        - Primary metrics (for checkpointing) use horizon 0 (next-step prediction)
        """
        pl_module.eval()
        
        # Collect predictions and actuals for all horizons
        # predictions_by_horizon[h] = list of batch predictions for horizon h
        predictions_by_horizon = {h: [] for h in range(self.max_prediction_length)}
        actuals_by_horizon = {h: [] for h in range(self.max_prediction_length)}
        
        with torch.no_grad():
            batch_count = 0
            for batch in self.val_dataloader:
                batch_count += 1
                x, y = batch
                
                # Move batch to device
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Get predictions - handle different output formats
                # Shape: [batch, max_prediction_length, num_quantiles]
                output = pl_module(x)
                if isinstance(output, dict):
                    pred_tensor = output['prediction']
                elif hasattr(output, 'prediction'):
                    pred_tensor = output.prediction
                else:
                    pred_tensor = output
                
                # Extract median predictions for each horizon
                for h in range(self.max_prediction_length):
                    preds_h = pred_tensor[:, h, self.median_idx]
                    predictions_by_horizon[h].append(preds_h.cpu().numpy())
                
                # Collect actuals - shape depends on prediction length
                y_tensor = y[0] if isinstance(y, (list, tuple)) else y
                # y_tensor shape: [batch, prediction_length] for multi-horizon, [batch, 1] for single
                for h in range(self.max_prediction_length):
                    if y_tensor.ndim == 2 and y_tensor.shape[1] > h:
                        actuals_h = y_tensor[:, h].cpu().numpy()
                    else:
                        actuals_h = y_tensor.flatten().cpu().numpy()
                    actuals_by_horizon[h].append(actuals_h)

        # Concatenate predictions and actuals for each horizon
        predictions_by_horizon = {h: np.concatenate(preds) for h, preds in predictions_by_horizon.items()}
        actuals_by_horizon = {h: np.concatenate(acts) for h, acts in actuals_by_horizon.items()}
        
        # Initialize per-horizon history keys if needed
        if self.max_prediction_length > 1 and 'prediction_std_by_horizon' not in self.history:
            self.history['prediction_std_by_horizon'] = {f'h{h}': [] for h in range(self.max_prediction_length)}
            self.history['directional_accuracy_by_horizon'] = {f'h{h}': [] for h in range(self.max_prediction_length)}
            self.history['num_unique_by_horizon'] = {f'h{h}': [] for h in range(self.max_prediction_length)}
        
        # Compute metrics for each horizon
        metrics_by_horizon = {}
        for h in range(self.max_prediction_length):
            predictions = predictions_by_horizon[h]
            actuals = actuals_by_horizon[h]
            
            pred_std = np.std(predictions)
            pred_range = np.ptp(predictions)
            pred_mean = np.mean(predictions)
            pct_pos = np.mean(predictions > 0) * 100
            pct_neg = np.mean(predictions < 0) * 100
            n_unique = len(np.unique(np.round(predictions, decimals=6)))
            
            # Directional accuracy
            pred_signs = np.sign(predictions)
            actual_signs = np.sign(actuals)
            dir_acc = np.mean(pred_signs == actual_signs)
            
            # Prediction Sharpe
            if pred_std > 1e-10:
                pred_sharpe = (pred_mean / pred_std) * np.sqrt(252)
            else:
                pred_sharpe = 0.0
            
            # Composite metric
            actual_pct_positive = np.mean(actuals > 0)
            pred_pct_positive = np.mean(predictions > 0)
            distribution_match = 1.0 - abs(pred_pct_positive - actual_pct_positive)
            composite = dir_acc * (0.5 + 0.5 * distribution_match)
            
            metrics_by_horizon[h] = {
                'pred_std': pred_std,
                'pred_range': pred_range,
                'pred_mean': pred_mean,
                'pct_pos': pct_pos,
                'pct_neg': pct_neg,
                'n_unique': n_unique,
                'dir_acc': dir_acc,
                'pred_sharpe': pred_sharpe,
                'composite': composite,
                'distribution_match': distribution_match,
                'actual_pct_positive': actual_pct_positive,
            }
            
            # Store per-horizon metrics if multi-horizon
            if self.max_prediction_length > 1:
                self.history['prediction_std_by_horizon'][f'h{h}'].append(float(pred_std))
                self.history['directional_accuracy_by_horizon'][f'h{h}'].append(float(dir_acc))
                self.history['num_unique_by_horizon'][f'h{h}'].append(int(n_unique))
        
        # Use horizon 0 (next-step) as primary metrics for checkpointing
        h0 = metrics_by_horizon[0]
        predictions = predictions_by_horizon[0]
        actuals = actuals_by_horizon[0]
        
        # Store primary metrics in history (horizon 0)
        self.history['epoch'].append(trainer.current_epoch)
        self.history['prediction_std'].append(float(h0['pred_std']))
        self.history['prediction_range'].append(float(h0['pred_range']))
        self.history['prediction_mean'].append(float(h0['pred_mean']))
        self.history['pct_positive'].append(float(h0['pct_pos']))
        self.history['pct_negative'].append(float(h0['pct_neg']))
        self.history['num_unique_predictions'].append(int(h0['n_unique']))
        
        if 'directional_accuracy' not in self.history:
            self.history['directional_accuracy'] = []
        if 'prediction_sharpe' not in self.history:
            self.history['prediction_sharpe'] = []
        if 'composite' not in self.history:
            self.history['composite'] = []
        if 'distribution_match' not in self.history:
            self.history['distribution_match'] = []
        if 'actual_pct_positive' not in self.history:
            self.history['actual_pct_positive'] = []
            
        self.history['directional_accuracy'].append(float(h0['dir_acc']))
        self.history['prediction_sharpe'].append(float(h0['pred_sharpe']))
        self.history['composite'].append(float(h0['composite']))
        self.history['distribution_match'].append(float(h0['distribution_match']))
        self.history['actual_pct_positive'].append(float(h0['actual_pct_positive'] * 100))
        
        # Log primary metrics to pl_module for ModelCheckpoint
        pl_module.log('val_pred_std', float(h0['pred_std']), on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log('val_pct_positive', float(h0['pct_pos']), on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log('val_num_unique', int(h0['n_unique']), on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log('val_dir_acc', float(h0['dir_acc']), on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log('val_sharpe', float(h0['pred_sharpe']), on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log('val_composite', float(h0['composite']), on_step=False, on_epoch=True, prog_bar=False)
        
        # Compute and log aggregate metrics if multi-horizon
        if self.max_prediction_length > 1:
            avg_pred_std = np.mean([m['pred_std'] for m in metrics_by_horizon.values()])
            avg_dir_acc = np.mean([m['dir_acc'] for m in metrics_by_horizon.values()])
            avg_composite = np.mean([m['composite'] for m in metrics_by_horizon.values()])
            
            pl_module.log('val_pred_std_avg', float(avg_pred_std), on_step=False, on_epoch=True, prog_bar=False)
            pl_module.log('val_dir_acc_avg', float(avg_dir_acc), on_step=False, on_epoch=True, prog_bar=False)
            pl_module.log('val_composite_avg', float(avg_composite), on_step=False, on_epoch=True, prog_bar=False)
            
            # Store aggregates in history
            if 'prediction_std_avg' not in self.history:
                self.history['prediction_std_avg'] = []
                self.history['directional_accuracy_avg'] = []
                self.history['composite_avg'] = []
            self.history['prediction_std_avg'].append(float(avg_pred_std))
            self.history['directional_accuracy_avg'].append(float(avg_dir_acc))
            self.history['composite_avg'].append(float(avg_composite))
        
        # Compute anti-collapse penalty (using horizon 0)
        collapse_penalty = None
        if hasattr(pl_module.loss, 'collapse_weight') and hasattr(pl_module.loss, 'collapse_threshold'):
            loss_fn = pl_module.loss
            if h0['pred_std'] < loss_fn.collapse_threshold:
                collapse_penalty = loss_fn.collapse_weight * (loss_fn.collapse_threshold - h0['pred_std']) ** 2
            else:
                collapse_penalty = 0.0
            self.history['collapse_penalty'] = self.history.get('collapse_penalty', [])
            self.history['collapse_penalty'].append(float(collapse_penalty))
        
        # Print summary
        if self.max_prediction_length == 1:
            # Original verbose output for single-horizon
            print(f"  [DEBUG] Processed {batch_count} batches from val_dataloader")
            print(f"  [DEBUG] Predictions shape after concat: {predictions.shape}")
            print(f"  Pred std: {h0['pred_std']:.6f}, range: {h0['pred_range']:.6f}, mean: {h0['pred_mean']:.6f}")
            print(f"  Pos: {h0['pct_pos']:.1f}%, Neg: {h0['pct_neg']:.1f}%, Unique: {h0['n_unique']}")
            print(f"  Dir Acc: {h0['dir_acc']*100:.2f}%, Pred Sharpe: {h0['pred_sharpe']:.4f}")
            print(f"  Composite: {h0['composite']:.4f} (dist_match: {h0['distribution_match']:.4f}, "
                  f"actual_pos: {h0['actual_pct_positive']*100:.1f}%, pred_pos: {h0['pct_pos']:.1f}%)")
        else:
            # Multi-horizon output with key metrics per horizon
            print(f"  [DEBUG] Processed {batch_count} batches, {self.max_prediction_length} horizon(s)")
            print(f"  Multi-horizon metrics ({self.max_prediction_length} horizons):")
            for h in range(min(self.max_prediction_length, 5)):  # Show first 5 horizons
                m = metrics_by_horizon[h]
                print(f"    h{h+1}: std={m['pred_std']:.4f}, dir_acc={m['dir_acc']*100:.1f}%, "
                      f"pos={m['pct_pos']:.0f}%, unique={m['n_unique']}, sharpe={m['pred_sharpe']:.2f}")
            if self.max_prediction_length > 5:
                print(f"    ... ({self.max_prediction_length - 5} more horizons)")
            print(f"  Aggregate: std_avg={avg_pred_std:.4f}, dir_acc_avg={avg_dir_acc*100:.1f}%, composite_avg={avg_composite:.4f}")
            # Also print h1 details for direct comparison with single-horizon runs
            print(f"  H1 detail: range={h0['pred_range']:.4f}, mean={h0['pred_mean']:.4f}, "
                  f"dist_match={h0['distribution_match']:.4f}")
        
        if collapse_penalty is not None:
            print(f"  Collapse penalty: {collapse_penalty:.6f} (threshold: {loss_fn.collapse_threshold:.6f})")
        
        # Print directional penalty if model uses it
        if hasattr(pl_module.loss, 'directional_weight') and hasattr(pl_module.loss, 'last_directional_penalty'):
            dir_weight = pl_module.loss.directional_weight
            dir_penalty = pl_module.loss.last_directional_penalty
            dir_threshold = pl_module.loss.directional_threshold
            if dir_weight > 0:
                if dir_penalty is not None:
                    print(f"  Directional penalty: {dir_penalty:.6f} (weight: {dir_weight:.2f}, threshold: {dir_threshold:.2f})")
                else:
                    print(f"  Directional penalty: not computed yet (weight: {dir_weight:.2f}, threshold: {dir_threshold:.2f})")
        
        # Regime attention diagnostics
        if hasattr(pl_module, 'multihead_attn') and hasattr(pl_module.multihead_attn, 'get_regime_diagnostics'):
            from train.regime_attention_training import get_regime_diagnostics
            diag = get_regime_diagnostics(pl_module)
            if diag.get('regime_distribution'):
                print(f"  Regime distribution: {diag['regime_distribution']}")
            if diag.get('gate_statistics'):
                print(f"  Regime gates: {diag['gate_statistics']}")

        # Save predictions for debugging (all horizons)
        pred_save_path = self.log_dir / f'val_predictions_epoch{trainer.current_epoch}.npy'
        if self.max_prediction_length == 1:
            np.save(pred_save_path, predictions)
        else:
            # Save as dict with horizon keys
            np.savez(pred_save_path.with_suffix('.npz'), **{f'h{h}': predictions_by_horizon[h] for h in range(self.max_prediction_length)})
            pred_save_path = pred_save_path.with_suffix('.npz')
        print(f"  Saved validation predictions to: {pred_save_path}")

    def _log_gradient_flow(self, pl_module):
        """Log gradient magnitudes by layer using stored gradients from training."""
        epoch = len(self.history['epoch']) - 1
        
        # Debug: Print attention parameter names to verify pattern
        attn_params = [n for n, p in pl_module.named_parameters() if 'attention' in n.lower()]
        if epoch == 0 and attn_params:  # Only print on first epoch to avoid spam
            print(f"  [DEBUG] Attention param names: {attn_params[:5]}")  # Show first 5
        
        # Use the gradients we captured during training
        for name, grad_norms in self._current_epoch_gradients.items():
            # Average gradient norm across all batches in this epoch
            avg_grad_norm = np.mean(grad_norms)
            
            if name not in self.history['gradient_norms']:
                self.history['gradient_norms'][name] = []
                
            self.history['gradient_norms'][name].append(float(avg_grad_norm))
        
        # Print summary of key layers
        key_layers = ['lstm_encoder', 'lstm_decoder', 'multihead_attention', 
                      'output_layer', 'classification_head']
        print("  Gradient norms:")
        for layer_name in key_layers:
            matching = [k for k in self.history['gradient_norms'].keys() 
                       if layer_name in k]
            if matching:
                norms = [self.history['gradient_norms'][k][-1] for k in matching]
                avg_norm = np.mean(norms)
                print(f"    {layer_name}: {avg_norm:.6f}")
        
        # In _log_gradient_flow, replace the existing regime gate grad print:
        if hasattr(self, '_regime_gate_grads_epoch') and self._regime_gate_grads_epoch:
            avg_grad = torch.stack(self._regime_gate_grads_epoch).mean(dim=0)
            print(f"  Regime gate grads (epoch avg): {avg_grad}")
                    
            # Save to history
            if 'regime_gate_grads' not in self.history:
                self.history['regime_gate_grads'] = []
            self.history['regime_gate_grads'].append(avg_grad.tolist())
    
            self._regime_gate_grads_epoch = []  # Reset for next epoch

        # Clear buffer for next epoch
        self._current_epoch_gradients = {}
                
    def _log_weight_statistics(self, pl_module):
        """Log weight matrix statistics."""
        epoch = len(self.history['epoch']) - 1
        
        for name, param in pl_module.named_parameters():
            if 'weight' in name:
                weight_norm = param.norm().item()
                weight_std = param.std().item()
                
                if name not in self.history['weight_norms']:
                    self.history['weight_norms'][name] = []
                    self.history['weight_stds'][name] = []
                    
                self.history['weight_norms'][name].append(float(weight_norm))
                self.history['weight_stds'][name].append(float(weight_std))
                
    def _log_vsn_activity(self, trainer, pl_module):
        """Log variable selection network output statistics."""
        pl_module.eval()
        
        vsn_outputs = {
            'encoder': [],
        }
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= 10:  # Sample first 10 batches
                    break
                    
                x, y = batch
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Get output - could be dict, namedtuple, or plain tensor
                output = pl_module(x)
                
                # Try to extract encoder_variables using different methods
                encoder_vsn = None
                
                # Method 1: Check if it's a dict with 'encoder_variables' key
                if isinstance(output, dict) and 'encoder_variables' in output:
                    encoder_vsn = output['encoder_variables']
                # Method 2: Check if it's a namedtuple with encoder_variables attribute
                elif hasattr(output, 'encoder_variables'):
                    encoder_vsn = output.encoder_variables
                # Method 3: Custom TFT model - use getter method
                elif hasattr(pl_module, 'get_encoder_vsn_output'):
                    vsn_result = pl_module.get_encoder_vsn_output()
                    if vsn_result is not None:
                        encoder_vsn, _ = vsn_result  # (output, weights) tuple
                
                if encoder_vsn is not None:
                    vsn_outputs['encoder'].append(encoder_vsn.cpu().numpy())
        
        # Compute std for each VSN
        epoch = len(self.history['epoch']) - 1
        print("  VSN output std:")
        for vsn_name, outputs in vsn_outputs.items():
            if outputs:
                concatenated = np.concatenate(outputs)
                vsn_std = np.std(concatenated)
                
                if vsn_name not in self.history['vsn_output_std']:
                    self.history['vsn_output_std'][vsn_name] = []
                    
                self.history['vsn_output_std'][vsn_name].append(float(vsn_std))
                print(f"    {vsn_name}: {vsn_std:.6f}")
            else:
                print(f"    {vsn_name}: (no data captured)")
                if vsn_name not in self.history['vsn_output_std']:
                    self.history['vsn_output_std'][vsn_name] = []
                self.history['vsn_output_std'][vsn_name].append(None)
                
    def _log_attention_patterns(self, trainer, pl_module):
        """
        Log attention weight entropy.
        
        For pytorch-forecasting TFT: Use interpret_output() method
        For custom TFT: Use get_attention_weights() getter method
        """
        pl_module.eval()
        attention_entropies = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= 10:  # Sample first 10 batches
                    break
                    
                x, y = batch
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Get predictions first
                output = pl_module(x)
                
                attn_weights = None
                
                # Method 1: Custom TFT model - use getter method
                if hasattr(pl_module, 'get_attention_weights'):
                    attn_weights = pl_module.get_attention_weights()
                    
                    if attn_weights is not None:
                        # Shape: [batch, num_heads, max_prediction_length, encoder_length + max_prediction_length]
                        # Compute per-head entropy instead of averaging heads first
                        
                        # For single-step prediction, extract first (only) prediction timestep
                        # Shape: [batch, num_heads, encoder_length + max_prediction_length]
                        if attn_weights.size(2) == 1:
                            attn_weights = attn_weights[:, :, 0, :]
                        else:
                            # If multiple prediction steps, use first one
                            attn_weights = attn_weights[:, :, 0, :]
                        
                        # Move to CPU and convert to numpy
                        attn = attn_weights.cpu().numpy()
                        
                        # Compute entropy per head, then average
                        # attn shape: [batch, num_heads, encoder_length + max_prediction_length]
                        per_head_entropies = []
                        num_heads = attn.shape[1]
                        
                        for h in range(num_heads):
                            # Extract attention for this head: [batch, seq_len]
                            head_attn = attn[:, h, :]
                            
                            # Normalize (should already be normalized, but verify)
                            head_attn_norm = head_attn / (head_attn.sum(axis=-1, keepdims=True) + 1e-10)
                            
                            # Compute entropy: [batch]
                            head_entropy = -np.sum(head_attn_norm * np.log(head_attn_norm + 1e-10), axis=-1)
                            per_head_entropies.append(head_entropy)
                        
                        # Average entropy across heads: [batch]
                        avg_entropy_per_batch = np.mean(per_head_entropies, axis=0)
                        attention_entropies.append(avg_entropy_per_batch)
                        
                # Method 2: pytorch-forecasting TFT model - use interpret_output
                elif hasattr(pl_module, 'interpret_output'):
                    try:
                        interpretation = pl_module.interpret_output(
                            output, 
                            reduction='none',
                            attention_prediction_horizon=0  # Focus on first prediction step
                        )
                        
                        # Extract attention weights from interpretation dict
                        # Note: interpret_output may already average across heads internally
                        # We cannot change pytorch-forecasting's internal behavior, but keep
                        # consistent structure with custom TFT branch above
                        if 'attention' in interpretation:
                            attn = interpretation['attention'].cpu().numpy()
                            # Compute entropy of attention distribution
                            # attn shape: [batch, encoder_length] typically (already head-averaged)
                            # Normalize if not already (attention should sum to 1)
                            attn_norm = attn / (attn.sum(axis=-1, keepdims=True) + 1e-10)
                            entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10), axis=-1)
                            attention_entropies.append(entropy)
                    except Exception as e:
                        # interpret_output might fail for various reasons
                        print(f"    (interpret_output failed: {type(e).__name__})")
                        break
                else:
                    # No method available to extract attention
                    print(f"    (no attention extraction method available)")
                    break
        
        if attention_entropies:
            avg_entropy = np.mean(np.concatenate(attention_entropies))
            self.history['attention_entropy'].append(float(avg_entropy))
            print(f"  Attention entropy: {avg_entropy:.6f}")
        else:
            self.history['attention_entropy'].append(None)
            print(f"  Attention entropy: (no data captured)")
    
    def _log_regime_attention_diagnostics(self, pl_module):
        """Log regime attention gate values if enabled."""
        if not hasattr(pl_module, 'multihead_attn'):
            return
        if not hasattr(pl_module.multihead_attn, 'regime_gates'):
            return
        
        gate_values = torch.sigmoid(pl_module.multihead_attn.regime_gates).detach().cpu()
        
        if 'regime_attention_gate_values' not in self.history:
            self.history['regime_attention_gate_values'] = []
        self.history['regime_attention_gate_values'].append(gate_values.tolist())
        
        print(f"  Regime attention gates: {gate_values.tolist()}")

    def _log_regime_diagnostics(self, trainer, pl_module):
        """
        Log regime-conditional output statistics if enabled.
        
        Tracks routing behavior and per-expert predictions by reading cached
        diagnostics from RegimeConditionalOutput layer (no recomputation needed).
        
        Metrics:
        - Routing entropy: H=0 (single expert dominates) to H=log(K) (uniform)
        - Dominant regime %: Percentage assigned to most common regime
        - Per-expert prediction diversity: Std/mean for each expert
        - VIX correlation: Correlation between routing weights and VIX (if available)
        """
        # check if regime output is enabled
        if not hasattr(pl_module, 'output_layer'):
            return
        
        # importing RegimeConditionalOutput
        try:
            import sys
            src_path = Path(__file__).parent / 'src'
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from src.regime_output import RegimeConditionalOutput
        except ImportError:
            return  # regime_output module not available
        
        if not isinstance(pl_module.output_layer, RegimeConditionalOutput):
            return  # Baseline output layer
        
        if pl_module.output_layer.routing_mode == 'disabled':
            return  # Single expert mode, nothing to diagnose
        
        pl_module.eval()
        
        routing_weights_list = []
        expert_preds_lists = [[] for _ in range(pl_module.output_layer.num_regimes)]
        vix_values = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= 10:
                    break
            
                x, y = batch
                x = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                     for k, v in x.items()}
                
                # Forward pass - regime output caches diagnostics internally
                output = pl_module(x)
                
                # Read from cache (already detached and on CPU)
                if pl_module.output_layer._cached_routing_weights is None:
                    continue
                
                routing_weights = pl_module.output_layer._cached_routing_weights
                
                """
                if i == 0:
                    print(f"[DEBUG Monitor] Cached routing shape: {routing_weights.shape}")
                    print(f"[DEBUG Monitor] First 5 samples: {routing_weights[:5].numpy()}")
                """

                # Routing weights shape: [batch, seq_len, num_regimes]
                # Extract first timestep for single-step prediction
                if routing_weights.ndim == 3:
                    routing_weights = routing_weights[:, 0, :]  # [batch, num_regimes]
                elif routing_weights.ndim == 2:
                    pass  # Already [batch, num_regimes]
                
                routing_weights_list.append(routing_weights.numpy())
                
                # Extract per-expert predictions from cache
                for regime_idx in range(pl_module.output_layer.num_regimes):
                    cached_pred = getattr(
                        pl_module.output_layer, 
                        f'_cached_expert_preds_{regime_idx}',
                        None
                    )
                    if cached_pred is None:
                        continue
                    
                    # Extract median quantile at first timestep
                    # Shape: [batch, seq_len, output_size] -> [batch]
                    if cached_pred.ndim == 3:
                        preds = cached_pred[:, 0, self.median_idx].numpy()
                    elif cached_pred.ndim == 2:
                        preds = cached_pred[:, self.median_idx].numpy()
                    else:
                        continue
                    
                    expert_preds_lists[regime_idx].append(preds)
               
                # Extract raw VIX (same method as training)
                if 'decoder_time_idx' in x:
                    time_idx = x['decoder_time_idx'][:, 0]
                    offset_idx = time_idx.cpu() - len(pl_module._raw_vix_train)
                    vix_val = pl_module._raw_vix_val[offset_idx].numpy()
                    vix_values.append(vix_val)
                    
                """
                # Extract VIX if available
                # VIX could be in encoder_cont or decoder_cont depending on feature config
                # Try multiple possible locations
                vix_extracted = False
                
                # Check encoder continuous features (most likely location)
                if 'encoder_cont' in x and x['encoder_cont'].numel() > 0:
                    # Try to find VIX in feature names if available
                    # Otherwise assume it's first feature (configurable)
                    # Extract from last encoder timestep: [batch, time, features]
                    encoder_cont = x['encoder_cont']
                    if encoder_cont.ndim == 3 and encoder_cont.size(-1) > 0:
                        # Use first continuous feature as VIX (customize this index as needed)
                        vix_val = encoder_cont[:, -1, 0].cpu().numpy()
                        vix_values.append(vix_val)
                        vix_extracted = True
                
                # Fallback: try decoder_cont
                if not vix_extracted and 'decoder_cont' in x and x['decoder_cont'].numel() > 0:
                    decoder_cont = x['decoder_cont']
                    if decoder_cont.ndim == 3 and decoder_cont.size(-1) > 0:
                        vix_val = decoder_cont[:, 0, 0].cpu().numpy()  # First decoder timestep
                        vix_values.append(vix_val)
                """
        if not routing_weights_list:
            print("  Regime diagnostics: (no data captured)")
            return
        
        # Concatenate across batches
        routing_weights = np.concatenate(routing_weights_list, axis=0)  # [N, num_regimes]
        expert_preds = [
            np.concatenate(preds, axis=0) if preds else np.array([])
            for preds in expert_preds_lists
        ]
        
        # 1. Routing entropy (H=0 if one regime dominates, H=log(K) if uniform)
        eps = 1e-10
        per_sample_entropy = -np.sum(
            routing_weights * np.log(routing_weights + eps), 
            axis=1
        )
        routing_entropy = per_sample_entropy.mean()
        max_entropy = np.log(pl_module.output_layer.num_regimes)
        normalized_entropy = routing_entropy / max_entropy  # [0, 1]
        
        # 2. Dominant regime percentage
        assigned_regimes = np.argmax(routing_weights, axis=1)  # [N]
        regime_counts = np.bincount(
            assigned_regimes, 
            minlength=pl_module.output_layer.num_regimes
        )
        dominant_regime_pct = regime_counts.max() / len(assigned_regimes) * 100
        
        # 3. Per-expert prediction diversity
        expert_stds = [np.std(preds) if len(preds) > 0 else 0.0 for preds in expert_preds]
        expert_means = [np.mean(preds) if len(preds) > 0 else 0.0 for preds in expert_preds]
        
        # 4. VIX correlation (if available)
        vix_corr = None
        if vix_values and len(vix_values) == len(routing_weights_list):
            vix = np.concatenate(vix_values, axis=0)  # [N]
            # Correlate routing weight for regime 1 (assumed high-volatility) with VIX
            if routing_weights.shape[1] > 1:
                # Check for valid variance before computing correlation
                vix_std = np.std(vix)
                regime1_std = np.std(routing_weights[:, 1])
                
                if vix_std > 1e-6 and regime1_std > 1e-6:
                    vix_corr = np.corrcoef(routing_weights[:, 1], vix)[0, 1]
                    # Check for NaN (can happen with constant series)
                    if np.isnan(vix_corr):
                        vix_corr = None
        
        # Initialize history keys if needed
        if 'regime_entropy' not in self.history:
            self.history['regime_entropy'] = []
        if 'regime_entropy_normalized' not in self.history:
            self.history['regime_entropy_normalized'] = []
        if 'dominant_regime_pct' not in self.history:
            self.history['dominant_regime_pct'] = []
        if 'expert_stds' not in self.history:
            self.history['expert_stds'] = {}
            for i in range(pl_module.output_layer.num_regimes):
                self.history['expert_stds'][f'expert_{i}'] = []
        if 'expert_means' not in self.history:
            self.history['expert_means'] = {}
            for i in range(pl_module.output_layer.num_regimes):
                self.history['expert_means'][f'expert_{i}'] = []
        if 'vix_correlation' not in self.history:
            self.history['vix_correlation'] = []
        if 'regime_assignments' not in self.history:
            self.history['regime_assignments'] = {}
            for i in range(pl_module.output_layer.num_regimes):
                self.history['regime_assignments'][f'regime_{i}_pct'] = []
        
        # Store metrics
        self.history['regime_entropy'].append(float(routing_entropy))
        self.history['regime_entropy_normalized'].append(float(normalized_entropy))
        self.history['dominant_regime_pct'].append(float(dominant_regime_pct))
        
        for i, (std, mean) in enumerate(zip(expert_stds, expert_means)):
            self.history['expert_stds'][f'expert_{i}'].append(float(std))
            self.history['expert_means'][f'expert_{i}'].append(float(mean))
        
        # Store regime assignment percentages
        for i, count in enumerate(regime_counts):
            pct = (count / len(assigned_regimes) * 100) if len(assigned_regimes) > 0 else 0.0
            self.history['regime_assignments'][f'regime_{i}_pct'].append(float(pct))
        
        if vix_corr is not None:
            self.history['vix_correlation'].append(float(vix_corr))
        else:
            self.history['vix_correlation'].append(None)
        
        # Print summary
        print("  Regime diagnostics:")
        print(f"    Routing entropy: {routing_entropy:.4f} (normalized: {normalized_entropy:.4f})")
        print(f"    Dominant regime: {dominant_regime_pct:.1f}%")
        print(f"    Regime assignments: " + ", ".join(
            f"{i}={regime_counts[i]/len(assigned_regimes)*100:.1f}%" 
            for i in range(len(regime_counts))
        ))
        
        # Routing weight statistics (soft probabilities)
        avg_routing_weights = routing_weights.mean(axis=0)
        print(f"    Avg routing weights: " + ", ".join(
            f"R{i}={avg_routing_weights[i]:.4f}"
            for i in range(len(avg_routing_weights))
        ))
        
        for i, (std, mean) in enumerate(zip(expert_stds, expert_means)):
            print(f"    Expert {i}: std={std:.6f}, mean={mean:.6f}")
            if std < 0.05:
                print(f"      WARNING: Expert {i} collapsed (std < 0.05)")
        if vix_corr is not None:
            print(f"    VIX correlation (regime_1): {vix_corr:.3f}")
        else:
            print(f"    VIX correlation: (VIX not found in batch)")
    
    def _log_expert_weight_divergence(self, pl_module):
        """
        Track if expert weights are diverging (learning different functions).
        
        Monitors weight differences between experts to detect:
        - Experts learning identical functions (no divergence = routing doesn't matter)
        - Healthy specialization (moderate divergence)
        - Extreme divergence (potential training instability)
        
        Metrics logged:
        - weight_diff: L2 norm of weight difference between expert pairs
        - weight_cosine: Cosine similarity (1.0 = identical, 0.0 = orthogonal)
        """
        import torch.nn.functional as F
        
        if not hasattr(pl_module, 'output_layer'):
            return
        if not hasattr(pl_module.output_layer, 'experts'):
            return
        
        experts = pl_module.output_layer.experts
        num_experts = len(experts)
        
        if num_experts < 2:
            return
        
        # Helper to extract first layer weights from Linear or Sequential experts
        def get_first_layer_weight(expert):
            if isinstance(expert, torch.nn.Sequential):
                return expert[0].weight
            return expert.weight
        
        # Initialize history keys if needed
        if 'expert_weight_diff' not in self.history:
            self.history['expert_weight_diff'] = []
        if 'expert_weight_cosine' not in self.history:
            self.history['expert_weight_cosine'] = []
        
        # For >2 experts, track pairwise metrics as dict
        if num_experts > 2:
            if 'expert_weight_diff_pairs' not in self.history:
                self.history['expert_weight_diff_pairs'] = {}
            if 'expert_weight_cosine_pairs' not in self.history:
                self.history['expert_weight_cosine_pairs'] = {}
        
        # Compute metrics for first two experts (backward compatible)
        w0 = get_first_layer_weight(experts[0])
        w1 = get_first_layer_weight(experts[1])
        
        weight_diff = (w0 - w1).norm().item()
        weight_cosine = F.cosine_similarity(
            w0.flatten().unsqueeze(0), 
            w1.flatten().unsqueeze(0)
        ).item()
        
        self.history['expert_weight_diff'].append(weight_diff)
        self.history['expert_weight_cosine'].append(weight_cosine)
        
        print(f"  Expert weight divergence (0 vs 1): diff={weight_diff:.4f}, cosine={weight_cosine:.4f}")
        
        # For 3+ experts, also track all pairwise comparisons
        if num_experts > 2:
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    if i == 0 and j == 1:
                        continue  # Already computed above
                    
                    pair_key = f"{i}_vs_{j}"
                    wi = get_first_layer_weight(experts[i])
                    wj = get_first_layer_weight(experts[j])
                    
                    diff = (wi - wj).norm().item()
                    cos = F.cosine_similarity(
                        wi.flatten().unsqueeze(0),
                        wj.flatten().unsqueeze(0)
                    ).item()
                    
                    if pair_key not in self.history['expert_weight_diff_pairs']:
                        self.history['expert_weight_diff_pairs'][pair_key] = []
                        self.history['expert_weight_cosine_pairs'][pair_key] = []
                    
                    self.history['expert_weight_diff_pairs'][pair_key].append(diff)
                    self.history['expert_weight_cosine_pairs'][pair_key].append(cos)
                    
                    print(f"  Expert weight divergence ({i} vs {j}): diff={diff:.4f}, cosine={cos:.4f}")
        
        # Interpretation guidance
        if weight_cosine > 0.99:
            print(f"    WARNING: Experts nearly identical (cosine > 0.99) - routing may be ineffective")
        elif weight_cosine < 0.5:
            print(f"    INFO: Good expert divergence (cosine < 0.5) - experts learning different functions")
            
    def _save_history(self, epoch):
        """Save monitoring history to disk (overwrites each epoch)."""
        save_path = self.log_dir / 'collapse_monitor_latest.json'
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        print(f"  Saved to: {save_path}")