"""
PyTorch Lightning callbacks for TFT training.

Note: CollapseMonitor is in collapse_monitor.py due to its size/complexity.
AntiCollapse penalty logging is integrated into CollapseMonitor for consistency.
"""

import torch
import pytorch_lightning as pl


class EpochSummaryCallback(pl.Callback):
    """Print training summary after each epoch."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print training loss at end of epoch."""
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', metrics.get('train_loss_epoch'))
        
        if train_loss is not None:
            print(f"Epoch {trainer.current_epoch}: train_loss={train_loss}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Print validation loss at end of epoch."""
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss')
        
        if val_loss is not None:
            print(f"Epoch {trainer.current_epoch}: val_loss={val_loss}")
        else:
            print(f"Epoch {trainer.current_epoch}: val_loss=N/A")

class DistributionLossLogger(pl.Callback):
    """
    Log distribution penalty statistics during training.
    
    Works with DistributionPenaltyWrapper to log prediction statistics.
    """
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print distribution statistics after each training epoch."""
        # Check if using wrapper
        if not hasattr(pl_module, 'is_enabled'):
            return
        
        if not pl_module.is_enabled():
            return
        
        # Print to console for monitoring
        if pl_module.last_pred_mean is not None:
            print(f"  [DistLoss] pred_mean={pl_module.last_pred_mean:.6f} "
                  f"(target={pl_module.target_mean:.6f}), "
                  f"pred_std={pl_module.last_pred_std:.6f} "
                  f"(target={pl_module.target_std:.6f})")

class GradientMonitorCallback(pl.Callback):
    """Monitor gradient norms for regime output diagnostics."""
    
    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def on_after_backward(self, trainer, pl_module):
        self.step_count += 1
        if self.step_count % self.log_every_n_steps != 0:
            return
        
        grad_norms = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        
        # Log output layer gradients specifically
        output_grads = {k: v for k, v in grad_norms.items() if 'output_layer' in k}
        if output_grads:
            expert_grads = [v for k, v in output_grads.items() if 'experts' in k]
            router_grads = [v for k, v in output_grads.items() if 'router' in k]
            
            if expert_grads:
                trainer.logger.log_metrics({
                    'grad/expert_mean': sum(expert_grads) / len(expert_grads),
                    'grad/expert_max': max(expert_grads),
                }, step=trainer.global_step)
            
            if router_grads:
                trainer.logger.log_metrics({
                    'grad/router_mean': sum(router_grads) / len(router_grads),
                }, step=trainer.global_step)
