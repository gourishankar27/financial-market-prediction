"""
Classification head extension for Temporal Fusion Transformer.

Adds parallel classification head alongside existing quantile regression.
Diagnostic tool to compare gradient flow between classification and regression heads.

Usage:
    from src.classification_tft import ClassificationTFT
    
    model = ClassificationTFT.from_dataset(
        training,
        classification=True,
        classification_mode='direction',
        classification_weight=1.0,
        regression_weight=1.0,
        **other_tft_kwargs
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Union
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


class ClassificationHead(nn.Module):
    """
    Simple classification head for direction/regime prediction.
    
    Single linear layer from encoder hidden state to class logits.
    Intentionally minimal - tests whether encoder representation is directly decodable.
    
    Parameters
    ----------
    hidden_size : int
        Input dimension (TFT hidden size)
    num_classes : int
        Number of output classes (2 for binary, 3+ for multi-class)
    dropout : float
        Dropout rate before linear layer
    """
    """ 
    def __init__(self, hidden_size: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    """
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1, 
                 head_hidden_size: int = None):
        super().__init__()
        head_hidden_size = head_hidden_size or hidden_size  # Default to same as encoder
        
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, hidden_size] - final encoder representation
            
        Returns
        -------
        torch.Tensor
            Shape [batch, num_classes] - class logits
        """
        #return self.head(x)
        return self.net(x)


def generate_labels(
    returns: torch.Tensor,
    mode: str = 'direction',
    thresholds: Optional[List[float]] = None,
    vix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Generate classification labels from return/VIX data.
    
    Parameters
    ----------
    returns : torch.Tensor
        Return values, any shape (will be flattened if needed)
    mode : str
        Label generation mode:
        - 'direction': binary up/down (2 classes)
        - 'direction_3class': down/neutral/up with thresholds (3 classes)
        - 'regime_volatility': binary low/high VIX (2 classes)
        - 'regime_volatility_3class': low/medium/high VIX (3 classes)
    thresholds : list of float, optional
        Thresholds for multi-class modes. Defaults:
        - direction_3class: [-0.01, 0.01]
        - regime_volatility: [20.0]
        - regime_volatility_3class: [15.0, 25.0]
    vix : torch.Tensor, optional
        VIX values, required for regime modes
        
    Returns
    -------
    torch.Tensor
        Integer class labels, same batch dimension as input
    """
    # Flatten to 1D if needed
    returns_flat = returns.view(-1)
    
    if mode == 'direction':
        return (returns_flat > 0).long()
    
    elif mode == 'direction_3class':
        thresholds = thresholds or [-0.01, 0.01]
        labels = torch.ones_like(returns_flat).long()  # neutral = 1
        labels[returns_flat < thresholds[0]] = 0       # down
        labels[returns_flat > thresholds[1]] = 2       # up
        return labels
    
    elif mode == 'regime_volatility':
        if vix is None:
            raise ValueError("regime_volatility mode requires vix tensor")
        thresholds = thresholds or [20.0]
        vix_flat = vix.view(-1)
        return (vix_flat > thresholds[0]).long()
    
    elif mode == 'regime_volatility_3class':
        if vix is None:
            raise ValueError("regime_volatility_3class mode requires vix tensor")
        thresholds = thresholds or [15.0, 25.0]
        vix_flat = vix.view(-1)
        labels = torch.ones_like(vix_flat).long()      # medium = 1
        labels[vix_flat < thresholds[0]] = 0           # low vol
        labels[vix_flat > thresholds[1]] = 2           # high vol
        return labels
    
    else:
        raise ValueError(f"Unknown label mode: {mode}")


class ClassificationTFT(TemporalFusionTransformer):
    """
    TFT with parallel classification head for diagnostic comparison.
    
    Adds classification head that reads from the same hidden representation
    as the regression output layer. Allows comparing gradient dynamics
    between cross-entropy (classification) and quantile loss (regression).
    
    Parameters
    ----------
    classification : bool
        Enable classification head (default: False for baseline compatibility)
    classification_mode : str
        Label generation mode (see generate_labels)
    classification_weight : float
        Weight for classification loss (beta). Default 1.0.
    regression_weight : float
        Weight for regression loss (alpha). Set to 0 for pure classification.
    num_classes : int
        Number of classification classes
    classification_thresholds : list of float, optional
        Thresholds for multi-class classification modes
    **kwargs
        All standard TFT arguments
    """
    
    def __init__(
        self,
        # Classification config
        classification: bool = False,
        classification_mode: str = 'direction',
        classification_weight: float = 1.0,
        regression_weight: float = 1.0,
        num_classes: int = 2,
        classification_thresholds: Optional[List[float]] = None,
        **kwargs
    ):
        # Remove classification args from kwargs FIRST (for checkpoint loading)
        # These might come through kwargs when loading from checkpoint
        kwargs.pop('classification', None)
        kwargs.pop('classification_mode', None)
        kwargs.pop('classification_weight', None)
        kwargs.pop('regression_weight', None)
        kwargs.pop('num_classes', None)
        kwargs.pop('classification_thresholds', None)
        
        # Store classification config
        self._classification_enabled = classification
        self._classification_mode = classification_mode
        self._classification_weight = classification_weight
        self._regression_weight = regression_weight
        self._num_classes = num_classes
        self._classification_thresholds = classification_thresholds
        
        super().__init__(**kwargs)
        
        # Add classification head if enabled
        if self._classification_enabled:
            self.classification_head = ClassificationHead(
                hidden_size=self.hparams.hidden_size,
                num_classes=num_classes,
                dropout=self.hparams.dropout
            )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional classification output.
        
        Overrides parent to capture hidden representation before output layer.
        """
        # === Begin: copied from parent forward() up to output_layer ===
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)
        max_encoder_length = int(encoder_lengths.max())
        
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Static embedding
        if len(self.static_variables) > 0:
            static_embedding = {
                name: input_vectors[name][:, 0] for name in self.static_variables
            }
            static_embedding, static_variable_selection = (
                self.static_variable_selection(static_embedding)
            )
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            static_variable_selection = torch.zeros(
                (x_cont.size(0), 0), dtype=self.dtype, device=self.device
            )

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        # Encoder variable selection
        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length]
            for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = (
            self.encoder_variable_selection(
                embeddings_varying_encoder,
                static_context_variable_selection[:, :max_encoder_length],
            )
        )

        # Decoder variable selection
        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:]
            for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = (
            self.decoder_variable_selection(
                embeddings_varying_decoder,
                static_context_variable_selection[:, max_encoder_length:],
            )
        )

        # LSTM
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )

        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder,
            (input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False,
        )

        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # Skip connections
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, embeddings_varying_encoder
        )

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, embeddings_varying_decoder
        )

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # Static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output,
            self.expand_static_context(static_context_enrichment, timesteps),
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths
            ),
        )

        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, max_encoder_length:]
        )

        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        
        # === HOOK POINT: output is [batch, decoder_len, hidden_size] ===
        # Store for classification head
        hidden_representation = output
        
        # Regression output (standard TFT path)
        if self.n_targets > 1:
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)

        # Build output 
        network_output = self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )
        
        # Convert to dict for modification
        #output_dict = dict(network_output._asdict())
        
        # Add classification logits if enabled
        if self._classification_enabled:
            # Convert to dict only when we need to add classification
            output_dict = dict(network_output._asdict())
            classification_input = hidden_representation[:, -1, :]
            output_dict['classification_logits'] = self.classification_head(classification_input)
            return output_dict
                
        return output_dict
    
    def _get_classification_labels(self, y, x=None) -> torch.Tensor:
        """Extract classification labels from target tensor and/or features."""
        # y is (target, weight) tuple or just target
        if isinstance(y, (tuple, list)):
            target = y[0]
        else:
            target = y
        
        # target shape: [batch, decoder_len] or [batch, decoder_len, 1]
        # For single-step prediction, use last timestep
        if target.dim() == 3:
            target = target[:, -1, 0]  # [batch]
        elif target.dim() == 2:
            target = target[:, -1]     # [batch]
        
        # Extract VIX for regime modes
        vix = None
        if self._classification_mode.startswith('regime') and x is not None:
            # Find VIX index in x_reals
            if 'VIX' in self.hparams.x_reals:
                vix_idx = self.hparams.x_reals.index('VIX')
                # Use last encoder timestep's VIX value
                # encoder_cont shape: [batch, encoder_len, num_features]
                vix = x['encoder_cont'][:, -1, vix_idx]  # [batch]
            else:
                raise ValueError("VIX not found in x_reals, cannot use regime classification mode")
        
        return generate_labels(
            target,
            mode=self._classification_mode,
            thresholds=self._classification_thresholds,
            vix=vix
        )
    
    def training_step(self, batch, batch_idx):
        """Training step with combined regression + classification loss."""
        x, y = batch
        out = self(x)
        
        # Regression loss (standard TFT)
        y_target = y[0] if isinstance(y, (tuple, list)) else y
        regression_loss = self.loss(out['prediction'], y_target)
        
        if self._classification_enabled:
            # Classification loss
            labels = self._get_classification_labels(y, x)
            classification_logits = out['classification_logits']
            classification_loss = F.cross_entropy(classification_logits, labels)
            
            # Combined loss
            total_loss = (
                self._regression_weight * regression_loss + 
                self._classification_weight * classification_loss
            )
            
            # Logging
            self.log('train_regression_loss', regression_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_classification_loss', classification_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
            
            # Classification accuracy
            preds = classification_logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            self.log('train_classification_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
            
            return total_loss
        else:
            self.log('train_loss', regression_loss, on_step=False, on_epoch=True, prog_bar=True)
            return regression_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - tracks classification metrics separately."""
        if self._classification_enabled:
            # Do our own forward and logging
            x, y = batch
            out = self(x)  # Returns dict
            
            # Regression loss (this is what early stopping monitors)
            y_target = y[0] if isinstance(y, (tuple, list)) else y
            regression_loss = self.loss(out['prediction'], y_target)
            self.log('val_loss', regression_loss, on_step=False, on_epoch=True, prog_bar=True)
            
            # Classification metrics
            labels = self._get_classification_labels(y, x)
            classification_logits = out['classification_logits']
            classification_loss = F.cross_entropy(classification_logits, labels)
            
            self.log('val_classification_loss', classification_loss, on_step=False, on_epoch=True, prog_bar=True)
            
            preds = classification_logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            self.log('val_classification_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
            
            # Return minimal dict - skip parent's interpretation logging
            return {'loss': regression_loss, 'interpretation': {}}
        else:
            # Baseline: use parent's full validation with interpretation
            return super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        """Override to skip interpretation logging when classification enabled."""
        if self._classification_enabled:
            return
        else:
            return super().on_validation_epoch_end()

    def on_train_epoch_end(self):
        """Override to skip interpretation logging when classification enabled."""
        if self._classification_enabled:
            return
        else:
            return super().on_train_epoch_end()

    def get_classification_head_grad_norm(self) -> Optional[float]:
        """Get gradient norm of classification head for diagnostics."""
        if not self._classification_enabled:
            return None
        
        total_norm = 0.0
        for p in self.classification_head.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def get_output_layer_grad_norm(self) -> float:
        """Get gradient norm of regression output layer for diagnostics."""
        total_norm = 0.0
        if isinstance(self.output_layer, nn.ModuleList):
            for layer in self.output_layer:
                for p in layer.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
        else:
            for p in self.output_layer.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
