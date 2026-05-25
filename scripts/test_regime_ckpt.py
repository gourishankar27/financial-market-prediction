# test_integration.py - run from project root
import torch
from pytorch_forecasting import TemporalFusionTransformer

from src.regime_attention import replace_attention_module
from train.regime_attention_training import patch_forward_for_regime

# Load your best baseline checkpoint
ckpt_path = 'experiments/00_baseline_exploration/sweep2_h16_drop_0.25/checkpoints/last.ckpt'
model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

# Apply regime attention
model = replace_attention_module(model, regime_mode='vix_threshold', vix_threshold=25.0)
model = patch_forward_for_regime(model, vix_feature_name='VIX')

print("Model modified successfully")
print(f"Attention module: {model.multihead_attn}")
