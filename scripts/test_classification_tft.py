"""Quick smoke test for ClassificationTFT."""

import sys
import torch
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

# Test 1: Import
print("Test 1: Import...")
try:
    from src.classification_tft import ClassificationTFT, ClassificationHead, generate_labels
    print("  OK: Import successful")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 2: ClassificationHead standalone
print("\nTest 2: ClassificationHead standalone...")
try:
    head = ClassificationHead(hidden_size=16, num_classes=2, dropout=0.1)
    x = torch.randn(32, 16)  # [batch, hidden]
    out = head(x)
    assert out.shape == (32, 2), f"Expected (32, 2), got {out.shape}"
    print(f"  OK: output shape {out.shape}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 3: generate_labels
print("\nTest 3: generate_labels...")
try:
    returns = torch.tensor([0.01, -0.02, 0.005, -0.001, 0.0])
    
    # direction mode
    labels = generate_labels(returns, mode='direction')
    expected = torch.tensor([1, 0, 1, 0, 0])
    assert torch.equal(labels, expected), f"direction: expected {expected}, got {labels}"
    print(f"  OK: direction mode")
    
    # direction_3class mode
    labels = generate_labels(returns, mode='direction_3class')
    # -0.02 < -0.01 -> 0, 0.01 == threshold -> 1, etc
    print(f"  OK: direction_3class mode, labels={labels.tolist()}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 4: Create minimal dataset and instantiate ClassificationTFT
print("\nTest 4: ClassificationTFT instantiation...")
try:
    # Create minimal synthetic data
    N = 200
    df = pd.DataFrame({
        'time_idx': range(N),
        'group': ['A'] * N,
        'target': np.random.randn(N) * 0.01,
        'feature1': np.random.randn(N),
        'feature2': np.random.randn(N),
    })
    
    # Create TimeSeriesDataSet
    max_encoder_length = 20
    max_prediction_length = 1
    
    training_cutoff = N - max_prediction_length
    
    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx='time_idx',
        target='target',
        group_ids=['group'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_reals=['target', 'feature1', 'feature2'],
    )
    
    # Instantiate with classification disabled (baseline mode)
    model_baseline = ClassificationTFT.from_dataset(
        training,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        classification=False,
    )
    print(f"  OK: Baseline mode (classification=False)")
    
    # Instantiate with classification enabled
    model_clf = ClassificationTFT.from_dataset(
        training,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        classification=True,
        classification_mode='direction',
        num_classes=2,
    )
    print(f"  OK: Classification mode (classification=True)")
    assert hasattr(model_clf, 'classification_head'), "Missing classification_head"
    print(f"  OK: classification_head exists")
    
except Exception as e:
    import traceback
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass
print("\nTest 5: Forward pass...")
try:
    train_dataloader = training.to_dataloader(train=True, batch_size=32)
    batch = next(iter(train_dataloader))
    x, y = batch
    
    # Baseline forward
    out_baseline = model_baseline(x)
    print(type(out_baseline))
    print(dir(out_baseline))
    print(out_baseline.keys() if hasattr(out_baseline, 'keys') else 'no keys method')
    assert 'prediction' in out_baseline, "Missing 'prediction' in baseline output"
    assert 'classification_logits' not in out_baseline, "Unexpected 'classification_logits' in baseline"
    print(f"  OK: Baseline forward, prediction shape {out_baseline['prediction'].shape}")
    
    # Classification forward
    out_clf = model_clf(x)
    assert 'prediction' in out_clf, "Missing 'prediction' in classification output"
    assert 'classification_logits' in out_clf, "Missing 'classification_logits' in classification output"
    print(f"  OK: Classification forward, prediction shape {out_clf['prediction'].shape}")
    print(f"  OK: classification_logits shape {out_clf['classification_logits'].shape}")
    
except Exception as e:
    import traceback
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Training step
print("\nTest 6: Training step...")
try:
    # Baseline training step
    loss_baseline = model_baseline.training_step(batch, 0)
    print(f"  OK: Baseline training_step, loss={loss_baseline.item():.6f}")
    
    # Classification training step
    loss_clf = model_clf.training_step(batch, 0)
    print(f"  OK: Classification training_step, loss={loss_clf.item():.6f}")
    
except Exception as e:
    import traceback
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Validation step
print("\nTest 7: Validation step...")
try:
    val_out = model_clf.validation_step(batch, 0)
    assert 'loss' in val_out, "Missing 'loss' in validation output"
    print(f"  OK: Validation step, loss={val_out['loss'].item():.6f}")
    
except Exception as e:
    import traceback
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("ALL TESTS PASSED")
print("="*50)
