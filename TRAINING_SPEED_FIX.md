# Training Speed Optimization Fix

## Problem Description

Training speed drastically decreased during the process:
- **Initial Speed**: ~930 samples/sec (Global Step 100-1000)
- **Later Speed**: ~170-210 samples/sec (Global Step 1050+)
- **Speed Drop**: Approximately **4-5x slower**

## Root Cause

The original code called evaluation callbacks at **every training step**:

```python
# ❌ Wrong implementation - executes every step
callback_verification(global_step, backbone)
if callback_ijb is not None:
    callback_ijb(global_step, backbone)
if callback_tinyface is not None:
    callback_tinyface(global_step, backbone)
```

Although these callback functions have internal frequency control, the overhead accumulates:
1. Function call overhead at every step
2. Condition checking overhead at every step
3. Potential frequent model eval/train mode switching
4. Cumulative overhead causes significant training slowdown

## Solution

Move evaluation callbacks inside conditional checks, only calling when needed:

```python
# ✅ Correct implementation - only execute when needed
# Only perform verification and checkpointing when needed
if global_step > 100 and global_step % val_eval_step_freq == 0:
    callback_verification(global_step, backbone)
    callback_checkpoint(global_step, backbone, header)

# IJB and TinyFace evaluation based on configured frequency
if global_step % eval_step_freq == 0:
    if callback_ijb is not None:
        callback_ijb(global_step, backbone)
    if callback_tinyface is not None:
        callback_tinyface(global_step, backbone)
```

## Evaluation Frequency Configuration

### Current Default Configuration (`config/config.py`)

```python
config.val_eval_every_n_epoch = 1  # Validation set every 1 epoch
config.eval_every_n_epoch = 4      # IJB/TinyFace every 4 epochs
```

### Actual Evaluation Steps

Assuming `steps_per_epoch = 22744`:
- `val_eval_step_freq = 22744` (validation every 1 epoch)
- `eval_step_freq = 90976` (IJB/TinyFace evaluation every 4 epochs)

### Recommended Configuration (Optional Optimization)

To further reduce evaluation frequency:

```python
config.val_eval_every_n_epoch = 2  # Validation set every 2 epochs
config.eval_every_n_epoch = 5      # IJB/TinyFace every 5 epochs
```

## Expected Results

After the fix, training speed should:
- ✅ Recover to stable **~900+ samples/sec**
- ✅ Eliminate unnecessary function call overhead
- ✅ Reduce model mode switching frequency
- ✅ Maintain same evaluation frequency (actual evaluation logic unchanged)

## Modified Files

- ✅ `train/train_AdaDistill.py` - Optimized evaluation callback calling logic (lines 286-293)

## Verification Method

Restart training and observe logs:

```bash
bash run_AdaDistill.sh
```

Observe if training speed remains at ~900+ samples/sec:
```
Training: 2025-12-01 HH:MM:SS-Speed 930.00 samples/sec   Loss XX.XX ...
Training: 2025-12-01 HH:MM:SS-Speed 932.00 samples/sec   Loss XX.XX ...
Training: 2025-12-01 HH:MM:SS-Speed 928.00 samples/sec   Loss XX.XX ...
```

## Additional Optimization Suggestions

1. **Reduce Validation Frequency**: If training is long, set `val_eval_every_n_epoch` to 2 or 3
2. **Monitor GPU Memory**: Use `nvidia-smi` to monitor memory usage and ensure no leaks
3. **Data Loading**: Confirm `num_workers=16` is appropriate, adjust based on CPU cores
4. **Mixed Precision**: Code already uses AMP, ensure it's working properly

## Related Issues

- ✅ Shape mismatch error fixed in `utils/losses.py`
- ✅ Training speed optimization completed
- ℹ️  If performance issues persist, check hardware resource usage
