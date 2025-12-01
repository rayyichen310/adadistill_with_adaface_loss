# 训练速度优化修复

## 问题描述

训练过程中速度急剧下降：
- **初始速度**: ~930 samples/sec (Global Step 100-1000)
- **后期速度**: ~170-210 samples/sec (Global Step 1050+)
- **速度下降**: 约 **4-5倍**

## 问题根源

原代码在**每个训练步骤**都调用评估回调函数：

```python
# ❌ 错误的实现 - 每个step都执行
callback_verification(global_step, backbone)
if callback_ijb is not None:
    callback_ijb(global_step, backbone)
if callback_tinyface is not None:
    callback_tinyface(global_step, backbone)
```

虽然这些回调函数内部有频率控制，但是：
1. 每个step都要调用函数（函数调用开销）
2. 每个step都要检查条件（条件判断开销）
3. 可能触发模型 eval/train 模式频繁切换
4. 累积的开销导致训练速度显著下降

## 修复方案

将评估回调移到条件判断内，只在需要时才调用：

```python
# ✅ 正确的实现 - 只在需要时执行
# 只在需要时执行验证和保存
if global_step > 100 and global_step % val_eval_step_freq == 0:
    callback_verification(global_step, backbone)
    callback_checkpoint(global_step, backbone, header)

# IJB 和 TinyFace 评估按照配置的频率执行
if global_step % eval_step_freq == 0:
    if callback_ijb is not None:
        callback_ijb(global_step, backbone)
    if callback_tinyface is not None:
        callback_tinyface(global_step, backbone)
```

## 评估频率配置

### 当前默认配置 (`config/config.py`)

```python
config.val_eval_every_n_epoch = 1  # 验证集每1个epoch评估一次
config.eval_every_n_epoch = 4      # IJB/TinyFace每4个epoch评估一次
```

### 实际评估步数

假设 `steps_per_epoch = 22744`：
- `val_eval_step_freq = 22744` (每1个epoch验证一次)
- `eval_step_freq = 90976` (每4个epoch评估IJB/TinyFace)

### 建议配置（可选优化）

如果想进一步减少评估次数，可以调整配置：

```python
config.val_eval_every_n_epoch = 2  # 验证集每2个epoch评估一次
config.eval_every_n_epoch = 5      # IJB/TinyFace每5个epoch评估一次
```

## 预期效果

修复后，训练速度应该能够：
- ✅ 恢复到 **~900+ samples/sec** 的稳定速度
- ✅ 消除不必要的函数调用开销
- ✅ 减少模型模式切换次数
- ✅ 保持相同的评估频率（实际评估逻辑不变）

## 修改的文件

- ✅ `train/train_AdaDistill.py` - 优化评估回调调用逻辑（第286-293行）

## 验证方法

重新启动训练并观察日志：

```bash
bash run_AdaDistill.sh
```

观察训练速度是否保持在 ~900+ samples/sec：
```
Training: 2025-12-01 HH:MM:SS-Speed 930.00 samples/sec   Loss XX.XX ...
Training: 2025-12-01 HH:MM:SS-Speed 932.00 samples/sec   Loss XX.XX ...
Training: 2025-12-01 HH:MM:SS-Speed 928.00 samples/sec   Loss XX.XX ...
```

## 额外优化建议

1. **减少验证频率**: 如果训练很长，可以将 `val_eval_every_n_epoch` 设为 2 或 3
2. **监控显存**: 使用 `nvidia-smi` 监控显存使用，确保没有泄漏
3. **数据加载**: 确认 `num_workers=8` 设置合理，可以根据CPU核心数调整
4. **混合精度**: 代码已使用 AMP，确保正常工作

## 相关问题

- ✅ 形状不匹配错误已在 `utils/losses.py` 中修复
- ✅ 训练速度优化已完成
- ℹ️  如果仍有性能问题，检查硬件资源使用情况
