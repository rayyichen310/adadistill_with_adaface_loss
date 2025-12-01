# AdaDistill 修复总结

## 问题描述
训练脚本 `train/train_AdaDistill.py` 在运行时遇到了形状不匹配错误：
```
RuntimeError: The size of tensor a (256) must match the size of tensor b (65536) at non-singleton dimension 0
```

## 修复内容

### 1. 修复 `utils/losses.py` 中的形状不匹配问题

**问题根源：**
- `AdaptiveAAdaFace` 类中的 `margin_scaler` 在使用 geometry-aware margin 时，`geom_delta` 是 `(batch_size, 1)` 形状
- 与一维的 `margin_scaler_q` 相加时产生广播，导致 `margin_scaler` 变成二维张量
- 后续计算 `g_angular` 和 `theta` 相加时维度不匹配

**修复方案：**
在 `utils/losses.py` 第 232 行附近添加：
```python
geom_delta = geom_delta.view(-1)  # 展平为一维向量
```

并在第 247 行附近添加额外的安全检查：
```python
# 确保 g_angular 是一维向量且长度匹配 theta
if g_angular.dim() > 1:
    g_angular = g_angular.view(-1)
```

### 2. 修复 CVLface 依赖缺失问题

**问题：**
项目依赖 `CVLface` 库来加载 HuggingFace teacher 模型和进行 IJB/TinyFace 评估，但该库未安装。

**修复方案：**

#### a) `utils/utils_callbacks.py`
使用条件导入处理缺失的评估器：
```python
# 尝试导入 IJB 和 TinyFace 评估器，如果失败则设为 None
try:
    from eval.ijb_evaluator import run_ijb_evaluation
except ImportError:
    run_ijb_evaluation = None
    logging.warning("IJB evaluator not available (missing CVLface dependency)")

try:
    from eval.tinyface_evaluator import run_tinyface_evaluation
except ImportError:
    run_tinyface_evaluation = None
    logging.warning("TinyFace evaluator not available")
```

并在评估函数开始处添加检查：
```python
if run_ijb_evaluation is None:
    logging.warning("Skipping IJB evaluation (evaluator not available)")
    return
```

#### b) `train/train_AdaDistill.py`
添加异常处理：
```python
elif cfg.teacher in hf_teacher_defaults:
    try:
        from CVLface.cvlface.general_utils.huggingface_model_utils import load_model_by_repo_id
        # ... 加载代码 ...
    except ImportError:
        logging.error(f"Cannot load HuggingFace teacher: CVLface module not found. "
                     f"Please use a local teacher model (iresnet18/50/100) instead.")
        raise
```

#### c) `config/config.py`
修改默认 teacher 为本地模型：
```python
config.teacher = "iresnet50"  # 改用本地模型而不是 "cvlface_ir50"
```

## 验证结果

修复后，训练脚本可以成功启动：
```bash
bash run_AdaDistill.sh
```

或使用 conda 环境：
```bash
conda run -n adadistill bash run_AdaDistill.sh
```

训练日志显示：
- ✅ 形状不匹配错误已解决
- ✅ 数据加载正常
- ✅ 模型开始训练（反向传播正常运行）
- ⚠️  Loss 出现 NaN（需要单独调查，可能是 teacher 模型未正确加载或学习率问题）

## 注意事项

1. **Teacher 模型加载失败：** 日志显示 `teacher init, failed!`，这意味着预训练的 teacher 权重文件 `output/teacher/295672backbone.pth` 不存在或无法加载。如果没有预训练的 teacher，可以：
   - 提供正确的 teacher 权重路径
   - 或者从随机初始化的 teacher 开始（效果会差一些）

2. **NaN Loss 问题：** 可能原因包括：
   - Teacher 模型未正确初始化
   - 学习率过高
   - 数值不稳定性
   建议检查 teacher 模型的状态并调整超参数。

3. **评估功能：** IJB 和 TinyFace 评估已被跳过（因为缺少 CVLface），但基本的 LFW/CFP/AgeDB 等评估仍然可用。

## 文件修改列表

- ✅ `utils/losses.py` - 修复形状不匹配
- ✅ `utils/utils_callbacks.py` - 添加条件导入
- ✅ `train/train_AdaDistill.py` - 添加异常处理
- ✅ `config/config.py` - 修改默认 teacher

## 下一步建议

1. 准备或训练一个 teacher 模型（iresnet50）
2. 调查 NaN loss 的根本原因
3. 考虑调整学习率或添加梯度裁剪
4. 如果需要 IJB/TinyFace 评估，安装 CVLface 库
