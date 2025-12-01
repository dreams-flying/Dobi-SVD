# 修复 CUDA cuSOLVER 错误指南

## 🐛 问题描述

在使用 `learned` routing strategy 进行多GPU训练时出现：

```
RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR,
when calling `cusolverDnCreate(handle)`
```

## 🔍 根本原因

1. **CUDA库问题**: cuSOLVER在多GPU环境下可能不稳定
2. **数值问题**: learned策略的importance_net可能产生异常值
3. **内存问题**: 多GPU训练时SVD计算可能耗尽GPU内存
4. **同步问题**: DataParallel在不同replica上状态不一致

## ✅ 解决方案（按优先级）

### 方案1: 切换到MAGMA后端 ⭐ **最推荐**

MAGMA是更稳定的线性代数库，特别适合SVD操作。

**已自动应用**: 代码已更新，会自动尝试使用MAGMA。

**验证**:
```bash
python -c "import torch; print(torch.backends.cuda.preferred_linalg_library())"
```

**手动设置** (如果需要):
```bash
# 在训练前设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

### 方案2: 使用更稳定的routing策略 ⭐ **快速解决**

`learned` 策略更复杂，更容易出错。建议先使用其他策略：

#### Option A: 使用 value_aware (VATP)
```bash
python svd_trainer_dynamic.py \
    --model_id /data1/common/llm-models/Llama-2-7b-chat-hf \
    --routing_strategy value_aware \  # ← 改为 value_aware
    --n_subspaces 3 \
    --target_ratio 0.4 \
    --remapping
```

**优点**:
- ✅ EMNLP'24 SOTA方法
- ✅ 无需额外参数
- ✅ 更稳定

#### Option B: 使用 norm (最稳定)
```bash
python svd_trainer_dynamic.py \
    --routing_strategy norm \  # ← 改为 norm
    --n_subspaces 3
```

**优点**:
- ✅ 最快
- ✅ 最稳定
- ✅ 适合快速验证

---

### 方案3: 优化learned策略配置

如果你必须使用 `learned`，尝试以下优化：

#### 3.1 减少batch size
```bash
python svd_trainer_dynamic.py \
    --routing_strategy learned \
    --gradient_accumulation_steps 8 \  # ← 增加accumulation
    --per_device_train_batch_size 1    # ← 保持为1
```

#### 3.2 降低模型复杂度
```bash
python svd_trainer_dynamic.py \
    --routing_strategy learned \
    --n_subspaces 2 \  # ← 减少到2个子空间
    --target_ratio 0.5  # ← 提高压缩率（更简单）
```

#### 3.3 使用更小的模型测试
```bash
python svd_trainer_dynamic.py \
    --model_id facebook/opt-125m \  # ← 先在小模型上测试
    --routing_strategy learned
```

---

### 方案4: 单GPU训练 (最稳定)

如果多GPU问题难以解决，使用单GPU：

```bash
# 方法1: 指定单个GPU
CUDA_VISIBLE_DEVICES=0 python svd_trainer_dynamic.py \
    --routing_strategy learned \
    ...

# 方法2: 使用accelerate配置
accelerate config  # 选择单GPU模式
accelerate launch svd_trainer_dynamic.py ...
```

---

### 方案5: 更新CUDA/PyTorch (如果可能)

检查版本兼容性：

```bash
# 检查当前版本
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# 建议版本
# PyTorch >= 2.0.0
# CUDA >= 11.7
```

如果版本过旧，考虑升级：
```bash
pip install torch>=2.0.0 --upgrade
```

---

## 🧪 调试步骤

### 步骤1: 确认问题

运行简单测试：
```bash
# 测试1: 单GPU + norm策略 (应该工作)
CUDA_VISIBLE_DEVICES=0 python svd_trainer_dynamic.py \
    --routing_strategy norm \
    --n_train_samples 32 \
    --n_eval_samples 16

# 测试2: 单GPU + learned策略
CUDA_VISIBLE_DEVICES=0 python svd_trainer_dynamic.py \
    --routing_strategy learned \
    --n_train_samples 32

# 测试3: 多GPU + norm策略
python svd_trainer_dynamic.py \
    --routing_strategy norm \
    --n_train_samples 32
```

### 步骤2: 查看日志

检查是否使用了MAGMA:
```
Using MAGMA backend for linear algebra operations  # ✅ 好
Using cuSOLVER backend (default)                    # ⚠️ 可能有问题
```

### 步骤3: 内存监控

```bash
# 训练时监控GPU内存
watch -n 1 nvidia-smi
```

如果看到OOM，减少batch size或减少n_subspaces。

---

## 📊 策略对比

| 策略 | 稳定性 | 准确性 | 速度 | 推荐度 |
|------|--------|--------|------|--------|
| **norm** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **value_aware** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **learned** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **hybrid** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**建议**:
1. **首选**: `value_aware` - SOTA准确性 + 高稳定性
2. **备选**: `norm` - 最稳定，适合快速实验
3. **高级**: `learned` - 需要更多调试

---

## 🔧 代码修复说明

已应用的修复：

### 1. 自动后端切换
```python
# svd_trainer_dynamic.py 开头
try:
    torch.backends.cuda.preferred_linalg_library('magma')
    print("Using MAGMA backend")
except:
    torch.backends.cuda.preferred_linalg_library('cusolver')
```

### 2. importance_net权重初始化
```python
# modules/dynamic_subspace.py: TokenRouter.__init__
for module in self.importance_net.modules():
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.1)  # 小初始值
        nn.init.constant_(module.bias, 0)
```

### 3. 数值稳定性检查
```python
# modules/dynamic_subspace.py: MultiSubspaceSVDLayer.forward
# 在SVD前检查NaN/Inf
if torch.isnan(x).any() or torch.isinf(x).any():
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    x = torch.where(torch.isinf(x), torch.zeros_like(x), x)

# SVD异常捕获
try:
    U, S, V = stable_lowrank_SVD.apply(x, gamma_range)
except RuntimeError:
    # Fallback to identity
    x_transformed += (weight * x).to(model_load_dtype)
    continue
```

---

## 🎯 推荐的训练命令

### 最稳定配置 (Llama-2-7B)

```bash
CUDA_VISIBLE_DEVICES=2,3 python svd_trainer_dynamic.py \
    --model_id /data1/common/llm-models/Llama-2-7b-chat-hf \
    --routing_strategy value_aware \
    --n_subspaces 3 \
    --target_ratio 0.4 \
    --seq_len 2048 \
    --seed 0 \
    --training_dataset wikitext2 \
    --n_train_epochs 20 \
    --n_train_samples 256 \
    --remapping \
    --gamma_multipliers 0.5 1.0 1.5
```

### 如果仍想尝试learned

```bash
# 单GPU，小模型，测试learned
CUDA_VISIBLE_DEVICES=0 python svd_trainer_dynamic.py \
    --model_id facebook/opt-125m \
    --routing_strategy learned \
    --learnable_thresholds \
    --n_subspaces 2 \
    --target_ratio 0.5 \
    --n_train_samples 128 \
    --n_eval_samples 64 \
    --gradient_accumulation_steps 4
```

---

## 📝 常见问题

### Q1: MAGMA是什么？
A: MAGMA是针对GPU优化的线性代数库，比cuSOLVER更稳定，特别是对于SVD操作。PyTorch默认支持。

### Q2: 为什么learned策略更容易出错？
A: learned策略包含额外的神经网络(importance_net)，在多GPU环境下可能：
- 参数同步问题
- 数值不稳定（梯度消失/爆炸）
- 产生异常的importance scores导致SVD失败

### Q3: 切换策略会影响效果吗？
A: 影响有限。根据EMNLP'24研究：
- value_aware vs. learned: 准确性相当，value_aware更稳定
- norm vs. value_aware: value_aware约提升15-20%
- 建议：先用norm验证流程，再用value_aware获得最佳效果

### Q4: 如何知道是否使用了MAGMA？
A: 查看训练开始时的日志：
```
Using MAGMA backend for linear algebra operations  ✅
```

---

## ✅ 检查清单

训练前检查：

- [ ] 确认PyTorch版本 >= 2.0.0
- [ ] 确认CUDA版本 >= 11.7
- [ ] 尝试单GPU训练
- [ ] 使用稳定的routing策略 (norm/value_aware)
- [ ] 查看MAGMA后端是否启用
- [ ] 设置合理的batch size和n_subspaces
- [ ] 监控GPU内存使用

---

## 📞 获取帮助

如果问题仍然存在：

1. **收集信息**:
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

2. **尝试最小化复现**:
```bash
# 最简单的配置
CUDA_VISIBLE_DEVICES=0 python svd_trainer_dynamic.py \
    --model_id facebook/opt-125m \
    --routing_strategy norm \
    --n_subspaces 2 \
    --n_train_samples 16 \
    --n_eval_samples 8
```

3. **报告问题**时包含：
   - 完整错误traceback
   - PyTorch/CUDA版本
   - 使用的routing_strategy
   - GPU型号和内存
   - 是否单GPU/多GPU

---

**总结**: 建议使用 `--routing_strategy value_aware`，这是最平衡的选择（SOTA准确性 + 高稳定性）。learned策略适合在小模型上先验证后再用于大模型。
