# 评测函数动态秩预测兼容性分析

## 结论：✅ 完全兼容

评测函数 `evaluate_matryoshka_svdllm.py` **完全支持动态秩预测**，测试验证了以下功能：

---

## 工作原理

### 1. 自适应 Rank 模式

```python
# evaluate_matryoshka_svdllm.py, line 204-205
if eval_rank == 'adaptive':
    set_model_rank(model, None)  # 启用动态预测
```

**关键函数**:
```python
def set_model_rank(model: nn.Module, rank: Optional[int]):
    """
    Set fixed rank for all MatryoshkaSVDLayers in model.

    Args:
        rank: Fixed rank to use (None for adaptive/dynamic)
    """
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            module.set_fixed_rank(rank)  # None = 启用动态预测
```

**工作流程**:
1. `set_fixed_rank(None)` → `MatryoshkaSVDLayer.fixed_rank = None`
2. 前向传播时检查 `if self.use_rank_predictor and self.fixed_rank is None`
3. 调用 `rank_predictor(x)` 预测每个 token 的 rank
4. 生成动态掩码（训练时软掩码，推理时硬掩码）

### 2. Rank 追踪

```python
# evaluate_matryoshka_svdllm.py, line 251-254
if eval_rank == 'adaptive':
    avg_rank = get_average_rank(model)  # 获取当前样本的平均 rank
    if avg_rank is not None:
        ranks_collected.append(avg_rank)  # 收集所有样本的 rank
```

**get_average_rank() 实现**:
```python
def get_average_rank(model: nn.Module) -> float:
    ranks = []
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            rank = module.get_effective_rank()  # 获取预测的 rank
            if rank is not None:
                ranks.append(rank)
    return np.mean(ranks) if ranks else None
```

---

## 测试验证

### 测试 1: 自适应 Rank 预测

```
[Test 1: Rank Predictor Mode (Matryoshka)]
    Sample 0: predicted rank = 84.49
    Sample 1: predicted rank = 83.62
    Sample 2: predicted rank = 84.67
    Sample 3: predicted rank = 84.18
    Sample 4: predicted rank = 84.93

    Statistics:
      Average rank: 84.38
      Std dev:      0.45  ✅ 动态变化
```

**结论**: 不同输入样本预测不同的 rank，动态预测正常工作。

### 测试 2: 固定 Rank 强制

```
[Test 2: Fixed Rank Evaluation (rank=64)]
    All samples: [64, 64, 64, 64, 64]
    ✅ Fixed rank correctly enforced
```

**结论**: 固定 rank 模式正确工作，可以对比不同 rank 的性能。

### 测试 3: 多 Rank 评测

```
[Test 3: Multi-Rank Evaluation]
  adaptive       : avg rank =  42.17  ✅ 动态预测
  r_max=64       : avg rank =  64.00  ✅ 固定为 r_max
  r_mid=40       : avg rank =  40.00  ✅ 固定为 r_mid
  r_min=16       : avg rank =  16.00  ✅ 固定为 r_min
```

**结论**: `--multi_rank_eval` 正确支持自适应和固定 rank 的对比评测。

---

## 两种预测模式的兼容性

### 模式 1: Rank Predictor (Matryoshka 嵌套)

```python
predictor_mode='rank'
hard_inference=True
```

**评测时行为**:
1. **预测**: RankPredictor 输出标量 rank ∈ [r_min, r_max]
2. **掩码**:
   - 训练时: `gates = sigmoid((rank - k) / tau)` (软掩码)
   - 推理时: `gates = (k < rank).float()` (硬掩码，二值化)
3. **报告**: `get_effective_rank()` 返回预测的 rank（在硬掩码之前）

**特点**:
- ✅ 完美兼容动态预测
- ✅ 推理时使用硬掩码（模拟真实部署）
- ✅ 每个样本独立预测不同 rank

### 模式 2: Dimension-Wise (稀疏选择)

```python
predictor_mode='dimension_wise'
use_gumbel=False
```

**评测时行为**:
1. **预测**: DimensionWisePredictor 输出每个维度的重要性
2. **掩码**:
   - 训练时: `mask = sigmoid(logits)` (deterministic=True)
   - 推理时: `mask = sigmoid(logits)` 或 Gumbel-Softmax
3. **报告**: `get_effective_rank()` 返回 `mask.sum(dim=-1).mean()`

**特点**:
- ✅ 完全兼容动态预测
- ✅ 稀疏维度选择（非嵌套）
- ✅ 有效 rank = 激活维度的总数

---

## 硬推理 vs 软推理

### Hard Inference (推荐用于 Matryoshka)

```python
hard_inference=True  # 默认
```

**推理时行为**:
```python
# compute_soft_gating(), line 305-309
if self.hard_inference and self.predictor_mode == 'rank':
    gates = (diff > 0).float()  # 完全二值化
    # gates = [1, 1, ..., 1, 0, 0, ..., 0]
```

**优势**:
- ✅ 完美 Matryoshka 嵌套结构
- ✅ 可以物理切片矩阵 `W[:rank, :]`
- ✅ 真正的推理加速
- ✅ 模拟真实部署场景

**rank 追踪**:
```python
# 即使使用硬掩码，get_effective_rank() 仍然返回预测的 rank
# 因为它在 forward 中缓存了 rank.mean().item()（硬掩码之前）
self._last_avg_rank = rank.mean().item()
```

### Soft Inference

```python
hard_inference=False
```

**推理时行为**:
```python
gates = torch.sigmoid(diff / self.gating_tau)  # 软掩码
# gates = [0.99, 0.98, ..., 0.5, ..., 0.02, 0.01]
```

**特点**:
- ✅ 更平滑的 rank 统计
- ⚠️ 无法物理切片（不能真正加速）
- ⚠️ 不适合部署

---

## 典型评测场景

### 场景 1: 验证动态压缩能力

**目标**: 看模型是否学会了根据输入调整 rank

```bash
python evaluate_matryoshka_svdllm.py \
    --checkpoint ./output/final \
    --eval_rank adaptive \
    --n_eval_samples 512
```

**期望结果**:
```
Perplexity:  12.5
Avg rank:    127.3  # 介于 r_min 和 r_max 之间
```

**分析**:
- 如果 avg_rank ≈ r_max: 模型总是使用最大 rank（压缩不够激进）
- 如果 avg_rank ≈ r_min: 模型总是使用最小 rank（可能牺牲了性能）
- 如果 avg_rank 在中间: 模型学会了动态平衡（✅ 理想状态）

### 场景 2: 对比不同压缩率

**目标**: 看固定不同 rank 时的性能 vs 压缩 trade-off

```bash
python evaluate_matryoshka_svdllm.py \
    --checkpoint ./output/final \
    --multi_rank_eval \
    --save_results
```

**期望结果**:
```
adaptive:    PPL=12.5, avg_rank=127.3
r_max=256:   PPL=11.8, avg_rank=256.0
r_mid=128:   PPL=12.4, avg_rank=128.0
r_min=64:    PPL=14.1, avg_rank=64.0
```

**分析**:
- adaptive 的 PPL 应该接近对应 avg_rank 的固定 rank
- 验证 Matryoshka 嵌套结构: PPL(r_min) > PPL(r_mid) > PPL(r_max)

### 场景 3: 部署前性能测试

**目标**: 测试目标 rank 下的实际性能

```bash
# 假设部署时想用 rank=128
python evaluate_matryoshka_svdllm.py \
    --checkpoint ./output/final \
    --eval_rank 128
```

**期望结果**:
```
Perplexity:  12.4
Avg rank:    128.0
Compression: 50% (128/256)
```

---

## 潜在问题 & 解决方案

### 问题 1: Rank 预测方差过大

**现象**:
```
Sample 0: rank = 50
Sample 1: rank = 200
Sample 2: rank = 80
...
Avg rank: 110, Std: 60  # 方差很大
```

**原因**: RankPredictor 训练不充分

**解决**:
1. 增加训练 epochs
2. 调整 `lambda_rank` 正则化
3. 使用更多 multi-scale 训练

### 问题 2: Adaptive 和固定 rank 的 PPL 差异很大

**现象**:
```
adaptive:  PPL=15.0, avg_rank=128
r=128:     PPL=12.0
```

**原因**:
- 硬掩码 vs 软掩码的差异
- RankPredictor 的预测误差

**解决**:
1. 检查 `hard_inference` 设置
2. 分析 rank 分布的方差
3. 可能需要更多训练

### 问题 3: 所有样本的 rank 都相同

**现象**:
```
Sample 0: rank = 128.0
Sample 1: rank = 128.0
...
Std dev: 0.0  # 完全没有变化
```

**原因**: RankPredictor 退化成了常数预测

**检查**:
1. `use_rank_predictor=True` 是否设置
2. RankPredictor 的权重是否被正确加载
3. 训练时是否用了 `freeze_uv` (只训练 predictor)

---

## 使用建议

### 推荐配置（论文实验）

```bash
# 1. 完整多 rank 评测
python evaluate_matryoshka_svdllm.py \
    --checkpoint ./output/final \
    --eval_dataset wikitext2 \
    --multi_rank_eval \
    --n_eval_samples 512 \
    --save_results

# 2. 在多个数据集上评测
for dataset in wikitext2 c4 ptb; do
    python evaluate_matryoshka_svdllm.py \
        --checkpoint ./output/final \
        --eval_dataset $dataset \
        --multi_rank_eval \
        --save_results
done
```

### 关键指标

| 指标 | 说明 | 期望值 |
|------|------|--------|
| **Perplexity** | 模型质量 | 越低越好 |
| **Avg Rank (adaptive)** | 动态压缩率 | r_min < avg < r_max |
| **Rank Std Dev** | 动态调整能力 | > 0（有变化）|
| **PPL vs Rank 曲线** | 性能权衡 | 单调递减 |

---

## 总结

### ✅ 确认兼容

1. **Adaptive rank 评测**: 完全支持，`--eval_rank adaptive`
2. **Fixed rank 评测**: 完全支持，`--eval_rank 128`
3. **Multi-rank 评测**: 完全支持，`--multi_rank_eval`
4. **Rank 追踪**: 正确收集每个样本的预测 rank
5. **两种模式**: rank (Matryoshka) 和 dimension_wise (Sparse) 都兼容
6. **硬/软推理**: 两种模式都支持

### 🎯 核心机制

```python
# 启用动态预测
set_model_rank(model, None)

# 前向传播
for sample in dataset:
    output = model(sample)
    rank = get_average_rank(model)  # 获取预测的 rank
    ranks_collected.append(rank)

# 统计
avg_rank = np.mean(ranks_collected)
```

### 📊 测试结果

所有测试通过，包括：
- ✅ 自适应 rank 预测（rank 在样本间动态变化）
- ✅ 固定 rank 强制（所有样本使用相同 rank）
- ✅ 多 rank 评测工作流
- ✅ 硬推理模式兼容性
- ✅ 两种预测模式兼容性

**评测函数完全兼容动态秩预测！可以放心使用。**
