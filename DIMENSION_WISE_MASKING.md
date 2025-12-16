# Dimension-Wise Soft Masking for Matryoshka SVD

## 概述

这个文档介绍了 Matryoshka SVD 的新特性：**维度级软掩码 (Dimension-Wise Soft Masking)**。

## 两种预测模式对比

### 模式 1: 标量 Rank 预测 (原始方法)

**工作原理:**
```
Input x → RankPredictor → scalar rank r ∈ [r_min, r_max]
                                    ↓
                         Soft Gating: gate_k = σ((r - k) / τ)
                                    ↓
                         Gate vector: [g_1, g_2, ..., g_r_max]
```

**特点:**
- ✅ 保持嵌套 Matryoshka 结构 (前 k 个维度最重要)
- ✅ 适合需要严格嵌套结构的场景
- ⚠️ 灵活性受限：所有维度的重要性必须单调递减

**使用场景:**
- 传统的 Matryoshka 训练
- 需要嵌套表示的应用 (可以在任意 rank 截断)
- 与原始 Matryoshka 论文保持一致

### 模式 2: 维度级掩码预测 (新方法) ⭐

**工作原理:**
```
Input x → DimensionWisePredictor → logits ∈ ℝ^r_max
                                        ↓
                          Sigmoid/Gumbel-Softmax
                                        ↓
                          Soft mask: [m_1, m_2, ..., m_r_max]
```

**特点:**
- ✅ 每个维度独立决定是否保留
- ✅ 更强的表达能力：可以学习"维度 5 重要但维度 2 不重要"
- ✅ **完全兼容梯度检查点** (确定性 soft masking)
- ✅ 端到端可微
- ⚠️ 不保证嵌套结构

**使用场景:**
- 需要最大化压缩效率的场景
- 不需要严格嵌套结构的应用
- **训练时启用梯度检查点以节省内存**

## 关键优势：解耦预测与计算

### 问题：梯度检查点冲突

```python
# 传统做法（可能导致形状不匹配）
rank = predictor(x)  # 随机性导致每次不同
z = v_proj(x)[:, :, :rank]  # 物理切片 → 形状改变 ❌

# 前向：z.shape = [batch, seq, 128]
# 重计算：z.shape = [batch, seq, 256]  → 梯度检查点错误！
```

### 解决方案：Soft Masking

```python
# 新方法（形状始终不变）
mask = predictor(x, deterministic=True)  # [batch, seq, r_max]
z = v_proj(x)  # [batch, seq, r_max]
z_masked = z * mask  # 元素级掩码，形状不变 ✅

# 前向：z_masked.shape = [batch, seq, r_max]
# 重计算：z_masked.shape = [batch, seq, r_max]  → 完美兼容！
```

**核心思想：**
- 保留完整的 U, V 矩阵 (维度 r_max)
- 使用 0/1 掩码向量来"软选择"哪些维度激活
- 所有张量形状固定，梯度检查点正常工作

## 使用方法

### 基础使用

```python
from modules.matryoshka_svd_layer import MatryoshkaSVDLayer

# 方法 1: 标量 rank 预测 (原始)
layer_rank = MatryoshkaSVDLayer(
    in_features=4096,
    out_features=4096,
    r_max=256,
    r_min=64,
    use_rank_predictor=True,
    predictor_mode='rank',  # 标量 rank
    gating_tau=0.1
)

# 方法 2: 维度级掩码 (新方法)
layer_dimwise = MatryoshkaSVDLayer(
    in_features=4096,
    out_features=4096,
    r_max=256,
    r_min=64,
    use_rank_predictor=True,
    predictor_mode='dimension_wise',  # 维度级掩码
    use_gumbel=False,  # 确定性 sigmoid (推荐用于梯度检查点)
    gating_tau=0.1
)

# 前向传播
x = torch.randn(2, 128, 4096)
output = layer_dimwise(x)  # 形状: [2, 128, 4096]
```

### 使用 Gumbel-Softmax (可选)

```python
# 启用 Gumbel-Softmax 以获得更稀疏的掩码
layer_gumbel = MatryoshkaSVDLayer(
    in_features=4096,
    out_features=4096,
    r_max=256,
    use_rank_predictor=True,
    predictor_mode='dimension_wise',
    use_gumbel=True,  # 启用 Gumbel-Softmax
    gumbel_tau=0.5,   # 温度 (越低越稀疏)
    gumbel_hard=False  # False=soft, True=hard (straight-through)
)
```

**注意:**
- 训练时会自动使用确定性模式 (`deterministic=True`)
- 评估时可以启用 Gumbel 采样 (如果 `use_gumbel=True`)
- 这确保了梯度检查点兼容性

### 在训练脚本中使用

更新 `train_matryoshka_from_svdllm.py` 中的 `create_matryoshka_layer_from_svd` 函数：

```python
def create_matryoshka_layer_from_svd(u_proj, v_proj, r_max, r_min,
                                     predictor_mode='dimension_wise'):
    """
    Create MatryoshkaSVDLayer from SVD-LLM projections.

    Args:
        predictor_mode: 'rank' or 'dimension_wise'
    """
    # ... 现有代码 ...

    matryoshka = MatryoshkaSVDLayer(
        in_features=in_features,
        out_features=out_features,
        r_max=effective_r_max,
        r_min=r_min,
        use_rank_predictor=True,
        predictor_mode=predictor_mode,  # 新参数！
        gating_tau=0.1,
        bias=False
    )

    # ... 权重复制 ...
    return matryoshka
```

## 性能对比

### 内存占用

| 组件 | 形状 | 标量 Rank | 维度级掩码 |
|------|------|-----------|------------|
| Predictor | - | 很小 | 稍大 (多 r_max 参数) |
| 前向激活 | [B, S, R] | 固定 | 固定 |
| 梯度检查点 | - | ✅ (需要 fixed rank) | ✅ (原生支持) |

### 表达能力

```python
# 标量 Rank 模式
# 只能学习: dim_1 > dim_2 > dim_3 > ... > dim_r_max
# 例如: rank=5 → 保留前 5 个维度

# 维度级掩码模式
# 可以学习: 任意维度组合
# 例如: [1, 0, 1, 0, 1] → 保留 dim_1, dim_3, dim_5
#       实现"跳跃式"重要性分布
```

## 训练建议

### 1. 使用确定性模式训练 (推荐)

```python
# 在 train_matryoshka_from_svdllm.py 中
training_args = TrainingArguments(
    # ...
    gradient_checkpointing=True,  # 启用梯度检查点
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# MatryoshkaSVDLayer 会自动在训练时使用确定性模式
# (self.training=True → deterministic=True)
```

### 2. 多尺度训练仍然有效

```python
# MatryoshkaTrainer.compute_loss() 仍然可以使用 fixed rank
# 这会覆盖 predictor，确保梯度检查点兼容

if use_multiscale:
    for rank in sampled_ranks:
        set_model_rank(model, rank)  # 强制所有层使用固定 rank
        outputs = model(**inputs)
        # ...
```

### 3. 监控有效 Rank

```python
# 在 dimension_wise 模式下，有效 rank = sum(mask)
effective_rank = layer.get_effective_rank()
print(f"Average effective rank: {effective_rank:.1f} / {r_max}")
```

## 理论基础

### Gumbel-Softmax 梯度

```
∂L/∂logits = ∂L/∂mask * ∂mask/∂logits

其中:
mask_i = exp((logit_i + g_i) / τ) / Σ_j exp((logit_j + g_j) / τ)
g_i ~ Gumbel(0, 1)
```

- `τ → 0`: 接近 argmax (one-hot)
- `τ → ∞`: 接近 uniform (平滑)
- `hard=True`: Straight-through estimator (前向 one-hot, 反向 softmax)

### 与 L0 正则化的关系

Dimension-wise masking 类似于 L0 正则化，但使用可微的软掩码：

```python
# L0 正则化 (不可微)
mask = (importance > threshold).float()  # 硬阈值 ❌

# Soft masking (可微)
mask = sigmoid(importance)  # 软阈值 ✅

# Gumbel-Softmax (可微 + 稀疏)
mask = gumbel_softmax(importance, tau=0.1)  # 可微稀疏采样 ✅
```

## 总结

| 特性 | 标量 Rank | 维度级掩码 |
|------|-----------|-----------|
| 嵌套结构 | ✅ 保证 | ⚠️ 不保证 |
| 表达能力 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 梯度检查点兼容 | ⚠️ 需要 fixed rank | ✅ 原生支持 |
| 端到端可微 | ✅ | ✅ |
| 稀疏性控制 | 通过 rank | 通过 Gumbel-τ |
| 推荐场景 | Matryoshka 应用 | 内存受限训练 |

**推荐配置:**
- **内存充足**: 使用 `predictor_mode='rank'` + fixed rank 训练
- **内存紧张**: 使用 `predictor_mode='dimension_wise'` + 梯度检查点
- **需要稀疏性**: 添加 `use_gumbel=True, gumbel_tau=0.5`

## 参考文献

1. Matryoshka Representation Learning (Kusupati et al., 2022)
2. Gumbel-Softmax (Jang et al., 2016; Maddison et al., 2016)
3. Gradient Checkpointing (Chen et al., 2016)
4. SVD-LLM (Wang et al., 2024)
