# 改进的Rank Predictor设计文档

## 🎯 核心改进

### 设计理念：关注点分离

```
旧设计 (V1):
  RankPredictor → 预测 + 门控 (耦合)
                ↓
            混杂的逻辑

新设计 (V2):
  RankPredictor → 只预测连续rank ∈ [r_min, r_max]
                ↓
  build_nested_mask → 只构建嵌套mask
                ↓
            清晰分离
```

## 📊 新设计的优势

### 1. **更清晰的架构**

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `ImprovedRankPredictor` | 预测rank | x: [B, L, D] | rank: [B, L, 1] ∈ [r_min, r_max] |
| `build_nested_mask` | 构建mask | rank, r_max, tau | gates: [B, L, r_max] |
| `MatryoshkaSVDLayerV2` | 前向传播 | x | output |

### 2. **易于实验**

```python
# 训练时：使用soft mask (可微)
gates = build_nested_mask(rank, r_max, tau=1.0, hard=False)

# 推理时：使用hard mask (加速)
gates = build_nested_mask(rank, r_max, tau=1.0, hard=True)

# 实验Gumbel-Softmax：更好的梯度
gates = build_nested_mask_gumbel(rank, r_max, tau=0.5, hard=True, training=True)
```

### 3. **内置Rank正则化**

```python
# 创建正则化损失
rank_reg_loss = RankRegularizationLoss(target_avg_rank=384.0, weight=0.001)

# 在训练循环中
predicted_ranks = rank_predictor(x)
reg_loss = rank_reg_loss(predicted_ranks)

total_loss = lm_loss + reg_loss  # 鼓励使用目标rank
```

### 4. **支持上下文感知预测**

```python
# 简单预测器：per-token MLP
predictor = ImprovedRankPredictor(
    in_features=4096,
    r_max=512,
    r_min=256,
    use_context=False  # 每个token独立预测
)

# 上下文预测器：考虑周围tokens
predictor = ImprovedRankPredictor(
    in_features=4096,
    r_max=512,
    r_min=256,
    use_context=True,       # 使用Conv1D聚合上下文
    context_window=3        # 看前后3个tokens
)
```

## 🔧 使用指南

### 基础使用

```python
from modules.matryoshka_svd_layer_v2 import MatryoshkaSVDLayerV2

# 创建层
layer = MatryoshkaSVDLayerV2(
    in_features=4096,
    out_features=4096,
    r_max=512,
    r_min=256,
    use_rank_predictor=True,
    use_contextual_predictor=True,  # 推荐开启
    use_gumbel=True,                # 推荐开启：更好的梯度
    gating_tau=1.0,
    hard_inference=True
)

# 前向传播
output = layer(x)  # [batch, seq_len, 4096]

# 获取有效rank（用于监控）
avg_rank = layer.get_effective_rank(x)
print(f"Average rank: {avg_rank:.2f}")
```

### 多阶段训练

```python
# Stage 1: 固定高rank训练（容易）
layer.set_fixed_rank(480)
for epoch in range(5):
    # Training loop...
    pass

# Stage 2: 固定中rank训练
layer.set_fixed_rank(384)
for epoch in range(5):
    # Training loop...
    pass

# Stage 3: 动态rank训练（最终目标）
layer.set_fixed_rank(None)  # 启用动态预测
for epoch in range(10):
    # Training loop...
    pass
```

### 添加Rank正则化

```python
from modules.improved_rank_predictor import RankRegularizationLoss

# 创建正则化
rank_reg = RankRegularizationLoss(
    target_avg_rank=384.0,  # 目标平均rank
    weight=0.001            # 正则化权重
)

# 训练循环
for batch in dataloader:
    # Forward
    output = model(batch)
    lm_loss = criterion(output, labels)

    # 计算rank正则化
    predicted_ranks = []
    for layer in model.modules():
        if isinstance(layer, MatryoshkaSVDLayerV2):
            if layer.use_rank_predictor:
                ranks = layer.rank_predictor(batch)
                predicted_ranks.append(ranks)

    if predicted_ranks:
        all_ranks = torch.cat(predicted_ranks, dim=0)
        reg_loss = rank_reg(all_ranks)
    else:
        reg_loss = 0

    # 总损失
    total_loss = lm_loss + reg_loss

    # Backward
    total_loss.backward()
    optimizer.step()
```

### 从Linear层转换

```python
from modules.matryoshka_svd_layer_v2 import create_matryoshka_from_linear

# 假设你有一个标准Linear层
linear_layer = nn.Linear(4096, 4096)

# 转换为Matryoshka层
matryoshka_layer = create_matryoshka_from_linear(
    linear_layer,
    r_max=512,
    r_min=256,
    use_contextual_predictor=True,
    use_gumbel=True
)

# 此时权重已经用SVD初始化
# 可以直接用于训练或推理
```

## 📈 性能优势

### 训练时

| 特性 | V1 | V2 | 优势 |
|------|----|----|------|
| 梯度流 | 一般 | ✅ 好 (Gumbel-Softmax) | 更快收敛 |
| 可控性 | 低 | ✅ 高 (tau, target_rank) | 精确控制 |
| 实验灵活性 | 低 | ✅ 高 (模块化) | 快速迭代 |

### 推理时

```python
# 硬mask + 物理切片加速
layer.eval()  # 自动启用hard_inference

# 理论加速：
# 如果average_rank = 384 (而r_max=512)
# 计算量减少：384/512 = 75%
# 实际加速：~1.3x (考虑overhead)
```

## 🎓 高级特性

### 1. Gumbel-Softmax

```python
# 为什么用Gumbel-Softmax？
# - 更好的梯度流（相比sigmoid）
# - 训练时soft，推理时hard
# - Straight-through estimator支持

layer = MatryoshkaSVDLayerV2(
    ...,
    use_gumbel=True  # 启用Gumbel-Softmax
)
```

### 2. 渐进式温度

```python
# 训练早期：高温度 (soft)
# 训练后期：低温度 (接近hard)

for epoch in range(num_epochs):
    # 渐进降低温度
    tau = max(0.1, 1.0 - epoch / num_epochs)

    for layer in model.modules():
        if isinstance(layer, MatryoshkaSVDLayerV2):
            layer.gating_tau = tau

    # Training...
```

### 3. 自定义Rank分布

```python
# 你可以修改RankRegularizationLoss来鼓励不同的分布

class CustomRankLoss(nn.Module):
    def forward(self, ranks):
        # 鼓励双峰分布：低rank (256) 或高rank (512)
        target_low = 256
        target_high = 512

        dist_to_low = (ranks - target_low).abs()
        dist_to_high = (ranks - target_high).abs()

        # 希望接近其中之一
        loss = torch.min(dist_to_low, dist_to_high).mean()

        return loss
```

## 🔬 理论基础

### 为什么这个设计更好？

**1. 数学清晰性**

```
旧设计：rank → gates (隐式耦合)

新设计：
  rank = f_θ(x)                    # 连续预测
  gates[i] = σ((rank - i) / τ)     # 显式mask构建

这让我们可以独立优化：
  - f_θ: 预测器架构
  - mask函数：sigmoid vs gumbel vs 其他
  - τ: 温度调度
```

**2. 嵌套性质保证**

```python
# Hard mask保证嵌套性：
gates[i] = 1 if i < rank else 0

# 因此：
# gates = [1, 1, 1, ..., 1, 0, 0, ..., 0]
#                    ↑
#                  cutoff at rank

# 这正是Matryoshka的核心：
# 前k个维度包含最重要信息
```

**3. 优化景观**

```
分离设计 → 更平滑的优化：
  - Rank predictor学习：何时可以降低rank
  - U/V学习：如何在给定rank下最优表示
  - 两者解耦 → 更容易优化
```

## 🚀 迁移指南

### 从V1迁移到V2

```python
# 旧代码 (V1)
from modules.matryoshka_svd_layer import MatryoshkaSVDLayer

layer_v1 = MatryoshkaSVDLayer(
    in_features=4096,
    out_features=4096,
    r_max=512,
    r_min=256
)

# 新代码 (V2)
from modules.matryoshka_svd_layer_v2 import MatryoshkaSVDLayerV2

layer_v2 = MatryoshkaSVDLayerV2(
    in_features=4096,
    out_features=4096,
    r_max=512,
    r_min=256,
    use_contextual_predictor=True,  # 新特性！
    use_gumbel=True                 # 新特性！
)

# API兼容：
output_v1 = layer_v1(x)
output_v2 = layer_v2(x)  # 相同的接口
```

### 权重转换

```python
# 如果你有V1训练的checkpoint：
checkpoint_v1 = torch.load("checkpoint_v1.pt")

# 创建V2层
layer_v2 = MatryoshkaSVDLayerV2(...)

# 手动加载V1权重
layer_v2.v_proj.load_state_dict(checkpoint_v1['v_proj'])
layer_v2.u_proj.load_state_dict(checkpoint_v1['u_proj'])

# Rank predictor需要重新训练
# （因为架构不同，但这通常很快）
```

## 💡 最佳实践

### 1. 训练策略

```python
# 推荐三阶段训练：
# Stage 1: 冻结U/V，只训练rank predictor
# Stage 2: 解冻U/V，小学习率联合训练
# Stage 3: 添加rank正则化，精调

# Stage 1
for param in layer.v_proj.parameters():
    param.requires_grad = False
for param in layer.u_proj.parameters():
    param.requires_grad = False
# Train rank predictor...

# Stage 2
for param in layer.parameters():
    param.requires_grad = True

optimizer = AdamW([
    {'params': layer.rank_predictor.parameters(), 'lr': 1e-4},
    {'params': [layer.v_proj.weight, layer.u_proj.weight], 'lr': 1e-5}
])

# Stage 3: 添加rank regularization
```

### 2. 超参数推荐

```python
config = {
    'r_max': 512,                    # 最大rank
    'r_min': 256,                    # 最小rank
    'gating_tau': 1.0,               # 温度（训练时）
    'use_contextual_predictor': True, # 启用上下文
    'context_window': 3,             # 上下文窗口
    'use_gumbel': True,              # 启用Gumbel-Softmax
    'rank_reg_weight': 0.001,        # Rank正则化权重
    'target_avg_rank': 384,          # 目标平均rank
}
```

### 3. 监控指标

```python
# 训练时监控：
# 1. 平均rank
# 2. Rank分布（标准差）
# 3. Soft vs Hard mask差异

def log_rank_statistics(model, x):
    ranks = []
    for layer in model.modules():
        if isinstance(layer, MatryoshkaSVDLayerV2):
            if layer.use_rank_predictor:
                rank = layer.rank_predictor(x)
                ranks.append(rank)

    if ranks:
        all_ranks = torch.cat(ranks, dim=0)
        print(f"Avg rank: {all_ranks.mean():.2f}")
        print(f"Std rank: {all_ranks.std():.2f}")
        print(f"Min rank: {all_ranks.min():.2f}")
        print(f"Max rank: {all_ranks.max():.2f}")
```

## 🎯 总结

### 为什么这个设计更好？

1. ✅ **关注点分离** - 预测 vs 门控解耦
2. ✅ **易于实验** - 可以独立调整每个组件
3. ✅ **更好的梯度** - Gumbel-Softmax支持
4. ✅ **内置正则化** - 控制平均rank
5. ✅ **上下文感知** - 考虑周围tokens
6. ✅ **理论清晰** - 数学上更优雅

### 与核心创新的对齐

**Per-token动态秩预测**仍然是核心：
- ✅ 每个token独立预测rank
- ✅ 预测连续rank值（更灵活）
- ✅ 嵌套性质保证（Matryoshka）
- ✅ 训练soft，推理hard（加速）

这个设计**增强**了核心创新，而不是改变它！

---

**准备好使用新设计了吗？**
参考上面的代码示例开始吧！如有问题，查看 `modules/improved_rank_predictor.py` 和 `modules/matryoshka_svd_layer_v2.py` 的源代码。
