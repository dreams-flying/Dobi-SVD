# Matryoshka SVD: 统一自适应 Rank 理论框架

## 核心思想

**Matryoshka 表示学习**（MRL）是一种嵌套表示方案，应用于 SVD 压缩可以：
1. 计算**一次**完整 SVD 分解
2. 每个 token 使用基于重要性的自适应截断
3. **无需路由**：直接使用不同长度的前缀
4. 数学上严格：嵌套结构保证一致性

---

## 数学表述

### 1. 标准 SVD 分解

对于权重矩阵 W ∈ R^(m×n)，计算低秩分解：

```
W ≈ U @ S @ V^T
```

其中：
- U ∈ R^(m×r_max): 左奇异向量
- S ∈ R^r_max: 奇异值（对角）
- V ∈ R^(n×r_max): 右奇异向量
- r_max: 最大 rank（如 r_max = 256）

### 2. Matryoshka 嵌套结构

关键性质：对任意 r_k < r_max，使用前 r_k 个分量：

```
W_k = U[:, :r_k] @ S[:r_k] @ V[:r_k, :]
```

满足**嵌套性质**：
```
W_1 ⊂ W_2 ⊂ ... ⊂ W_{r_max}
```

这保证了：
- 较小的 rank 是较大 rank 的前缀
- 无需重新计算 SVD
- 训练和推理使用相同的分解

### 3. 自适应 Rank 选择

对于输入 x ∈ R^(batch×seq×d)，计算每个 token 的重要性：

```
importance(x_i) ∈ [0, 1]
```

将重要性映射到 rank：

```
r_i = r_min + (r_max - r_min) × importance(x_i)
```

其中：
- r_min: 最小 rank（如 32）
- r_max: 最大 rank（如 256）
- r_i ∈ [r_min, r_max]: token i 的自适应 rank

### 4. 可微软截断

为了端到端训练，使用可微的软截断函数：

```
truncation_i(k) = σ((r_i - k) / τ)
```

其中：
- σ: sigmoid 函数
- τ: 温度参数（控制硬度，默认 τ=0.1）
- k ∈ [0, r_max): SVD 分量索引

截断后的奇异值：

```
S_adaptive[i, k] = S[k] × truncation_i(k)
```

### 5. 前向传播

完整的前向传播公式：

```
x_transformed = (x @ V^T) @ diag(S_adaptive) @ U^T
```

展开：
```
对于每个 token i:
  xV_i = x_i @ V^T                    # [d] → [r_max]
  xVS_i = xV_i × S_adaptive[i, :]     # 逐元素乘法
  output_i = xVS_i @ U^T              # [r_max] → [m]
```

---

## 重要性计算策略

### 策略 1：Norm-based（最简单）

```
importance(x) = ||x||_2 / (max_i ||x_i||_2)
```

- 优点：O(d) 复杂度，无参数
- 缺点：忽略语义信息

### 策略 2：Learned Predictor（推荐）

```
importance(x) = σ(MLP(x))

MLP: d → 128 → 64 → 1
```

- 优点：学习语义重要性
- 缺点：增加少量参数（~8K for d=4096）

### 策略 3：Attention-based

```
importance(x) = mean(attention_scores)
```

- 优点：利用已有的 attention
- 缺点：需要 hook attention 层

---

## 复杂度分析

### 时间复杂度

**初始化（一次性）**：
```
SVD 分解: O(min(m,n)^2 × max(m,n)) ≈ O(d^3)
```

**前向传播（每次）**：
```
重要性计算:
  - Norm: O(n × d)
  - Learned: O(n × d × h) where h=128

Rank 映射: O(n)

截断计算: O(n × r_max)

矩阵乘法:
  - x @ V^T: O(n × d × r_max)
  - × S_adaptive: O(n × r_max)
  - @ U^T: O(n × r_max × m)

总计: O(n × d × r_max)
```

**对比多子空间路由**：
```
Value-aware 路由: O(n^2 × d)  ← 二次！
Matryoshka: O(n × d × r_max)  ← 线性
```

对于 n=2048, d=4096, r_max=256：
- Matryoshka: 2048 × 4096 × 256 = 2.1B ops
- Value-aware: 2048^2 × 4096 = 17.2B ops
- **加速: 8.2x**

### 空间复杂度

**参数量**：
```
SVD 参数: m×r_max + r_max + n×r_max = r_max×(m+n+1)

Learned predictor: d×128 + 128×64 + 64×1 ≈ d×128

总计: r_max×(m+n+1) + d×128
```

对于 m=n=d=4096, r_max=256：
```
SVD: 256 × 8,193 = 2,097,408
Predictor: 4096 × 128 = 524,288
总计: 2,621,696 参数

vs 原始: 4096^2 = 16,777,216
压缩率: 2.6M / 16.7M = 15.6% (84.4% 压缩)
```

**激活内存**：
```
输入: n × d
重要性: n
截断矩阵: n × r_max
中间结果: n × r_max
输出: n × m

总计: n×(d + r_max + m) + n
```

vs 多子空间：
```
多子空间: n×(d + K×r + m + K) 其中 K=3
Matryoshka: n×(d + r_max + m)

内存相近，但 Matryoshka 无路由开销
```

---

## 理论优势

### 1. 消除路由开销

**多子空间路由**需要：
- 重要性计算：O(n^2×d)（value-aware）
- 路由决策：O(n×K)
- 条件分支：GPU 不友好
- 训练/测试不匹配

**Matryoshka** 只需：
- 重要性计算：O(n×d)（norm）或 O(n×d×h)（learned）
- Rank 映射：O(n)
- 无分支：完全向量化

### 2. 数学严格性

**多子空间问题**：
- 不同子空间可能有重叠/冲突的表示
- 软路由混合不同 SVD 分解
- 理论上不保证最优性

**Matryoshka 优势**：
- 嵌套结构：r_small ⊂ r_large
- 单一 SVD：全局最优分解
- 可证明：使用更多分量总是不会更差

### 3. 训练稳定性

**多子空间问题**：
- 路由崩溃 → 均匀分布
- Gamma 梯度消失
- NaN/Inf 级联

**Matryoshka 优势**：
- 无离散路由决策
- 梯度平滑流动
- 数值稳定（单次 SVD）

### 4. 灵活性

可以在推理时动态调整压缩率：

```python
# 低延迟模式：使用低 rank
importance_scale = 0.5
r_i = r_min + (r_max - r_min) × importance × importance_scale

# 高质量模式：使用高 rank
importance_scale = 1.5
r_i = r_min + (r_max - r_min) × importance × importance_scale
```

无需重新训练或切换模型。

---

## 训练策略

### 多尺度损失

借鉴 Matryoshka 表示学习，使用多尺度损失：

```
L_total = Σ_k w_k × L_task(output_k)
```

其中：
- output_k: 使用固定 rank=k 的输出
- k ∈ {r_min, r_max/2, r_max}
- w_k: 权重（如 [0.3, 0.3, 0.4]）

这确保模型在**所有 rank 级别**都表现良好。

### Rank 正则化

鼓励模型使用低 rank（提高压缩）：

```
L_rank = λ × mean(r_i) / r_max
```

其中 λ=0.001（小权重，避免过度压缩）

### 完整损失函数

```
L_total = L_task + λ_rank × L_rank + λ_multiscale × L_multiscale
```

---

## 理论保证

### 定理 1：嵌套性质

对于 Matryoshka SVD，有：

```
∀ r_1 < r_2 ≤ r_max:
  ||W - W_{r_1}||_F ≥ ||W - W_{r_2}||_F
```

即：使用更多 SVD 分量，重构误差单调递减。

**证明**：由 SVD 的最优性质直接得出。

### 定理 2：计算复杂度

Matryoshka SVD 的前向传播复杂度为：

```
O(n × d × r_avg)
```

其中 r_avg = E[r_i] 是平均 rank。

当 r_avg < r_max 时，相比固定 rank=r_max 的 SVD 有加速。

### 定理 3：梯度存在性

软截断函数：

```
truncation(r, k) = σ((r - k) / τ)
```

对 r 可微，梯度为：

```
∂truncation/∂r = σ'((r-k)/τ) / τ
```

保证了端到端训练的可行性。

---

## 与现有方法对比

| 特性 | 多子空间路由 | Matryoshka SVD |
|------|-------------|----------------|
| **参数量** | 1.0M + routing | 2.6M (learned) / 2.1M (norm) |
| **前向复杂度** | O(n²d) 或 O(ndr) | O(ndr) |
| **训练稳定性** | ❌ 路由崩溃 | ✅ 稳定 |
| **数学严格性** | ⚠️ 无保证 | ✅ 嵌套性质 |
| **实现复杂度** | 1,239 行 | ~300 行 |
| **调试难度** | 高（NaN级联） | 低 |
| **灵活性** | 固定 K 个子空间 | 连续 rank 范围 |

---

## 实现计划

### 阶段 1：核心实现

1. `MatryoshkaSVDLayer` 类
   - SVD 初始化
   - 软截断函数
   - 前向传播

2. 重要性计算
   - Norm-based（默认）
   - Learned predictor（可选）

### 阶段 2：训练支持

3. 多尺度损失
4. Rank 正则化
5. 与现有 trainer 集成

### 阶段 3：优化与评估

6. 数值优化（混合精度）
7. 与多子空间对比实验
8. 性能分析报告

---

## 预期性能

基于理论分析，预期：

### 压缩率
- **目标**: 85% 压缩（15% 参数保留）
- **方法**: r_max=256, r_avg≈128（自适应）
- **对比**: 多子空间 6.25%（但有容量瓶颈）

### 质量
- **目标**: PPL < 7.0（LLaMA-2-7B on WikiText-2）
- **方法**: 多尺度训练确保各 rank 级别质量
- **对比**: 多子空间未验证（路由崩溃）

### 速度
- **训练**: 2-3x 快于 value-aware 路由
- **推理**: 与标准 SVD 持平（无路由开销）

### 稳定性
- **数值**: 单次 SVD，无 NaN 级联
- **训练**: 平滑梯度，无路由崩溃
- **收敛**: 预期更快（无矛盾目标）

---

## 下一步

1. ✅ 理论框架完成
2. → 实现 `MatryoshkaSVDLayer` 核心
3. → 实现 `AdaptiveRankPredictor`
4. → 集成到训练框架
5. → 运行对比实验
6. → 发表结果

---

## 参考文献

1. **Matryoshka Representation Learning** (NeurIPS 2022)
   - Aditya Kusupati et al.
   - https://arxiv.org/abs/2205.13147

2. **Dobi: SVD-based Low-Rank Compression** (ACL 2024)
   - Original Dobi-SVD paper

3. **Optimal SVD Truncation** (Linear Algebra Theory)
   - Eckart–Young–Mirsky theorem
