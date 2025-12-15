# Matryoshka SVD 最佳训练超参数

## 🎯 快速推荐配置

### 配置 A：快速实验（推荐用于验证想法）

```bash
python train_matryoshka_from_svdllm.py \
    --model meta-llama/Llama-2-7b-hf \
    --svdllm_model /path/to/svd_llm_output \
    --r_max 256 \
    --r_min 64 \
    --freeze_uv \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lambda_rank 0.001 \
    --multiscale_frequency 0.5 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 200 \
    --output_dir ./results/matryoshka_quick
```

**适用场景**：
- ✅ 快速验证 Matryoshka 思路
- ✅ 调试代码和流程
- ✅ 资源有限（单卡 24GB）

**预期效果**：
- 训练时间：~4 小时（7B 模型，1 epoch）
- PPL 提升：相比 SVD-LLM 基线 +5-10%
- 内存占用：~20GB

---

### 配置 B：最佳性能（推荐用于论文实验）

```bash
python train_matryoshka_from_svdllm.py \
    --model meta-llama/Llama-2-7b-hf \
    --svdllm_model /path/to/svd_llm_output \
    --r_max 256 \
    --r_min 64 \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lambda_rank 0.0005 \
    --multiscale_frequency 0.6 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --logging_steps 5 \
    --save_steps 500 \
    --eval_steps 500 \
    --load_best_model_at_end \
    --output_dir ./results/matryoshka_best
```

**适用场景**：
- ✅ 论文最终结果
- ✅ 追求最佳性能
- ✅ 充足的计算资源（单卡 40GB+ 或多卡）

**预期效果**：
- 训练时间：~16 小时（7B 模型，2 epochs）
- PPL 提升：相比 SVD-LLM 基线 +2-5%
- 内存占用：~35GB（联合微调）

---

## 📊 核心超参数详解

### 1. Rank 范围（r_max, r_min）

**理论依据**：
- `r_max`：必须 ≤ SVD-LLM 的有效秩
- `r_min`：平衡压缩率与性能

| SVD-LLM ratio | 有效秩 | 推荐 r_max | 推荐 r_min | 压缩率范围 |
|--------------|--------|-----------|-----------|-----------|
| 0.3 | ~153 | 128-192 | 32-64 | 高压缩 |
| 0.5 | ~256 | 256-320 | 64-128 | 中等压缩 |
| 0.7 | ~358 | 320-384 | 128-192 | 保守压缩 |

**选择建议**：
```python
# 激进压缩（追求速度）
r_max = 128
r_min = 32

# 平衡配置（推荐）
r_max = 256
r_min = 64

# 保守配置（追求精度）
r_max = 384
r_min = 128
```

**实验技巧**：
```bash
# 先用小范围测试
--r_max 128 --r_min 64

# 如果效果好，扩大范围
--r_max 256 --r_min 32

# 观察 rank predictor 的实际输出分布
# 如果大多数 token 预测在 [100, 120]，说明范围设置合理
```

---

### 2. 学习率（learning_rate）

**理论依据**：
- Rank predictor 是新训练的小网络，可以用较大学习率
- U, V 是预训练的，需要小心微调

| 训练模式 | Rank Predictor LR | U, V LR | 总 LR |
|---------|------------------|---------|-------|
| **冻结 U,V** | 1e-4 ~ 5e-4 | N/A | 1e-4 |
| **联合微调** | 1e-4 ~ 5e-4 | 1e-6 ~ 1e-5 | **分层设置** |

**推荐配置**：

**模式 A（冻结 U,V）**：
```bash
--freeze_uv \
--learning_rate 1e-4  # 只影响 rank predictor
```

**模式 B（联合微调）**：
需要在代码中实现分层学习率：

```python
# train_matryoshka_from_svdllm.py 中添加
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() 
                   if 'rank_predictor' in n],
        'lr': 1e-4  # Rank predictor 用较大学习率
    },
    {
        'params': [p for n, p in model.named_parameters() 
                   if ('_u_proj' in n or '_v_proj' in n) and p.requires_grad],
        'lr': 5e-6  # U, V 用很小的学习率
    }
]
```

**调试技巧**：
```bash
# 如果训练不稳定（loss 波动大）
--learning_rate 5e-5

# 如果收敛太慢（loss 几乎不降）
--learning_rate 5e-4

# 观察训练曲线，调整到 loss 平滑下降
```

---

### 3. Rank 正则化（lambda_rank）

**理论依据**：
$$L_{\text{total}} = L_{\text{task}} + \lambda_{\text{rank}} \cdot \frac{1}{N} \sum_{i=1}^{N} \frac{r_i - r_{\min}}{r_{\max} - r_{\min}}$$

**作用**：
- 鼓励 rank predictor 输出更低的秩
- 平衡精度与速度

**推荐值**：

| 目标 | lambda_rank | 预期平均秋 | PPL 影响 |
|------|------------|-----------|---------|
| **高精度优先** | 0.0001 | ~0.7 × r_max | +2% |
| **平衡** | 0.001 | ~0.5 × r_max | +5% |
| **高速度优先** | 0.01 | ~0.3 × r_max | +10% |

**选择策略**：
```python
# 第一阶段：先不用正则化，看 rank 分布
--lambda_rank 0

# 观察：如果 90% token 都预测 r_max，说明需要正则化
# 第二阶段：加入正则化
--lambda_rank 0.001

# 第三阶段：根据平均秋调整
# 如果平均秋还是太高（> 0.7 × r_max）
--lambda_rank 0.005

# 如果平均秋太低（< 0.3 × r_max），PPL 太差
--lambda_rank 0.0005
```

**实验技巧**：
```bash
# 训练后分析 rank 分布
python -c "
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('./results/checkpoint-1000')
# 在测试数据上运行，记录所有 token 的预测秩
# 画直方图，看分布是否合理
"
```

---

### 4. 多尺度训练频率（multiscale_frequency）

**理论依据**：
- 多尺度训练强制模型在不同秩下都能工作
- 频率太高：rank predictor 学不到动态选择
- 频率太低：嵌套属性学习不足

**推荐值**：

| 阶段 | 频率 | 说明 |
|------|------|------|
| **早期（0-30%）** | 0.7 | 强制学习嵌套结构 |
| **中期（30-70%）** | 0.5 | 平衡动态和嵌套 |
| **后期（70-100%）** | 0.3 | 更多动态预测 |

**推荐配置**：
```bash
# 简单方案：固定频率
--multiscale_frequency 0.5

# 高级方案：在代码中实现动态调整
# train_matryoshka_from_svdllm.py 中修改
def compute_loss(self, model, inputs, return_outputs=False):
    # 根据训练进度调整频率
    progress = self.state.global_step / self.state.max_steps
    if progress < 0.3:
        frequency = 0.7
    elif progress < 0.7:
        frequency = 0.5
    else:
        frequency = 0.3
    
    use_multiscale = random.random() < frequency
    ...
```

---

### 5. Gating 温度（gating_tau）

**理论依据**：
$$\text{gate}_k = \sigma\left(\frac{r_i - k}{\tau}\right)$$

- `τ` 小：截断更锐利（接近硬截断）
- `τ` 大：截断更平滑（梯度更稳定）

**推荐值**：

| 任务特点 | tau | 效果 |
|---------|-----|------|
| **需要精确控制秩** | 0.05 | 锐利截断 |
| **平衡（推荐）** | 0.1 | 中等平滑 |
| **训练不稳定** | 0.2 | 很平滑 |

**设置位置**：
```python
# modules/matryoshka_svd_layer.py:109
def __init__(..., gating_tau: float = 0.1):
    self.gating_tau = gating_tau
```

**调试技巧**：
```python
# 可视化不同 tau 的效果
import matplotlib.pyplot as plt
tau_values = [0.05, 0.1, 0.2]
r_i = 128
k = torch.arange(1, 257)

for tau in tau_values:
    gate = torch.sigmoid((r_i - k) / tau)
    plt.plot(k, gate, label=f'tau={tau}')

plt.axvline(r_i, color='red', linestyle='--', label='predicted rank')
plt.legend()
plt.show()
```

---

### 6. Batch Size 与 Gradient Accumulation

**理论依据**：
- 有效 batch size = `per_device_batch_size × num_gpus × gradient_accumulation_steps`
- 太小：训练不稳定，rank predictor 难收敛
- 太大：泛化性差

**推荐配置**：

| 模型大小 | 目标有效 batch | 单卡 batch | Accum steps | 内存需求 |
|---------|--------------|-----------|------------|---------|
| **1-3B** | 32 | 8 | 4 | 16GB |
| **7B** | 16-32 | 4 | 4-8 | 24GB |
| **13B** | 16 | 2 | 8 | 40GB |
| **70B** | 8-16 | 1 | 8-16 | 80GB |

**配置示例**：
```bash
# 单卡 A100 40GB，训练 7B 模型
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
# 有效 batch size = 4 × 1 × 4 = 16

# 多卡 4×A100，训练 7B 模型
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
# 有效 batch size = 4 × 4 × 2 = 32
```

---

### 7. 训练 Epochs

**理论依据**：
- Rank predictor 是小网络，1 epoch 即可学会基本模式
- 联合微调需要更多 epochs 让 U,V 适应嵌套结构

| 训练模式 | 推荐 epochs | 说明 |
|---------|------------|------|
| **冻结 U,V** | 1 | Predictor 快速收敛 |
| **联合微调** | 2-3 | U,V 需要慢慢调整 |

**Early stopping**：
```bash
--num_train_epochs 3 \
--load_best_model_at_end \
--metric_for_best_model eval_loss \
--greater_is_better False \
--eval_steps 500 \
--save_total_limit 3
```

---

## 🔬 不同场景的完整配置

### 场景 1：快速验证（4小时内出结果）

```bash
#!/bin/bash
# quick_test.sh

python train_matryoshka_from_svdllm.py \
    --model meta-llama/Llama-2-7b-hf \
    --svdllm_model ./svd_llm_output \
    --r_max 128 \
    --r_min 64 \
    --freeze_uv \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lambda_rank 0.001 \
    --multiscale_frequency 0.5 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 200 \
    --warmup_ratio 0.05 \
    --dataset wikitext2 \
    --output_dir ./results/quick_test

# 立即评估
python evaluate_matryoshka.py \
    --checkpoint ./results/quick_test/final \
    --dataset wikitext2 \
    --max_samples 100
```

---

### 场景 2：论文实验（最佳性能）

```bash
#!/bin/bash
# paper_experiment.sh

# 阶段1：冻结 U,V 训练（快速收敛）
python train_matryoshka_from_svdllm.py \
    --model meta-llama/Llama-2-7b-hf \
    --svdllm_model ./svd_llm_output \
    --r_max 256 \
    --r_min 64 \
    --freeze_uv \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lambda_rank 0.001 \
    --multiscale_frequency 0.5 \
    --output_dir ./results/stage1_frozen

# 阶段2：联合微调（精细调整）
python train_matryoshka_from_svdllm.py \
    --model ./results/stage1_frozen/final \
    --svdllm_model ./svd_llm_output \
    --r_max 256 \
    --r_min 64 \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lambda_rank 0.0005 \
    --multiscale_frequency 0.4 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --output_dir ./results/stage2_finetuned

# 全面评估
python evaluate_matryoshka.py \
    --checkpoint ./results/stage2_finetuned/final \
    --dataset wikitext2 \
    --max_samples 1000 \
    --measure_latency
```

---

### 场景 3：大模型（70B）

```bash
#!/bin/bash
# large_model.sh

# 需要使用 DeepSpeed ZeRO-3
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train_matryoshka_from_svdllm.py \
    --model meta-llama/Llama-2-70b-hf \
    --svdllm_model ./svd_llm_70b \
    --r_max 384 \
    --r_min 128 \
    --freeze_uv \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lambda_rank 0.0005 \
    --multiscale_frequency 0.5 \
    --max_grad_norm 0.5 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ds_config_zero3.json \
    --output_dir ./results/llama70b_matryoshka
```

---

## 📊 超参数调优流程

### Step 1: 基线测试（不加正则化）

```bash
--lambda_rank 0 \
--multiscale_frequency 0.3
```

观察：
- Rank predictor 的输出分布
- 大部分 token 预测什么秩？

### Step 2: 加入适度正则化

```bash
--lambda_rank 0.001 \
--multiscale_frequency 0.5
```

观察：
- 平均秋是否降低？
- PPL 提升多少？

### Step 3: 微调平衡点

根据 Step 2 的结果：
- 如果平均秋仍 > 0.7×r_max：增加 lambda_rank 到 0.005
- 如果 PPL 提升 > 10%：降低 lambda_rank 到 0.0005

### Step 4: 联合微调（可选）

如果性能还不够好：
```bash
# 不使用 --freeze_uv
--learning_rate 5e-6 \
--num_train_epochs 2
```

---

## 🎯 常见问题

### Q1: Rank predictor 总是输出 r_max，怎么办？

**原因**：正则化太弱，模型"懒惰"
**解决**：
```bash
--lambda_rank 0.01  # 增加10倍
```

### Q2: PPL 提升太多（>20%），不可接受

**原因**：正则化太强 或 r_min 太小
**解决**：
```bash
--lambda_rank 0.0001  # 减少正则化
--r_min 128  # 提高最小秋
```

### Q3: 训练不稳定，loss 波动大

**原因**：学习率太大 或 batch size 太小
**解决**：
```bash
--learning_rate 5e-5  # 降低学习率
--gradient_accumulation_steps 8  # 增加有效 batch
--max_grad_norm 0.5  # 更激进的梯度裁剪
```

### Q4: 联合微调时模型崩溃

**原因**：U,V 的学习率太大
**解决**：
```bash
--learning_rate 1e-6  # 极小的学习率
--warmup_ratio 0.2  # 更长的 warmup
```

---

## 📝 最终推荐配置总结

**快速实验**（4小时，单卡 24GB）：
```bash
--r_max 128 --r_min 64 --freeze_uv \
--learning_rate 1e-4 --num_train_epochs 1 \
--lambda_rank 0.001 --multiscale_frequency 0.5
```

**论文结果**（16小时，单卡 40GB）：
```bash
--r_max 256 --r_min 64 \
--learning_rate 5e-6 --num_train_epochs 2 \
--lambda_rank 0.0005 --multiscale_frequency 0.6
```

**大模型**（24小时，8卡 80GB）：
```bash
--r_max 384 --r_min 128 --freeze_uv \
--learning_rate 5e-5 --num_train_epochs 1 \
--lambda_rank 0.0005 --multiscale_frequency 0.5 \
--deepspeed ds_config_zero3.json
```

---

## 🔬 实验建议

1. **先快速验证**：用小范围（r_max=128）快速跑1 epoch
2. **观察 rank 分布**：看 predictor 是否学到了动态选择
3. **调整正则化**：根据平均秋和 PPL 的平衡点
4. **联合微调**：如果需要最佳性能，再用小学习率联合微调
5. **多次实验**：超参数对不同模型/数据集敏感，需要调整

Good luck! 🎯
