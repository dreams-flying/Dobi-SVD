# 显存消耗分析与优化方案

## 🔴 严重问题：权重重复 (CRITICAL)

### 问题1: MatryoshkaSVDLayer权重复制
**位置**: `train_matryoshka_from_svdllm.py:228-274` (`create_matryoshka_layer_from_svd`)

**当前代码**:
```python
def create_matryoshka_layer_from_svd(u_proj, v_proj, r_max, r_min):
    # 创建新的MatryoshkaSVDLayer
    matryoshka = MatryoshkaSVDLayer(...)

    # 复制权重 - 问题在这里！
    matryoshka.v_proj.weight.data = v_proj.weight.data[:effective_r_max, :].clone()
    matryoshka.u_proj.weight.data = u_proj.weight.data[:, :effective_r_max].clone()
```

**显存消耗**:
- 原始SVD-LLM: `u_proj.weight` + `v_proj.weight`
- 新Matryoshka层: `matryoshka.u_proj.weight` + `matryoshka.v_proj.weight`
- **结果**: 权重显存翻倍！

**每层消耗** (以LLaMA-7B为例):
- 注意力层: 4个投影 × 2份权重 = 8份权重
- MLP层: 3个投影 × 2份权重 = 6份权重
- 总计: 32层 × 14份权重 = **448份权重副本**

**显存浪费**:
```
假设每个投影平均 20MB
448个副本 × 20MB = 8.96 GB 纯浪费！
```

### 问题2: 原始投影未删除
**位置**: `train_matryoshka_from_svdllm.py:313-357`

**当前代码**:
```python
# 创建Matryoshka层
attn.q_matryoshka = create_matryoshka_layer_from_svd(...)

# 但是原始的 attn.q_u_proj 和 attn.q_v_proj 仍然存在！
# 从未被删除或设为None
```

**影响**:
- 模型中同时保留：
  - `attn.q_u_proj`, `attn.q_v_proj` (原始)
  - `attn.q_matryoshka.u_proj`, `attn.q_matryoshka.v_proj` (新的)
- 前向传播只用Matryoshka层，但原始层占用显存

---

## 🟡 次要问题

### 问题3: 梯度检查点未启用
**位置**: `train_matryoshka_from_svdllm.py:570`

```python
gradient_checkpointing=False,  # 激活值显存消耗大
```

**影响**:
- 不启用梯度检查点时，所有中间激活值都保存
- LLaMA-7B: ~32层 × batch_size × seq_len × hidden_dim
- 估算: 4 batch × 2048 seq × 4096 dim × 32层 × 2 bytes ≈ 2GB

### 问题4: Optimizer内存
**位置**: 默认使用AdamW

```python
# Trainer默认使用AdamW
# AdamW存储: parameters + momentum + variance = 3x参数量
```

**影响**:
- 如果模型3.5B参数 (fp16: 7GB)
- Optimizer状态: 7GB × 2 = 14GB
- 总计: 7GB + 14GB = 21GB

### 问题5: 数据集全部加载到内存
**位置**: `train_matryoshka_from_svdllm.py:547-549`

```python
train_dataset = Dataset.from_list(tokenized_traindata)  # 全部加载
eval_dataset = Dataset.from_list(tokenized_valdata)
```

**影响**:
- 256 samples × 2048 tokens × 4 bytes ≈ 2MB (可忽略)

---

## ✅ 优化方案

### 方案1: 直接复用权重而非复制 (节省 ~9GB)

**修改**: `create_matryoshka_layer_from_svd`

```python
def create_matryoshka_layer_from_svd(u_proj, v_proj, r_max, r_min):
    in_features = v_proj.in_features
    out_features = u_proj.out_features
    current_rank = v_proj.out_features
    effective_r_max = min(r_max, current_rank)

    # 创建层但不初始化权重
    matryoshka = MatryoshkaSVDLayer(...)

    # 方案A: 直接替换权重张量 (零拷贝)
    matryoshka.v_proj.weight = nn.Parameter(
        v_proj.weight[:effective_r_max, :].contiguous()
    )
    matryoshka.u_proj.weight = nn.Parameter(
        u_proj.weight[:, :effective_r_max].contiguous()
    )

    # 不需要 .clone()，直接使用切片视图
    return matryoshka
```

### 方案2: 删除原始投影 (节省 ~9GB)

**修改**: `convert_to_matryoshka`

```python
for layer_idx, layer in enumerate(layers):
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn

        # 创建Matryoshka层
        attn.q_matryoshka = create_matryoshka_layer_from_svd(...)

        # 删除原始投影以释放显存
        del attn.q_u_proj
        del attn.q_v_proj
        attn.q_u_proj = None
        attn.q_v_proj = None

        # 对所有投影重复
```

### 方案3: 启用梯度检查点 (节省 ~2GB)

```python
training_args = TrainingArguments(
    gradient_checkpointing=True,  # 改为True
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

### 方案4: 使用8-bit Optimizer (节省 ~14GB)

```python
pip install bitsandbytes

training_args = TrainingArguments(
    optim="adamw_bnb_8bit",  # 8-bit AdamW
)
```

### 方案5: 减少batch size，增加梯度累积

```python
--per_device_train_batch_size 1  # 从4降到1 (节省3/4激活显存)
--gradient_accumulation_steps 16  # 从4增到16 (保持有效batch=16)
```

### 方案6: 使用bf16代替fp16

```python
training_args = TrainingArguments(
    bf16=True,  # 更稳定，显存相同
    fp16=False,
)
```

---

## 📊 预期节省

| 优化项 | 节省显存 | 难度 |
|--------|----------|------|
| 1. 零拷贝权重复用 | ~9 GB | 低 |
| 2. 删除原始投影 | ~9 GB | 低 |
| 3. 梯度检查点 | ~2 GB | 低 |
| 4. 8-bit optimizer | ~14 GB | 低 |
| 5. Batch size优化 | ~3 GB | 低 |
| **总计** | **~37 GB** | - |

---

## 🚀 立即行动

### 关键修改优先级:
1. ✅ **立即**: 修复权重重复问题 (节省18GB)
2. ✅ **立即**: 启用梯度检查点 (节省2GB)
3. ✅ **推荐**: 使用8-bit optimizer (节省14GB)
4. ⚠️ **可选**: 调整batch size (节省3GB)

### 实施步骤:
```bash
# 1. 修改代码 (见下方实现)
# 2. 运行优化后的训练
CUDA_VISIBLE_DEVICES=3 python train_matryoshka_from_svdllm.py \
    --model MODEL_ID \
    --svdllm_model ./svd_llm_output/model.pt \
    --r_max 256 --r_min 64 \
    --output_dir ./output \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --freeze_uv \
    --num_train_epochs 1
```
