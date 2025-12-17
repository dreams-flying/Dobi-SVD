#!/usr/bin/env python3
"""
检查重建后的模型结构，诊断为什么PPL很高
"""

import torch
from pathlib import Path
from matryoshka_model_utils import load_matryoshka_model
from modules.matryoshka_svd_layer import MatryoshkaSVDLayer

# 加载模型
checkpoint_path = "matryoshka_output0/final"
print(f"Loading model from {checkpoint_path}...")

model, tokenizer = load_matryoshka_model(
    checkpoint_path=checkpoint_path,
    base_model=None,
    device='cpu'
)

print("\n" + "="*80)
print("检查模型结构")
print("="*80)

# 检查第一层的结构
layer0 = model.model.layers[0]

print(f"\nLayer 0 类型: {type(layer0).__name__}")
print(f"Layer 0.self_attn 类型: {type(layer0.self_attn).__name__}")

# 检查attention的属性
attn = layer0.self_attn
print(f"\nAttention 属性:")
for attr_name in dir(attn):
    if not attr_name.startswith('_'):
        attr = getattr(attn, attr_name)
        if isinstance(attr, (torch.nn.Module, MatryoshkaSVDLayer)):
            print(f"  {attr_name}: {type(attr).__name__}")

# 检查是否有matryoshka属性
print(f"\n检查Matryoshka层:")
if hasattr(attn, 'q_matryoshka'):
    print(f"  ✅ q_matryoshka: {type(attn.q_matryoshka).__name__}")
else:
    print(f"  ❌ q_matryoshka 不存在")

if hasattr(attn, 'q_proj'):
    print(f"  ⚠️  q_proj: {type(attn.q_proj).__name__}")
else:
    print(f"  ✓ q_proj 不存在（正常，已被替换）")

# 检查forward方法
print(f"\n检查forward方法:")
print(f"  attn.forward: {attn.forward}")

# 测试一个简单的forward pass
print(f"\n测试forward pass:")
x = torch.randn(1, 4, 4096)

try:
    with torch.no_grad():
        output = attn(x)
    print(f"  ✅ Forward pass成功")
    print(f"  输出形状: {output[0].shape}")
except Exception as e:
    print(f"  ❌ Forward pass失败: {e}")
    import traceback
    traceback.print_exc()

# 检查MLP
print(f"\n" + "="*80)
print("检查MLP结构")
print("="*80)

mlp = layer0.mlp
print(f"\nMLP 类型: {type(mlp).__name__}")

print(f"\nMLP 属性:")
for attr_name in dir(mlp):
    if not attr_name.startswith('_'):
        attr = getattr(mlp, attr_name)
        if isinstance(attr, (torch.nn.Module, MatryoshkaSVDLayer)):
            print(f"  {attr_name}: {type(attr).__name__}")

if hasattr(mlp, 'gate_matryoshka'):
    print(f"  ✅ gate_matryoshka: {type(mlp.gate_matryoshka).__name__}")
else:
    print(f"  ❌ gate_matryoshka 不存在")

# 测试MLP forward
print(f"\n测试MLP forward pass:")
try:
    with torch.no_grad():
        output = mlp(x)
    print(f"  ✅ MLP Forward pass成功")
    print(f"  输出形状: {output.shape}")
except Exception as e:
    print(f"  ❌ MLP Forward pass失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("结论")
print("="*80)

# 统计MatryoshkaSVDLayer
matryoshka_count = sum(1 for m in model.modules() if isinstance(m, MatryoshkaSVDLayer))
print(f"\n总MatryoshkaSVDLayer数量: {matryoshka_count}")

# 检查第一层是否正确使用matryoshka
if hasattr(attn, 'q_matryoshka') and hasattr(mlp, 'gate_matryoshka'):
    print(f"✅ MatryoshkaSVDLayer已创建")

    # 但是否被使用？
    print(f"\n⚠️  关键问题：forward方法是否使用了matryoshka层？")
    print(f"需要检查forward方法的实现")
else:
    print(f"❌ MatryoshkaSVDLayer未创建")
