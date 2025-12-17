#!/usr/bin/env python3
"""
测试模型forward pass，找出tuple index out of range的原因
"""

import torch
from matryoshka_model_utils import load_matryoshka_model

print("Loading model...")
model, tokenizer = load_matryoshka_model(
    checkpoint_path="matryoshka_output0/final",
    base_model=None,
    device='cpu'
)

print("\n" + "="*80)
print("Testing forward pass")
print("="*80)

# 创建测试输入
test_input = tokenizer("Hello world", return_tensors="pt")
print(f"\nInput IDs shape: {test_input['input_ids'].shape}")

# 测试模型forward
print("\nTesting full model forward...")
try:
    with torch.no_grad():
        outputs = model(**test_input)
    print(f"✅ Model forward succeeded")
    print(f"   Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ Model forward failed: {e}")
    import traceback
    traceback.print_exc()

# 测试单个层的forward
print("\n" + "="*80)
print("Testing individual layer forward")
print("="*80)

layer0 = model.model.layers[0]
x = torch.randn(1, 4, 4096)

print("\nTesting layer 0 attention...")
try:
    with torch.no_grad():
        attn_out = layer0.self_attn(x)
    print(f"✅ Attention forward succeeded")
    print(f"   Return type: {type(attn_out)}")
    if isinstance(attn_out, tuple):
        print(f"   Tuple length: {len(attn_out)}")
        for i, elem in enumerate(attn_out):
            if elem is not None:
                if isinstance(elem, torch.Tensor):
                    print(f"   Element {i}: Tensor with shape {elem.shape}")
                else:
                    print(f"   Element {i}: {type(elem)}")
            else:
                print(f"   Element {i}: None")
except Exception as e:
    print(f"❌ Attention forward failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting layer 0 MLP...")
try:
    with torch.no_grad():
        mlp_out = layer0.mlp(x)
    print(f"✅ MLP forward succeeded")
    print(f"   Output shape: {mlp_out.shape}")
except Exception as e:
    print(f"❌ MLP forward failed: {e}")
    import traceback
    traceback.print_exc()

# 检查attention的属性
print("\n" + "="*80)
print("Checking attention attributes")
print("="*80)

attn = layer0.self_attn
required_attrs = ['num_heads', 'head_dim', 'hidden_size', 'rotary_emb',
                  'q_matryoshka', 'k_matryoshka', 'v_matryoshka', 'o_matryoshka']

for attr in required_attrs:
    if hasattr(attn, attr):
        val = getattr(attn, attr)
        print(f"✅ {attr}: {type(val).__name__}")
    else:
        print(f"❌ {attr}: MISSING")
