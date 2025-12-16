#!/usr/bin/env python3
"""
全面诊断Matryoshka SVD训练和评测流程

检查项目：
1. SVD-LLM模型是否正确加载
2. MatryoshkaSVDLayer是否正确创建
3. 动态秩预测是否实现
4. 模型保存是否包含metadata
5. 模型加载是否正确重建MatryoshkaSVDLayer
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Dict, List


def check_svdllm_loading(svd_model_path: str):
    """检查SVD-LLM模型加载"""
    print("\n" + "="*80)
    print("1. 检查SVD-LLM模型加载")
    print("="*80)

    if not os.path.exists(svd_model_path):
        print(f"❌ SVD-LLM模型文件不存在: {svd_model_path}")
        return False

    print(f"✅ 模型文件存在: {svd_model_path}")

    try:
        # 尝试加载
        pruned_dict = torch.load(svd_model_path, weights_only=False, map_location='cpu')

        # 检查必需的keys
        required_keys = ['model', 'tokenizer']
        missing_keys = [k for k in required_keys if k not in pruned_dict]

        if missing_keys:
            print(f"❌ 模型文件缺少必需的keys: {missing_keys}")
            print(f"   实际包含的keys: {list(pruned_dict.keys())}")
            return False

        print(f"✅ 模型文件包含所有必需的keys: {required_keys}")

        # 检查模型结构
        model = pruned_dict['model']
        print(f"✅ 模型类型: {type(model).__name__}")

        # 检查是否有SVD layers
        has_svd_layers = False
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if 'SVD' in module_type:
                has_svd_layers = True
                print(f"✅ 找到SVD层: {name} ({module_type})")
                break

        if not has_svd_layers:
            print(f"⚠️  警告: 未找到SVD层，这可能不是SVD-LLM模型")

        # 检查参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型参数量: {total_params / 1e9:.2f}B")

        return True

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_matryoshka_layer_creation():
    """检查MatryoshkaSVDLayer是否正确实现动态秩预测"""
    print("\n" + "="*80)
    print("2. 检查MatryoshkaSVDLayer实现")
    print("="*80)

    try:
        from modules.matryoshka_svd_layer import MatryoshkaSVDLayer

        # 创建测试layer
        layer = MatryoshkaSVDLayer(
            in_features=4096,
            out_features=4096,
            r_max=256,
            r_min=64,
            use_rank_predictor=True,
            predictor_mode='rank',
            hard_inference=True
        )

        print(f"✅ MatryoshkaSVDLayer创建成功")
        print(f"   - in_features: {layer.in_features}")
        print(f"   - out_features: {layer.out_features}")
        print(f"   - r_max: {layer.r_max}")
        print(f"   - r_min: {layer.r_min}")
        print(f"   - predictor_mode: {layer.predictor_mode}")
        print(f"   - hard_inference: {layer.hard_inference}")

        # 检查是否有rank predictor
        if layer.use_rank_predictor:
            print(f"✅ Rank predictor已启用")
            print(f"   - Predictor类型: {type(layer.rank_predictor).__name__}")
        else:
            print(f"❌ Rank predictor未启用")
            return False

        # 测试动态秩预测
        print(f"\n测试动态秩预测...")
        layer.eval()

        # 测试输入
        x = torch.randn(2, 10, 4096)

        # 自适应模式（动态预测）
        layer.set_fixed_rank(None)
        output_adaptive = layer(x)

        # 获取预测的秩
        if layer.predictor_mode == 'rank':
            with torch.no_grad():
                predicted_rank = layer.rank_predictor(x.mean(dim=1))
                avg_rank = predicted_rank.mean().item()
            print(f"✅ 动态秩预测工作正常")
            print(f"   - 平均预测秩: {avg_rank:.2f}")
            print(f"   - 预测秩范围: [{predicted_rank.min().item():.2f}, {predicted_rank.max().item():.2f}]")

        # 固定秩模式
        layer.set_fixed_rank(128)
        output_fixed = layer(x)

        print(f"✅ 固定秩模式工作正常")
        print(f"   - 固定秩: {layer.fixed_rank}")

        # 检查hard inference模式
        if layer.hard_inference and layer.predictor_mode == 'rank':
            print(f"✅ Hard inference模式已启用（推理时使用二值门控）")
        else:
            print(f"⚠️  Hard inference模式未启用")

        return True

    except Exception as e:
        print(f"❌ MatryoshkaSVDLayer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_saving(checkpoint_path: str):
    """检查模型保存是否包含metadata"""
    print("\n" + "="*80)
    print("3. 检查模型保存机制")
    print("="*80)

    checkpoint_path = Path(checkpoint_path)

    # 检查checkpoint是否存在
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint不存在: {checkpoint_path}")
        print(f"   这是正常的，因为还没有训练")
        return None

    print(f"✅ Checkpoint存在: {checkpoint_path}")

    # 检查必需文件
    required_files = ['config.json', 'matryoshka_metadata.json']
    optional_files = ['model.safetensors', 'pytorch_model.bin', 'tokenizer.json']

    print(f"\n检查必需文件:")
    for file in required_files:
        file_path = checkpoint_path / file
        if file_path.exists():
            print(f"✅ {file} 存在")

            if file == 'matryoshka_metadata.json':
                # 检查metadata内容
                import json
                with open(file_path, 'r') as f:
                    metadata = json.load(f)

                num_layers = len(metadata.get('matryoshka_layers', []))
                print(f"   - Matryoshka层数量: {num_layers}")

                if num_layers > 0:
                    example = metadata['matryoshka_layers'][0]
                    print(f"   - 示例层: {example['name']}")
                    print(f"   - r_max: {example['r_max']}, r_min: {example['r_min']}")
                    print(f"   - predictor_mode: {example['predictor_mode']}")
                else:
                    print(f"❌ metadata中没有Matryoshka层信息!")
                    return False
        else:
            print(f"❌ {file} 不存在!")
            return False

    print(f"\n检查权重文件:")
    has_weights = False
    for file in optional_files:
        file_path = checkpoint_path / file
        if file_path.exists():
            print(f"✅ {file} 存在")
            has_weights = True

    # 检查分片的safetensors
    sharded_files = list(checkpoint_path.glob('model-*.safetensors'))
    if sharded_files:
        print(f"✅ 找到 {len(sharded_files)} 个分片的safetensors文件")
        has_weights = True

    if not has_weights:
        print(f"❌ 未找到权重文件!")
        return False

    return True


def check_model_loading(checkpoint_path: str):
    """检查模型加载是否正确重建MatryoshkaSVDLayer"""
    print("\n" + "="*80)
    print("4. 检查模型加载机制")
    print("="*80)

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint不存在: {checkpoint_path}")
        print(f"   跳过加载测试")
        return None

    try:
        from matryoshka_model_utils import load_matryoshka_model, verify_matryoshka_structure

        # 先验证结构
        print(f"\n验证checkpoint结构...")
        result = verify_matryoshka_structure(checkpoint_path)

        print(f"   - 是否有metadata: {result['has_metadata']}")
        print(f"   - Matryoshka层数量: {result['num_layers']}")
        print(f"   - config存在: {result['config_exists']}")
        print(f"   - 权重存在: {result['weights_exist']}")

        if not result['has_metadata']:
            print(f"❌ Checkpoint缺少matryoshka_metadata.json!")
            print(f"   这个checkpoint是用旧版训练脚本生成的")
            print(f"   需要重新训练才能使用新的加载机制")
            return False

        # 尝试加载模型
        print(f"\n尝试加载模型...")
        model, tokenizer = load_matryoshka_model(
            checkpoint_path=str(checkpoint_path),
            base_model=None,
            device='cpu'
        )

        # 统计MatryoshkaSVDLayer数量
        from modules.matryoshka_svd_layer import MatryoshkaSVDLayer

        matryoshka_count = 0
        matryoshka_layers = []

        for name, module in model.named_modules():
            if isinstance(module, MatryoshkaSVDLayer):
                matryoshka_count += 1
                matryoshka_layers.append(name)

        print(f"\n✅ 模型加载成功!")
        print(f"   - MatryoshkaSVDLayer数量: {matryoshka_count}")

        if matryoshka_count == 0:
            print(f"❌ 严重错误: 加载后模型中没有MatryoshkaSVDLayer!")
            return False

        if matryoshka_count != result['num_layers']:
            print(f"⚠️  警告: 加载的层数({matryoshka_count})与metadata中的数量({result['num_layers']})不匹配")

        # 显示前几个层
        print(f"\n前5个MatryoshkaSVDLayer:")
        for name in matryoshka_layers[:5]:
            print(f"   - {name}")

        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_script():
    """检查训练脚本是否使用了正确的保存函数"""
    print("\n" + "="*80)
    print("5. 检查训练脚本")
    print("="*80)

    script_path = Path('/home/user/Dobi-SVD/train_matryoshka_from_svdllm.py')

    if not script_path.exists():
        print(f"❌ 训练脚本不存在: {script_path}")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    # 检查是否使用了save_matryoshka_model
    if 'save_matryoshka_model' in content:
        print(f"✅ 训练脚本使用了save_matryoshka_model()")

        # 检查导入
        if 'from matryoshka_model_utils import save_matryoshka_model' in content:
            print(f"✅ 正确导入了save_matryoshka_model")
        else:
            print(f"⚠️  未找到save_matryoshka_model的导入语句")
    else:
        print(f"❌ 训练脚本未使用save_matryoshka_model()!")
        print(f"   这意味着训练后的checkpoint不会包含matryoshka_metadata.json")
        return False

    # 检查是否还在使用trainer.save_model()
    if 'trainer.save_model' in content:
        # 检查是否在save_matryoshka_model之前
        save_matryoshka_pos = content.find('save_matryoshka_model')
        trainer_save_pos = content.find('trainer.save_model')

        if trainer_save_pos > 0 and trainer_save_pos < save_matryoshka_pos:
            print(f"⚠️  警告: 代码中仍然调用了trainer.save_model()")
            print(f"   这可能会导致问题，应该只使用save_matryoshka_model()")

    return True


def check_evaluation_script():
    """检查评测脚本是否使用了正确的加载函数"""
    print("\n" + "="*80)
    print("6. 检查评测脚本")
    print("="*80)

    script_path = Path('/home/user/Dobi-SVD/evaluate_matryoshka_svdllm.py')

    if not script_path.exists():
        print(f"❌ 评测脚本不存在: {script_path}")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    # 检查是否使用了matryoshka_model_utils
    if 'from matryoshka_model_utils import load_matryoshka_model' in content:
        print(f"✅ 评测脚本导入了load_matryoshka_model")
    else:
        print(f"❌ 评测脚本未导入matryoshka_model_utils!")
        return False

    # 检查load_matryoshka_model函数是否调用了custom load
    if 'load_matryoshka_custom' in content or 'load_matryoshka_model as load' in content:
        print(f"✅ 评测脚本使用了自定义加载函数")
    else:
        print(f"⚠️  评测脚本可能没有正确使用自定义加载函数")

    return True


def generate_recommendations(results: Dict[str, bool]):
    """根据检查结果生成建议"""
    print("\n" + "="*80)
    print("诊断总结与建议")
    print("="*80)

    all_passed = all(v for v in results.values() if v is not None)

    if all_passed:
        print("\n✅ 所有检查都通过!")
        print("\n后续步骤:")
        print("1. 使用更新后的训练脚本训练模型")
        print("2. 训练完成后，验证checkpoint包含matryoshka_metadata.json")
        print("3. 使用更新后的评测脚本评测模型")
    else:
        print("\n⚠️  发现以下问题:")

        for check, passed in results.items():
            if passed is False:
                print(f"   ❌ {check}")

        print("\n修复建议:")

        if results.get('training_script') is False:
            print("\n1. 训练脚本问题:")
            print("   - 确保使用了save_matryoshka_model()而不是trainer.save_model()")
            print("   - 检查是否正确导入了matryoshka_model_utils")

        if results.get('evaluation_script') is False:
            print("\n2. 评测脚本问题:")
            print("   - 确保使用了load_matryoshka_model()从matryoshka_model_utils")
            print("   - 检查load_matryoshka_model函数是否调用了自定义加载逻辑")

        if results.get('model_saving') is False:
            print("\n3. 模型保存问题:")
            print("   - 当前checkpoint缺少matryoshka_metadata.json")
            print("   - 需要使用更新后的训练脚本重新训练")

        if results.get('model_loading') is False:
            print("\n4. 模型加载问题:")
            print("   - 加载后模型中没有MatryoshkaSVDLayer")
            print("   - 确保checkpoint有matryoshka_metadata.json文件")
            print("   - 重新训练模型以生成正确的checkpoint")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='诊断Matryoshka SVD训练和评测流程')
    parser.add_argument('--svd_model', type=str,
                       default='/data1/lichangqun/SVD-LLM/svd_llm_output/MODEL_ID_whitening_then_update_0.5.pt',
                       help='SVD-LLM模型路径')
    parser.add_argument('--checkpoint', type=str,
                       default='/data1/lichangqun/Dobi-SVD-Matryoshka/matryoshka_output/final',
                       help='Matryoshka checkpoint路径（用于检查保存/加载）')

    args = parser.parse_args()

    print("="*80)
    print("Matryoshka SVD 全流程诊断工具")
    print("="*80)

    results = {}

    # 1. 检查SVD-LLM模型加载
    results['svdllm_loading'] = check_svdllm_loading(args.svd_model)

    # 2. 检查MatryoshkaSVDLayer实现
    results['matryoshka_layer'] = check_matryoshka_layer_creation()

    # 3. 检查训练脚本
    results['training_script'] = check_training_script()

    # 4. 检查评测脚本
    results['evaluation_script'] = check_evaluation_script()

    # 5. 检查模型保存（如果checkpoint存在）
    results['model_saving'] = check_model_saving(args.checkpoint)

    # 6. 检查模型加载（如果checkpoint存在）
    results['model_loading'] = check_model_loading(args.checkpoint)

    # 生成建议
    generate_recommendations(results)

    # 返回是否所有检查都通过
    return 0 if all(v for v in results.values() if v is not None) else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
