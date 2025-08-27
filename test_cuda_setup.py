#!/usr/bin/env python3
"""
CUDA兼容性测试脚本
验证CUDA环境、数据集加载和模型兼容性
"""

import os
import sys
import torch
import numpy as np

def test_cuda_environment():
    """测试CUDA环境"""
    print("🔍 测试CUDA环境...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✅ CUDA可用，版本: {torch.version.cuda}")
    print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    # 测试GPU内存
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({total_memory:.1f} GB)")
        
        # 测试GPU计算
        torch.cuda.set_device(i)
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"   ✅ GPU {i} 计算测试通过")
    
    return True

def test_dataset_loading():
    """测试数据集加载"""
    print("\n🔍 测试数据集加载...")
    
    try:
        from ltr.dataset import PdesDataset
        from ltr.data import sampler
        
        # 检查数据集路径
        pedestrain_dir = os.environ.get('PEDESTRAIN_DIR', '/Users/xingqiangchen/TASK/pytracking-pdestrain/pedestrain')
        if not os.path.exists(pedestrain_dir):
            print(f"   ⚠️  行人数据集路径不存在: {pedestrain_dir}")
            return False
        
        print(f"   ✅ 行人数据集路径: {pedestrain_dir}")
        
        # 创建数据集
        dataset = PdesDataset(root=pedestrain_dir)
        print(f"   ✅ 数据集创建成功，序列数量: {len(dataset.sequence_list)}")
        
        # 测试数据加载
        if len(dataset.sequence_list) > 0:
            seq_info = dataset.get_sequence_info(0)
            print(f"   ✅ 序列信息获取成功: {seq_info}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 数据集测试失败: {e}")
        return False

def test_model_compatibility():
    """测试模型兼容性"""
    print("\n🔍 测试模型兼容性...")
    
    try:
        # 测试RoI池化
        try:
            from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
            print("   ✅ PreciseRoIPooling导入成功")
        except ImportError:
            from ltr.models.layers.roi_pool_mac import PrRoIPool2D
            print("   ✅ Mac兼容RoI池化导入成功")
        
        # 测试基本模型组件
        roi_pool = PrRoIPool2D(3, 3, 1.0/16.0)
        features = torch.randn(1, 256, 32, 32)
        rois = torch.tensor([[0, 10, 10, 20, 20]], dtype=torch.float32)
        
        if torch.cuda.is_available():
            roi_pool = roi_pool.cuda()
            features = features.cuda()
            rois = rois.cuda()
        
        output = roi_pool(features, rois)
        print(f"   ✅ RoI池化测试成功，输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模型兼容性测试失败: {e}")
        return False

def test_training_config():
    """测试训练配置"""
    print("\n🔍 测试训练配置...")
    
    try:
        # 测试CUDA训练配置
        from ltr.train_settings.dimp import dimp18_pdes_cuda, dimp50_pdes_cuda
        
        print("   ✅ CUDA训练配置导入成功")
        
        # 创建模拟设置对象
        class MockSettings:
            def __init__(self):
                self.batch_size = 8
                self.num_workers = 0
                self.multi_gpu = False
        
        settings = MockSettings()
        
        # 测试配置函数
        dimp18_pdes_cuda.run(settings)
        print(f"   ✅ DiMP18 CUDA配置测试成功")
        print(f"      批次大小: {settings.batch_size}")
        print(f"      工作器数量: {settings.num_workers}")
        print(f"      多GPU: {settings.multi_gpu}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 训练配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 CUDA兼容性测试")
    print("=" * 60)
    
    tests = [
        ("CUDA环境", test_cuda_environment),
        ("数据集加载", test_dataset_loading),
        ("模型兼容性", test_model_compatibility),
        ("训练配置", test_training_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！CUDA环境配置成功！")
        print("\n📋 下一步:")
        print("1. 运行训练: python train_dimp_pdes_cuda.py --model dimp18")
        print("2. 检查GPU状态: nvidia-smi")
        print("3. 监控训练进度")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查配置")
        print("\n🔧 故障排除建议:")
        print("- 检查CUDA驱动安装")
        print("- 验证PyTorch CUDA版本")
        print("- 确认数据集路径正确")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
