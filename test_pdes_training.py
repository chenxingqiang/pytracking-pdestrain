#!/usr/bin/env python3
"""
简化的PdesDataset训练测试脚本
用于验证数据集加载和基本训练流程是否正常
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('/Users/xingqiangchen/TASK/pytracking-pdestrain')

def test_dataset_loading():
    """测试数据集加载"""
    print("🧪 测试数据集加载...")
    
    try:
        from ltr.dataset.pdesdataset import PdesDataset
        
        # 创建数据集实例
        dataset = PdesDataset()
        print(f"✅ 数据集创建成功，包含 {len(dataset.sequence_list)} 个序列")
        
        # 测试获取第一个序列
        if len(dataset.sequence_list) > 0:
            seq_info = dataset.get_sequence_info(0)
            print(f"✅ 第一个序列信息获取成功: {dataset.sequence_list[0]}")
            
            if seq_info['bbox'] is not None:
                print(f"   - 边界框数量: {len(seq_info['bbox'])}")
                print(f"   - 有效帧数: {seq_info['valid'].sum().item()}")
                print(f"   - 可见帧数: {seq_info['visible'].sum().item()}")
            else:
                print("   - 警告: 未找到边界框标注")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n🧪 测试数据加载器...")
    
    try:
        from ltr.dataset.pdesdataset import PdesDataset
        from ltr.data import processing, sampler, LTRLoader
        import ltr.data.transforms as tfm
        
        # 创建数据集
        dataset = PdesDataset()  # 使用全部有效数据进行测试
        
        # 创建数据变换
        transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                       tfm.RandomHorizontalFlip(0.5),
                                       tfm.Normalize(mean=[0.485, 0.456, 0.406], 
                                                   std=[0.229, 0.224, 0.225]))
        
        # 创建数据处理
        data_processing = processing.DiMPProcessing(search_area_factor=5.0,
                                                   output_sz=288,
                                                   center_jitter_factor={'train': 3, 'test': 4.5},
                                                   scale_jitter_factor={'train': 0.25, 'test': 0.5},
                                                   crop_type='replicate',
                                                   max_scale_change=1.5,
                                                   mode='sequence',
                                                   transform=transform_train)
        
        # 创建采样器
        dataset_sampler = sampler.DiMPSampler([dataset], [1.0],
                                            samples_per_epoch=10,  # 减少样本数用于测试
                                            max_gap=30,
                                            num_test_frames=3,
                                            num_train_frames=3,
                                            processing=data_processing)
        
        # 创建数据加载器
        loader = LTRLoader('train', dataset_sampler, 
                          training=True,
                          batch_size=2,  # 小批次
                          num_workers=0,  # 单进程
                          drop_last=True,
                          stack_dim=1)
        
        print("✅ 数据加载器创建成功")
        
        # 尝试加载一个批次
        print("🧪 测试加载一个数据批次...")
        data_iter = iter(loader)
        batch = next(data_iter)
        
        print("✅ 数据批次加载成功")
        print(f"   - 训练图像形状: {batch['train_images'].shape}")
        print(f"   - 测试图像形状: {batch['test_images'].shape}")
        print(f"   - 训练标注形状: {batch['train_anno'].shape}")
        print(f"   - 测试标注形状: {batch['test_anno'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_compatibility():
    """测试设备兼容性"""
    print("\n🧪 测试设备兼容性...")
    
    try:
        # 检查MPS可用性
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✅ MPS设备可用")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("✅ CUDA设备可用")
        else:
            device = torch.device('cpu')
            print("✅ 使用CPU设备")
        
        # 测试张量操作
        x = torch.randn(2, 3, 224, 224).to(device)
        y = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
        z = y(x)
        
        print(f"✅ 设备张量操作测试成功: {device}")
        print(f"   - 输入形状: {x.shape}")
        print(f"   - 输出形状: {z.shape}")
        
        return True, device
        
    except Exception as e:
        print(f"❌ 设备兼容性测试失败: {e}")
        return False, torch.device('cpu')

def main():
    """主测试函数"""
    print("🚀 开始PdesDataset训练兼容性测试")
    print("=" * 50)
    
    # 测试1: 数据集加载
    success1 = test_dataset_loading()
    
    # 测试2: 数据加载器
    success2 = test_data_loader()
    
    # 测试3: 设备兼容性
    success3, device = test_device_compatibility()
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"   - 数据集加载: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"   - 数据加载器: {'✅ 通过' if success2 else '❌ 失败'}")
    print(f"   - 设备兼容性: {'✅ 通过' if success3 else '❌ 失败'}")
    
    if success1 and success2 and success3:
        print("\n🎉 所有测试通过！可以开始正式训练。")
        print(f"   推荐使用设备: {device}")
        return True
    else:
        print("\n⚠️  部分测试失败，需要进一步调试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
