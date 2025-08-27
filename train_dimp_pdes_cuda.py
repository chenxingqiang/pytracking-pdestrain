#!/usr/bin/env python3
"""
CUDA兼容的DiMP行人跟踪训练启动脚本
基于Mac M3兼容版本，增加了CUDA环境支持
"""

import argparse
import os
import sys
import torch

def check_cuda_environment():
    """检查CUDA环境"""
    print("🔍 检查CUDA环境...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请检查CUDA安装")
        return False
    
    print(f"✅ CUDA可用，版本: {torch.version.cuda}")
    print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='CUDA兼容的DiMP行人跟踪训练')
    parser.add_argument('--model', choices=['dimp18', 'dimp50'], default='dimp18',
                       help='模型变体 (默认: dimp18)')
    parser.add_argument('--config', default=None,
                       help='自定义配置文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='从检查点恢复训练')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 DiMP CUDA兼容训练启动")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"配置: {args.config or f'{args.model}_pdes_cuda'}")
    print(f"恢复训练: {'是' if args.resume else '否'}")
    print(f"调试模式: {'是' if args.debug else '否'}")
    print("=" * 60)
    
    # 检查CUDA环境
    if not check_cuda_environment():
        sys.exit(1)
    
    # 检查环境设置
    print("\n🔍 检查环境设置...")
    pedestrain_dir = os.environ.get('PEDESTRAIN_DIR', '/Users/xingqiangchen/TASK/pytracking-pdestrain/pedestrain')
    if os.path.exists(pedestrain_dir):
        print(f"   ✅ 行人数据集路径: {pedestrain_dir}")
    else:
        print(f"   ⚠️  行人数据集路径不存在: {pedestrain_dir}")
    
    # 设置CUDA设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"   ✅ 使用CUDA设备: {device}")
        print(f"   ✅ 当前GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   ❌ CUDA不可用")
        sys.exit(1)
    
    # 启动训练
    print(f"\n🚀 开始训练 {args.model}_pdes_cuda...")
    
    try:
        # 导入训练模块
        from ltr.run_training import run_training
        
        # 确定训练配置
        if args.config:
            train_module = 'dimp'
            train_name = args.config
        else:
            train_module = 'dimp'
            train_name = f'{args.model}_pdes_cuda'
        
        # 运行训练
        run_training(train_module, train_name)
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 训练错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
