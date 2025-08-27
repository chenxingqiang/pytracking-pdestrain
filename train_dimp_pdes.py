#!/usr/bin/env python3
"""
DiMP在PdesDataset上的训练脚本
针对Mac M3优化的行人跟踪模型训练
"""

import sys
import os
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Train DiMP on Pedestrian Dataset')
    parser.add_argument('--model', type=str, default='dimp18', 
                       choices=['dimp18', 'dimp50'],
                       help='DiMP model variant (default: dimp18)')
    parser.add_argument('--config', type=str, default='pdes',
                       help='Training configuration (default: pdes)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 构建配置名称
    config_name = f"{args.model}_{args.config}"
    
    print("=" * 60)
    print("🚀 DiMP Pedestrian Tracking Training")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"配置: {config_name}")
    print(f"恢复训练: {'是' if args.resume else '否'}")
    print(f"调试模式: {'是' if args.debug else '否'}")
    print("=" * 60)
    
    # 检查环境
    print("🔍 检查环境设置...")
    try:
        from ltr.admin.environment import env_settings
        env = env_settings()
        
        # 检查数据集路径
        if hasattr(env, 'pedestrain_dir') and env.pedestrain_dir:
            print(f"   ✅ 行人数据集路径: {env.pedestrain_dir}")
            if not os.path.exists(env.pedestrain_dir):
                print(f"   ⚠️  警告: 数据集路径不存在: {env.pedestrain_dir}")
        else:
            print("   ❌ 未设置行人数据集路径 (pedestrain_dir)")
            print("   请在 ltr/admin/local.py 中设置 pedestrain_dir")
            return 1
            
        # 检查其他必要路径
        paths_to_check = [
            ('got10k_dir', 'GOT-10k数据集'),
            ('coco_dir', 'COCO数据集'),
        ]
        
        for attr, desc in paths_to_check:
            if hasattr(env, attr) and getattr(env, attr):
                path = getattr(env, attr)
                status = "存在" if os.path.exists(path) else "不存在"
                print(f"   {desc}: {path} ({status})")
            else:
                print(f"   ⚠️  {desc}: 未设置")
                
    except Exception as e:
        print(f"   ❌ 环境检查失败: {e}")
        return 1
    
    # 检查设备
    print("\n🔍 检查计算设备...")
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("   ✅ 使用MPS设备 (Mac M3优化)")
        elif torch.cuda.is_available():
            print("   ✅ 使用CUDA设备")
        else:
            print("   ⚠️  使用CPU设备")
    except Exception as e:
        print(f"   ❌ 设备检查失败: {e}")
    
    # 导入并运行训练
    print(f"\n🚀 开始训练 {config_name}...")
    try:
        # 动态导入训练配置
        from ltr.run_training import run_training
        
        # 运行训练
        run_training('dimp', config_name)
        
        print("\n🎉 训练完成！")
        print("\n📋 后续步骤:")
        print("1. 检查训练日志和checkpoints")
        print("2. 使用训练好的模型进行测试")
        print("3. 评估模型性能")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
