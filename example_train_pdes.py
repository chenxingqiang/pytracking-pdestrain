#!/usr/bin/env python3
"""
使用新的PdesDataset进行训练的示例脚本
Mac M3 兼容，支持MPS加速
"""

import torch
import torch.nn as nn
import torch.optim as optim
from ltr.dataset import PdesDataset
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss as ltr_losses
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import ltr.admin.settings as ws_settings


def create_pdes_dimp18_training():
    """创建使用PdesDataset的DiMP18训练配置"""
    
    # 创建设置对象
    settings = ws_settings.Settings()
    settings.module_name = 'dimp'
    settings.script_name = 'dimp18_pdes'
    settings.project_path = 'ltr/dimp/dimp18_pdes'
    
    # 基本设置
    settings.description = 'DiMP-18 with ResNet18 backbone trained on PdesDataset (Mac M3 optimized).'
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    
    # Mac M3 优化设置
    # 检测设备并调整批处理大小
    if torch.backends.mps.is_available():
        print("检测到MPS设备，使用Mac M3优化配置")
        settings.batch_size = 8  # Mac M3内存优化
        settings.num_workers = 4  # Mac M3 CPU核心优化
        print(f"批处理大小: {settings.batch_size}")
        print(f"工作进程数: {settings.num_workers}")
    else:
        settings.batch_size = 26  # 原始设置
        settings.num_workers = 8
    
    # 创建数据集
    try:
        pdes_train = PdesDataset(settings.env.pedestrain_dir)
        print(f"成功加载PdesDataset，序列数量: {len(pdes_train.sequence_list)}")
        
        # 数据划分 - 80%训练，20%验证
        train_frac = 0.8
        pdes_train_len = int(len(pdes_train) * train_frac)
        pdes_val_len = len(pdes_train) - pdes_train_len
        
        # 分割序列ID
        train_seq_ids = list(range(0, pdes_train_len))
        val_seq_ids = list(range(pdes_train_len, len(pdes_train)))
        
        pdes_train = PdesDataset(settings.env.pedestrain_dir, seq_ids=train_seq_ids)
        pdes_val = PdesDataset(settings.env.pedestrain_dir, seq_ids=val_seq_ids)
        
        print(f"训练序列: {len(pdes_train.sequence_list)}")
        print(f"验证序列: {len(pdes_val.sequence_list)}")
        
    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("请确保在 ltr/admin/local.py 中正确配置了 pedestrain_dir 路径")
        return None
    
    # 数据变换
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))
    
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    
    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    
    # 处理参数
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 8, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    
    # 训练数据处理
    data_processing_train = processing.DiMPProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)
    
    # 验证数据处理
    data_processing_val = processing.DiMPProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=proposal_params,
                                                    label_function_params=label_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)
    
    # 训练采样器和加载器
    samples_per_epoch = 5000 if len(pdes_train) > 0 else 0
    dataset_train = sampler.DiMPSampler([pdes_train], [1],
                                      samples_per_epoch=samples_per_epoch, max_gap=30, num_test_frames=3, num_train_frames=3,
                                      processing=data_processing_train)
    
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)
    
    # 验证采样器和加载器
    samples_per_epoch_val = 1000 if len(pdes_val) > 0 else 0
    dataset_val = sampler.DiMPSampler([pdes_val], [1], 
                                     samples_per_epoch=samples_per_epoch_val, max_gap=30, num_test_frames=3, num_train_frames=3,
                                     processing=data_processing_val)
    
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)
    
    # 创建网络
    net = dimpnet.dimpnet18(filter_size=settings.target_filter_sz, backbone_pretrained=True, optim_iter=5,
                            clf_feat_norm=True, final_conv=True, optim_init_step=0.9, optim_init_reg=0.1,
                            init_gauss_sigma=output_sigma * settings.feature_sz, num_dist_bins=100,
                            bin_displacement=0.1, mask_init_factor=3.0, target_mask_act='sigmoid', score_act='relu')
    
    # 多GPU包装（通常在Mac M3上不需要）
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)
    
    # 损失函数
    objective = {'iou': nn.MSELoss(), 'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold)}
    loss_weight = {'iou': 1, 'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400}
    
    # 创建actor
    actor = actors.DiMPActor(net=net, objective=objective, loss_weight=loss_weight)
    
    # 优化器
    optimizer = optim.Adam([{'params': actor.net.classifier.filter_initializer.parameters(), 'lr': 5e-5},
                            {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': 5e-4},
                            {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': 5e-5},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 1e-3},
                            {'params': actor.net.feature_extractor.parameters()}],
                           lr=2e-4)
    
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    
    # 创建训练器
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    
    return trainer, settings


def main():
    """主函数"""
    print("=" * 60)
    print("Mac M3 PyTracking-PDESTrain 训练示例")
    print("=" * 60)
    
    # 检查设备
    if torch.backends.mps.is_available():
        print("✅ 检测到MPS设备，将使用Mac M3加速")
    elif torch.cuda.is_available():
        print("✅ 检测到CUDA设备")
    else:
        print("⚠️  将使用CPU训练")
    
    # 创建训练配置
    try:
        trainer, settings = create_pdes_dimp18_training()
        if trainer is None:
            print("❌ 训练配置创建失败")
            return
        
        print("✅ 训练配置创建成功")
        print(f"   描述: {settings.description}")
        print(f"   批处理大小: {settings.batch_size}")
        print(f"   工作进程数: {settings.num_workers}")
        
        # 开始训练（示例：只训练5个epoch）
        print("\n开始训练...")
        print("注意：这只是一个示例，实际训练请根据需要调整epoch数量")
        
        # 这里可以开始实际训练
        # trainer.train(5, load_latest=True, fail_safe=True)
        
        print("✅ 训练配置验证完成")
        print("\n要开始实际训练，请取消注释 trainer.train() 行")
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
