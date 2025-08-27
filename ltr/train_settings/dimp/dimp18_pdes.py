import os
import torch.nn as nn
import torch.optim as optim
from ltr.dataset import PdesDataset, Got10k, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
import ltr.models.loss as ltr_losses
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU


def run(settings):
    settings.description = 'DiMP18 training on PdesDataset (Pedestrian tracking) with Mac M3 optimization.'
    
    # Mac M3 优化设置
    settings.batch_size = 8  # 减小批次大小适应Mac M3内存
    settings.num_workers = 0  # 使用单进程避免多进程问题
    settings.multi_gpu = False
    settings.print_interval = 1
    
    # 标准化参数
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    
    # DiMP网络参数
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05

    # 数据集设置
    # 主要数据集：PdesDataset（行人数据集）
    pdes_train = PdesDataset(settings.env.pedestrain_dir, split='train')
    
    # 辅助数据集：仅在路径设置时加载
    datasets_train = [pdes_train]
    dataset_weights = [1.0]
    
    try:
        if hasattr(settings.env, 'got10k_dir') and settings.env.got10k_dir and os.path.exists(settings.env.got10k_dir):
            got10k_train = Got10k(settings.env.got10k_dir, split='vottrain', data_fraction=0.1)
            datasets_train.append(got10k_train)
            dataset_weights.append(0.3)
    except Exception as e:
        print(f"⚠️  跳过GOT-10k数据集: {e}")
    
    try:
        if hasattr(settings.env, 'coco_dir') and settings.env.coco_dir and os.path.exists(settings.env.coco_dir):
            coco_train = MSCOCOSeq(settings.env.coco_dir, data_fraction=0.05)
            datasets_train.append(coco_train)
            dataset_weights.append(0.3)
    except Exception as e:
        print(f"⚠️  跳过COCO数据集: {e}")

    # 验证数据集
    pdes_val = PdesDataset(settings.env.pedestrain_dir, split='val')
    if len(pdes_val) == 0:  # 如果没有验证集，使用部分训练数据
        pdes_val = PdesDataset(settings.env.pedestrain_dir, data_fraction=0.2)

    # 数据变换
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # DiMP处理参数
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 8, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    
    data_processing_train = processing.DiMPProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

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
    # 重点关注行人数据，辅以少量通用数据
    dataset_train = sampler.DiMPSampler(datasets_train, 
                                        dataset_weights,  # 动态权重
                                        samples_per_epoch=100,  # 大幅减少样本数用于快速测试
                                        max_gap=10,  # 减少帧间隔
                                        num_test_frames=2,  # 减少测试帧数
                                        num_train_frames=2,  # 减少训练帧数
                                        processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, 
                             batch_size=settings.batch_size, 
                             num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # 验证采样器和加载器
    dataset_val = sampler.DiMPSampler([pdes_val], [1], 
                                      samples_per_epoch=1000,  # 减少验证样本
                                      max_gap=30,
                                      num_test_frames=3, 
                                      num_train_frames=3,
                                      processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, 
                           batch_size=settings.batch_size, 
                           num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, 
                           epoch_interval=5, stack_dim=1)

    # 创建网络和actor
    net = dimpnet.dimpnet18(filter_size=settings.target_filter_sz, 
                            backbone_pretrained=True, 
                            optim_iter=5,
                            clf_feat_norm=True, 
                            final_conv=True, 
                            optim_init_step=0.9, 
                            optim_init_reg=0.1,
                            init_gauss_sigma=output_sigma * settings.feature_sz, 
                            num_dist_bins=100,
                            bin_displacement=0.1, 
                            mask_init_factor=3.0, 
                            target_mask_act='sigmoid', 
                            score_act='relu')

    # 多GPU支持（Mac M3上通常不需要）
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    # 损失函数
    objective = {'iou': nn.MSELoss(), 'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold)}

    # 损失权重（针对行人跟踪稍作调整）
    loss_weight = {'iou': 1, 'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400}

    actor = actors.DiMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # 优化器（Mac M3适配的学习率）
    optimizer = optim.Adam([{'params': actor.net.classifier.filter_initializer.parameters(), 'lr': 3e-5},
                            {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': 3e-4},
                            {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': 3e-5},
                            {'params': actor.net.bb_regressor.parameters(), 'lr': 5e-4},
                            {'params': actor.net.feature_extractor.parameters()}],
                           lr=1e-4)  # 稍微降低基础学习率

    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.2)

    # 训练器
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # 开始训练（减少epoch数适应Mac M3）
    trainer.train(30, load_latest=True, fail_safe=True)
