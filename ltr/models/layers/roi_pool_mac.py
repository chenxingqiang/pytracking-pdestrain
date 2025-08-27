"""
Mac M3兼容的RoI池化实现
替代PreciseRoIPooling，使用PyTorch内置的RoIPool
"""

import torch
import torch.nn as nn
from torchvision.ops import roi_pool


class MacCompatibleRoIPool(nn.Module):
    """Mac M3兼容的RoI池化层，使用PyTorch内置的roi_pool替代PreciseRoIPooling"""
    
    def __init__(self, output_size, spatial_scale):
        """
        Args:
            output_size: 输出尺寸 (height, width)
            spatial_scale: 空间缩放因子
        """
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.spatial_scale = spatial_scale
        
    def forward(self, features, rois):
        """
        Args:
            features: 输入特征图 [N, C, H, W]
            rois: RoI坐标 [K, 5], 格式为 [batch_index, x1, y1, x2, y2]
        
        Returns:
            池化后的特征 [K, C, output_height, output_width]
        """
        # 使用torchvision的roi_pool
        return roi_pool(
            input=features,
            boxes=rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale
        )


class PrRoIPool2D(MacCompatibleRoIPool):
    """PreciseRoIPooling的Mac兼容替代实现"""
    
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        """
        兼容原始PrRoIPool2D接口
        
        Args:
            pooled_height: 池化后高度
            pooled_width: 池化后宽度  
            spatial_scale: 空间缩放因子
        """
        super().__init__(
            output_size=(pooled_height, pooled_width),
            spatial_scale=spatial_scale
        )
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        
    def forward(self, features, rois):
        """
        保持与原始PrRoIPool2D相同的接口
        
        Args:
            features: 输入特征图 [N, C, H, W]
            rois: RoI坐标 [K, 5], 格式为 [batch_index, x1, y1, x2, y2]
        
        Returns:
            池化后的特征 [K, C, pooled_height, pooled_width]
        """
        return super().forward(features, rois)
