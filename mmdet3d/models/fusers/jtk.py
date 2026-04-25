from typing import List, Tuple

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

import torch.nn.functional as F

from .fuser_blocks import ImprovedMSNetFusion


@FUSERS.register_module()
class JTK(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channel = 256
        in_channels = [80, 256]

        self.cam_conv = nn.Conv2d(in_channels[0], mid_channel, 3, 1, 1)
        self.lidar_conv = nn.Conv2d(in_channels[1], mid_channel, 3, 1, 1)
        self.align_block = JointTaskFusion(mid_channel)
        self.out_conv = nn.Sequential(nn.Conv2d(mid_channel, out_channels, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True))
        self.task_vector = torch.tensor([[0., 1.]])

        # self.sensor_mask_prob = 0.0
        # self.sensor_mask = 'random'

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        feat_channels = [0] + [x.shape[1] for x in inputs]  # [0, 80, 256]  
        feat_channels = torch.cumsum(torch.tensor(feat_channels), dim=0)  
        x_mm = torch.cat(inputs, dim=1)

        # if self.sensor_mask_prob > 0.0:
        #     sensor_mask = torch.rand(1, device=x_mm.device) < self.sensor_mask_prob  # 决定是否应用掩码
        #     # condition = x_t.clone()
        #     if sensor_mask:  # 应用掩码
        #         if self.sensor_mask == 'random':
        #             idx = torch.randint(1, 3, (1,), device=x_mm.device).long()  # 随机选择传感器（1或2）
        #         elif self.sensor_mask == 'lidar':
        #             idx = torch.tensor([0], device=x_mm.device).long()  # 屏蔽 LiDAR
        #         elif self.sensor_mask == 'camera':
        #             idx = torch.tensor([1], device=x_mm.device).long()  # 屏蔽 Camera
        #         if not (self.sensor_mask == 'random' and not self.training):
        #             # 根据通道索引置零特定传感器特征
        #             x_mm[:, feat_channels[idx-1]:feat_channels[idx], :, :] = x_mm[:, feat_channels[idx-1]:feat_channels[idx], :, :] * 0.0

        x_cam = self.cam_conv(x_mm[:, :feat_channels[1]]) 
        x_lidar = self.lidar_conv(x_mm[:, feat_channels[1]:])

        # visualize_single_bev_feature(x_cam, channel_mode='max', save_path='./vis_cam.jpg')
        # visualize_single_bev_feature(x_lidar, channel_mode='max', save_path='./vis_lidar.jpg')
        
        task_vec = self.task_vector.repeat(x_cam.size(0), 1).cuda() 
        fused_bev = self.align_block(x_cam, x_lidar, task_vector=task_vec)

        # visualize_single_bev_feature(fused_bev, channel_mode='max', save_path='./vis_fused.jpg')

        fused_bev = self.out_conv(fused_bev)

        return fused_bev
    

from einops import rearrange  
class JointTaskFusion(nn.Module):
    def __init__(self, in_channels, task_channels=32, num_heads=8):
        """
        联合任务融合模块 (目标检测+BEV地图分割并行)
        Args:
            in_channels: 输入特征通道数
            task_channels: 任务嵌入维度
            num_heads: CIT注意力头数
        """
        super().__init__()
        
        # 1. 任务感知特征解耦层 (共享)
        self.geo_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)  # 几何增强
        self.sem_se = nn.Sequential(  # 语义增强
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 2. 非对称CIT交互模块 (双路独立)
        self.cit_det = CrossModalInteraction(in_channels, num_heads, mode='lidar_guided')
        self.cit_seg = CrossModalInteraction(in_channels, num_heads, mode='camera_guided')
        
        # 3. 联合任务门控
        self.task_embed = nn.Linear(2, task_channels)  # 输入为双任务激活状态
        self.gate_generator = nn.Sequential(
            nn.Linear(task_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 输出4个权重[w_det_geo, w_det_sem, w_seg_geo, w_seg_sem]
        )

        self.out = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, 1, 1, bias=False),
                                 nn.BatchNorm2d(in_channels),
                                 nn.ReLU(inplace=True))

    def forward(self, camera_feat, lidar_feat, task_vector=None):
        """
        Args:
            camera_feat: [B, C, H, W] 相机BEV特征
            lidar_feat:  [B, C, H, W] LiDAR BEV特征
            task_vector: [B, 2] 任务激活向量 
                        [1,0]: 仅检测, [0,1]: 仅分割, [1,1]: 双任务
        Returns:
            det_feat: [B, C, H, W] 检测任务特征
            seg_feat: [B, C, H, W] 分割任务特征
        """
        # 任务向量示例: 
        #   task_vector = torch.tensor([[1., 0.]])  # 仅检测
        #   task_vector = torch.tensor([[0., 1.]])  # 仅分割
        #   task_vector = torch.tensor([[1., 1.]])  # 双任务
        
        # 1. 特征解耦 (共享计算)
        lidar_geo = self.geo_conv(lidar_feat)
        camera_sem = camera_feat * self.sem_se(camera_feat)
        
        # 2. 跨模态交互 (双路并行)
        feat_det_geo, feat_det_sem = self.cit_det(lidar_geo, camera_sem)  # 检测路径
        feat_seg_geo, feat_seg_sem = self.cit_seg(lidar_geo, camera_sem)  # 分割路径


        # 3. 联合任务门控
        task_emb = self.task_embed(task_vector)  # [B, task_channels]
        weights = self.gate_generator(task_emb)  # [B, 4]
        w_det_geo, w_det_sem, w_seg_geo, w_seg_sem = weights.chunk(4, dim=1)

        save_weights(w_det_geo[0][0].cpu().numpy(), w_det_sem[0][0].cpu().numpy(), w_seg_geo[0][0].cpu().numpy(), w_seg_sem[0][0].cpu().numpy())
        
        # 扩展权重到空间维度
        w_det_geo = w_det_geo.view(-1, 1, 1, 1)
        w_det_sem = w_det_sem.view(-1, 1, 1, 1)
        w_seg_geo = w_seg_geo.view(-1, 1, 1, 1)
        w_seg_sem = w_seg_sem.view(-1, 1, 1, 1)
        
        # 4. 任务自适应融合
        det_feat = w_det_geo * feat_det_geo + w_det_sem * feat_det_sem
        seg_feat = w_seg_geo * feat_seg_geo + w_seg_sem * feat_seg_sem
        # det_feat = feat_det_geo + feat_det_sem
        # seg_feat = feat_seg_geo + feat_seg_sem

        fused_bev = self.out(torch.cat([det_feat, seg_feat], 1))
        
        return fused_bev

import os
def save_weights(w_det_geo, w_det_sem, w_seg_geo, w_seg_sem):
    file = os.path.join('weights.txt')
    with open(file, 'a') as f:
        f.write(f"{w_det_geo:.5f} | {w_det_sem:.5f} | {w_seg_geo:.5f} | {w_seg_sem:.5f}\n")


class CrossModalInteraction(nn.Module):
    """轻量化的跨模态交互模块（使用CAB重构）"""
    def __init__(self, dim, num_heads, mode='lidar_guided'):
        super().__init__()
        self.mode = mode
        
        # 使用轻量化的交叉注意力块替代原始投影层
        self.cab = CAB(
            dim=dim,
            num_heads=num_heads,
            bias=True
        )
        
        # 保留模式特定的缩放因子 (可选)
        if mode == 'lidar_guided':
            self.lambda_scale = nn.Parameter(torch.tensor(1.2))
        else:
            self.lambda_scale = 1.0

    def forward(self, lidar_feat, camera_feat):
        # CAB直接处理特征图，无需展平操作
        if self.mode == 'lidar_guided':
            out = self.cab(lidar_feat, camera_feat) + lidar_feat * self.lambda_scale
            return out, camera_feat
        else:  # camera_guided
            out = self.cab(camera_feat, lidar_feat) + camera_feat * self.lambda_scale
            return lidar_feat, out

  
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        # 调用父类的构造函数
        super(CAB, self).__init__()
        # 注意力头的数量
        self.num_heads = num_heads
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 查询卷积层
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 查询的深度可分离卷积层
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 键值卷积层
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # 键值的深度可分离卷积层
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # 获取输入特征图的形状
        b, c, h, w = x.shape
        # 计算查询
        q = self.q_dwconv(self.q(x))

        # 计算键值
        kv = self.kv_dwconv(self.kv(y))
        # 将键值在通道维度上拆分为键和值
        k, v = kv.chunk(2, dim=1)

        # 对查询、键和值进行维度重排
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对查询和键进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 对注意力分数进行 softmax 操作
        attn = nn.functional.softmax(attn, dim=-1)
        # 计算注意力输出
        out = (attn @ v)
        # 对注意力输出进行维度重排
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 输出投影
        out = self.project_out(out)
        return out
    

class AdaptiveModule(nn.Module):
    """自适应调整模块 类似SE模块"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DDF(nn.Module):
    """Dual Dynamic Fusion模块"""
    def __init__(self, in_channels):
        super().__init__()
        # 动态权重生成部分
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        
        # 3x3卷积融合
        self.conv = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自适应调整
        self.adaptive = AdaptiveModule(in_channels)
    
    def forward(self, camera_feat, lidar_feat):
        """
        输入:
            camera_feat: [B, C, H, W] 相机BEV特征
            lidar_feat:  [B, C, H, W] LiDAR BEV特征
        输出:
            fused_feat:  [B, C, H, W] 融合后的特征
        """
        # 特征相加
        combined = camera_feat + lidar_feat

        # 生成动态权重 [B, C]
        weights = self.weight_generator(combined)  # [B, C]
        
        # 维度调整用于广播 [B, C, 1, 1]
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        
        # 加权融合
        weighted_cam = camera_feat * weights
        weighted_lidar = lidar_feat * (1 - weights)
        
        # 拼接特征 [B, 2C, H, W]
        concat_feat = torch.cat([weighted_cam, weighted_lidar], dim=1)
        
        # 3x3卷积融合
        fused_feat = self.conv(concat_feat)
        
        # 自适应调整
        fused_feat = self.adaptive(fused_feat)
        
        return fused_feat
    



#########################################################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def visualize_single_bev_feature(feature_map,
                                 channel_mode='mean',
                                 channel_idx=None,
                                 save_path=None,
                                 dpi=100,
                                 color_range='auto'):
    """
    可视化单个BEV特征激活热力图 (PyTorch tensor输入)

    参数:
        feature_map: 输入特征图, 形状为[B, C, H, W]
        channel_mode: 通道处理方式 ('mean'取均值 | 'max'取最大值 | 'sum'求和 | 'select'选择特定通道)
        channel_idx: 当channel_mode='select'时指定的通道索引
        save_path: 图片保存路径 (None则不保存)
        dpi: 输出图像分辨率
        color_range: 颜色范围 ('auto'自动归一化 | tuple(min, max)指定范围)

    返回:
        matplotlib图像对象
    """
    # ===== 1. 输入验证 =====
    assert len(feature_map.shape) == 4, "输入必须是4D张量 [B, C, H, W]"
    assert channel_mode in ['mean', 'max', 'sum', 'select'], "无效的通道处理方式"

    if channel_mode == 'select':
        assert channel_idx is not None, "选择通道模式必须指定channel_idx"
        assert 0 <= channel_idx < feature_map.shape[1], f"通道索引超出范围 [0, {feature_map.shape[1] - 1}]"

    # ===== 2. 通道处理 =====
    sample_idx = 0  # 默认使用batch中第一个样本
    with torch.no_grad():
        if channel_mode == 'mean':
            feat = torch.mean(feature_map[sample_idx], dim=0)  # [H, W]
        elif channel_mode == 'max':
            feat, _ = torch.max(feature_map[sample_idx], dim=0)  # [H, W]
        elif channel_mode == 'sum':
            feat = torch.sum(feature_map[sample_idx], dim=0)  # [H, W]
        else:  # select
            feat = feature_map[sample_idx, channel_idx]  # [H, W]

        # 转换到CPU并转为numpy
        feat = feat.cpu().numpy()

    # ===== 3. 数据归一化 =====
    if color_range == 'auto':
        vmin, vmax = np.min(feat), np.max(feat)
    else:
        vmin, vmax = color_range

    # 归一化到[0,1]范围 (保留原始数值关系)
    normalized_feat = (feat - vmin) / (vmax - vmin + 1e-8)

    # ===== 4. 创建专业蓝白红色彩映射 =====
    colors = [
        "#00008B", "#1E90FF", "#00BFFF", "#87CEFA",  # 深蓝到浅蓝
        "#F0F8FF", "#FFFFFF", "#FFFFE0",  # 中性白到淡黄
        "#FFD700", "#FFA500", "#FF6347", "#FF0000"  # 黄到亮红
    ]
    cmap = LinearSegmentedColormap.from_list("bev_feature", colors, N=256)

    # ===== 5. 创建专业可视化 =====
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi, facecolor='white')

    # 绘制热力图
    im = ax.imshow(normalized_feat,
                   cmap=cmap,
                   vmin=0,
                   vmax=1)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Level', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # 设置坐标轴
    ax.set_xticks(np.linspace(0, feat.shape[1], 5))
    ax.set_yticks(np.linspace(0, feat.shape[0], 5))
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('BEV X-axis', fontsize=12)
    ax.set_ylabel('BEV Y-axis', fontsize=12)

    # 添加标题
    title = f"BEV Feature Map - {channel_mode.capitalize()}"
    if channel_mode == 'select':
        title += f" (Ch:{channel_idx})"
    ax.set_title(title, fontsize=14, pad=20)

    # ===== 6. 保存或展示 =====
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi, transparent=False)
        print(f"图像已保存至: {save_path}")

    # plt.show()
    plt.close()
    return fig