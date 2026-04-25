import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from torchvision.ops import DeformConv2d

import matplotlib.pyplot as plt
import numpy as np

import os

from matplotlib.colors import LightSource


class DoubleInputSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.seq = nn.Sequential(*args)

    def forward(self, x, c):
        for module in self.seq:
            x, c = module(x, c)
        return x, c


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm=False,
                 bias=False,
                 activation=False,
                 onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        stride=1, padding=0)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.1, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.act = nn.SiLU()
            # self.swish = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.act(x)

        return x


class Modulate(nn.Module):
    """
    Simple example of a 'modulate' block that can incorporate shift, scale, or gating.
    """

    def __init__(self, shift_scale_gate='shift_scale_gate'):
        super(Modulate, self).__init__()
        self.shift_scale_gate = shift_scale_gate

    @force_fp32()
    def forward(self, x, gate, shift, scale):
        if 'scale' in self.shift_scale_gate:
            x = x * (scale + 1)
        if 'shift' in self.shift_scale_gate:
            x = x + shift
        if 'gate' in self.shift_scale_gate:
            x = torch.sigmoid(gate) * x

        return x


class AttentionBlock(nn.Module):
    """
    Custom block that does 'attention-like' or feature fusion logic.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 first_layer=False,
                 shift_scale_gate='shift_scale_gate',
                 **block_kwargs):
        super().__init__()
        self.first_block = first_layer
        self.modulate = Modulate(shift_scale_gate)

        # 首层特殊处理
        if first_layer:
            self.x_down_sample = nn.Sequential(  # 主路径下采样  Conv2d(in→out, stride=stride)
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            )
            self.c_down_sample = nn.Sequential(  # 条件路径下采样  Conv2d(in→out) → BN → SiLU()
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.BatchNorm2d(out_channels, eps=1e-3),
                nn.SiLU(),
            )
        else:
            self.x_down_sample = None
            self.c_down_sample = None

        self.norm1 = nn.BatchNorm2d(out_channels, eps=1e-3)

        # 条件调制模块
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, 3 * out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            # 生成shift, scale, gate参数
        )

        # 主处理块
        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.SiLU(),
        )
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, c):
        # If this is the first block, special downsampling of x and c
        if self.first_block:
            x = self.x_down_sample(x)  # 主路径下采样  [B, 128, Hb/2, Wb/2] | [B, 256, 32, 32]
            c = self.c_down_sample(c)  # 条件路径下采样  [B, 128, 64, 64] | [B, 256, 32, 32]

        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)  # [B, 336, 128, 128]
        h = self.modulate(self.norm1(x), gate_mlp, shift_mlp, scale_mlp)  # 公式: h = gate*(norm(x)*scale + shift)
        h = F.silu(h)
        h = self.block(h)
        x = h + self.conv1(F.silu(x))  # 残差连接
        return x, c


class SeparableConvBlock(nn.Module):
    """
    Depthwise + Pointwise Convolution block, commonly used in lightweight networks.
    """

    def __init__(self, in_channels,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm=False,
                 bias=False,
                 activation=False,
                 onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        stride=1, padding=0)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.1, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.act(x)

        return x


class DoubleInputSequential(nn.Module):
    """
    A small wrapper to sequentially apply blocks that each return (x, c).
    """

    def __init__(self, *args):
        super().__init__()
        self.seq = nn.Sequential(*args)

    def forward(self, x, c):
        for module in self.seq:
            x, c = module(x, c)
        return x, c


##################################

class GuidanceBlock(nn.Module):
    """
    Custom block that does 'attention-like' or feature fusion logic.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 first_layer=False,
                 shift_scale_gate='shift_scale_gate',
                 **block_kwargs):
        super().__init__()
        self.first_block = first_layer
        # self.modulate = Modulate(shift_scale_gate)

        # 首层特殊处理
        if first_layer:
            self.x_down_sample = nn.Sequential(  # 主路径下采样  Conv2d(in→out, stride=stride)
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            )
            self.c_down_sample = nn.Sequential(  # 条件路径下采样  Conv2d(in→out) → BN → SiLU()
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.BatchNorm2d(out_channels, eps=1e-3),
                nn.SiLU(),
            )
        else:
            self.x_down_sample = None
            self.c_down_sample = None

        ##########################
        # 差异提取卷积
        self.conv_sub = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        # 差异增强卷积
        self.conv_diff_enh = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        # MSFF模块
        self.MPFL = MSFF(inchannel=out_channels, mid_channel=64)
        # 混合卷积
        self.convmix = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, groups=out_channels, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        # 降维卷积
        self.conv_dr = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x, c):
        # If this is the first block, special downsampling of x and c
        if self.first_block:
            x = self.x_down_sample(x)  # 主路径下采样  [B, 128, Hb/2, Wb/2] | [B, 256, 32, 32]
            c = self.c_down_sample(c)  # 条件路径下采样  [B, 128, 64, 64] | [B, 256, 32, 32]

        #########################
        # 差异增强
        x_sub = torch.abs(x - c)
        x_att = torch.sigmoid(self.conv_sub(x_sub))
        x1 = (x * x_att) + self.MPFL(self.conv_diff_enh(x))
        x2 = (c * x_att) + self.MPFL(self.conv_diff_enh(c))

        # 融合
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (x1.size(0), -1, x1.size(2), x1.size(3)))
        x_f = self.convmix(x_f)

        # 应用注意力
        x_f = x_f * x_att
        x = self.conv_dr(x_f)

        return x, c


# 多尺度特征融合模块 (MSFF)
class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()
        # 定义不同尺寸的卷积序列
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 混合卷积
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 通过不同的卷积序列
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # 在通道维度上拼接
        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        # 通过混合卷积
        out = self.convmix(x_f)

        return out


class FeatureAlign(nn.Module):
    """可学习特征对齐模块, 参考FlowNetSimple结构"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 3, padding=1)  # 生成x,y偏移量
        )

    def forward(self, x_low, x_high):
        # 计算偏移量 [B,2,H,W]
        offset = self.conv(torch.cat([x_low, x_high], dim=1))

        # 双线性采样对齐
        B, C, H, W = x_high.shape
        grid = self._get_grid(B, H, W, x_high.device)
        warped = F.grid_sample(x_high, grid + offset.permute(0, 2, 3, 1), align_corners=False)
        return warped

    def _get_grid(self, B, H, W, device):
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        grid = torch.meshgrid(x, y, indexing='xy')  # pytorch 1.10+
        grid = torch.stack(grid[::-1], dim=-1)  # [H,W,2]
        return grid.unsqueeze(0).repeat(B, 1, 1, 1).to(device)


class DynamicScaleConv(nn.Module):
    """动态多尺度卷积，结合可变形卷积与通道注意力"""

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 可变形卷积
        offset = torch.zeros(x.shape[0], 2 * 3 * 3, x.shape[2], x.shape[3], device=x.device)
        x_deform = self.deform_conv(x, offset)  # 这里其实就相当于普通卷积，offset置为0

        # 通道注意力加权
        scale = self.se(x_deform)
        return x_deform * scale


class EnhancedSubtractionUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.align = FeatureAlign(in_channels)
        self.scale_convs = nn.ModuleList([
            DynamicScaleConv(in_channels) for _ in range(3)
        ])

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_low, x_high):
        # 特征对齐
        aligned_high = self.align(x_low, x_high)  # [B, 128, 64, 64]

        # 多尺度差异计算
        diff = x_low - aligned_high

        # 动态多尺度处理
        scale_feats = []
        for conv in self.scale_convs:
            scale_feats.append(conv(diff))
        fused = sum(scale_feats)  # 多尺度特征融合

        # 空间注意力加权
        attn = self.spatial_attn(fused)
        return attn * diff + x_low  # 残差连接


class ImprovedMSNetFusion(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.channels = channels

        # ===== 分辨率变换模块 =====
        self.p3_to_p2 = self._build_upsample(channels)
        self.p2_to_p1 = self._build_upsample(channels)
        self.p1_to_p2 = self._build_downsample(channels)
        self.p2_to_p3 = self._build_downsample(channels)

        # ===== 增强减法单元 =====
        self.su_p3 = EnhancedSubtractionUnit(channels)
        self.su_p2 = EnhancedSubtractionUnit(channels)
        self.su_p1 = EnhancedSubtractionUnit(channels)

        # ===== 轻量化融合模块 =====
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels, channels)  # 深度可分离卷积
        )

    def _build_upsample(self, channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(channels, channels)
        )

    def _build_downsample(self, channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(channels, channels)
        )

    def forward(self, p1, p2, p3):
        # 特征传播路径
        p3_up = self.p3_to_p2(p3)  # [B, 128, 64, 64]
        su_p32 = self.su_p3(p2, p3_up)  # [B, 128, 64, 64]

        p32_down = self.p2_to_p3(su_p32)  # [B, 128, 32, 32]
        p32_up = self.p3_to_p2(p32_down)  # [B, 128, 64, 64]
        su_p21 = self.su_p2(p1, self.p2_to_p1(p32_up))  # [B, 128, 128, 128]

        # 跨层级融合
        p1_out = self._fuse_features(su_p21, p1)
        p2_out = self._fuse_features(su_p32, p2)
        p3_out = self._fuse_features(p32_down, p3)

        return p1_out, p2_out, p3_out

    def _fuse_features(self, fused_feat, origin_feat):
        """残差融合模块"""
        return self.fusion_conv(
            torch.cat([fused_feat, origin_feat, origin_feat - fused_feat], dim=1)
        ) + origin_feat  # 残差连接


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，降低计算量"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3,
                                   padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


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
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

        # 3x3卷积融合
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 自适应调整
        self.adaptive = AdaptiveModule(in_channels)

    def forward(self, camera_feat, lidar_feat):
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


def draw_fusion_heatmap(cam_feat, lidar_feat, save_path="fusion_heatmap.png"):
    assert cam_feat.shape == lidar_feat.shape, "输入特征维度必须一致"
    cam_map = cam_feat[0].mean(0).detach().cpu().numpy()  # 通道平均
    lidar_map = lidar_feat[0].mean(0).detach().cpu().numpy()

    def robust_normalize(x):
        q_low, q_high = np.percentile(x, [1, 99])
        return np.clip((x - q_low) / (q_high - q_low), 0, 1)

    cam_norm = robust_normalize(cam_map)
    lidar_norm = robust_normalize(lidar_map)

    plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.labelpad': 15
    })

    plt.imshow(cam_norm, cmap='Reds', alpha=0.6, vmin=0.1, vmax=0.9)
    img = plt.imshow(lidar_norm, cmap='Blues', alpha=0.6, vmin=0.1, vmax=0.9)

    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Activation Intensity (a.u.)",
                   rotation=270, labelpad=25)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=12)

    plt.title("Fused BEV Features (Red=Camera, Blue=LiDAR)",
              fontweight='bold', pad=20)

    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#666666')
    plt.gca().spines['left'].set_color('#666666')

    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_fused_feature(x_fused, save_path="fused_feature_heatmap.png"):
    assert len(x_fused.shape) == 4, "输入必须是4D张量"

    # 多通道特征聚合策略
    if x_fused.shape[1] > 1:  # 多通道特征
        # 通道加权平均（可替换为注意力权重）
        channel_weights = torch.softmax(x_fused.mean(dim=(2, 3)), dim=1)
        feat_map = (x_fused[0] * channel_weights[0].view(-1, 1, 1)).sum(dim=0)
    else:  # 单通道特征
        feat_map = x_fused[0, 0]

    feat_np = feat_map.detach().cpu().numpy()

    vmin, vmax = np.percentile(feat_np, [0.5, 99.5])
    feat_norm = np.clip((feat_np - vmin) / (vmax - vmin), 0, 1)

    plt.figure(figsize=(10, 8), dpi=300)
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlepad': 20,
        'font.family': 'DejaVu Sans'
    })

    im = plt.imshow(feat_norm,
                    cmap='jet',  # 改用jet色阶增强细节
                    interpolation='bicubic',  # 双三次插值平滑
                    vmin=0.1,
                    vmax=0.9)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
    cbar.set_label("Feature Activation Level (σ-normalized)",
                   rotation=270,
                   labelpad=25)

    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=12)

    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)

    plt.title("Fused BEV Feature Activation Map", fontweight='bold')
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


class SMC_FusionGate(nn.Module):
    def __init__(self, in_channels, H=180, W=180, base_k=1.0, eta=1.0):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.InstanceNorm2d(in_channels)
        )

        self.spatial_weight = None
        self.spatial_initialized = False

        self.channel_att = ChannelAttention(in_channels)

        self.alpha_gate = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        self.beta_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Softplus()
        )

        self.k = nn.Parameter(torch.tensor(base_k))
        self.eta = nn.Parameter(torch.tensor(eta))
        self.k_gamma = nn.Parameter(torch.tensor(0.1))

        # 可学习空间权重矩阵
        self.spatial_weight = nn.Parameter(torch.ones(1, 1, H, W))  # 初始化为随机

        # 可学习近场偏置矩阵
        self.near_bias = nn.Parameter(torch.ones(1, 1, H, W))  # 初始全1

    def _init_spatial_param(self, H, W):
        """空间权重初始化（Xavier+近场偏置）"""
        nn.init.xavier_normal_(self.spatial_weight)
        with torch.no_grad():
            h_mid = H // 2
            self.spatial_weight[:, :, :h_mid, :] *= 0.8  # 远场初始权重较低
            self.spatial_weight[:, :, h_mid:, :] *= 1.5  # 近场初始权重较高

    def _init_near_bias(self, H, W):
        """近场偏置初始化（高斯分布）"""
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        near_mask = torch.exp(-5 * yy ** 2)  # 假设y轴负方向为近场
        self.near_bias.data = near_mask.view(1, 1, H, W)

    def forward(self, x_cam, x_lidar):
        # 特征投影（保持原样）
        x_lidar_proj = self.projection(x_lidar)

        # 差异计算（关键修改：显式包含k）
        diff = self.k * self.channel_att(x_cam - x_lidar_proj)  # k进入计算图

        # 空间增强（保持原结构）
        spatial_enhance = self.spatial_weight * self.near_bias
        enhanced_diff = diff * spatial_enhance

        # 滑模面计算（保持原逻辑）
        alpha = self.alpha_gate(enhanced_diff.mean(dim=(2, 3)))  # [B,C]
        beta = self.beta_gate(enhanced_diff)  # [B,1,H,W]

        S = (alpha.unsqueeze(-1).unsqueeze(-1) * beta * enhanced_diff).abs().mean(dim=1)
        S_global = S.mean()  # 简化全局量

        # 动态融合（保持原结构）
        fusion_gate = torch.sigmoid(self.k * S_global)
        fused_feat = fusion_gate * x_cam + (1 - fusion_gate) * x_lidar_proj

        # 稳定性计算（关键修改：直接使用S_global）
        if self.training:
            V = 0.5 * (self.k * S_global) ** 2  # 显式依赖k
            grad_V = torch.autograd.grad(V, self.k, create_graph=True)[0]
            loss = F.relu(grad_V + self.eta * S_global.detach())
        else:
            loss = 0.0

        return fused_feat, loss


class ChannelAttention(nn.Module):
    """改进的通道注意力机制"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        channel_att = self.fc(torch.cat([avg_out, max_out], dim=1))
        return x * channel_att.view(x.size(0), -1, 1, 1)


import torchvision


class PIDFusionNet(nn.Module):
    def __init__(self, cam_channels, lidar_channels):
        super().__init__()
        # ------------------- 特征对齐模块 -------------------
        # 可变形投影网络（激光雷达→相机BEV）
        self.projection = DeformableConvBlock(cam_channels, lidar_channels)

        # ------------------- PID三分支 -------------------
        # 比例分支（P）：直接特征差异
        self.p_branch = P_Branch(lidar_channels)

        # 积分分支（I）：全局上下文建模
        self.i_branch = I_Branch(lidar_channels)

        # 微分分支（D）：边缘特征提取
        self.d_branch = D_Branch(lidar_channels)

        # ------------------- 动态融合门控 -------------------
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(3 * lidar_channels, lidar_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # ------------------- 稳定性约束参数 -------------------
        self.k_p = nn.Parameter(torch.tensor(1.0))  # 比例增益
        self.k_i = nn.Parameter(torch.tensor(1.0))  # 积分增益
        self.k_d = nn.Parameter(torch.tensor(1.0))  # 微分增益

        # ------------------- 边界增强损失 -------------------
        self.edge_conv = SpatialGradientConv()

    def forward(self, x_cam, x_lidar):
        # ================== 特征对齐 ==================
        x_cam_proj = self.projection(x_cam)  # [B,C,H,W]

        # ================== PID三支路计算 ==================
        # ------ 比例项 P ------  TODO: 更复杂的差异增强网络
        p_term = self.k_p * self.p_branch(x_cam_proj - x_lidar)  # 即时差异

        # ------ 积分项 I ------  TODO: 更复杂的全局增强网络
        i_term = self.k_i * self.i_branch(x_cam_proj + x_lidar)  # 全局上下文

        # ------ 微分项 D ------  TODO: 更复杂的边缘增强网络
        d_term = self.k_d * self.d_branch(x_cam_proj, x_lidar)  # 边缘特征

        # ================== 动态融合 ==================
        pid_features = torch.cat([p_term, i_term, d_term], dim=1)  # [B,3C,H,W]
        fusion_gates = self.fusion_gate(pid_features)  # [B,C,H,W]
        fused_feat = fusion_gates * x_cam_proj + (1 - fusion_gates) * x_lidar

        return fused_feat, None


# ------------------- 核心模块实现 -------------------
class DeformableConvBlock(nn.Module):
    """可变形特征投影"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_c, 2 * 3 * 3, 3, padding=1)
        self.main_conv = nn.Conv2d(in_c, out_c, 3, padding=1)

    def forward(self, x):
        offsets = self.offset_conv(x)
        return torchvision.ops.deform_conv2d(
            x, offsets, self.main_conv.weight,
            self.main_conv.bias, padding=1
        )


class P_Branch(nn.Module):
    """比例项分支：通道注意力增强差异"""

    def __init__(self, channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, diff):
        return self.att(diff) * diff


class P_Branch_Enh(nn.Module):
    """比例项分支：通道注意力增强差异"""

    def __init__(self, channels):
        super().__init__()

        self.conv_sub = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.conv_diff_enh = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.MPFL = MSFF(inchannel=channels, mid_channel=64)
        self.convmix = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, groups=channels, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True), )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x_sub = torch.abs(x1 - x2)
        x_att = torch.sigmoid(self.conv_sub(x_sub))
        x1 = (x1 * x_att) + self.MPFL(self.conv_diff_enh(x1))
        x2 = (x2 * x_att) + self.MPFL(self.conv_diff_enh(x2))
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)

        x_f = x_f * x_att
        out = self.conv_dr(x_f)
        return out


class I_Branch(nn.Module):
    """积分项分支：全局上下文建模"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x) * x


import math


class Mix(nn.Module):
    # 初始化方法，设置初始混合权重m
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class I_Branch_Enh(nn.Module):

    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

        self.fc = nn.Conv2d(channels, channels, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, x):
        U = self.avg_pool(x)
        Ugc = self.conv1(U.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (batch_size, channels, 1)
        Ulc = self.fc(U).squeeze(-1).transpose(-1, -2)  # (batch_size, 1, channels)
        out1 = torch.sum(torch.matmul(Ugc, Ulc), dim=1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(Ulc.transpose(-1, -2), Ugc.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        W = self.sigmoid(out)
        return x * W


class D_Branch(nn.Module):
    """微分项分支：边缘特征提取"""

    def __init__(self, channels):
        super().__init__()
        self.edge_conv = SpatialGradientConv()
        self.adapt_conv = nn.Conv2d(2, channels, 3, padding=1, bias=True)

    def forward(self, x_cam, x_lidar):
        # 多模态边缘融合
        edge_cam = self.edge_conv(x_cam)
        edge_lidar = self.edge_conv(x_lidar)
        return self.adapt_conv(edge_cam + edge_lidar)


class D_Branch_Enh(nn.Module):
    """微分项分支：边缘特征提取"""

    def __init__(self, channels, width=4):
        super().__init__()
        in_dim = channels
        hidden_dim = channels // 2
        self.width = width

        self.conv = nn.Sequential(nn.Conv2d(in_dim * 2, in_dim, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(in_dim),
                                  nn.Sigmoid())
        self.in_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                                     nn.BatchNorm2d(hidden_dim),
                                     nn.Sigmoid())

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()

        for i in range(width - 1):  # 遍历宽度
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, nn.BatchNorm2d, nn.ReLU))  # 添加边缘增强模块
        self.out_conv = nn.Sequential(nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
                                      nn.BatchNorm2d(in_dim),
                                      nn.ReLU())

    def forward(self, x_cam, x_lidar):
        x = self.conv(torch.cat([x_cam, x_lidar], 1))
        mid = self.in_conv(x)
        out = mid
        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)
            edge = self.edge_enhance[i](mid)
            out = torch.cat([out, edge], dim=1)
        out = self.out_conv(out)
        return out


class EdgeEnhancer(nn.Module):  # 边缘增强模块
    def __init__(self, in_dim, norm, act):  # 初始化函数，接收输入维度、归一化层和激活函数
        super().__init__()  # 调用父类构造函数
        self.out_conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 1, bias=False),
                                      norm(in_dim),
                                      nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)  # 定义平均池化层

    def forward(self, x):  # 前向传播函数
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class SpatialGradientConv(nn.Module):
    """Sobel边缘检测"""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('kernel', torch.stack([sobel_x, sobel_y]).unsqueeze(1))

    def forward(self, x):
        # 输入: [B,C,H,W] 输出: [B,2,H,W]
        return F.conv2d(x.mean(dim=1, keepdim=True), self.kernel, padding=1)


class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        # 调用父类的构造函数
        super(CAB, self).__init__()
        # 自适应平均池化，将特征图缩小到1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义通道注意力机制的卷积序列
        self.conv_du = nn.Sequential(
            nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),  # 降维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),  # 升维
            nn.Sigmoid()  # Sigmoid激活生成注意力权重
        )

    def forward(self, x):
        # 对输入进行池化
        y = self.avg_pool(x)
        # 通过卷积序列
        y = self.conv_du(y)
        # 将注意力权重应用于输入特征图
        return x * y


class HDRAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        # 调用父类的构造函数
        super(HDRAB, self).__init__()
        # 定义卷积核大小
        kernel_size = 3
        # 通道注意力的降维率
        reduction = 8

        # 定义通道注意力模块
        self.cab = CAB(in_channels, reduction, bias)

        # 定义一系列卷积层和激活层，使用不同的扩张率和填充
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)

        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv_tail = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)

    def forward(self, y):
        # 通过第一组卷积和激活
        y1 = self.conv1(y)
        y1_1 = self.relu1(self.bn(y1))

        # 通过第二个卷积层并进行跳跃连接
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y

        # 通过第三组卷积和激活
        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)

        # 通过第四个卷积层并进行跳跃连接
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        # 通过第五组卷积和激活
        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)

        # 通过第六个卷积层并进行跳跃连接
        y6 = self.conv2_1(y5_1 + y3)
        y6_1 = y6 + y4_1

        # 通过第七组卷积和激活
        y7 = self.conv1_1(y6_1 + y2_1)
        y7_1 = self.relu1_1(y7)

        # 通过尾部卷积层并进行跳跃连接
        y8 = self.conv_tail(y7_1 + y1)
        y8_1 = y8 + y6_1

        # 通过通道注意力模块
        y9 = self.cab(y8_1)
        y9_1 = y + y9

        # 返回最终结果
        return y9_1