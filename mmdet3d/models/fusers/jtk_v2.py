from typing import List, Tuple

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

import torch.nn.functional as F

from .fuser_blocks import ImprovedMSNetFusion


@FUSERS.register_module()
class JTKV2(nn.Sequential):
    def __init__(self, in_channels, mid_planes, out_channels) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_planes = [256, 128]
        in_channels = [80, 256]
        num_upsample_filters=[128, 128, 128]
        upsample_strides=[1, 2, 4]

        self.cam_conv = nn.Conv2d(in_channels[0], mid_planes[0], 3, 1, 1)
        self.lidar_conv = nn.Conv2d(in_channels[1], mid_planes[0], 3, 1, 1)
        # self.align_block = JointTaskFusion(mid_channel)

        self.cit_geo = CrossModalInteraction(mid_planes[0], 8, mode='lidar_guided')
        self.cit_sem = CrossModalInteraction(mid_planes[0], 8, mode='camera_guided')

        self.out_conv = nn.Sequential(nn.Conv2d(mid_planes[0]*2, mid_planes[0], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(mid_planes[0]),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(mid_planes[0], mid_planes[0], 3, 1, 1, bias=True))
        
        ################################################

        self.conv_dr1 = nn.Sequential(
            nn.Conv2d(mid_planes[0], mid_planes[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_dr2 = nn.Sequential(
            nn.Conv2d(mid_planes[1], mid_planes[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_dr3 = nn.Sequential(
            nn.Conv2d(mid_planes[1], mid_planes[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )

        self.conv_ds1 = nn.Sequential(
            nn.Conv2d(mid_planes[1], mid_planes[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_ds2 = nn.Sequential(
            nn.Conv2d(mid_planes[1], mid_planes[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_ds11 = nn.Sequential(
            nn.Conv2d(mid_planes[1], mid_planes[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_ds22 = nn.Sequential(
            nn.Conv2d(mid_planes[1], mid_planes[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )

        self.geo_down = nn.Sequential(
            nn.Conv2d(in_channels[1], mid_planes[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )
        self.sem_down = nn.Sequential(
            nn.Conv2d(in_channels[1], mid_planes[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes[1]),
            nn.ReLU(inplace=True),
        )

        # self.convmix1 = DDF(mid_planes[0])
        self.convmix2 = DDF(mid_planes[1])
        self.convmix3 = DDF(mid_planes[1])

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[0],
                               int(upsample_strides[0]),
                               stride=int(upsample_strides[0]), bias=False
                               ) if upsample_strides[0] >= 1 else  # 转置卷积：当 upsample_strides[i] ≥1 时，用于放大特征图。
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[0],
                      kernel_size=int(1 / upsample_strides[0]),
                      stride=int(1 / upsample_strides[0]),
                      bias=False),  
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[1],
                               int(upsample_strides[1]),
                               stride=int(upsample_strides[1]), bias=False
                               ) if upsample_strides[1] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[1],
                      kernel_size=int(1 / upsample_strides[1]),
                      stride=int(1 / upsample_strides[1]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[1], eps=1e-3, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1],
                               num_upsample_filters[2],
                               int(upsample_strides[2]),
                               stride=int(upsample_strides[2]), bias=False
                               ) if upsample_strides[2] >= 1 else
            nn.Conv2d(mid_planes[1],
                      num_upsample_filters[2],
                      kernel_size=int(1 / upsample_strides[2]),
                      stride=int(1 / upsample_strides[2]),
                      bias=False),
            nn.BatchNorm2d(num_upsample_filters[2], eps=1e-3, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_last = nn.Sequential(nn.Conv2d(sum(num_upsample_filters), out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True))
        
        # self.pyramid_fusion = CrossAttentionFusion(in_channels=mid_planes[1])
        self.pyramid_fusion = ImprovedMSNetFusion(mid_planes[1])


    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        feat_channels = [0] + [x.shape[1] for x in inputs]  # [0, 80, 256]  
        feat_channels = torch.cumsum(torch.tensor(feat_channels), dim=0)  
        x_mm = torch.cat(inputs, dim=1)

        x_cam = self.cam_conv(x_mm[:, :feat_channels[1]]) 
        x_lidar = self.lidar_conv(x_mm[:, feat_channels[1]:])

        # task_vec = self.task_vector.repeat(x_cam.size(0), 1).cuda() 
        # fused_bev = self.align_block(x_cam, x_lidar, task_vector=task_vec)

        x_l_geo, _ = self.cit_geo(x_lidar, x_cam)  # 几何增强
        _, x_c_sem = self.cit_sem(x_lidar, x_cam)  # 语义增强

        # x_geo = x_l_geo + x_c_geo  # C=256
        # x_sem = x_l_sem + x_c_sem
        x_f = self.out_conv(torch.cat([x_l_geo, x_c_sem], 1))
        # x_f = self.convmix1(x1, x2)  # [B, 256, 128, 128]

        ## DA-FPN
        p1 = self.conv_dr1(x_f)   # [B, 128, 180, 180]  

        x1 = self.conv_ds1(p1 + self.geo_down(x_l_geo))  # [B, 128, 90, 90]
        x2 = self.conv_ds2(p1 + self.sem_down(x_c_sem))
        x_f2 = self.convmix2(x1, x2)  # [B, 128, 90, 90]

        p2 = self.conv_dr2(x_f2)    # [B, 128, 90, 90]  

        x1 = self.conv_ds11(p2 + x1)  # [B, 128, 45, 45]
        x2 = self.conv_ds22(p2 + x2)
        x_f3 = self.convmix3(x1, x2)  # [B, 128, 45, 45]

        p3 = self.conv_dr3(x_f3)    # [B, 128, 45, 45]

        p1_out, p2_out, p3_out = self.pyramid_fusion(p1, p2, p3)

        ## 解码阶段
        up1 = self.deconv1(p1_out)  # [B, 128, 128, 128] => [B, 128, 128, 128]
        up2 = self.deconv2(p2_out)  # [B, 128, 64, 64] => [B, 128, 128, 128] 
        up3 = self.deconv3(p3_out)  # [B, 128, 32, 32] => [B, 128, 128, 128]

        x = F.silu(torch.cat([up1, up2, up3], dim=1))  # 拼接后激活  [B, 384, 128, 128]

        fused_bev = self.conv_last(x)


        return fused_bev
    

from einops import rearrange  
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
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels=128):
        """
        改进的跨尺度注意力特征融合模块
        Args:
            in_channels (int): 输入特征图的通道数，默认为128
        """
        super().__init__()
        self.in_channels = in_channels

        # 动态权重生成组件
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, 3, kernel_size=1),  # 输入3倍通道，输出3个权重
            nn.Sigmoid()
        )
        self.channel_attn_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, 2, kernel_size=1),  # 输入3倍通道，输出3个权重
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=1),  # 空间注意力
            nn.Sigmoid()
        )

        # 特征上/下采样组件
        self.dw_conv_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
        self.dw_conv_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )

        # 特征增强组件 (使用深度可分离卷积减少参数量)
        self.feature_enhancer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 全局上下文组件
        self.global_threshold = 32  # 特征图尺寸小于32时注入全局信息
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.LayerNorm([in_channels, 1, 1])
        )

        # 残差投影
        self.residual_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def dynamic_weights(self, features, num_multi=False):
        """
        生成特征融合的动态权重
        Args:
            features: 待融合的特征图列表 (每张特征图形状: [B, C, H, W])
        Returns:
            list: 归一化的权重张量列表
        """
        # 1. 通道注意力: 捕获跨尺度通道关联
        concat_features = torch.cat([
            F.adaptive_avg_pool2d(feat, 1) for feat in features
        ], dim=1)

        if num_multi:
            channel_weights = self.channel_attn(concat_features)  # [B, 3, 1, 1]
        else:
            channel_weights = self.channel_attn_2(concat_features)
        all_weights = []
        for i, feat in enumerate(features):
            # 2. 空间注意力: 保留位置敏感信息
            spatial_weight = self.spatial_attn(feat)  # [B, 1, H, W]

            # 3. 综合通道+空间权重
            channel_weight = channel_weights[:, i:i + 1]  # [B, 1, 1, 1]
            fused_weight = torch.sigmoid(channel_weight * spatial_weight)
            all_weights.append(fused_weight)

        # 4. 归一化权重 (确保总和为1)
        weight_sum = torch.stack(all_weights, dim=0).sum(dim=0)
        normalized_weights = [w / (weight_sum + 1e-6) for w in all_weights]

        return normalized_weights

    def cross_scale_fusion(self, features):

        # 确保有特征需要融合
        assert len(features) > 0, "At least one feature map is required"
        num_multi = False

        # 如果只有一个特征图，直接返回增强后的特征
        if len(features) == 1:
            return self.enhance_features(features[0])
        if len(features) == 3:
            num_multi = True

        # 对齐不同分辨率特征图
        aligned_features = []
        target_size = features[len(features) // 2].shape[2:]  # 以中间特征图大小为基准

        for i, feat in enumerate(features):
            if feat.shape[2:] != target_size:
                if feat.shape[2] < target_size[0]:  # 上采样小尺寸特征
                    aligned_feat = self.dw_conv_up(feat)
                else:  # 下采样大尺寸特征
                    aligned_feat = self.dw_conv_down(feat)
            else:
                aligned_feat = feat
            aligned_features.append(aligned_feat)

        # 生成动态权重并进行融合
        weights = self.dynamic_weights(aligned_features, num_multi)
        fused = sum(w * feat for w, feat in zip(weights, aligned_features))

        # 特征增强
        return self.enhance_features(fused)

    def enhance_features(self, x):
        """
        特征增强组件
        Args:
            x (torch.Tensor): 输入特征图
        Returns:
            torch.Tensor: 增强后的特征图
        """
        # 1. 特征增强
        enhanced = self.feature_enhancer(x)

        # 2. 门控机制: 学习特征重要性
        gate = self.gate(x)
        gated = gate * enhanced

        # 3. 残差连接: 保留原始信息
        return self.residual_proj(gated) + x

    def add_global_context(self, feat):
        """
        为高层特征注入全局上下文信息
        Args:
            feat (torch.Tensor): 输入特征图
        Returns:
            torch.Tensor: 增强全局上下文后的特征图
        """
        # 仅当特征图尺寸足够小时注入全局信息
        if min(feat.shape[2:]) <= self.global_threshold:
            global_feat = self.global_proj(feat)
            # 自适应调整全局特征的尺寸
            if global_feat.shape[2:] != feat.shape[2:]:
                global_feat = F.interpolate(
                    global_feat, size=feat.shape[2:],
                    mode='bilinear', align_corners=True
                )
            return feat + global_feat
        return feat

    def forward(self, p1, p2, p3):

        # Step 1: 高层特征注入全局上下文 (改善大目标检测)
        p3_global = self.add_global_context(p3)

        # Step 2: 双向融合路径
        # 自底向上融合 (增强细节信息)
        p2_up = self.cross_scale_fusion([p3_global, p2])
        p1_up = self.cross_scale_fusion([p2_up, p1])

        # 自顶向下融合 (增强语义信息)
        p2_down = self.cross_scale_fusion([p1_up, p2])
        p3_down = self.cross_scale_fusion([p2_down, p3_global])

        # Step 3: 多尺度特征聚合
        # 第二级融合 (结合双向特征)
        p2_fused = self.cross_scale_fusion([p3_down, p2, p1_up])
        p1_fused = self.cross_scale_fusion([p2_fused, p1])
        p3_fused = self.cross_scale_fusion([p2_fused, p3_global])

        # 最终输出 (添加残差连接增强稳定性)
        p1_out = p1_fused + p1_up
        p2_out = p2_fused + p2
        p3_out = p3_fused + p3_global

        return p1_out, p2_out, p3_out