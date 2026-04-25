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
    def __init__(self, dim, num_heads, mode='lidar_guided'):
        super().__init__()
        self.mode = mode

        self.cab = CAB(
            dim=dim,
            num_heads=num_heads,
            bias=True
        )

        if mode == 'lidar_guided':
            self.lambda_scale = nn.Parameter(torch.tensor(1.2))
        else:
            self.lambda_scale = 1.0

    def forward(self, lidar_feat, camera_feat):
        if self.mode == 'lidar_guided':
            out = self.cab(lidar_feat, camera_feat) + lidar_feat * self.lambda_scale
            return out, camera_feat
        else:  # camera_guided
            out = self.cab(camera_feat, lidar_feat) + camera_feat * self.lambda_scale
            return lidar_feat, out

  
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):

        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class AdaptiveModule(nn.Module):

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
    def __init__(self, in_channels):
        super().__init__()
        # 动态权重生成部分
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.adaptive = AdaptiveModule(in_channels)
    
    def forward(self, camera_feat, lidar_feat):

        combined = camera_feat + lidar_feat
        weights = self.weight_generator(combined)  # [B, C]
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        weighted_cam = camera_feat * weights
        weighted_lidar = lidar_feat * (1 - weights)

        concat_feat = torch.cat([weighted_cam, weighted_lidar], dim=1)
        fused_feat = self.conv(concat_feat)
        fused_feat = self.adaptive(fused_feat)
        
        return fused_feat
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels=128):

        super().__init__()
        self.in_channels = in_channels

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


        self.dw_conv_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
        self.dw_conv_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )

        self.feature_enhancer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.global_threshold = 32  # 特征图尺寸小于32时注入全局信息
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.LayerNorm([in_channels, 1, 1])
        )

        self.residual_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def dynamic_weights(self, features, num_multi=False):
        concat_features = torch.cat([
            F.adaptive_avg_pool2d(feat, 1) for feat in features
        ], dim=1)

        if num_multi:
            channel_weights = self.channel_attn(concat_features)  # [B, 3, 1, 1]
        else:
            channel_weights = self.channel_attn_2(concat_features)
        all_weights = []
        for i, feat in enumerate(features):
            spatial_weight = self.spatial_attn(feat)  # [B, 1, H, W]
            channel_weight = channel_weights[:, i:i + 1]  # [B, 1, 1, 1]
            fused_weight = torch.sigmoid(channel_weight * spatial_weight)
            all_weights.append(fused_weight)

        weight_sum = torch.stack(all_weights, dim=0).sum(dim=0)
        normalized_weights = [w / (weight_sum + 1e-6) for w in all_weights]

        return normalized_weights

    def cross_scale_fusion(self, features):
        assert len(features) > 0, "At least one feature map is required"
        num_multi = False

        if len(features) == 1:
            return self.enhance_features(features[0])
        if len(features) == 3:
            num_multi = True

        aligned_features = []
        target_size = features[len(features) // 2].shape[2:]

        for i, feat in enumerate(features):
            if feat.shape[2:] != target_size:
                if feat.shape[2] < target_size[0]:  # 上采样小尺寸特征
                    aligned_feat = self.dw_conv_up(feat)
                else:  # 下采样大尺寸特征
                    aligned_feat = self.dw_conv_down(feat)
            else:
                aligned_feat = feat
            aligned_features.append(aligned_feat)

        weights = self.dynamic_weights(aligned_features, num_multi)
        fused = sum(w * feat for w, feat in zip(weights, aligned_features))

        return self.enhance_features(fused)

    def enhance_features(self, x):

        enhanced = self.feature_enhancer(x)
        gate = self.gate(x)
        gated = gate * enhanced

        return self.residual_proj(gated) + x

    def add_global_context(self, feat):

        if min(feat.shape[2:]) <= self.global_threshold:
            global_feat = self.global_proj(feat)

            if global_feat.shape[2:] != feat.shape[2:]:
                global_feat = F.interpolate(
                    global_feat, size=feat.shape[2:],
                    mode='bilinear', align_corners=True
                )
            return feat + global_feat
        return feat

    def forward(self, p1, p2, p3):

        p3_global = self.add_global_context(p3)
        p2_up = self.cross_scale_fusion([p3_global, p2])
        p1_up = self.cross_scale_fusion([p2_up, p1])
        p2_down = self.cross_scale_fusion([p1_up, p2])
        p3_down = self.cross_scale_fusion([p2_down, p3_global])

        p2_fused = self.cross_scale_fusion([p3_down, p2, p1_up])
        p1_fused = self.cross_scale_fusion([p2_fused, p1])
        p3_fused = self.cross_scale_fusion([p2_fused, p3_global])

        p1_out = p1_fused + p1_up
        p2_out = p2_fused + p2
        p3_out = p3_fused + p3_global

        return p1_out, p2_out, p3_out
