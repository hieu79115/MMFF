import torch
import torch.nn as nn
import timm 
from models.attention import CrossModalAttention


class RGBRefinementBlock(nn.Module):
    """
    Lightweight conv block cho RGB stream tự refine feature ĐỘC LẬP
    trước khi nhận guidance từ skeleton.
    Giúp RGB không bị phụ thuộc hoàn toàn vào cross-attention.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = channels // reduction
        self.block = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),  # depthwise
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # SE-style channel attention to reweight
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        w = self.se(out).unsqueeze(-1).unsqueeze(-1)
        return residual + w * out


class RGBStream_Base(nn.Module):
    def __init__(self, skel_channels=256):
        super(RGBStream_Base, self).__init__()
        
        # Load Xception từ timm (pretrained=True để lấy trọng số đã học ImageNet)
        self.backbone = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        
        out_channels = 2048
        
        # ---- RGB-only refinement (independent of skeleton) ----
        self.rgb_refine = RGBRefinementBlock(out_channels, reduction=4)
        
        # ---- Cross-Attention Module (skeleton guides RGB) ----
        self.cross_att = CrossModalAttention(rgb_channels=out_channels, skel_channels=skel_channels)
        
        # Pooling để chuyển về vector
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_rgb, x_skel_feature_map):
        features = self.backbone(x_rgb)
        f_rgb_map = features[-1]   # (B, 2048, H, W)
        
        # 1. Refine RGB features independently (không cần skeleton)
        f_rgb_refined = self.rgb_refine(f_rgb_map)
        
        # 2. Cross-Attention: skeleton hướng dẫn RGB (additive)
        f_rgb_guided = self.cross_att(f_rgb_refined, x_skel_feature_map)
        
        # Pooling & Flatten
        f_rgb_vec = self.avg_pool(f_rgb_guided).flatten(1)
        
        return f_rgb_vec