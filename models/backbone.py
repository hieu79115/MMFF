import torch
import torch.nn as nn
import timm 
from models.attention import CrossModalAttention, ReversedCrossModalAttention

class RGBStream_Base(nn.Module):
    def __init__(self, skel_channels=256, cross_attention_mode: str = 'normal'):
        super(RGBStream_Base, self).__init__()
        
        # Load Xception từ timm (pretrained=True để lấy trọng số đã học ImageNet)
        # features_only=True: Chỉ lấy feature maps, bỏ lớp phân loại cuối
        self.backbone = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        
        # Xception trả về feature map có 2048 channels ở lớp cuối cùng
        out_channels = 2048
        
        self.cross_attention_mode = (cross_attention_mode or 'normal').lower()

        # Cross-Attention Module
        if self.cross_attention_mode == 'normal':
            self.cross_att = CrossModalAttention(rgb_channels=out_channels, skel_channels=skel_channels)
        elif self.cross_attention_mode == 'reversed':
            self.cross_att = ReversedCrossModalAttention(rgb_channels=out_channels, skel_channels=skel_channels)
        elif self.cross_attention_mode == 'none':
            self.cross_att = None
        else:
            raise ValueError(f"Unsupported cross_attention_mode: {self.cross_attention_mode}")
        
        # Pooling để chuyển về vector
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_rgb, x_skel_feature_map):
        # timm trả về một list các feature maps từ nông đến sâu
        features = self.backbone(x_rgb)
        
        # Ta lấy cái cuối cùng (feature map sâu nhất)
        f_rgb_map = features[-1] 
        # Shape dự kiến: (Batch, 2048, 10, 10) với ảnh đầu vào 299x299
        
        # Optional cross-attention for ablation.
        if self.cross_att is None:
            f_rgb_guided = f_rgb_map
        else:
            f_rgb_guided = self.cross_att(f_rgb_map, x_skel_feature_map)
        
        # Pooling & Flatten
        f_rgb_vec = self.avg_pool(f_rgb_guided).flatten(1)
        
        return f_rgb_vec