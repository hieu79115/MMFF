import torch
import torch.nn as nn
import timm 
from models.attention import CrossModalAttention

class RGBStream_Base(nn.Module):
    def __init__(self, skel_channels=256):
        super(RGBStream_Base, self).__init__()
        
        # Load Xception từ timm (pretrained=True để lấy trọng số đã học ImageNet)
        # features_only=True: Chỉ lấy feature maps, bỏ lớp phân loại cuối
        self.backbone = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        
        # Xception trả về feature map có 2048 channels ở lớp cuối cùng
        out_channels = 2048
        
        # Cross-Attention Module
        self.cross_att = CrossModalAttention(rgb_channels=out_channels, skel_channels=skel_channels)
        
        # Pooling để chuyển về vector
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_rgb, x_skel_feature_map):
        # timm trả về một list các feature maps từ nông đến sâu
        features = self.backbone(x_rgb)
        
        # Ta lấy cái cuối cùng (feature map sâu nhất)
        f_rgb_map = features[-1] 
        # Shape dự kiến: (Batch, 2048, 10, 10) với ảnh đầu vào 299x299
        
        # Áp dụng Cross-Attention (Skel hướng dẫn RGB)
        f_rgb_guided = self.cross_att(f_rgb_map, x_skel_feature_map)
        
        # Pooling & Flatten
        f_rgb_vec = self.avg_pool(f_rgb_guided).flatten(1)
        
        return f_rgb_vec