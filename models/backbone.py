import torch
import torch.nn as nn
import timm 
from models.attention import CrossModalAttention

class RGBStream_Base(nn.Module):
    """
    Nhánh RGB hỗ trợ multi-frame input (ví dụ: 3 frame đầu-giữa-cuối).
    Input shape: (B, num_frames, 3, H, W)
    Output shape: (B, 2048)
    """
    def __init__(self, skel_channels=256):
        super(RGBStream_Base, self).__init__()
        
        # Load Xception từ timm (pretrained=True để lấy trọng số đã học ImageNet)
        # features_only=True: Chỉ lấy feature maps, bỏ lớp phân loại cuối
        self.backbone = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        
        # Xception trả về feature map có 2048 channels ở lớp cuối cùng
        out_channels = 2048
        
        # Cross-Attention Module (áp dụng cho mỗi frame)
        self.cross_att = CrossModalAttention(rgb_channels=out_channels, skel_channels=skel_channels)
        
        # Pooling spatial để chuyển feature map -> vector cho mỗi frame
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal Attention: học cách tổng hợp thông tin từ nhiều frame
        # Mỗi frame được đánh trọng số attention trước khi gộp
        self.temporal_attn = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, 1),
        )

    def forward(self, x_rgb, x_skel_feature_map):
        """
        Args:
            x_rgb: (B, num_frames, 3, H, W) - Nhiều frame RGB
            x_skel_feature_map: (B, C_skel, T_skel, V_skel) - Feature map từ nhánh skeleton
        Returns:
            f_rgb_vec: (B, 2048) - Vector đặc trưng RGB tổng hợp
        """
        B, T, C, H, W = x_rgb.shape
        
        # Gộp batch và frame để chạy backbone 1 lần duy nhất
        x = x_rgb.reshape(B * T, C, H, W)  # (B*T, 3, H, W)
        
        # Trích xuất feature map qua Xception
        features = self.backbone(x)
        f_rgb_map = features[-1]  # (B*T, 2048, h, w)
        _, C_out, h, w = f_rgb_map.shape
        
        # Tách lại thành (B, T, 2048, h, w)
        f_rgb_map = f_rgb_map.view(B, T, C_out, h, w)
        
        # Áp dụng Cross-Attention cho từng frame + Spatial Pooling
        frame_vecs = []
        for t in range(T):
            frame_feat = f_rgb_map[:, t]  # (B, 2048, h, w)
            guided = self.cross_att(frame_feat, x_skel_feature_map)  # (B, 2048, h, w)
            pooled = self.avg_pool(guided).flatten(1)  # (B, 2048)
            frame_vecs.append(pooled)
        
        # Stack thành (B, T, 2048)
        frame_features = torch.stack(frame_vecs, dim=1)
        
        # Temporal Attention Aggregation
        attn_logits = self.temporal_attn(frame_features)  # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T, 1)
        
        # Weighted sum qua chiều thời gian
        f_rgb_vec = (frame_features * attn_weights).sum(dim=1)  # (B, 2048)
        
        return f_rgb_vec