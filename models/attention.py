import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    CẢI TIẾN (Reversed): Cross-Attention với Skeleton làm Query, RGB làm Key/Value.
    Mục tiêu: Dùng đặc trưng RGB (Key/Value) để hướng dẫn/tăng cường đặc trưng Skeleton (Query).
    
    Input:
        x_skel: (B, C_s, T, V) - Skeleton feature map từ ST-GCN
        x_rgb:  (B, C_r, H, W) - RGB feature map từ Xception backbone
    Output:
        Enhanced skeleton feature map (B, C_s, T, V)
    """
    def __init__(self, skel_channels, rgb_channels, inter_channels=512):
        super(CrossModalAttention, self).__init__()
        
        # Skeleton là Query
        self.query_conv = nn.Conv2d(skel_channels, inter_channels, kernel_size=1)
        # RGB là Key và Value
        self.key_conv = nn.Conv2d(rgb_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(rgb_channels, skel_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_skel, x_rgb):
        B, C_s, T, V = x_skel.size()
        _, C_r, H, W = x_rgb.size()
        
        # Query from Skeleton: (B, inter, T, V) → (B, T*V, inter)
        proj_query = self.query_conv(x_skel).view(B, -1, T * V).permute(0, 2, 1)
        
        # Key from RGB: (B, inter, H, W) → (B, inter, H*W)
        proj_key = self.key_conv(x_rgb).view(B, -1, H * W)
        
        # Attention map: (B, T*V, inter) x (B, inter, H*W) → (B, T*V, H*W)
        # Mỗi vị trí skeleton (joint × time) attend tới toàn bộ vùng không gian RGB
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Value from RGB: (B, C_s, H*W) — project RGB channels xuống skel_channels
        proj_value = self.value_conv(x_rgb).view(B, -1, H * W)
        
        # Output: (B, C_s, H*W) x (B, H*W, T*V) → (B, C_s, T*V)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C_s, T, V)
        
        # Residual connection: giữ nguyên thông tin skeleton gốc
        out = self.gamma * out + x_skel
        return out