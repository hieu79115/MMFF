import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossModalAttention(nn.Module):
    def __init__(self, rgb_channels, skel_channels, inter_channels=512):
        super(CrossModalAttention, self).__init__()
        
        # 1. Thêm BatchNorm2d 
        self.norm_rgb = nn.BatchNorm2d(rgb_channels)
        self.norm_skel = nn.BatchNorm2d(skel_channels)
        
        self.query_conv = nn.Conv2d(rgb_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(skel_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(skel_channels, rgb_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 2. Thêm Scale-factor theo chuẩn Scaled Dot-Product Attention
        self.scale = inter_channels ** -0.5

    def forward(self, x_rgb, x_skel):
        B, C_r, H, W = x_rgb.size()
        
        # Đi qua Normalize trước khi thực hiện Attention Mapping
        x_rgb_norm = self.norm_rgb(x_rgb)
        x_skel_norm = self.norm_skel(x_skel)
        
        # Average Pool theo thời gian T (đưa về 1), giữ nguyên số khớp V (x_skel.size(3))
        x_skel_pool = F.adaptive_avg_pool2d(x_skel_norm, (1, x_skel.size(3))) 
        
        proj_query = self.query_conv(x_rgb_norm).view(B, -1, H*W).permute(0, 2, 1)
        proj_key = self.key_conv(x_skel_pool).view(B, -1, x_skel.size(3))
        
        # Nhân scale để chia nhỏ lại độ lớn của Energy, tránh vanishing gradient ở hàm Softmax
        energy = torch.bmm(proj_query, proj_key) * self.scale
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x_skel_pool).view(B, -1, x_skel.size(3))
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C_r, H, W)
        
        # Residual Connection (cộng với x_rgb gốc)
        out = self.gamma * out + x_rgb
        return out