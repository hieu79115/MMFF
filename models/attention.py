import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, rgb_channels, skel_channels, inter_channels=512, dropout=0.0):
        super(CrossModalAttention, self).__init__()
        
        self.query_conv = nn.Conv2d(rgb_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(skel_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(skel_channels, rgb_channels, kernel_size=1)
        
        self.scale = inter_channels ** -0.5
        self.softmax = nn.Softmax(dim=-1)

        self.attn_dropout = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_rgb, x_skel):
        B, C_r, H, W = x_rgb.size()
        # x_skel: (B, C_s, T, V) – giữ nguyên chiều thời gian để attention học tương quan spatio-temporal
        B_s, C_s, T, V = x_skel.size()
        assert B_s == B, "RGB and skeleton batch size must match"

        # Query: mỗi vị trí trên feature map RGB là một token
        proj_query = self.query_conv(x_rgb).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, D)

        # Key/Value: toàn bộ chuỗi thời gian T và khớp V của skeleton -> T*V tokens
        proj_key = self.key_conv(x_skel).view(B, -1, T * V)                      # (B, D, T*V)
        proj_value = self.value_conv(x_skel).view(B, C_r, T * V)                 # (B, C_r, T*V)

        # Dot-product attention giữa từng vị trí RGB và toàn bộ (T,V) skeleton
        energy = torch.bmm(proj_query, proj_key) * self.scale                    # (B, HW, T*V)
        attention = self.softmax(energy)
        attention = self.attn_dropout(attention)

        # Ánh xạ ngược về không gian RGB: kết hợp value skeleton cho từng vị trí RGB
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                  # (B, C_r, HW)
        out = out.view(B, C_r, H, W)

        out = self.gamma * out + x_rgb
        return out