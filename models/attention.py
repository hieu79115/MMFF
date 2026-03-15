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
        
        # Fix: Pool the temporal dimension (T) to 1 and preserve the joint dimension (V)
        # Assuming x_skel is (B, C, T, V). T is height, V is width.
        x_skel_pool = F.adaptive_avg_pool2d(x_skel, (1, x_skel.size(3)))
        
        # We need to transpose to match (B, -1, V) because view() after a 1x1 conv
        # on (B, C, 1, V) naturally gives (B, C, V)
        proj_query = self.query_conv(x_rgb).view(B, -1, H*W).permute(0, 2, 1)
        proj_key = self.key_conv(x_skel_pool).view(B, -1, x_skel.size(3))
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x_skel_pool).view(B, -1, x_skel.size(3))
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C_r, H, W)
        
        out = self.gamma * out + x_rgb
        return out