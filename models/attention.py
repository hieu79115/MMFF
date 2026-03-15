import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionModule(nn.Module):
    """
    CBAM-style Spatial Attention. 
    Helps the RGB network further refine where to look after Cross-Attention.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class MultiHeadCrossModalAttention(nn.Module):
    """
    Advanced Multi-Head Cross-Attention that uses Skeleton Joints (V) to query the RGB spatial patches (H*W)
    or RGB spatial patches to query Skeleton Joints, allowing learning of semantic mapping.
    Includes Spatial Attention refinement.
    """
    def __init__(self, rgb_channels, skel_channels, inter_channels=512, num_heads=8, dropout=0.1):
        super(MultiHeadCrossModalAttention, self).__init__()
        
        self.num_heads = num_heads
        self.inter_channels = inter_channels
        
        assert inter_channels % num_heads == 0, "inter_channels must be divisible by num_heads"
        self.head_dim = inter_channels // num_heads
        
        self.q_conv = nn.Conv2d(rgb_channels, inter_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(skel_channels, inter_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(skel_channels, inter_channels, kernel_size=1)
        
        self.out_conv = nn.Conv2d(inter_channels, rgb_channels, kernel_size=1)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Spatial Attention Module to refine the RGB tensor
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x_rgb, x_skel):
        B, C_r, H, W = x_rgb.size()
        
        # 1. Prepare Skeleton token (reduce time, preserve joints)
        # x_skel: (B, C_s, T', V) -> (B, C_s, 1, V)
        x_skel_pool = F.adaptive_avg_pool2d(x_skel, (1, x_skel.size(3)))
        V = x_skel_pool.size(3)
        
        # 2. Linear Projections
        # Q (from RGB): (B, num_heads, H*W, head_dim)
        q = self.q_conv(x_rgb).view(B, self.num_heads, self.head_dim, H*W).permute(0, 1, 3, 2)
        
        # K, V (from Skeleton): (B, num_heads, V, head_dim)
        k = self.k_conv(x_skel_pool).view(B, self.num_heads, self.head_dim, V).permute(0, 1, 3, 2)
        v = self.v_conv(x_skel_pool).view(B, self.num_heads, self.head_dim, V).permute(0, 1, 3, 2)
        
        # 3. Scaled Dot-Product Attention: Q * K^T
        scale = self.head_dim ** -0.5
        # (B, num_heads, H*W, head_dim) * (B, num_heads, head_dim, V) -> (B, num_heads, H*W, V)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Softmax over the joints mapping (V)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 4. Multiply with Value
        # (B, num_heads, H*W, V) * (B, num_heads, V, head_dim) -> (B, num_heads, H*W, head_dim)
        out = torch.matmul(attn, v)
        
        # 5. Concatenate Heads and Reshape Back
        # (B, num_heads, H*W, head_dim) -> (B, H*W, inter_channels) -> (B, inter_channels, H, W)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H*W, self.inter_channels)
        out = out.permute(0, 2, 1).contiguous().view(B, self.inter_channels, H, W)
        
        # 6. Output Projection
        out = self.out_conv(out)
        
        # 7. Refine with Spatial Attention 
        # Apply spatial attention on the guided features to "mask" background noise further
        spatial_mask = self.spatial_attention(out)
        out = out * spatial_mask
        
        # 8. Residual connection
        out = x_rgb + self.gamma * out
        
        return out