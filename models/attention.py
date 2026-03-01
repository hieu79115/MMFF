import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Multi-Head Gated Cross-Attention.
    Dùng đặc trưng Skeleton (Key/Value) để hướng dẫn đặc trưng RGB (Query).

    Cải tiến so với phiên bản cũ:
      - Multi-head attention (num_heads) để capture nhiều dạng quan hệ cross-modal.
      - Gating mechanism: mô hình tự học mức độ fusion thay vì gamma cố định.
      - LayerNorm trên Q, K, V để ổn định training.
      - Output projection sau attention.
      - Proper scaling (sqrt(d_k)).
      - Attention dropout.
    """

    def __init__(
        self,
        rgb_channels: int,
        skel_channels: int,
        inter_channels: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(CrossModalAttention, self).__init__()
        assert inter_channels % num_heads == 0, "inter_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = inter_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # --- Projections ---
        self.query_conv = nn.Conv2d(rgb_channels, inter_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(skel_channels, inter_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(skel_channels, inter_channels, kernel_size=1, bias=False)

        # Output projection: map back to rgb_channels
        self.out_proj = nn.Conv2d(inter_channels, rgb_channels, kernel_size=1, bias=False)

        # --- Normalization ---
        self.norm_q = nn.GroupNorm(1, rgb_channels)   # acts like LayerNorm per sample
        self.norm_k = nn.GroupNorm(1, skel_channels)

        # --- Gating (replaces scalar gamma) ---
        # Learns a per-channel gate value in [0, 1]
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(rgb_channels, rgb_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(rgb_channels // 4, rgb_channels),
            nn.Sigmoid(),
        )

        # --- Dropout ---
        self.attn_dropout = nn.Dropout(dropout)

    # -----------------------------------------------------------------
    def forward(self, x_rgb: torch.Tensor, x_skel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb:  (B, C_rgb, H, W)   – RGB feature map (e.g. 2048×10×10)
            x_skel: (B, C_skel, T, V)  – Skeleton feature map (e.g. 256×T×V)
        Returns:
            Tensor of shape (B, C_rgb, H, W) – guided RGB features.
        """
        B, C_r, H, W = x_rgb.size()

        # Pool skeleton over temporal dim, keep joints V
        x_skel_pool = F.adaptive_avg_pool2d(x_skel, (1, x_skel.size(3)))  # (B, C_s, 1, V)

        # Normalize before projection
        q_in = self.norm_q(x_rgb)          # (B, C_r, H, W)
        k_in = self.norm_k(x_skel_pool)    # (B, C_s, 1, V)

        # Project Q, K, V
        Q = self.query_conv(q_in)  # (B, D, H, W)
        K = self.key_conv(k_in)    # (B, D, 1, V)
        V = self.value_conv(x_skel_pool)  # (B, D, 1, V)

        D = Q.size(1)  # inter_channels
        S_q = H * W
        S_k = K.size(2) * K.size(3)  # 1 * V = V

        # Reshape to multi-head: (B, num_heads, head_dim, S)
        Q = Q.view(B, self.num_heads, self.head_dim, S_q)   # (B, nh, hd, H*W)
        K = K.view(B, self.num_heads, self.head_dim, S_k)   # (B, nh, hd, V)
        V = V.view(B, self.num_heads, self.head_dim, S_k)   # (B, nh, hd, V)

        # Attention: (B, nh, H*W, V)
        attn = torch.einsum("bhds,bhdk->bhsk", Q, K) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum: (B, nh, hd, H*W)
        out = torch.einsum("bhsk,bhdk->bhds", attn, V)

        # Merge heads -> (B, D, H, W)
        out = out.reshape(B, D, H, W)

        # Output projection -> (B, C_r, H, W)
        out = self.out_proj(out)

        # Gated residual connection
        g = self.gate(out).unsqueeze(-1).unsqueeze(-1)  # (B, C_r, 1, 1)
        out = g * out + x_rgb

        return out