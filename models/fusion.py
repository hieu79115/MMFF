import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FusionTransformer(nn.Module):
    """
    Cải tiến Fusion module:
      1. Modality embeddings — Transformer biết token nào là skel / rgb.
      2. Positional embeddings — cho thứ tự [CLS, skel, rgb].
      3. CLS token init bằng trunc_normal (gradient flow tốt hơn).
      4. Cross-Attention layer trước self-attention: skel attend vào rgb và ngược lại,
         tạo cross-modal interaction mạnh hơn self-attention 3-token.
      5. Bilinear pooling shortcut: capture multiplicative interaction giữa 2 modality,
         bổ sung cho additive interaction của Transformer.
      6. Deeper MLP head giữ nguyên.
    """

    def __init__(
        self,
        skel_dim: int,
        rgb_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 60,
        dropout: float = 0.3,
    ):
        super(FusionTransformer, self).__init__()
        self.embed_dim = embed_dim

        # --- Token projections ---
        self.skel_proj = nn.Sequential(
            nn.Linear(skel_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # --- Learnable special tokens & embeddings ---
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embeddings for 3 positions: [CLS=0, skel=1, rgb=2]
        self.pos_embed = nn.Parameter(torch.empty(1, 3, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Modality type embeddings: 0=CLS, 1=skeleton, 2=rgb
        self.modality_embed = nn.Parameter(torch.empty(1, 3, embed_dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

        # --- Cross-Attention layers (skel ↔ rgb explicit interaction) ---
        self.cross_attn_s2r = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_r2s = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm_s = nn.LayerNorm(embed_dim)
        self.cross_norm_r = nn.LayerNorm(embed_dim)

        # --- Self-Attention Transformer (fuse everything + CLS) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # --- Bilinear pooling shortcut ---
        # Compact bilinear: project down, element-wise product, project up
        self.bilinear_proj_s = nn.Linear(embed_dim, embed_dim)
        self.bilinear_proj_r = nn.Linear(embed_dim, embed_dim)
        self.bilinear_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Final classifier (merges CLS output + bilinear) ---
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, f_skel: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_skel: (B, skel_dim)  — skeleton global vector
            f_rgb:  (B, rgb_dim)   — RGB global vector
        Returns:
            logits: (B, num_classes)
        """
        B = f_skel.shape[0]

        # Project to embed_dim  →  (B, 1, D)
        token_s = self.skel_proj(f_skel).unsqueeze(1)
        token_r = self.rgb_proj(f_rgb).unsqueeze(1)

        # ---- Cross-Attention: explicit cross-modal interaction ----
        # Skeleton attends to RGB
        s_cross, _ = self.cross_attn_s2r(token_s, token_r, token_r)
        token_s = self.cross_norm_s(token_s + self.drop(s_cross))

        # RGB attends to Skeleton
        r_cross, _ = self.cross_attn_r2s(token_r, token_s, token_s)
        token_r = self.cross_norm_r(token_r + self.drop(r_cross))

        # ---- Assemble sequence [CLS, skel, rgb] ----
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, token_s, token_r), dim=1)  # (B, 3, D)

        # Add positional + modality embeddings
        x = x + self.pos_embed + self.modality_embed

        # ---- Self-Attention Transformer ----
        x = self.transformer(x)  # (B, 3, D)

        cls_out = x[:, 0]  # (B, D)

        # ---- Bilinear pooling shortcut ----
        s_out = x[:, 1]  # (B, D) — refined skeleton token
        r_out = x[:, 2]  # (B, D) — refined rgb token
        bilinear = self.bilinear_proj_s(s_out) * self.bilinear_proj_r(r_out)
        bilinear = self.bilinear_out(bilinear)  # (B, D)

        # ---- Merge & classify ----
        fused = torch.cat((cls_out, bilinear), dim=-1)  # (B, 2*D)
        logits = self.mlp_head(fused)

        return logits