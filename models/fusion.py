import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(self, skel_dim, rgb_dim, embed_dim=512, num_heads=8, num_classes=60, dropout=0.3):
        super(FusionTransformer, self).__init__()
        
        self.skel_proj = nn.Linear(skel_dim, embed_dim)
        self.rgb_proj = nn.Linear(rgb_dim, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Upgrade: 3 layers with larger FFN (dim_feedforward = embed_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # 2048 for embed_dim=512
            batch_first=True, 
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Deeper MLP head with GELU activation and intermediate layer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, f_skel, f_rgb):
        B = f_skel.shape[0]
        token_skel = self.skel_proj(f_skel).unsqueeze(1)
        token_rgb = self.rgb_proj(f_rgb).unsqueeze(1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, token_skel, token_rgb), dim=1)
        
        x = self.transformer(x)
        
        cls_out = x[:, 0]
        logits = self.mlp_head(cls_out)
        
        return logits


class FusionElementwise(nn.Module):
    def __init__(
        self,
        skel_dim: int,
        rgb_dim: int,
        embed_dim: int = 512,
        num_classes: int = 60,
        op: str = 'add',
        dropout: float = 0.3,
    ):
        super().__init__()

        op = op.lower().strip()
        if op not in {'add'}:
            raise ValueError(f"FusionElementwise op must be 'add', got: {op}")

        self.op = op
        self.skel_proj = nn.Linear(skel_dim, embed_dim)
        self.rgb_proj = nn.Linear(rgb_dim, embed_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, f_skel: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        x_skel = self.skel_proj(f_skel)
        x_rgb = self.rgb_proj(f_rgb)

        fused = x_skel + x_rgb

        return self.mlp_head(fused)


class FusionConcat(nn.Module):
    def __init__(
        self,
        skel_dim: int,
        rgb_dim: int,
        embed_dim: int = 512,
        num_classes: int = 60,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.skel_proj = nn.Linear(skel_dim, embed_dim)
        self.rgb_proj = nn.Linear(rgb_dim, embed_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_classes),
        )

    def forward(self, f_skel: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        x_skel = self.skel_proj(f_skel)
        x_rgb = self.rgb_proj(f_rgb)
        fused = torch.cat([x_skel, x_rgb], dim=1)
        return self.mlp_head(fused)