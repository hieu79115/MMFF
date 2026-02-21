import torch
import torch.nn as nn

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

        # Concat dimensional: embed_dim + embed_dim = 2 * embed_dim
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