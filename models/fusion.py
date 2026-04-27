import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(
        self,
        skel_dim,
        rgb_dim,
        embed_dim=512,
        num_heads=8,
        num_classes=60,
        dropout=0.3,
        fusion_type: str = 'cmaf',
    ):
        super(FusionTransformer, self).__init__()

        self.fusion_type = (fusion_type or 'cmaf').lower()

        if self.fusion_type == 'cmaf':
            self.skel_proj = nn.Linear(skel_dim, embed_dim)
            self.rgb_proj = nn.Linear(rgb_dim, embed_dim)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                dropout=dropout
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )
        elif self.fusion_type in {'sum', 'average'}:
            self.skel_proj = nn.Linear(skel_dim, embed_dim)
            self.rgb_proj = nn.Linear(rgb_dim, embed_dim)
            self.simple_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        elif self.fusion_type == 'concat':
            concat_dim = skel_dim + rgb_dim
            self.concat_head = nn.Sequential(
                nn.LayerNorm(concat_dim),
                nn.Linear(concat_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def forward(self, f_skel, f_rgb):
        if self.fusion_type == 'cmaf':
            batch_size = f_skel.shape[0]
            token_skel = self.skel_proj(f_skel).unsqueeze(1)
            token_rgb = self.rgb_proj(f_rgb).unsqueeze(1)

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, token_skel, token_rgb), dim=1)

            x = self.transformer(x)
            cls_out = x[:, 0]
            return self.mlp_head(cls_out)

        if self.fusion_type == 'sum':
            fused = self.skel_proj(f_skel) + self.rgb_proj(f_rgb)
            return self.simple_head(fused)

        if self.fusion_type == 'average':
            fused = 0.5 * (self.skel_proj(f_skel) + self.rgb_proj(f_rgb))
            return self.simple_head(fused)

        fused = torch.cat([f_skel, f_rgb], dim=1)
        return self.concat_head(fused)