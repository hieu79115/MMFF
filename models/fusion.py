import torch
import torch.nn as nn


class FusionTransformer(nn.Module):
    """
    Improved Fusion Transformer for multi-modal action recognition.

    Key improvements over the original:
    1. Multi-token projection: each modality is split into multiple tokens
       so the sequence is long enough for self-attention to be meaningful.
    2. Pre-LN Transformer (norm_first=True) for more stable training.
    3. LayerNorm on raw inputs before projection to handle the large
       scale difference between skeleton (256-d) and RGB (2048-d).
    4. Learnable positional encoding + modality-type embeddings
       (analogous to BERT segment embeddings).
    5. CLS token initialized with small random values instead of zeros.
    6. Model size reduced to match small-dataset regimes
       (embed_dim=256, 4 heads, 2 layers by default).
    """

    def __init__(
        self,
        skel_dim,
        rgb_dim,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_skel_tokens=4,
        num_rgb_tokens=8,
        num_classes=60,
        dropout=0.3,
    ):
        super(FusionTransformer, self).__init__()

        self.num_skel_tokens = num_skel_tokens
        self.num_rgb_tokens = num_rgb_tokens
        total_tokens = 1 + num_skel_tokens + num_rgb_tokens   # CLS + skel + rgb

        # --- Input normalisation (handles 256-d vs 2048-d scale gap) ---
        self.skel_norm = nn.LayerNorm(skel_dim)
        self.rgb_norm = nn.LayerNorm(rgb_dim)

        # --- Multi-token projection ---
        # Each modality is projected into *multiple* tokens so that the
        # transformer sees a sequence of 13 tokens instead of just 3.
        self.skel_proj = nn.Linear(skel_dim, embed_dim * num_skel_tokens)
        self.rgb_proj = nn.Linear(rgb_dim, embed_dim * num_rgb_tokens)

        # --- Special tokens & embeddings ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, total_tokens, embed_dim) * 0.02
        )
        # 3 modality types: 0 = CLS, 1 = skeleton, 2 = RGB
        self.modality_embed = nn.Parameter(torch.randn(3, 1, embed_dim) * 0.02)

        # --- Pre-LN Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
            norm_first=True,           # Pre-LN: more stable gradients
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # --- Classification head ---
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, f_skel, f_rgb):
        B = f_skel.shape[0]

        # 1. Normalise raw features
        f_skel = self.skel_norm(f_skel)
        f_rgb = self.rgb_norm(f_rgb)

        # 2. Create multi-token representations
        skel_tokens = self.skel_proj(f_skel).view(
            B, self.num_skel_tokens, -1
        )  # (B, 4, embed_dim)
        rgb_tokens = self.rgb_proj(f_rgb).view(
            B, self.num_rgb_tokens, -1
        )  # (B, 8, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)

        # 3. Build sequence: [CLS, skel_1..skel_k, rgb_1..rgb_m]
        x = torch.cat([cls_tokens, skel_tokens, rgb_tokens], dim=1)

        # 4. Add positional + modality embeddings
        x = x + self.pos_embed

        s_end = 1 + self.num_skel_tokens
        x[:, 0:1] = x[:, 0:1] + self.modality_embed[0]        # CLS
        x[:, 1:s_end] = x[:, 1:s_end] + self.modality_embed[1] # skeleton
        x[:, s_end:] = x[:, s_end:] + self.modality_embed[2]   # RGB

        # 5. Transformer
        x = self.transformer(x)

        # 6. Classify from CLS token
        cls_out = self.final_norm(x[:, 0])
        logits = self.mlp_head(cls_out)

        return logits