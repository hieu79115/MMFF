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
        
        # Deeper MLP head with GELU activation and intermediate layer for CLS token
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # Modality-specific classification head (shared for skeleton & RGB token)
        self.modality_head = nn.Linear(embed_dim, num_classes)

        # Gating network để học trọng số giữa hai modality dựa trên token sau fusion
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 2)
        )

    def forward(self, f_skel, f_rgb):
        B = f_skel.shape[0]
        token_skel = self.skel_proj(f_skel).unsqueeze(1)
        token_rgb = self.rgb_proj(f_rgb).unsqueeze(1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, token_skel, token_rgb), dim=1)
        
        x = self.transformer(x)

        # Token [CLS] dùng cho classification toàn cục
        cls_out = x[:, 0]
        logits_cls = self.mlp_head(cls_out)

        # Token modality sau fusion
        skel_token = x[:, 1]
        rgb_token = x[:, 2]

        # Logits riêng cho từng modality
        logits_skel = self.modality_head(skel_token)
        logits_rgb = self.modality_head(rgb_token)

        # Gating theo độ tin cậy tương đối của hai modality
        gate_input = torch.cat([skel_token, rgb_token], dim=-1)
        gates = torch.softmax(self.gate(gate_input), dim=-1)  # (B, 2)

        # Kết hợp logits của hai modality theo trọng số học được
        gates = gates.unsqueeze(-1)  # (B, 2, 1)
        logits_modal = gates[:, 0] * logits_skel + gates[:, 1] * logits_rgb

        # Tổng hợp: CLS head + gated modality head
        logits = logits_cls + logits_modal

        return logits