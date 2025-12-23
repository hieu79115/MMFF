import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(self, skel_dim, rgb_dim, embed_dim=256, num_heads=4, num_classes=60, dropout=0.5):
        super(FusionTransformer, self).__init__()
        
        self.skel_proj = nn.Linear(skel_dim, embed_dim)
        self.rgb_proj = nn.Linear(rgb_dim, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Thêm Dropout vào Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Thêm Dropout trước lớp phân loại cuối cùng
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),  # Quan trọng: Dropout mạnh (0.5)
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