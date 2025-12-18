import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    """
    CẢI TIẾN: Thay thế Late Fusion nối chuỗi của baseline bằng Transformer Encoder.
    Mục tiêu: Học sự tương tác phi tuyến tính giữa vector Skeleton và vector RGB.
    """
    def __init__(self, skel_dim, rgb_dim, embed_dim=256, num_heads=4, num_classes=60):
        super(FusionTransformer, self).__init__()
        
        # Project cả 2 đặc trưng về cùng không gian chiều (embed_dim)
        self.skel_proj = nn.Linear(skel_dim, embed_dim)
        self.rgb_proj = nn.Linear(rgb_dim, embed_dim)
        
        # Token phân loại [CLS] (Learnable Parameter)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer Encoder Layer (1 lớp là đủ cho Late Fusion)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Đầu ra phân loại
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, f_skel, f_rgb):
        # f_skel: (B, skel_dim)
        # f_rgb: (B, rgb_dim)
        B = f_skel.shape[0]
        
        # 1. Tạo tokens từ đặc trưng đầu vào
        token_skel = self.skel_proj(f_skel).unsqueeze(1) # (B, 1, E)
        token_rgb = self.rgb_proj(f_rgb).unsqueeze(1)    # (B, 1, E)
        
        # 2. Tạo CLS token cho từng batch
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 3. Ghép thành chuỗi sequence: [CLS, Skel, RGB]
        x = torch.cat((cls_tokens, token_skel, token_rgb), dim=1) # (B, 3, E)
        
        # 4. Cho qua Transformer để trộn thông tin
        x = self.transformer(x)
        
        # 5. Lấy token đầu tiên (CLS) để phân loại
        cls_out = x[:, 0]
        logits = self.mlp_head(cls_out)
        
        return logits