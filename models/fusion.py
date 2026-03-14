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

class GatedMultimodalFusion(nn.Module):
    """
    Gated Multimodal Fusion (GMF)
    Phương pháp này sử dụng cổng (gate) để kiểm soát lượng thông tin luân chuyển 
    từ mỗi modality dựa trên sigmoid. Rất ổn định trên tập dữ liệu nhỏ và khắc phục 
    sự nhạy cảm nhiễu của Transformer.
    """
    def __init__(self, skel_dim, rgb_dim, embed_dim=512, num_classes=60, dropout=0.3):
        super(GatedMultimodalFusion, self).__init__()
        
        self.skel_proj = nn.Sequential(
            nn.Linear(skel_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # Cổng Gate để đánh trọng số (attention channel-wise)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, f_skel, f_rgb):
        # 1. Chiếu về cùng không gian
        h_skel = self.skel_proj(f_skel)
        h_rgb = self.rgb_proj(f_rgb)
        
        # 2. Tính toán ma trận Gate dựa trên sự kết hợp của 2 đặc trưng
        cat_feat = torch.cat([h_skel, h_rgb], dim=-1)
        z = self.gate(cat_feat)
        
        # 3. Gated Fusion: Tính tổng có trọng số (Gating)
        h_fused = z * h_skel + (1 - z) * h_rgb
        
        # 4. Phân loại
        logits = self.mlp_head(h_fused)
        return logits

class CrossModalModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for Multimodal fusion.
    Sử dụng Skeleton (vốn mang thông tin cấu trúc mạnh) để làm "prior" 
    điều chuẩn (modulate) đặc trưng của RGB. 
    """
    def __init__(self, skel_dim, rgb_dim, embed_dim=512, num_classes=60, dropout=0.3):
        super(CrossModalModulation, self).__init__()
        
        self.rgb_proj = nn.Linear(rgb_dim, embed_dim)
        self.skel_proj = nn.Linear(skel_dim, embed_dim)
        
        # Layer tạo ra hệ số gamma (scale) và beta (shift) từ Skeleton
        self.film_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2)
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, f_skel, f_rgb):
        # Chiếu về cùng không gian
        h_rgb = self.rgb_proj(f_rgb)
        h_skel = self.skel_proj(f_skel)
        
        # Sinh hệ số gamma, beta từ Skeleton
        film_params = self.film_generator(h_skel)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        # Điều chuẩn không gian RGB bằng thông tin Skeleton
        h_fused = (1 + gamma) * h_rgb + beta
        
        # Có thể cộng thêm residual để giữ lại chút gốc
        h_fused = h_fused + h_skel 
        
        h_fused = self.dropout(h_fused)
        logits = self.mlp_head(h_fused)
        return logits