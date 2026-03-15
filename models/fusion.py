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

class DisentangledMultimodalFusion(nn.Module):
    """
    💎 HIGH CONTRIBUTION (TÍNH ĐÓNG GÓP CAO CHO LUẬN VĂN) 💎
    Disentangled Representation Multimodal Fusion (Hợp nhất theo biểu diễn tháo gỡ)
    
    Phương pháp này giải quyết một vấn đề cực kỳ học thuật: 
    "Làm sao để mô hình biết được thông tin nào là cốt lõi dùng chung (Shared) 
    và thông tin nào là đặc trưng riêng của mỗi modality (Private)?"
    
    Cách hoạt động:
    1. Tách đặc trưng RGB thành: RGB_shared + RGB_private
    2. Tách đặc trưng Skeleton thành: Skel_shared + Skel_private
    3. Trích xuất Consensus (Sự đồng thuận) từ 2 nhánh Shared thông qua Cross-Attention đơn giản.
    4. Fuse kết quả: Consensus + RGB_private + Skel_private.
    
    Trong luận văn, bạn có thể lập luận rằng phương pháp này chống lại sự dư thừa 
    thông tin (Redundancy) và giúp mô hình học bản chất hành động thực sự thay vì 
    học thuộc lòng nhiễu từ mô trường.
    """
    def __init__(self, skel_dim, rgb_dim, embed_dim=512, num_classes=60, dropout=0.3):
        super(DisentangledMultimodalFusion, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 1. Projectors để đưa về không gian trung gian
        self.skel_base = nn.Sequential(nn.Linear(skel_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.rgb_base = nn.Sequential(nn.Linear(rgb_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        
        # 2. Extractors tách không gian Feature (Shared vs Private)
        # Shared là những gì chung nhất (ví dụ: tư thế con người tồn tại ở cả hình và xương)
        self.skel_shared_extractor = nn.Linear(embed_dim, embed_dim // 2)
        self.skel_private_extractor = nn.Linear(embed_dim, embed_dim // 2)
        
        self.rgb_shared_extractor = nn.Linear(embed_dim, embed_dim // 2)
        self.rgb_private_extractor = nn.Linear(embed_dim, embed_dim // 2)
        
        # 3. Consensus Module (Đoạt lấy sự đồng thuận giữa 2 shared features)
        # Dùng một nhánh Multi-head Attention nhỏ chỉ trên nhánh Shared (rất ít tham số)
        self.consensus_attention = nn.MultiheadAttention(embed_dim // 2, num_heads=4, batch_first=True, dropout=dropout)
        
        # 4. Final MLP Classifier (Nhận vào Shared_consensus + Skel_private + RGB_private)
        # Tổng dimension = (embed_dim // 2) + (embed_dim // 2) + (embed_dim // 2) = embed_dim * 1.5
        fused_dim = int(embed_dim * 1.5)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, f_skel, f_rgb, return_features_for_loss=False):
        # Biến đổi cơ bản
        h_skel = self.skel_base(f_skel)
        h_rgb = self.rgb_base(f_rgb)
        
        # Tách nhánh (Disentanglement)
        s_shared = self.skel_shared_extractor(h_skel)
        s_priv = self.skel_private_extractor(h_skel)
        
        r_shared = self.rgb_shared_extractor(h_rgb)
        r_priv = self.rgb_private_extractor(h_rgb)
        
        # Tìm sự đồng thuận (Consensus) bằng Cross-Attention giữa các nhánh shared
        # Biến thành sequence length = 1 để đưa vào attention: (Batch, Seq, Feature)
        s_seq = s_shared.unsqueeze(1)
        r_seq = r_shared.unsqueeze(1)
        
        # Skel query RGB Context
        consensus, _ = self.consensus_attention(query=s_seq, key=r_seq, value=r_seq)
        consensus = consensus.squeeze(1) # (Batch, embed_dim // 2)
        
        # Hợp nhất: Consensus (Chung) + Skel_private (Đặc trưng xương riêng) + RGB_private (Bối cảnh hình ảnh riêng)
        fused_vector = torch.cat([consensus, s_priv, r_priv], dim=-1)
        
        # Phân loại
        logits = self.classifier(fused_vector)
        
        if return_features_for_loss:
            # Trả về thêm để tính Orthogonality Loss và Contrastive Loss trong train.py
            return logits, (s_shared, r_shared, s_priv, r_priv)
            
        return logits