import torch
import torch.nn as nn
import torch.nn.functional as F
from models.st_gcn import SkeletonStream_STGCN
from models.backbone import RGBStream_Base
from models.fusion import FusionTransformer
from models.attention import CrossModalAttention

class MMFF_Net_Advanced(nn.Module):
    def __init__(self, num_classes=60, dataset='ntu', edge_importance_weighting: bool = False, stgcn_dropout: float = 0.0):
        super(MMFF_Net_Advanced, self).__init__()
        
        # 1. Nhánh Skeleton
        self.skel_encoder = SkeletonStream_STGCN(
            in_channels=3,
            num_class=num_classes,
            dataset=dataset,
            edge_importance_weighting=edge_importance_weighting,
            dropout=stgcn_dropout,
        )
        # Đầu ra phụ cho Skeleton (để train riêng)
        self.skel_head = nn.Linear(256, num_classes)
        
        # 2. Nhánh RGB (không còn cross-attention bên trong)
        self.rgb_encoder = RGBStream_Base() 
        # Đầu ra phụ cho RGB (để train riêng)
        self.rgb_head = nn.Linear(2048, num_classes)
        
        # 3. Cross-Attention: Skeleton (Query) được hướng dẫn bởi RGB (Key/Value)
        self.cross_att = CrossModalAttention(skel_channels=256, rgb_channels=2048)
        
        # 4. Fusion
        self.fusion_head = FusionTransformer(
            skel_dim=256, 
            rgb_dim=2048, 
            embed_dim=512,      # Updated from 256
            num_heads=8,        # Updated from 4
            num_classes=num_classes,
            dropout=0.3         # Updated from 0.5
        )

    def forward(self, skel_input, rgb_input, stage='fusion'):
        """
        stage: 'skeleton', 'rgb', hoặc 'fusion'
        
        Thứ tự train mới (reversed cross-attention):
            1. Train 'rgb' trước (RGB encoder đơn lẻ)
            2. Train 'skeleton' (Skeleton + Cross-Attention, freeze RGB)
            3. Train 'fusion' (Toàn bộ mạng)
        """
        # --- Stage 1: Train riêng RGB (train trước vì skeleton cần RGB feature map) ---
        if stage == 'rgb':
            rgb_vec, _ = self.rgb_encoder(rgb_input)
            return self.rgb_head(rgb_vec)
        
        # --- Stage 2: Train Skeleton với Cross-Attention ---
        # Cần chạy RGB encoder (đóng băng) để lấy feature map cho Cross-Attention
        if stage == 'skeleton':
            with torch.no_grad():  # Đóng băng nhánh RGB
                _, rgb_map = self.rgb_encoder(rgb_input)
            
            _, skel_map = self.skel_encoder(skel_input)
            
            # Cross-Attention: Skeleton (Query) attend tới RGB (Key/Value)
            skel_enhanced = self.cross_att(skel_map, rgb_map)
            skel_vec_enhanced = F.adaptive_avg_pool2d(skel_enhanced, 1).flatten(1)  # (B, 256)
            
            return self.skel_head(skel_vec_enhanced)

        # --- Stage 3: Fusion (Chạy cả 2 + Cross-Attention) ---
        _, skel_map = self.skel_encoder(skel_input)
        rgb_vec, rgb_map = self.rgb_encoder(rgb_input)
        
        # Cross-Attention: Skeleton được tăng cường bởi RGB
        skel_enhanced = self.cross_att(skel_map, rgb_map)
        skel_vec_enhanced = F.adaptive_avg_pool2d(skel_enhanced, 1).flatten(1)  # (B, 256)
        
        logits = self.fusion_head(skel_vec_enhanced, rgb_vec)
        return logits