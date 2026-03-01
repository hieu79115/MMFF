import torch
import torch.nn as nn
from models.st_gcn import SkeletonStream_STGCN
from models.backbone import RGBStream_Base
from models.fusion import FusionTransformer

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
        # Đầu ra phụ cho Skeleton (để train riêng + auxiliary loss)
        self.skel_head = nn.Linear(256, num_classes)
        
        # 2. Nhánh RGB
        self.rgb_encoder = RGBStream_Base(skel_channels=256) 
        # Đầu ra phụ cho RGB (để train riêng + auxiliary loss)
        self.rgb_head = nn.Linear(2048, num_classes)
        
        # 3. Fusion
        self.fusion_head = FusionTransformer(
            skel_dim=256, 
            rgb_dim=2048, 
            embed_dim=512,
            num_heads=8,
            num_classes=num_classes,
            dropout=0.3,
        )

    def forward(self, skel_input, rgb_input, stage='fusion'):
        """
        stage: 'skeleton', 'rgb', hoặc 'fusion'
        
        Returns:
            - stage='skeleton': logits (B, C)
            - stage='rgb':      logits (B, C)
            - stage='fusion':   dict {
                  'logits':      fusion logits  (B, C),
                  'skel_logits': auxiliary skel  (B, C),
                  'rgb_logits':  auxiliary rgb   (B, C),
              }
        """
        # --- Stage 1: Train riêng Skeleton ---
        if stage == 'skeleton':
            skel_vec, _ = self.skel_encoder(skel_input)
            return self.skel_head(skel_vec)
            
        # --- Stage 2: Train riêng RGB ---
        if stage == 'rgb':
            with torch.no_grad():
                _, skel_map = self.skel_encoder(skel_input)
            rgb_vec = self.rgb_encoder(rgb_input, skel_map)
            return self.rgb_head(rgb_vec)

        # --- Stage 3: Fusion (trả thêm auxiliary logits) ---
        skel_vec, skel_map = self.skel_encoder(skel_input) 
        rgb_vec = self.rgb_encoder(rgb_input, skel_map)
        
        logits = self.fusion_head(skel_vec, rgb_vec)
        
        # Auxiliary predictions (giữ 2 nhánh riêng không bị degrade)
        skel_logits = self.skel_head(skel_vec)
        rgb_logits = self.rgb_head(rgb_vec)
        
        return {
            'logits': logits,
            'skel_logits': skel_logits,
            'rgb_logits': rgb_logits,
        }