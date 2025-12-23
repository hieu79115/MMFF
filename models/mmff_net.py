import torch
import torch.nn as nn
from models.st_gcn import SkeletonStream_STGCN
from models.backbone import RGBStream_Base
from models.fusion import FusionTransformer

class MMFF_Net_Advanced(nn.Module):
    def __init__(self, num_classes=60, dataset='ntu'):
        super(MMFF_Net_Advanced, self).__init__()
        
        # 1. Nhánh Skeleton
        self.skel_encoder = SkeletonStream_STGCN(in_channels=3, num_class=num_classes, dataset=dataset)
        # Đầu ra phụ cho Skeleton (để train riêng)
        self.skel_head = nn.Linear(256, num_classes)
        
        # 2. Nhánh RGB
        self.rgb_encoder = RGBStream_Base(skel_channels=256) 
        # Đầu ra phụ cho RGB (để train riêng)
        self.rgb_head = nn.Linear(2048, num_classes)
        
        # 3. Fusion
        self.fusion_head = FusionTransformer(skel_dim=256, rgb_dim=2048, num_classes=num_classes)

    def forward(self, skel_input, rgb_input, stage='fusion'):
        """
        stage: 'skeleton', 'rgb', hoặc 'fusion'
        """
        # --- Stage 1: Train riêng Skeleton ---
        if stage == 'skeleton':
            skel_vec, _ = self.skel_encoder(skel_input)
            return self.skel_head(skel_vec) # Chỉ trả về kết quả nhánh xương
            
        # --- Stage 2: Train riêng RGB ---
        # (Vẫn cần chạy Skeleton encoder để lấy Feature Map cho Cross-Attention, nhưng không update weight xương)
        if stage == 'rgb':
            with torch.no_grad(): # Đóng băng nhánh xương
                _, skel_map = self.skel_encoder(skel_input)
            
            rgb_vec = self.rgb_encoder(rgb_input, skel_map)
            return self.rgb_head(rgb_vec) # Chỉ trả về kết quả nhánh RGB

        # --- Stage 3: Fusion (Chạy cả 2) ---
        skel_vec, skel_map = self.skel_encoder(skel_input) 
        rgb_vec = self.rgb_encoder(rgb_input, skel_map)
        logits = self.fusion_head(skel_vec, rgb_vec)
        return logits