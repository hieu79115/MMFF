import torch
import torch.nn as nn
from models.st_gcn import SkeletonStream_STGCN
from models.backbone import RGBStream_Base       
from models.fusion import FusionTransformer

class MMFF_Net_Advanced(nn.Module):
    def __init__(self, num_classes=60, dataset='ntu'): # Thêm dataset
        super(MMFF_Net_Advanced, self).__init__()
        
        # 1. Skeleton Branch: ST-GCN
        # ST-GCN trả về cả vector (cho fusion) và feature map (cho attention)
        self.skel_encoder = SkeletonStream_STGCN(in_channels=3, num_class=num_classes, dataset=dataset)
        
        # 2. RGB Branch (Chứa ResNet + CrossAttention)
        # out_dim của ST-GCN block cuối cùng là 256
        self.rgb_encoder = RGBStream_Base(skel_channels=256) 
        
        # 3. Late Fusion: Transformer
        # ST-GCN output vector là 256, RGB ResNet output là 2048
        self.fusion_head = FusionTransformer(skel_dim=256, rgb_dim=2048, num_classes=num_classes)

    def forward(self, skel_input, rgb_input):
        # --- Skeleton Path ---
        # Nhận về 2 giá trị: Vector đặc trưng và Feature Map không gian-thời gian
        skel_vec, skel_map = self.skel_encoder(skel_input) 
        
        # --- RGB Path ---
        # Truyền Feature Map xương vào để RGB stream "nhìn" và chú ý
        rgb_vec = self.rgb_encoder(rgb_input, skel_map)
        
        # --- Fusion Path ---
        # Hợp nhất 2 vector đặc trưng cấp cao
        logits = self.fusion_head(skel_vec, rgb_vec)
        
        return logits