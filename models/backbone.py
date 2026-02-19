import torch
import torch.nn as nn
import timm 

class RGBStream_Base(nn.Module):
    """
    Nhánh RGB: Trích xuất đặc trưng bằng Xception backbone.
    Cross-Attention đã được chuyển sang phía Skeleton (trong mmff_net.py).
    Trả về cả vector và feature map để phục vụ Cross-Attention.
    """
    def __init__(self):
        super(RGBStream_Base, self).__init__()
        
        # Load Xception từ timm (pretrained=True để lấy trọng số đã học ImageNet)
        # features_only=True: Chỉ lấy feature maps, bỏ lớp phân loại cuối
        self.backbone = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        
        # Xception trả về feature map có 2048 channels ở lớp cuối cùng
        self.out_channels = 2048
        
        # Pooling để chuyển về vector
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_rgb):
        # timm trả về một list các feature maps từ nông đến sâu
        features = self.backbone(x_rgb)
        
        # Ta lấy cái cuối cùng (feature map sâu nhất)
        f_rgb_map = features[-1] 
        # Shape dự kiến: (Batch, 2048, 10, 10) với ảnh đầu vào 299x299
        
        # Pooling & Flatten → vector
        f_rgb_vec = self.avg_pool(f_rgb_map).flatten(1)  # (B, 2048)
        
        # Trả về cả vector (cho classifier/fusion) và feature map (cho Cross-Attention)
        return f_rgb_vec, f_rgb_map