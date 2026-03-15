import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.mmff_net import MMFF_Net_Advanced

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, skel_input):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.skel_input = skel_input

    def forward(self, rgb_input):
        # stage='fusion' để chạy cả hai nhánh phân loại
        logits = self.model(self.skel_input, rgb_input, stage='fusion')
        if isinstance(logits, dict):
            return logits['logits']
        return logits

def main():
    # Khởi tạo model
    model = MMFF_Net_Advanced(num_classes=60)
    model.eval()

    # Tạo dữ liệu giả (Dummy Data)
    # RGB 299x299
    rgb_input = torch.rand(1, 3, 299, 299)
    # Skeleton NTU: (1, 3, 32, 25, 2)
    skel_input = torch.rand(1, 3, 32, 25, 2)

    # Bọc model lại để chỉ truyền RGB qua forward (chuẩn đầu vào của pytorch_grad_cam)
    wrapper = ModelWrapper(model, skel_input)

    # Target Layer: Lớp Cross-Attention trong RGB encoder 
    # Tính Grad-CAM tại đầu ra của khối Cross-Attention
    target_layers_crossattention = [wrapper.model.rgb_encoder.cross_att]

    # Target Layer: Lớp Query Conv trong Cross-Attention 
    # (Đóng vai trò như feature map từ nhánh RGB TRƯỚC khi gộp với Skeleton)
    target_layers_before = [wrapper.model.rgb_encoder.cross_att.query_conv]

    # 1. Grad-CAM SAU Cross-Attention
    cam_crossattention = GradCAM(model=wrapper, target_layers=target_layers_crossattention)
    grayscale_cam_after = cam_crossattention(input_tensor=rgb_input, targets=None)[0, :]

    # 2. Grad-CAM TRƯỚC Cross-Attention
    cam_before = GradCAM(model=wrapper, target_layers=target_layers_before)
    grayscale_cam_before = cam_before(input_tensor=rgb_input, targets=None)[0, :]

    # Hiển thị
    rgb_img = np.transpose(rgb_input[0].numpy(), (1, 2, 0))
    # Normalize dummy RGB img for viewing
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

    visualization_after = show_cam_on_image(rgb_img, grayscale_cam_after, use_rgb=True)
    visualization_before = show_cam_on_image(rgb_img, grayscale_cam_before, use_rgb=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rgb_img)
    ax[0].set_title('Original Dummy RGB')
    
    ax[1].imshow(visualization_before)
    ax[1].set_title('Grad-CAM BEFORE Cross-Attention')
    
    ax[2].imshow(visualization_after)
    ax[2].set_title('Grad-CAM AFTER Cross-Attention\n(With Skeleton Guidance)')
    
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=300)
    print("Grad-CAM visualization saved to gradcam_visualization.png")

if __name__ == '__main__':
    main()
