import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.mmff_net import MMFF_Net_Advanced
from utils.dataset import MMFFDataset
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from test import get_class_names

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
    parser = argparse.ArgumentParser(description='Visualize Grad-CAM for MMFF Model')
    parser.add_argument('--data_dir', type=str, default='', help='Dataset directory containing train_data.pkl/test_data.pkl')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'utd', 'nw-ucla'], 
                        help='dataset name: ntu, utd, or nw-ucla')
    parser.add_argument('--model_path', type=str, default='', help='Path to trained model weights (.pth)')
    parser.add_argument('--is_dummy', action='store_true', help='Use dummy data for testing without real dataset')
    parser.add_argument('--target_class', type=int, default=None, help='Target class index to visualize. If None, uses the model prediction or the ground truth label.')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of skeleton frames')
    
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Xác định số lượng classes từ config
    from config import Config
    NUM_CLASSES = Config.get_num_classes(args.dataset)
    class_names = get_class_names(args.dataset, NUM_CLASSES)

    # Khởi tạo model
    model = MMFF_Net_Advanced(num_classes=NUM_CLASSES, dataset=args.dataset)
    
    if args.model_path:
        state = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {args.model_path}")
    else:
        print("Warning: No model path provided. Evaluating with random weights.")

    model.to(DEVICE)
    model.eval()

    # Dữ liệu
    if args.is_dummy or not args.data_dir:
        print("Using DUMMY data for visualization.")
        rgb_input = torch.rand(1, 3, 299, 299).to(DEVICE)
        
        if args.dataset == 'ntu':
            skel_input = torch.rand(1, 3, 32, 25, 2).to(DEVICE)
        else:
            skel_input = torch.rand(1, 3, 32, 20).to(DEVICE)
            
        target_label = 0 # Default target label for dummy data
    else:
        print(f"Loading real data from {args.data_dir} ({args.dataset})...")
        test_dataset = MMFFDataset(mode='test', is_dummy=False, 
                                   num_samples=10, num_classes=NUM_CLASSES, 
                                   dataset=args.dataset,
                                   stage='fusion',
                                   num_frames=args.num_frames,
                                   root_dir=args.data_dir)
                                   
        # Lấy 1 mẫu (batch đầu tiên)
        skel, rgb, _, labels = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)))
        
        rgb_input = rgb.to(DEVICE)
        skel_input = skel.to(DEVICE)
        target_label = labels[0].item()
        print(f"Successfully loaded a sample. Ground truth label: {target_label}")


    # Bọc model lại để chỉ truyền RGB qua forward (chuẩn đầu vào của pytorch_grad_cam)
    wrapper = ModelWrapper(model, skel_input).to(DEVICE)

    # Target Layer: Lớp Cross-Attention trong RGB encoder 
    target_layers_crossattention = [wrapper.model.rgb_encoder.cross_att]

    # Target Layer: Lớp Query Conv trong Cross-Attention 
    target_layers_before = [wrapper.model.rgb_encoder.cross_att.q_conv]

    # Quyết định target đánh dấu grad-cam
    if args.target_class is not None:
        target_category = [ClassifierOutputTarget(args.target_class)]
        print(f"Visualizing for specifically requested class: {args.target_class}")
    else:
        target_category = [ClassifierOutputTarget(target_label)]
        print(f"Visualizing for ground truth class: {target_label}")

    # 1. Grad-CAM SAU Cross-Attention
    print("Generating Grad-CAM AFTER Cross-Attention...")
    cam_crossattention = GradCAM(model=wrapper, target_layers=target_layers_crossattention)
    grayscale_cam_after = cam_crossattention(input_tensor=rgb_input, targets=target_category)[0, :]

    # 2. Grad-CAM TRƯỚC Cross-Attention
    print("Generating Grad-CAM BEFORE Cross-Attention...")
    cam_before = GradCAM(model=wrapper, target_layers=target_layers_before)
    grayscale_cam_before = cam_before(input_tensor=rgb_input, targets=target_category)[0, :]

    # Hiển thị
    rgb_img = np.transpose(rgb_input[0].cpu().numpy(), (1, 2, 0))
    # Denormalize ImageNet RGB (Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]) config from dataset.py
    rgb_img = rgb_img * 0.5 + 0.5 
    rgb_img = np.clip(rgb_img, 0, 1)

    visualization_after = show_cam_on_image(rgb_img, grayscale_cam_after, use_rgb=True)
    visualization_before = show_cam_on_image(rgb_img, grayscale_cam_before, use_rgb=True)

    class_name = class_names[target_label] if target_label < len(class_names) else f"Class {target_label}"

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rgb_img)
    ax[0].set_title(f'Original RGB Crop\n(Target: {class_name})')
    
    ax[1].imshow(visualization_before)
    ax[1].set_title('Grad-CAM BEFORE Cross-Attention')
    
    ax[2].imshow(visualization_after)
    ax[2].set_title('Grad-CAM AFTER Cross-Attention\n(With Skeleton Guidance)')
    
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    output_filename = f'gradcam_real_{args.dataset}.png' if not args.is_dummy else 'gradcam_dummy.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Grad-CAM visualization saved to {output_filename}")
    plt.show() # Display inline in Jupyter Notebook / Kaggle

if __name__ == '__main__':
    main()
