import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from PIL import Image
from torchvision import transforms

class MMFFDataset(Dataset):
    def __init__(self, 
                 root_dir='./data',     
                 mode='train', 
                 is_dummy=True, 
                 num_samples=100,
                 num_classes=60,
                 dataset='ntu'):
        
        self.mode = mode
        self.is_dummy = is_dummy
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.dataset_name = dataset
        self.root_dir = root_dir
        
        # Cấu hình kích thước
        self.num_frames = 32      
        self.img_size = 299       
        
        if self.dataset_name == 'utd':
            self.num_joints = 20
        else:
            self.num_joints = 25

        # Transform cho ảnh RGB
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- LOAD DỮ LIỆU THẬT ---
        # Logic: Chỉ load khi is_dummy=False
        if not self.is_dummy:
            self._load_real_data()

    def _load_real_data(self):
        prefix = 'train' if self.mode == 'train' else 'val'
        
        # Đường dẫn file
        data_path = os.path.join(self.root_dir, f'{prefix}_data.npy')
        label_path = os.path.join(self.root_dir, f'{prefix}_label.pkl')
        
        # 1. Load Label (.pkl) trước để biết độ dài
        try:
            with open(label_path, 'rb') as f:
                self.sample_name, self.labels = pickle.load(f)
        except FileNotFoundError:
             print(f"Lỗi: Không tìm thấy file {label_path}")
             # Tạo list rỗng để tránh crash ngay lập tức, nhưng sẽ báo lỗi sau
             self.labels = []
             raise

        # 2. Load Skeleton (.npy)
        try:
            self.skeleton_data = np.load(data_path, mmap_mode='r') 
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file {data_path}")
            raise
             
        print(f"Loaded {self.mode} data: {len(self.labels)} samples.")

    def __len__(self):
        if self.is_dummy:
            return self.num_samples
        else:
            # SỬA LỖI Ở ĐÂY: Trả về độ dài thật thay vì 0
            return len(self.labels)

    def __getitem__(self, idx):
        if self.is_dummy:
            return self._get_dummy_item()
        else:
            return self._get_real_item(idx)

    def _get_dummy_item(self):
        skeleton_feat = torch.randn(3, self.num_frames, self.num_joints)
        rgb_img = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return skeleton_feat, rgb_img, 0, label

    def _get_real_item(self, idx):
        # 1. Skeleton
        # Input shape: (N, 3, T, V, M) -> Lấy người đầu tiên [..., 0] -> (3, T, V)
        skel = self.skeleton_data[idx, :, :, :, 0] 
        skel_tensor = torch.from_numpy(skel).float()

        # 2. RGB Image
        video_name = self.sample_name[idx]
        img_path = os.path.join(self.root_dir, 'images', video_name + '.jpg')
        
        try:
            image = Image.open(img_path).convert('RGB')
            rgb_tensor = self.transform(image)
        except Exception:
            # Fallback nếu ảnh lỗi: Trả về ảnh đen
            rgb_tensor = torch.zeros(3, self.img_size, self.img_size)

        # 3. Label
        label = self.labels[idx]
        
        return skel_tensor, rgb_tensor, 0, label