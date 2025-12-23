import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from PIL import Image
from torchvision import transforms

class MMFFDataset(Dataset):
    def __init__(self, root_dir='./data', mode='train', is_dummy=True, 
                 num_samples=100, num_classes=60, dataset='ntu'):
        
        self.mode = mode
        self.is_dummy = is_dummy
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.dataset_name = dataset
        self.root_dir = root_dir
        self.num_frames = 32      
        self.img_size = 299       
        
        if self.dataset_name == 'utd': self.num_joints = 20
        else: self.num_joints = 25

        # Augmentation cho ảnh RGB (Mạnh hơn)
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Đổi màu nhẹ
                transforms.RandomHorizontalFlip(p=0.5), # Lật ảnh
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if not self.is_dummy:
            self._load_real_data()

    def _load_real_data(self):
        prefix = 'train' if self.mode == 'train' else 'val'
        data_path = os.path.join(self.root_dir, f'{prefix}_data.npy')
        label_path = os.path.join(self.root_dir, f'{prefix}_label.pkl')
        
        try:
            with open(label_path, 'rb') as f:
                self.sample_name, self.labels = pickle.load(f)
            self.skeleton_data = np.load(data_path, mmap_mode='r') 
        except Exception as e:
            print(f"Error loading data: {e}")
            self.labels = []

    def __len__(self):
        return self.num_samples if self.is_dummy else len(self.labels)

    def __getitem__(self, idx):
        if self.is_dummy: return self._get_dummy_item()
        
        # 1. Skeleton
        skel = self.skeleton_data[idx, :, :, :, 0] 
        
        # --- DATA AUGMENTATION CHO SKELETON (Chỉ áp dụng khi Train) ---
        if self.mode == 'train':
            # Thêm nhiễu Gaussian ngẫu nhiên
            noise = np.random.normal(0, 0.01, skel.shape)
            skel = skel + noise
            
        skel_tensor = torch.from_numpy(skel).float()

        # 2. RGB Image
        video_name = self.sample_name[idx]
        img_path = os.path.join(self.root_dir, 'images', video_name + '.jpg')
        try:
            image = Image.open(img_path).convert('RGB')
            rgb_tensor = self.transform(image)
        except:
            rgb_tensor = torch.zeros(3, self.img_size, self.img_size)

        label = self.labels[idx]
        return skel_tensor, rgb_tensor, 0, label

    def _get_dummy_item(self):
        # ... (giữ nguyên dummy)
        return torch.randn(3, 32, 20), torch.randn(3, 299, 299), 0, 0