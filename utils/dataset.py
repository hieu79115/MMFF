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

    @staticmethod
    def _parse_label_object(obj):
        # Common formats:
        # 1) (sample_name, labels)
        # 2) (sample_name, labels, ...extra)
        # 3) {'sample_name'/'names': [...], 'label'/'labels': [...]}
        # 4) [(name, label), (name, label), ...]

        if isinstance(obj, (tuple, list)):
            if len(obj) >= 2 and not (
                len(obj) > 0 and isinstance(obj[0], (tuple, list)) and len(obj[0]) == 2
            ):
                sample_name, labels = obj[0], obj[1]
                return list(sample_name), list(labels)

            if len(obj) > 0 and isinstance(obj[0], (tuple, list)) and len(obj[0]) == 2:
                sample_name, labels = zip(*obj)
                return list(sample_name), list(labels)

        if isinstance(obj, dict):
            name_keys = ('sample_name', 'sample_names', 'names', 'name')
            label_keys = ('labels', 'label', 'y')
            name_key = next((k for k in name_keys if k in obj), None)
            label_key = next((k for k in label_keys if k in obj), None)
            if name_key is not None and label_key is not None:
                return list(obj[name_key]), list(obj[label_key])

        raise ValueError(
            'Unsupported label pickle format. Expected (names, labels), (names, labels, ...), '
            'a dict with names/labels keys, or a list of (name, label) pairs.'
        )

    def _load_real_data(self):
        prefix = 'train' if self.mode == 'train' else 'val'
        data_path = os.path.join(self.root_dir, f'{prefix}_data.npy')
        label_path = os.path.join(self.root_dir, f'{prefix}_label.pkl')
        
        try:
            with open(label_path, 'rb') as f:
                obj = pickle.load(f)
                self.sample_name, self.labels = self._parse_label_object(obj)
            self.skeleton_data = np.load(data_path, mmap_mode='r') 

            if len(self.sample_name) != len(self.labels):
                raise ValueError(
                    f'Label file is inconsistent: names={len(self.sample_name)} labels={len(self.labels)}'
                )
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
            
        # If skeleton data is memory-mapped (mmap_mode='r') or a view, it can be non-writable.
        # PyTorch warns because writing to such a tensor is undefined behavior.
        skel_tensor = torch.from_numpy(np.array(skel, copy=True)).float()

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