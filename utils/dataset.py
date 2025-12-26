import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from PIL import Image
from torchvision import transforms

class MMFFDataset(Dataset):
    def __init__(self, root_dir='./data', mode='train', is_dummy=True, 
                 num_samples=100, num_classes=60, dataset='ntu',
                 val_ratio: float = 0.1, split_seed: int = 42):
        
        # Supported modes:
        # - 'train': training split (from train_* files)
        # - 'val'  : validation split (held-out from train_* files)
        # - 'test' : test split (from val_* files, kept for backward compatibility)
        self.mode = (mode or 'train').lower()
        self.is_dummy = is_dummy
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.dataset_name = dataset
        self.root_dir = root_dir
        self.num_frames = 32      
        self.img_size = 299       

        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        self._subset_indices = None
        
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
        if self.mode not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid mode '{self.mode}'. Expected one of: train, val, test")

        # Repo convention:
        # - train split stored as train_*
        # - held-out split stored as test_*
        # Backward compatibility:
        # - older preprocess exported held-out as val_* (we fall back)
        if self.mode == 'test':
            # Prefer new naming scheme
            data_path = os.path.join(self.root_dir, 'test_data.npy')
            label_path = os.path.join(self.root_dir, 'test_label.pkl')

            # Fall back to legacy naming if needed
            if not (os.path.exists(data_path) and os.path.exists(label_path)):
                data_path = os.path.join(self.root_dir, 'val_data.npy')
                label_path = os.path.join(self.root_dir, 'val_label.pkl')

            try:
                with open(label_path, 'rb') as f:
                    self.sample_name, self.labels = pickle.load(f)
                self.skeleton_data = np.load(data_path, mmap_mode='r')
            except Exception as e:
                print(f"Error loading data: {e}")
                self.sample_name, self.labels = [], []
            return

        # For 'train' and 'val': load full train_* and create a deterministic split.
        data_path = os.path.join(self.root_dir, 'train_data.npy')
        label_path = os.path.join(self.root_dir, 'train_label.pkl')

        try:
            with open(label_path, 'rb') as f:
                self.sample_name, self.labels = pickle.load(f)
            self.skeleton_data = np.load(data_path, mmap_mode='r')
        except Exception as e:
            print(f"Error loading data: {e}")
            self.sample_name, self.labels = [], []
            self._subset_indices = np.array([], dtype=np.int64)
            return

        n = len(self.labels)
        if n == 0:
            self._subset_indices = np.array([], dtype=np.int64)
            return

        # Clamp val_ratio to a safe range.
        vr = self.val_ratio
        if not np.isfinite(vr):
            vr = 0.1
        vr = max(0.0, min(0.5, float(vr)))

        val_count = int(round(vr * n))
        # Ensure both splits are non-empty when possible.
        if n >= 2:
            val_count = max(1, min(n - 1, val_count))
        else:
            val_count = 0

        rng = np.random.RandomState(self.split_seed)
        perm = rng.permutation(n)
        val_idx = perm[:val_count]
        train_idx = perm[val_count:]

        self._subset_indices = train_idx if self.mode == 'train' else val_idx

    def __len__(self):
        if self.is_dummy:
            return self.num_samples
        if self._subset_indices is not None:
            return int(len(self._subset_indices))
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_dummy: return self._get_dummy_item()

        real_idx = idx
        if self._subset_indices is not None:
            real_idx = int(self._subset_indices[idx])
        
        # 1. Skeleton
        skel = self.skeleton_data[real_idx, :, :, :, 0] 
        
        # --- DATA AUGMENTATION CHO SKELETON (Chỉ áp dụng khi Train) ---
        if self.mode == 'train':
            # Thêm nhiễu Gaussian ngẫu nhiên
            noise = np.random.normal(0, 0.01, skel.shape)
            skel = skel + noise
            
        skel_tensor = torch.from_numpy(skel).float()

        # 2. RGB Image
        video_name = self.sample_name[real_idx]
        video_name_str = str(video_name)
        if video_name_str.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_filename = video_name_str
        else:
            img_filename = video_name_str + '.jpg'
        img_path = os.path.join(self.root_dir, 'images', img_filename)
        try:
            image = Image.open(img_path).convert('RGB')
            rgb_tensor = self.transform(image)
        except:
            rgb_tensor = torch.zeros(3, self.img_size, self.img_size)

        label = self.labels[real_idx]
        return skel_tensor, rgb_tensor, 0, label

    def _get_dummy_item(self):
        # ... (giữ nguyên dummy)
        skel = torch.randn(3, self.num_frames, self.num_joints)
        rgb = torch.randn(3, self.img_size, self.img_size)
        label = int(np.random.randint(0, max(1, self.num_classes)))
        return skel, rgb, 0, label