import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from PIL import Image
from torchvision import transforms


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _skeleton_to_c_t_v(skel: np.ndarray, num_joints: int | None = None) -> np.ndarray:
    """Normalize skeleton array to shape (C, T, V) with C=3.

    Accepts common variants:
    - (T, V, C)
    - (C, T, V)
    - (T, C, V)
    - with extra leading dims (e.g., person/M): takes first element
    """
    skel = _to_numpy(skel)
    if skel is None:
        raise ValueError("Skeleton is None")

    # Peel off extra leading dims (e.g., M persons)
    while skel.ndim > 3:
        skel = skel[0]

    if skel.ndim != 3:
        raise ValueError(f"Unsupported skeleton ndim={skel.ndim}, shape={getattr(skel, 'shape', None)}")

    a, b, c = skel.shape

    # Heuristics based on where coordinate dimension (3) likely is.
    if c == 3:  # (T, V, C)
        skel = np.transpose(skel, (2, 0, 1))
    elif a == 3:  # (C, T, V)
        pass
    elif b == 3:  # (T, C, V)
        skel = np.transpose(skel, (1, 0, 2))
    else:
        # If none axis is 3, assume last axis is coords and truncate/pad.
        skel = skel.astype(np.float32)
        if skel.shape[-1] > 3:
            skel = skel[..., :3]
        elif skel.shape[-1] < 3:
            pad = [(0, 0)] * skel.ndim
            pad[-1] = (0, 3 - skel.shape[-1])
            skel = np.pad(skel, pad, mode='constant')
        # Now treat as (T, V, C)
        skel = np.transpose(skel, (2, 0, 1))

    skel = skel.astype(np.float32)
    if num_joints is not None and skel.shape[2] != num_joints:
        # If joint count mismatches, attempt to truncate or pad joints.
        v = skel.shape[2]
        if v > num_joints:
            skel = skel[:, :, :num_joints]
        else:
            pad = [(0, 0), (0, 0), (0, num_joints - v)]
            skel = np.pad(skel, pad, mode='constant')
    return skel


def _temporal_resample_c_t_v(skel_c_t_v: np.ndarray, target_t: int) -> np.ndarray:
    """Resample (C,T,V) to (C,target_t,V) using linear interpolation along time."""
    c, t, v = skel_c_t_v.shape
    if t == target_t:
        return skel_c_t_v
    if t <= 0 or target_t <= 0:
        return np.zeros((c, max(0, target_t), v), dtype=np.float32)
    if t == 1:
        return np.repeat(skel_c_t_v, target_t, axis=1)

    x_old = np.linspace(0.0, 1.0, t, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_t, dtype=np.float32)

    out = np.empty((c, target_t, v), dtype=np.float32)
    for ci in range(c):
        for vi in range(v):
            out[ci, :, vi] = np.interp(x_new, x_old, skel_c_t_v[ci, :, vi]).astype(np.float32)
    return out


def _to_pil_rgb(img):
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert('RGB')
    arr = _to_numpy(img)
    if arr is None:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert('RGB')

class MMFFDataset(Dataset):
    def __init__(self, root_dir='./data', mode='train', is_dummy=True, 
                 num_samples=100, num_classes=60, dataset='ntu',
                 val_ratio: float = 0.1, split_seed: int = 42,
                 stage: str | None = None,
                 num_frames: int = 32):
        
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
        self.num_frames = int(num_frames)
        self.img_size = 299       

        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        self._subset_indices = None

        # Training stage affects which skeleton stream to use.
        # - 'skeleton': uses augmented_skeletons (stage 1)
        # - 'rgb'/'fusion': uses normalized_skeleton (stage 2/3)
        self.stage = (stage or 'fusion').lower()
        
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

        # New convention (2026-01): train_data.pkl / test_data.pkl
        # Each is a list of dict samples with keys:
        #   - video_name
        #   - label
        #   - normalized_skeleton
        #   - augmented_skeletons
        #   - rgb_crop
        new_train_pkl = os.path.join(self.root_dir, 'train_data.pkl')
        new_test_pkl = os.path.join(self.root_dir, 'test_data.pkl')

        self.samples = None

        # If new pkls exist, prefer them.
        if self.mode == 'test' and os.path.exists(new_test_pkl):
            try:
                with open(new_test_pkl, 'rb') as f:
                    self.samples = pickle.load(f)
                if not isinstance(self.samples, (list, tuple)):
                    raise ValueError("test_data.pkl must contain a list of sample dicts")
            except Exception as e:
                print(f"Error loading new test_data.pkl: {e}")
                self.samples = []
            self._subset_indices = None
            return

        if self.mode in {'train', 'val'} and os.path.exists(new_train_pkl):
            try:
                with open(new_train_pkl, 'rb') as f:
                    self.samples = pickle.load(f)
                if not isinstance(self.samples, (list, tuple)):
                    raise ValueError("train_data.pkl must contain a list of sample dicts")
            except Exception as e:
                print(f"Error loading new train_data.pkl: {e}")
                self.samples = []
                self._subset_indices = np.array([], dtype=np.int64)
                return

            n = len(self.samples)
            if n == 0:
                self._subset_indices = np.array([], dtype=np.int64)
                return

            vr = self.val_ratio
            if not np.isfinite(vr):
                vr = 0.1
            vr = max(0.0, min(0.5, float(vr)))

            val_count = int(round(vr * n))
            if n >= 2:
                val_count = max(1, min(n - 1, val_count))
            else:
                val_count = 0

            rng = np.random.RandomState(self.split_seed)
            perm = rng.permutation(n)
            val_idx = perm[:val_count]
            train_idx = perm[val_count:]
            self._subset_indices = train_idx if self.mode == 'train' else val_idx
            return

        # ---- Legacy convention fallback (npy + label pkl) ----

        # Repo convention:
        # - train split stored as train_*
        # - held-out split stored as test_*
        # Backward compatibility:
        # - older preprocess exported held-out as val_* (we fall back)
        if self.mode == 'test':
            data_path = os.path.join(self.root_dir, 'test_data.npy')
            label_path = os.path.join(self.root_dir, 'test_label.pkl')

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
        if self.samples is not None:
            if self._subset_indices is not None:
                return int(len(self._subset_indices))
            return int(len(self.samples))
        if self._subset_indices is not None:
            return int(len(self._subset_indices))
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_dummy: return self._get_dummy_item()

        real_idx = idx
        if self._subset_indices is not None:
            real_idx = int(self._subset_indices[idx])

        # ---- New pkl schema path ----
        if self.samples is not None:
            sample = self.samples[real_idx]
            if not isinstance(sample, dict):
                raise ValueError(f"Sample at idx={real_idx} is not a dict")

            label = int(sample.get('label', 0))

            # 1) Skeleton: stage 'skeleton' uses augmented skeletons (train); stage 'rgb'/'fusion' use normalized.
            use_aug = (self.stage == 'skeleton') and (self.mode == 'train')
            if use_aug and isinstance(sample.get('augmented_skeletons', None), (list, tuple)) and len(sample['augmented_skeletons']) > 0:
                skel_raw = sample['augmented_skeletons'][np.random.randint(0, len(sample['augmented_skeletons']))]
            else:
                skel_raw = sample.get('normalized_skeleton', None)
                if skel_raw is None and isinstance(sample.get('augmented_skeletons', None), (list, tuple)) and len(sample['augmented_skeletons']) > 0:
                    skel_raw = sample['augmented_skeletons'][0]

            skel_c_t_v = _skeleton_to_c_t_v(skel_raw, num_joints=self.num_joints)
            skel_c_t_v = _temporal_resample_c_t_v(skel_c_t_v, self.num_frames)

            # Optional light noise augmentation (kept for parity with legacy pipeline)
            if self.mode == 'train':
                noise = np.random.normal(0, 0.01, skel_c_t_v.shape).astype(np.float32)
                skel_c_t_v = skel_c_t_v + noise

            skel_tensor = torch.from_numpy(skel_c_t_v).float()

            # 2) RGB crop directly from pkl if present
            img = sample.get('rgb_crop', None)
            pil = _to_pil_rgb(img)
            if pil is None:
                rgb_tensor = torch.zeros(3, self.img_size, self.img_size)
            else:
                rgb_tensor = self.transform(pil)

            return skel_tensor, rgb_tensor, 0, label

        # ---- Legacy path ----
        # 1. Skeleton
        skel = self.skeleton_data[real_idx, :, :, :, 0]

        # --- DATA AUGMENTATION CHO SKELETON (Chỉ áp dụng khi Train) ---
        if self.mode == 'train':
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