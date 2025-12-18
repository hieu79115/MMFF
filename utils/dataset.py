import torch
from torch.utils.data import Dataset
import numpy as np

class MMFFDataset(Dataset):
    def __init__(self, 
                 root_dir=None, 
                 mode='train', 
                 is_dummy=True, 
                 num_samples=100,
                 num_classes=60,      
                 dataset='ntu'):
        """
        Args:
            root_dir (str): Đường dẫn đến thư mục dữ liệu.
            mode (str): 'train' hoặc 'test'.
            is_dummy (bool): Nếu True, sinh dữ liệu ngẫu nhiên để test code.
            num_samples (int): Số lượng mẫu giả lập (nếu is_dummy=True).
            num_classes (int): Số lượng nhãn (NTU=60, UTD=27).
            dataset (str): 'ntu' hoặc 'utd'.
        """
        self.mode = mode
        self.is_dummy = is_dummy
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.dataset_name = dataset
        
        # Cấu hình kích thước theo bài báo NTU RGB+D
        # [cite_start]Tham chiếu từ bài báo: Skeleton sequence downsampled to 32 frames [cite: 407]
        self.num_frames = 32      
        self.img_size = 299       # Xception input size

        # Tự động chỉnh số khớp dựa trên tên bộ dữ liệu
        if self.dataset_name == 'utd':
            self.num_joints = 20
        else:
            self.num_joints = 25
        

    def __len__(self):
        if self.is_dummy:
            return self.num_samples
        else:
            # TODO: Trả về số lượng file thực tế trong dataset
            # Tạm thời return 0 hoặc raise NotImplementedError nếu chưa code
            return 0 

    def __getitem__(self, idx):
        if self.is_dummy:
            return self._get_dummy_item()
        else:
            return self._get_real_item(idx)

    def _get_dummy_item(self):
        """
        Sinh dữ liệu giả lập
        """
        # 1. Skeleton Sequence: (Channel, Time, Vertex)
        # Vertex sẽ tự động là 20 (UTD) hoặc 25 (NTU)
        skeleton_feat = torch.randn(3, self.num_frames, self.num_joints)
        
        # 2. RGB Frame: (3, 299, 299)
        rgb_img = torch.randn(3, self.img_size, self.img_size)
        
        # 3. Label
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        return skeleton_feat, rgb_img, 0, label

    def _get_real_item(self, idx):
        # TODO: Implement real data loading later
        pass