import os
import glob
import numpy as np
import pickle
import cv2
import scipy.io  # Cần cài scipy: pip install scipy
from tqdm import tqdm

def process_utd_mhad(raw_data_path, output_path, num_frames=32):
    """
    Chuyển đổi dữ liệu UTD-MHAD thô -> .npy và trích xuất ảnh.
    
    Cấu trúc thư mục raw_data_path mong đợi:
        raw_data_path/
            ├── Skeleton/   (Chứa file .mat)
            └── RGB/        (Chứa file .avi)
    """
    
    # Tạo thư mục output
    images_out_dir = os.path.join(output_path, 'images')
    os.makedirs(images_out_dir, exist_ok=True)
    
    # Lấy danh sách file skeleton
    skel_files = sorted(glob.glob(os.path.join(raw_data_path, 'Skeleton', '*.mat')))
    
    data_list = []   # Chứa skeleton
    labels_list = [] # Chứa label
    names_list = []  # Chứa tên file
    
    print(f"Tìm thấy {len(skel_files)} mẫu dữ liệu. Đang xử lý...")
    
    for skel_path in tqdm(skel_files):
        filename = os.path.basename(skel_path).replace('.mat', '')
        
        # 1. Parse tên file để lấy Label
        # Định dạng UTD: a1_s1_t1_skeleton.mat (a=action, s=subject, t=trial)
        try:
            parts = filename.split('_') # ['a1', 's1', 't1', 'skeleton']
            action_str = parts[0]       # 'a1'
            label = int(action_str.replace('a', '')) - 1 # Chuyển về 0-26
        except:
            print(f"Skip file lỗi tên: {filename}")
            continue

        # 2. Đọc Skeleton (.mat)
        try:
            mat = scipy.io.loadmat(skel_path)
            # UTD mat file thường có key 'd_skel' shape (20, 3, T)
            skeleton = mat['d_skel'] 
            # Chuyển về shape mong muốn (3, T, 20)
            skeleton = skeleton.transpose(1, 2, 0) 
        except Exception as e:
            print(f"Lỗi đọc skeleton {filename}: {e}")
            continue
            
        # Downsample/Pad về num_frames (32)
        C, T, V = skeleton.shape
        if T > num_frames:
            # Lấy mẫu cách đều
            indices = np.linspace(0, T-1, num_frames).astype(int)
            skeleton = skeleton[:, indices, :]
        elif T < num_frames:
            # Pad zero
            pad = np.zeros((C, num_frames - T, V))
            skeleton = np.concatenate((skeleton, pad), axis=1)
            
        # 3. Xử lý Video tương ứng để lấy Middle Frame
        # Tên video khớp: a1_s1_t1_color.avi
        video_name = filename.replace('_skeleton', '_color') + '.avi'
        video_path = os.path.join(raw_data_path, 'RGB', video_name)
        
        if not os.path.exists(video_path):
            # Thử tìm tên khác (đôi khi dataset đặt tên không nhất quán)
            # Logic này tùy thuộc dataset bạn tải về
            pass 
        
        # [cite_start]Trích xuất frame giữa [cite: 78, 412]
        cap = cv2.VideoCapture(video_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_f // 2)
            ret, frame = cap.read()
            if ret:
                # Lưu ảnh
                save_img_path = os.path.join(images_out_dir, filename + '.jpg')
                cv2.imwrite(save_img_path, frame) # Lưu file ảnh
            else:
                print(f"Lỗi đọc frame video {video_name}")
            cap.release()
        else:
             # Nếu không có video, tạo ảnh đen dummy (để code không crash)
             dummy_img = np.zeros((299, 299, 3), dtype=np.uint8)
             save_img_path = os.path.join(images_out_dir, filename + '.jpg')
             cv2.imwrite(save_img_path, dummy_img)

        # 4. Thêm vào list
        data_list.append(skeleton)
        labels_list.append(label)
        names_list.append(filename)

    # Convert sang numpy array
    # Shape cuối: (N, 3, 32, 20, 1) - Thêm dim 1 cuối cho giống format chuẩn ST-GCN (Person)
    all_data = np.stack(data_list) 
    all_data = all_data[:, :, :, :, np.newaxis] 
    
    # Chia Train/Test (Theo bài báo dùng Subject hoặc Random split)
    # Ở đây mình làm đơn giản Random Split 80/20
    total_samples = len(labels_list)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    split = int(0.8 * total_samples)
    train_idx, test_idx = indices[:split], indices[split:]
    
    # Lưu Train
    np.save(os.path.join(output_path, 'train_data.npy'), all_data[train_idx])
    with open(os.path.join(output_path, 'train_label.pkl'), 'wb') as f:
        pickle.dump(([names_list[i] for i in train_idx], [labels_list[i] for i in train_idx]), f)
        
    # Lưu Test (held-out)
    np.save(os.path.join(output_path, 'test_data.npy'), all_data[test_idx])
    with open(os.path.join(output_path, 'test_label.pkl'), 'wb') as f:
        pickle.dump(([names_list[i] for i in test_idx], [labels_list[i] for i in test_idx]), f)
        
    print("Hoàn tất! Dữ liệu đã sẵn sàng tại:", output_path)

if __name__ == "__main__":
    # SỬA ĐƯỜNG DẪN CỦA BẠN Ở ĐÂY
    raw_path = r"" 
    out_path = r"./data"
    
    process_utd_mhad(raw_path, out_path)