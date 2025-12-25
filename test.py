import argparse
import torch
from torch.utils.data import DataLoader
import os

def plot_confusion_matrix(cm, classes, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix to {filename}")

def main():
    # --- 1. Cấu hình tham số dòng lệnh ---
    parser = argparse.ArgumentParser(description='Test MMFF Model')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'utd'], 
                        help='dataset name: ntu or utd')
    parser.add_argument('--stage', type=str, default='fusion', choices=['skeleton', 'rgb', 'fusion'],
                        help="Which stage checkpoint to evaluate: 'skeleton', 'rgb', or 'fusion'")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--is_dummy', action='store_true', help='Use dummy data for testing')
    
    args = parser.parse_args()

    # Import module của dự án (defer until after argparse so --help works without ML deps installed)
    from models.mmff_net import MMFF_Net_Advanced
    from utils.dataset import MMFFDataset
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cấu hình số lớp
    if args.dataset == 'ntu':
        NUM_CLASSES = 60
    else:
        NUM_CLASSES = 27
        
    # Keep compatibility with both naming schemes:
    # - New (train.py): best_{stage}_{dataset}.pth
    # - Old: best_model_{dataset}.pth
    preferred_model_path = f"best_{args.stage}_{args.dataset}.pth"
    legacy_model_path = f"best_model_{args.dataset}.pth"
    MODEL_PATH = preferred_model_path if os.path.exists(preferred_model_path) else legacy_model_path
    
    print(f"Evaluating on {args.dataset.upper()} (Classes: {NUM_CLASSES})...")
    
    # --- 2. Load Test Data ---
    # is_dummy=True nếu bạn chỉ muốn test code, False nếu chạy thật
    test_dataset = MMFFDataset(mode='test', is_dummy=args.is_dummy, 
                               num_samples=50, num_classes=NUM_CLASSES, 
                               dataset=args.dataset)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- 3. Khởi tạo Model ---
    model = MMFF_Net_Advanced(num_classes=NUM_CLASSES, dataset=args.dataset)
    
    # Load weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        print(f"ERROR: Weight file {MODEL_PATH} not found. Train first!")
        return

    model.to(DEVICE)
    model.eval()
    
    # --- 4. Evaluation ---
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for skel, rgb, _, labels in test_loader:
            skel = skel.to(DEVICE)
            rgb = rgb.to(DEVICE)
            
            outputs = model(skel, rgb, stage=args.stage)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # --- 5. Report ---
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency for metrics/plots. Install with: pip install -r requirements.txt"
        ) from e

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n>>> Final Test Accuracy: {acc*100:.2f}%")
    
    if args.is_dummy:
        print("Note: Accuracy is random because you are using Dummy Data.")
    else:
        # Chỉ vẽ Confusion Matrix khi chạy dữ liệu thật hoặc muốn test
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, range(NUM_CLASSES), f'confusion_matrix_{args.dataset}.png')

if __name__ == "__main__":
    main()