import argparse
import torch
from torch.utils.data import DataLoader
import os

def plot_confusion_matrix(cm, classes, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Scale figure size a bit with the number of classes (helps readability for UTD/NTU)
    n = len(classes)
    fig_w = max(10, min(24, 0.6 * n))
    fig_h = max(8, min(24, 0.55 * n))
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to {filename}")


def get_class_names(dataset: str, num_classes: int):
    dataset = dataset.lower()
    if dataset == 'utd':
        # UTD-MHAD (27 classes)
        return [
            'swipe_left',
            'swipe_right',
            'wave',
            'clap',
            'throw',
            'arm_cross',
            'basketball_shoot',
            'draw_x',
            'draw_circle_CW',
            'draw_circle_CCW',
            'draw_triangle',
            'bowling',
            'boxing',
            'baseball_swing',
            'tennis_swing',
            'arm_curl',
            'tennis_serve',
            'push',
            'knock',
            'catch',
            'pickup_throw',
            'jog',
            'walk',
            'sit2stand',
            'stand2sit',
            'lunge',
            'squat',
        ]

    if dataset == 'ntu':
        # NTU RGB+D 60 classes
        return [
            'drink_water',
            'eat_meal',
            'brush_teeth',
            'brush_hair',
            'drop',
            'pick_up',
            'throw',
            'sit_down',
            'stand_up',
            'clapping',
            'reading',
            'writing',
            'tear_up_paper',
            'put_on_jacket',
            'take_off_jacket',
            'put_on_a_shoe',
            'take_off_a_shoe',
            'put_on_glasses',
            'take_off_glasses',
            'put_on_a_hat_cap',
            'take_off_a_hat_cap',
            'cheer_up',
            'hand_waving',
            'kicking_something',
            'reach_into_pocket',
            'hopping',
            'jump_up',
            'phone_call',
            'play_with_phone_tablet',
            'type_on_a_keyboard',
            'point_to_something',
            'taking_a_selfie',
            'check_time_from_watch',
            'rub_two_hands',
            'nod_head_bow',
            'shake_head',
            'wipe_face',
            'salute',
            'put_palms_together',
            'cross_hands_in_front',
            'sneeze_cough',
            'staggering',
            'falling_down',
            'headache',
            'chest_pain',
            'back_pain',
            'neck_pain',
            'nausea_vomiting',
            'fan_self',
            'punch_slap',
            'kicking',
            'pushing',
            'pat_on_back',
            'point_finger',
            'hugging',
            'giving_object',
            'touch_pocket',
            'shaking_hands',
            'walking_towards',
            'walking_apart',
        ]

    # Fallback: generic labels
    return [f'class_{i}' for i in range(num_classes)]

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

    class_names = get_class_names(args.dataset, NUM_CLASSES)
    if len(class_names) != NUM_CLASSES:
        print(
            f"WARNING: class_names length ({len(class_names)}) != NUM_CLASSES ({NUM_CLASSES}). "
            "Falling back to generic labels."
        )
        class_names = [f'class_{i}' for i in range(NUM_CLASSES)]
    
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
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing optional dependency for metrics/plots. Install with: pip install -r requirements.txt"
        ) from e

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n>>> Final Test Accuracy: {acc*100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=2,
            zero_division=0,
        )
    )
    
    if args.is_dummy:
        print("Note: Accuracy is random because you are using Dummy Data.")
    else:
        # Chỉ vẽ Confusion Matrix khi chạy dữ liệu thật hoặc muốn test
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
        plot_confusion_matrix(cm, class_names, f'confusion_matrix_{args.dataset}.png')

if __name__ == "__main__":
    main()