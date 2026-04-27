import argparse
import torch
from torch.utils.data import DataLoader
import os


def _ablation_tag(fusion_type: str, cross_attention: str) -> str:
    return f"{fusion_type}_attn-{cross_attention}"


def _checkpoint_candidates(stage: str, dataset: str, fusion_type: str, cross_attention: str) -> list[str]:
    candidates = []
    if stage == 'fusion':
        candidates.append(f"best_{stage}_{dataset}_{_ablation_tag(fusion_type, cross_attention)}.pth")
    else:
        candidates.append(f"best_{stage}_{dataset}_attn-{cross_attention}.pth")
        # Backward compatibility with old naming that included fusion_type.
        candidates.append(f"best_{stage}_{dataset}_{_ablation_tag(fusion_type, cross_attention)}.pth")

    candidates.extend([
        f"best_{stage}_{dataset}.pth",
        f"best_model_{dataset}.pth",
    ])
    return candidates


def _first_existing_path(candidates: list[str]) -> str | None:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def plot_confusion_matrix(cm, classes, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Scale figure size a bit with the number of classes (helps readability for UTD/NTU)
    n = len(classes)
    fig_w = max(10, min(24, 0.6 * n))
    fig_h = max(8, min(24, 0.55 * n))
    plt.figure(figsize=(fig_w, fig_h))
    
    # Tăng kích thước chữ dựa trên số class nhưng ở mức vừa phải
    font_size = 14 if n <= 10 else max(8, 14 - (n * 0.15))
    sns.set(font_scale=1.2 if n <= 10 else 0.8)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": font_size}
    )
    plt.xticks(rotation=45, ha='right', fontsize=font_size * 0.85)
    plt.yticks(rotation=0, fontsize=font_size * 0.85)
    plt.ylabel('True Label', fontsize=font_size)
    plt.xlabel('Predicted Label', fontsize=font_size)
    plt.title('Confusion Matrix', fontsize=font_size * 1.1)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    sns.reset_orig()
    print(f"Saved confusion matrix to {filename}")


def get_class_names(dataset: str, num_classes: int):
    dataset = dataset.lower()
    
    if dataset == 'sumv2':
        return [
            'raising_hand',
            'reading',
            'sleeping',
            'writing',
        ]

    if dataset == 'sgu-sb':
        return [
            'Reading',
            'Writting',
            'Hands_up',
            'Standing',
            'Focus_on_the_board',
            'Looking_around',
            'Sleep'
        ]

    if dataset == 'nw-ucla':
        # NW-UCLA (Northwestern-UCLA Multiview Action 3D) - 10 classes
        return [
            'pick_up_with_one_hand',
            'pick_up_with_two_hands',
            'drop_trash',
            'walk_around',
            'sit_down',
            'stand_up',
            'donning',
            'doffing',
            'throw',
            'carry',
        ]
    
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
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory containing train_data.pkl/test_data.pkl')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'utd', 'nw-ucla', 'sumv2', 'sgu-sb'], 
                        help='dataset name')
    parser.add_argument('--stage', type=str, default='fusion', choices=['skeleton', 'rgb', 'fusion'],
                        help="Which stage checkpoint to evaluate: 'skeleton', 'rgb', or 'fusion'")
    parser.add_argument('--fusion_type', type=str, default='cmaf', choices=['cmaf', 'sum', 'average', 'concat'])
    parser.add_argument('--cross_attention', type=str, default='normal', choices=['normal', 'none', 'reversed'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_frames', type=int, default=32, help='Number of skeleton frames after resampling')
    parser.add_argument('--edge_importance', type=int, default=0, choices=[0, 1], help='Enable Edge Importance Weighting in ST-GCN (0/1)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout for ST-GCN blocks (kept for run metadata)')
    parser.add_argument('--is_dummy', action='store_true', help='Use dummy data for testing')
    
    args = parser.parse_args()

    # Import module của dự án (defer until after argparse so --help works without ML deps installed)
    from models.mmff_net import MMFF_Net_Advanced
    from utils.dataset import MMFFDataset
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cấu hình số lớp
    from config import Config
    args.dataset = Config.normalize_dataset(args.dataset)
    NUM_CLASSES = Config.get_num_classes(args.dataset)
        
    # Keep compatibility with both naming schemes:
    # - New (train.py): best_{stage}_{dataset}.pth
    # - Old: best_model_{dataset}.pth
    MODEL_PATH = _first_existing_path(
        _checkpoint_candidates(args.stage, args.dataset, args.fusion_type, args.cross_attention)
    )
    
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
                               dataset=args.dataset,
                               stage=args.stage,
                               num_frames=args.num_frames,
                               root_dir=args.data_dir)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- 3. Khởi tạo Model ---
    model = MMFF_Net_Advanced(
        num_classes=NUM_CLASSES,
        dataset=args.dataset,
        edge_importance_weighting=bool(args.edge_importance),
        stgcn_dropout=float(args.dropout),
        fusion_type=args.fusion_type,
        cross_attention_mode=args.cross_attention,
    )
    
    # Load weights
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        incompatible = model.load_state_dict(state, strict=False)
        if getattr(incompatible, 'missing_keys', None):
            print(f"WARNING: Missing keys when loading checkpoint: {len(incompatible.missing_keys)}")
        if getattr(incompatible, 'unexpected_keys', None):
            print(f"WARNING: Unexpected keys when loading checkpoint: {len(incompatible.unexpected_keys)}")
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        print("ERROR: No matching weight file found. Train first!")
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
        dropout_tag = str(float(args.dropout)).replace('.', 'p')
        plot_confusion_matrix(
            cm,
            class_names,
            f'confusion_matrix_{args.stage}_{args.dataset}_T{args.num_frames}_ei{args.edge_importance}_do{dropout_tag}.png'
        )

if __name__ == "__main__":
    main()