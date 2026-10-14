import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt

from utils.dataset import MMFFDataset
from models.mmff_net import MMFF_Net_Advanced
from config import Config
from utils.losses import get_criterion
from utils.model_stats import print_model_stats


def _ablation_tag(fusion_type: str, cross_attention: str) -> str:
    return f"{fusion_type}_attn-{cross_attention}"


def _checkpoint_candidates(stage: str, dataset: str, fusion_type: str, cross_attention: str) -> list[str]:
    # Stage-specific naming:
    # - skeleton/rgb do not depend on fusion_type
    # - fusion keeps fusion_type + cross_attention tag
    candidates = []
    if stage == 'fusion':
        candidates.append(f"best_{stage}_{dataset}_{_ablation_tag(fusion_type, cross_attention)}.pth")
    else:
        candidates.append(f"best_{stage}_{dataset}_attn-{cross_attention}.pth")
        # Backward compatibility with old naming that included fusion_type.
        candidates.append(f"best_{stage}_{dataset}_{_ablation_tag(fusion_type, cross_attention)}.pth")

    candidates.append(f"best_{stage}_{dataset}.pth")
    return candidates


def _first_existing_path(candidates: list[str]) -> str | None:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def plot_history(history, save_path, eval_label='Eval'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train'); plt.plot(history['eval_acc'], label=eval_label)
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['train_loss'], label='Train'); plt.plot(history['eval_loss'], label=eval_label)
    plt.title('Loss'); plt.legend()
    plt.savefig(save_path); plt.close()

def train_epoch(model, loader, criterion, optimizer, device, stage):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Train {stage}", leave=False)
    for skel, rgb, _, labels in pbar:
        skel, rgb, labels = skel.to(device), rgb.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Truyền tham số stage vào model
        outputs = model(skel, rgb, stage=stage)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        pbar.set_postfix({'acc': 100.*correct/total})
    return total_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device, stage):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for skel, rgb, _, labels in loader:
            skel, rgb, labels = skel.to(device), rgb.to(device), labels.to(device)
            outputs = model(skel, rgb, stage=stage)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return total_loss/len(loader), 100.*correct/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory containing train_data.pkl/test_data.pkl')
    parser.add_argument('--dataset', type=str, default='ntu',
                        choices=['ntu', 'utd', 'nw-ucla', 'sumv2', 'sgu-sb'])
    parser.add_argument('--stage', type=str, default='fusion', choices=['skeleton', 'rgb', 'fusion'])
    parser.add_argument('--fusion_type', type=str, default='cmaf', choices=['cmaf', 'sum', 'average', 'concat'])
    parser.add_argument('--cross_attention', type=str, default='normal', choices=['normal', 'none', 'reversed'])
    parser.add_argument('--epochs', type=int, default=None, help='Epochs (uses config default if not specified)')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (uses config default per stage)')
    parser.add_argument('--edge_importance', type=int, default=0, choices=[0, 1], help='Enable Edge Importance Weighting in ST-GCN (0/1)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout for ST-GCN blocks (0.0-0.8 typical)')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of skeleton frames after resampling')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='Validation ratio split from training set (set to 0 to use all train data)')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for train/val split')
    parser.add_argument('--gaussian_noise', type=float, default=0.01, help='Standard deviation of Gaussian noise added during training (RGB & Skeleton)')
    args = parser.parse_args()
    args.dataset = Config.normalize_dataset(args.dataset)
    
    # Set defaults from config if not specified
    if args.epochs is None:
        args.epochs = Config.get_epochs(args.stage)
    if args.lr is None:
        args.lr = Config.get_lr(args.stage)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = Config.get_num_classes(args.dataset)
    
    # Dataset & Loader
    use_val = args.val_ratio > 0.0
    train_ds = MMFFDataset(
        root_dir=args.data_dir,
        mode='train',
        is_dummy=False,
        num_classes=NUM_CLASSES,
        dataset=args.dataset,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        stage=args.stage,
        num_frames=args.num_frames,
        noise_std=args.gaussian_noise,
    )
    # val_ratio > 0: epoch metrics & best checkpoint use validation split from train pool.
    # val_ratio == 0: use held-out test set (same split as test.py).
    eval_ds = MMFFDataset(
        root_dir=args.data_dir,
        mode='val' if use_val else 'test',
        is_dummy=False,
        num_classes=NUM_CLASSES,
        dataset=args.dataset,
        val_ratio=args.val_ratio if use_val else 0.0,
        split_seed=args.split_seed,
        stage=args.stage,
        num_frames=args.num_frames,
        noise_std=args.gaussian_noise,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    eval_label = 'Val' if use_val else 'Test'
    
    model = MMFF_Net_Advanced(
        num_classes=NUM_CLASSES,
        dataset=args.dataset,
        edge_importance_weighting=bool(args.edge_importance),
        stgcn_dropout=float(args.dropout),
        fusion_type=args.fusion_type,
        cross_attention_mode=args.cross_attention,
    )
    model.to(DEVICE)

    # --- LOGIC LOAD WEIGHTS THEO GIAI ĐOẠN ---
    if args.stage == 'rgb':
        # Load pre-trained Skeleton only when cross-attention is used.
        if args.cross_attention != 'none':
            skeleton_ckpt = _first_existing_path(
                _checkpoint_candidates('skeleton', args.dataset, args.fusion_type, args.cross_attention)
            )
            if skeleton_ckpt:
                print(">> Loading best SKELETON weights for RGB training...")
                model.load_state_dict(torch.load(skeleton_ckpt), strict=False)
        else:
            print(">> Cross-attention disabled: skip loading SKELETON checkpoint for RGB stage.")
        # Initially freeze backbone for gradual unfreezing
        for param in model.rgb_encoder.backbone.parameters():
            param.requires_grad = False
        print("RGB backbone frozen initially (will unfreeze at epoch {})".format(Config.RGB_UNFREEZE_EPOCH))
    
    elif args.stage == 'fusion': 
        # Load cả 2 thằng trước khi train tổng
        skeleton_ckpt = _first_existing_path(
            _checkpoint_candidates('skeleton', args.dataset, args.fusion_type, args.cross_attention)
        )
        if skeleton_ckpt:
            print(">> Loading best SKELETON weights...")
            model.load_state_dict(torch.load(skeleton_ckpt), strict=False)

        rgb_ckpt = _first_existing_path(
            _checkpoint_candidates('rgb', args.dataset, args.fusion_type, args.cross_attention)
        )
        if rgb_ckpt:
            print(">> Loading best RGB weights...")
            model.load_state_dict(torch.load(rgb_ckpt), strict=False)
    
    # Optimizer:  Use AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate schedulers
    # Warmup scheduler
    def warmup_lambda(epoch):
        if epoch < Config.WARMUP_EPOCHS:
            return (epoch + 1) / Config.WARMUP_EPOCHS
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Main scheduler (cosine annealing after warmup)
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - Config.WARMUP_EPOCHS, 
        eta_min=Config.LR_MIN
    )
    
    # Loss criterion with label smoothing or focal loss
    criterion = get_criterion(
        use_focal=Config.USE_FOCAL_LOSS,
        label_smoothing=Config.LABEL_SMOOTHING,
        focal_alpha=Config.FOCAL_ALPHA,
        focal_gamma=Config.FOCAL_GAMMA
    )

    best_acc = 0.0
    history = {'train_acc': [], 'eval_acc': [], 'train_loss': [], 'eval_loss': []}
    
    print(f"\n=== START TRAINING STAGE: {args.stage.upper()} ===")
    print(f"Epochs: {args.epochs}, Initial LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"Fusion type: {args.fusion_type}, Cross-attention: {args.cross_attention}")
    print(f"Loss function: {'Focal Loss' if Config.USE_FOCAL_LOSS else 'CrossEntropy with Label Smoothing'}")
    print(f"Epoch evaluation: {'validation split (val_ratio=' + str(args.val_ratio) + ')' if use_val else 'held-out test set'}")
    
    for epoch in range(args.epochs):
        # Gradual unfreezing for RGB stage
        if args.stage == 'rgb' and epoch == Config.RGB_UNFREEZE_EPOCH:
            print(f"\n>>> Unfreezing RGB backbone at epoch {epoch+1}...")
            for param in model.rgb_encoder.backbone.parameters():
                param.requires_grad = True
            # Reduce LR when unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * Config.RGB_UNFREEZE_LR_FACTOR
            print(f"Learning rate reduced to:  {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, args.stage)
        eval_loss, eval_acc = validate(model, eval_loader, criterion, DEVICE, args.stage)
        
        # Step schedulers
        if epoch < Config.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"Ep {epoch+1}/{args.epochs} | LR: {current_lr:.6f} | "
            f"Train:  {train_acc:.2f}% (loss {train_loss:.4f}) | "
            f"{eval_label}: {eval_acc:.2f}% (loss {eval_loss:.4f})"
        )
        
        history['train_acc'].append(train_acc); history['eval_acc'].append(eval_acc)
        history['train_loss'].append(train_loss); history['eval_loss'].append(eval_loss)
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            # Lưu tên file theo stage
            if args.stage == 'fusion':
                save_name = f"best_{args.stage}_{args.dataset}_{_ablation_tag(args.fusion_type, args.cross_attention)}.pth"
            else:
                save_name = f"best_{args.stage}_{args.dataset}_attn-{args.cross_attention}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"Saved {save_name}!")

    # Include stage in filename (and extra knobs to avoid overwriting across experiments)
    dropout_tag = str(float(args.dropout)).replace('.', 'p')
    plot_history(
        history,
        f'history_{args.stage}_{args.dataset}_{_ablation_tag(args.fusion_type, args.cross_attention)}_T{args.num_frames}_ei{args.edge_importance}_do{dropout_tag}.png',
        eval_label=eval_label,
    )

    print("\n" + "="*35)
    print("TRAINING COMPLETED!  Printing Model Statistics...")
    print("="*35)
    
    # Load model đã train tốt nhất
    best_checkpoint = _first_existing_path(
        _checkpoint_candidates(args.stage, args.dataset, args.fusion_type, args.cross_attention)
    )
    if best_checkpoint:
        print(f"\nLoading best checkpoint: {best_checkpoint}")
        model.load_state_dict(torch.load(best_checkpoint, map_location=DEVICE))
    
    print_model_stats(
        model=model,
        dataset=args.dataset,
        num_frames=args.num_frames,
        stage=args.stage,
        device=DEVICE,
        img_size=299, 
        verbose=True
    )

if __name__ == "__main__": 
    main()