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

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val')
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
    parser.add_argument('--dataset', type=str, default='ntu')
    parser.add_argument('--stage', type=str, default='fusion', choices=['skeleton', 'rgb', 'fusion'])
    parser.add_argument('--epochs', type=int, default=None, help='Epochs (uses config default if not specified)')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (uses config default per stage)')
    parser.add_argument('--edge_importance', type=int, default=0, choices=[0, 1], help='Enable Edge Importance Weighting in ST-GCN (0/1)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout for ST-GCN blocks (0.0-0.8 typical)')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of skeleton frames after resampling')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio split from training set')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for train/val split')
    args = parser.parse_args()
    
    # Set defaults from config if not specified
    if args.epochs is None:
        args.epochs = Config.get_epochs(args.stage)
    if args.lr is None:
        args.lr = Config.get_lr(args.stage)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 60 if args.dataset == 'ntu' else 27
    
    # Dataset & Loader
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
    )
    val_ds = MMFFDataset(
        root_dir=args.data_dir,
        mode='val',
        is_dummy=False,
        num_classes=NUM_CLASSES,
        dataset=args.dataset,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        stage=args.stage,
        num_frames=args.num_frames,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = MMFF_Net_Advanced(
        num_classes=NUM_CLASSES,
        dataset=args.dataset,
        edge_importance_weighting=bool(args.edge_importance),
        stgcn_dropout=float(args.dropout),
    )
    model.to(DEVICE)

    # --- LOGIC LOAD WEIGHTS THEO GIAI ĐOẠN ---
    if args.stage == 'rgb':
        # Load pre-trained Skeleton để hỗ trợ Attention (nhưng không train nó)
        if os.path.exists(f'best_skeleton_{args.dataset}.pth'):
            print(">> Loading best SKELETON weights for RGB training...")
            model.load_state_dict(torch.load(f'best_skeleton_{args.dataset}.pth'), strict=False)
        # Initially freeze backbone for gradual unfreezing
        for param in model.rgb_encoder.backbone.parameters():
            param.requires_grad = False
        print("RGB backbone frozen initially (will unfreeze at epoch {})".format(Config.RGB_UNFREEZE_EPOCH))
    
    elif args.stage == 'fusion': 
        # Load cả 2 thằng trước khi train tổng
        if os.path.exists(f'best_skeleton_{args.dataset}.pth'):
            print(">> Loading best SKELETON weights...")
            model.load_state_dict(torch.load(f'best_skeleton_{args.dataset}.pth'), strict=False)
        if os.path.exists(f'best_rgb_{args.dataset}.pth'):
            print(">> Loading best RGB weights...")
            model.load_state_dict(torch.load(f'best_rgb_{args.dataset}.pth'), strict=False)
    
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
    history = {'train_acc': [], 'val_acc':[], 'train_loss':[], 'val_loss':[]}
    
    print(f"\n=== START TRAINING STAGE: {args.stage.upper()} ===")
    print(f"Epochs: {args.epochs}, Initial LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"Loss function: {'Focal Loss' if Config.USE_FOCAL_LOSS else 'CrossEntropy with Label Smoothing'}")
    
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
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, args.stage)
        
        # Step schedulers
        if epoch < Config.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"Ep {epoch+1}/{args.epochs} | LR: {current_lr:.6f} | "
            f"Train:  {train_acc:.2f}% (loss {train_loss:.4f}) | "
            f"Val: {val_acc:.2f}% (loss {val_loss:.4f})"
        )
        
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        
        if val_acc > best_acc: 
            best_acc = val_acc
            # Lưu tên file theo stage
            save_name = f"best_{args.stage}_{args.dataset}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"Saved {save_name}!")

    # Include stage in filename (and extra knobs to avoid overwriting across experiments)
    dropout_tag = str(float(args.dropout)).replace('.', 'p')
    plot_history(
        history,
        f'history_{args.stage}_{args.dataset}_T{args.num_frames}_ei{args.edge_importance}_do{dropout_tag}.png'
    )

    print("\n" + "="*35)
    print("TRAINING COMPLETED!  Printing Model Statistics...")
    print("="*35)
    
    # Load model đã train tốt nhất
    best_checkpoint = f"best_{args.stage}_{args.dataset}.pth"
    if os.path.exists(best_checkpoint):
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