import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt

from utils.dataset import MMFFDataset
from models.mmff_net import MMFF_Net_Advanced

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
    parser.add_argument('--dataset', type=str, default='ntu')
    parser.add_argument('--stage', type=str, default='fusion', choices=['skeleton', 'rgb', 'fusion'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8) # Batch nhỏ tốt cho UTD
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 60 if args.dataset == 'ntu' else 27
    
    # Dataset & Loader
    train_ds = MMFFDataset(mode='train', is_dummy=False, num_classes=NUM_CLASSES, dataset=args.dataset)
    val_ds = MMFFDataset(mode='test', is_dummy=False, num_classes=NUM_CLASSES, dataset=args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = MMFF_Net_Advanced(num_classes=NUM_CLASSES, dataset=args.dataset)
    model.to(DEVICE)

    # --- LOGIC LOAD WEIGHTS THEO GIAI ĐOẠN ---
    if args.stage == 'rgb':
        # Load pre-trained Skeleton để hỗ trợ Attention (nhưng không train nó)
        if os.path.exists(f'best_skeleton_{args.dataset}.pth'):
            print(">> Loading best SKELETON weights for RGB training...")
            model.load_state_dict(torch.load(f'best_skeleton_{args.dataset}.pth'), strict=False)
    
    elif args.stage == 'fusion':
        # Load cả 2 thằng trước khi train tổng
        if os.path.exists(f'best_skeleton_{args.dataset}.pth'):
            print(">> Loading best SKELETON weights...")
            model.load_state_dict(torch.load(f'best_skeleton_{args.dataset}.pth'), strict=False)
        if os.path.exists(f'best_rgb_{args.dataset}.pth'):
            print(">> Loading best RGB weights...")
            model.load_state_dict(torch.load(f'best_rgb_{args.dataset}.pth'), strict=False)
    
    # Optimizer: Thêm Weight Decay để chống Overfit
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label Smoothing giúp ổn định

    best_acc = 0.0
    history = {'train_acc':[], 'val_acc':[], 'train_loss':[], 'val_loss':[]}
    
    print(f"\n=== START TRAINING STAGE: {args.stage.upper()} ===")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, args.stage)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, args.stage)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Ep {epoch+1} | LR: {current_lr:.6f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Lưu tên file theo stage
            save_name = f"best_{args.stage}_{args.dataset}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"Saved {save_name}!")

    plot_history(history, f'history_{args.stage}_{args.dataset}.png')

if __name__ == "__main__":
    main()