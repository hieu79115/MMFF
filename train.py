import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt

# Import các module của dự án
from utils.dataset import MMFFDataset
from models.mmff_net import MMFF_Net_Advanced

def plot_history(history, save_path='training_history.png'):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'r*-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ training vào: {save_path}")
    plt.close()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for i, (skel, rgb, _, labels) in enumerate(pbar):
        skel = skel.to(device)
        rgb = rgb.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(skel, rgb)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/(i+1), 'acc': 100.*correct/total})
        
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for skel, rgb, _, labels in dataloader:
            skel = skel.to(device)
            rgb = rgb.to(device)
            labels = labels.to(device)
            
            outputs = model(skel, rgb)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(dataloader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Train MMFF Model')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'utd'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--is_dummy', action='store_true')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'ntu': NUM_CLASSES = 60
    else: NUM_CLASSES = 27
    
    # --- Load Data ---
    train_dataset = MMFFDataset(mode='train', is_dummy=args.is_dummy, 
                                num_samples=100, num_classes=NUM_CLASSES, dataset=args.dataset)
    val_dataset = MMFFDataset(mode='test', is_dummy=args.is_dummy, 
                              num_samples=20, num_classes=NUM_CLASSES, dataset=args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- Model ---
    model = MMFF_Net_Advanced(num_classes=NUM_CLASSES, dataset=args.dataset)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # --- SỬA LỖI Ở ĐÂY: Bỏ verbose=True ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Cập nhật Scheduler
        scheduler.step(val_acc)
        
        # In ra LR hiện tại (thay cho verbose cũ)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f"best_model_{args.dataset}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}!")

    plot_history(history, save_path=f'history_{args.dataset}.png')

if __name__ == "__main__":
    main()