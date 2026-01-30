#!/usr/bin/env python3
"""
Test script to verify NW-UCLA dataset configuration.
This script tests all components of the MMFF framework with NW-UCLA dataset.
"""

import torch
from torch.utils.data import DataLoader
from config import Config
from utils.graph import Graph
from utils.dataset import MMFFDataset
from models.mmff_net import MMFF_Net_Advanced


def test_config():
    """Test configuration for NW-UCLA dataset."""
    print("=" * 60)
    print("TEST 1: Configuration")
    print("=" * 60)
    
    num_classes = Config.get_num_classes('nw-ucla')
    assert num_classes == 10, f"Expected 10 classes, got {num_classes}"
    print(f"✓ NW-UCLA has {num_classes} classes")
    
    # Test default epochs and learning rates
    epochs_skel = Config.get_epochs('skeleton')
    epochs_rgb = Config.get_epochs('rgb')
    epochs_fusion = Config.get_epochs('fusion')
    print(f"✓ Default epochs - skeleton: {epochs_skel}, rgb: {epochs_rgb}, fusion: {epochs_fusion}")
    
    lr_skel = Config.get_lr('skeleton')
    lr_rgb = Config.get_lr('rgb')
    lr_fusion = Config.get_lr('fusion')
    print(f"✓ Default learning rates - skeleton: {lr_skel}, rgb: {lr_rgb}, fusion: {lr_fusion}")


def test_graph():
    """Test skeleton graph for NW-UCLA dataset."""
    print("\n" + "=" * 60)
    print("TEST 2: Skeleton Graph Structure")
    print("=" * 60)
    
    graph = Graph(dataset='nw-ucla')
    assert graph.num_node == 21, f"Expected 21 joints, got {graph.num_node}"
    print(f"✓ Graph has {graph.num_node} joints (expected 21)")
    
    # Check edge count (21 self-links + 20 neighbor links)
    assert len(graph.edge) == 41, f"Expected 41 edges, got {len(graph.edge)}"
    print(f"✓ Graph has {len(graph.edge)} edges (21 self-links + 20 neighbor links)")
    
    # Check adjacency matrix shape
    assert graph.A.shape == (3, 21, 21), f"Expected (3, 21, 21), got {graph.A.shape}"
    print(f"✓ Adjacency matrix shape: {graph.A.shape}")


def test_dataset():
    """Test dataset for NW-UCLA."""
    print("\n" + "=" * 60)
    print("TEST 3: Dataset")
    print("=" * 60)
    
    dataset = MMFFDataset(
        mode='train',
        is_dummy=True,
        num_samples=20,
        num_classes=10,
        dataset='nw-ucla',
        num_frames=32
    )
    
    assert dataset.num_joints == 21, f"Expected 21 joints, got {dataset.num_joints}"
    print(f"✓ Dataset configured with {dataset.num_joints} joints")
    
    assert len(dataset) == 20, f"Expected 20 samples, got {len(dataset)}"
    print(f"✓ Dataset has {len(dataset)} samples")
    
    # Check sample shapes
    skel, rgb, _, label = dataset[0]
    assert skel.shape == (3, 32, 21), f"Expected skeleton shape (3, 32, 21), got {skel.shape}"
    assert rgb.shape == (3, 299, 299), f"Expected RGB shape (3, 299, 299), got {rgb.shape}"
    print(f"✓ Skeleton shape: {skel.shape}")
    print(f"✓ RGB shape: {rgb.shape}")
    print(f"✓ Label: {label} (range: 0-9)")


def test_model():
    """Test model with NW-UCLA configuration."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Architecture")
    print("=" * 60)
    
    model = MMFF_Net_Advanced(num_classes=10, dataset='nw-ucla')
    print("✓ Model created successfully")
    
    # Test forward pass for each stage
    device = torch.device('cpu')
    model.to(device)
    
    batch_size = 4
    batch_skel = torch.randn(batch_size, 3, 32, 21).to(device)
    batch_rgb = torch.randn(batch_size, 3, 299, 299).to(device)
    
    for stage in ['skeleton', 'rgb', 'fusion']:
        model.train()
        output = model(batch_skel, batch_rgb, stage=stage)
        assert output.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), got {output.shape}"
        print(f"✓ Stage '{stage}': output shape {output.shape}")


def test_training_setup():
    """Test training setup with NW-UCLA."""
    print("\n" + "=" * 60)
    print("TEST 5: Training Setup")
    print("=" * 60)
    
    # Create datasets
    train_ds = MMFFDataset(
        mode='train',
        is_dummy=True,
        num_samples=20,
        num_classes=10,
        dataset='nw-ucla',
        num_frames=32
    )
    
    val_ds = MMFFDataset(
        mode='val',
        is_dummy=True,
        num_samples=10,
        num_classes=10,
        dataset='nw-ucla',
        num_frames=32
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    
    print(f"✓ Train loader: {len(train_loader)} batches ({len(train_ds)} samples)")
    print(f"✓ Val loader: {len(val_loader)} batches ({len(val_ds)} samples)")
    
    # Test a training iteration
    model = MMFF_Net_Advanced(num_classes=10, dataset='nw-ucla')
    device = torch.device('cpu')
    model.to(device)
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for skel, rgb, _, labels in train_loader:
        skel, rgb, labels = skel.to(device), rgb.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(skel, rgb, stage='skeleton')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training iteration successful: loss={loss.item():.4f}")
        break


def test_class_names():
    """Test class names for confusion matrix."""
    print("\n" + "=" * 60)
    print("TEST 6: Class Names")
    print("=" * 60)
    
    from test import get_class_names
    
    class_names = get_class_names('nw-ucla', 10)
    assert len(class_names) == 10, f"Expected 10 class names, got {len(class_names)}"
    
    print(f"✓ {len(class_names)} class names defined:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NW-UCLA DATASET CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        test_config()
        test_graph()
        test_dataset()
        test_model()
        test_training_setup()
        test_class_names()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNW-UCLA dataset is properly configured and ready to use.")
        print("\nTo train with NW-UCLA dataset:")
        print("  python train.py --dataset nw-ucla --stage skeleton --batch_size 16")
        print("  python train.py --dataset nw-ucla --stage rgb --batch_size 16")
        print("  python train.py --dataset nw-ucla --stage fusion --batch_size 16")
        print("\nTo evaluate:")
        print("  python test.py --dataset nw-ucla --stage fusion --batch_size 4")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
