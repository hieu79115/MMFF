"""
Usage:
    # Kiểm tra model chưa train
    python check_model_stats.py --dataset ntu --stage fusion
    
    # Kiểm tra model đã train
    python check_model_stats.py --dataset ntu --stage fusion --checkpoint best_fusion_ntu.pth
    
    # So sánh tất cả các stage
    python check_model_stats.py --dataset ntu --compare_all
    
    # Kiểm tra model UTD-MHAD
    python check_model_stats.py --dataset utd --stage skeleton --checkpoint best_skeleton_utd.pth
"""
import torch
import argparse
import os
from models.mmff_net import MMFF_Net_Advanced
from utils.model_stats import (
    print_model_stats, 
    compare_stages, 
    save_stats_to_file,
    get_dataset_config
)


def main():
    parser = argparse.ArgumentParser(description='Check MMFF Model Parameters and FLOPs')
    
    # Dataset và Stage
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'utd'],
                       help='Dataset name:  ntu (60 classes, 25 joints) or utd (27 classes, 20 joints)')
    parser.add_argument('--stage', type=str, default='fusion', choices=['skeleton', 'rgb', 'fusion'],
                       help='Training stage to check')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (. pth file). If not provided, checks untrained model')
    
    # Model configuration
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of skeleton frames after resampling')
    parser.add_argument('--edge_importance', type=int, default=0, choices=[0, 1],
                       help='Enable Edge Importance Weighting in ST-GCN (0/1)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout for ST-GCN blocks (0.0-0.8)')
    parser.add_argument('--img_size', type=int, default=299,
                       help='RGB image size (default: 299 for Xception)')
    
    # Output options
    parser.add_argument('--save', action='store_true',
                       help='Save statistics to file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename for statistics (default: auto-generated)')
    
    # Compare all stages
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare all stages (skeleton, rgb, fusion)')
    
    args = parser.parse_args()
    
    # Setup device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataset config
    config = get_dataset_config(args.dataset)
    NUM_CLASSES = config['num_classes']
    NUM_JOINTS = config['num_joints']
    
    print(f"\n Configuration:")
    print(f"   Dataset: {config['dataset_name']} ({args.dataset.upper()})")
    print(f"   Classes: {NUM_CLASSES}, Joints: {NUM_JOINTS}")
    print(f"   Device: {DEVICE}")
    print(f"   Checkpoint: {args.checkpoint if args.checkpoint else 'None (untrained model)'}")
    
    # === OPTION 1: So sánh tất cả các stage ===
    if args.compare_all:
        print(f"\nComparing all stages for {config['dataset_name']}.. .\n")
        
        results = compare_stages(
            model_class=MMFF_Net_Advanced,
            dataset=args.dataset,
            num_frames=args.num_frames,
            device=DEVICE,
            edge_importance=bool(args.edge_importance),
            dropout=args.dropout
        )
        
        if args.save:
            output_file = args.output or f'model_stats_compare_{args.dataset}.txt'
            # Lưu kết quả so sánh
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write(f"MMFF MODEL COMPARISON - {config['dataset_name']}\n")
                f.write("="*70 + "\n\n")
                for stage, stats in results.items():
                    f.write(f"\n{'─'*70}\n")
                    f.write(f"Stage: {stage.upper()}\n")
                    f.write('─'*70 + "\n")
                    f.write(f"Total Parameters:      {stats['total_params']: ,}\n")
                    f.write(f"Trainable Parameters:  {stats['trainable_params']:,}\n")
                    if stats['flops']: 
                        f.write(f"FLOPs:                 {stats['flops']:.2e}\n")
                    f.write("\n")
            print(f"Comparison saved to: {output_file}")
        
        return
    
    # === OPTION 2: Kiểm tra một stage cụ thể ===
    print(f"\nChecking {args.stage.upper()} stage...\n")
    
    # Khởi tạo model
    model = MMFF_Net_Advanced(
        num_classes=NUM_CLASSES,
        dataset=args.dataset,
        edge_importance_weighting=bool(args.edge_importance),
        stgcn_dropout=args.dropout,
    )
    
    # Load checkpoint nếu có
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file not found: {args.checkpoint}")
            return
        
        print(f"Loading checkpoint: {args.checkpoint}")
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
            print(f"Checkpoint loaded successfully!")
        except Exception as e: 
            print(f"Warning: Could not load checkpoint:  {e}")
            print(f"Proceeding with untrained model...")
    
    model.to(DEVICE)
    
    # In thống kê
    stats = print_model_stats(
        model,
        dataset=args.dataset,
        num_frames=args.num_frames,
        stage=args.stage,
        device=DEVICE,
        img_size=args.img_size,
        verbose=True
    )
    
    # Lưu kết quả nếu được yêu cầu
    if args.save:
        if args.output:
            output_file = args.output
        else:
            checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0] if args.checkpoint else 'untrained'
            output_file = f'model_stats_{args.stage}_{args.dataset}_{checkpoint_name}.txt'
        
        save_stats_to_file(stats, output_file)


if __name__ == "__main__":
    main()