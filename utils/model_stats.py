"""
Utility functions to calculate model Parameters and FLOPs for multiple datasets
Supports:  NTU RGB+D (60 classes, 25 joints) và UTD-MHAD (27 classes, 20 joints)
"""
import torch
import numpy as np

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' not installed.  FLOPs calculation will be skipped.")
    print("Install with: pip install thop")


# ============================================================================
# Dataset Configurations
# ============================================================================
DATASET_CONFIG = {
    'ntu':  {
        'num_classes':  60,
        'num_joints': 25,
        'dataset_name': 'NTU RGB+D'
    },
    'utd':  {
        'num_classes':  27,
        'num_joints': 20,
        'dataset_name': 'UTD-MHAD'
    }
}


def get_dataset_config(dataset_name):
    """
    Args:
        dataset_name: 'ntu' hoặc 'utd'
    
    Returns:
        dict: {'num_classes', 'num_joints', 'dataset_name'}
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]


def count_parameters(model):
    """
    Đếm tổng số parameters của model
    Args: 
        model: PyTorch model
    Returns:
        tuple: (total_params, trainable_params, non_trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p. numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return total_params, trainable_params, non_trainable_params


def calculate_flops(model, dataset, num_frames=32, stage='fusion', device='cuda', img_size=299):
    """
    Tính FLOPs (Floating Point Operations) của model
    Args: 
        model: PyTorch model
        dataset: 'ntu' hoặc 'utd'
        num_frames: Số frame skeleton (default: 32)
        stage: 'skeleton', 'rgb', hoặc 'fusion'
        device: 'cuda' hoặc 'cpu'
        img_size: Kích thước ảnh RGB (default: 299 cho Xception)
    
    Returns:
        tuple: (flops, params) hoặc (None, None) nếu không có thop
    """
    if not THOP_AVAILABLE: 
        return None, None
    
    config = get_dataset_config(dataset)
    num_joints = config['num_joints']
    
    model.eval()
    model.to(device)
    
    # Tạo dummy inputs theo đúng format của dataset
    # Skeleton: (batch=1, channels=3, frames=T, joints=V)
    skel_input = torch.randn(1, 3, num_frames, num_joints).to(device)
    
    # RGB: (batch=1, channels=3, height=H, width=W)
    rgb_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    try:
        # Tính FLOPs với stage tương ứng
        flops, params = profile(model, inputs=(skel_input, rgb_input, stage), verbose=False)
        return flops, params
    except Exception as e:
        print(f"⚠️  Error calculating FLOPs: {e}")
        return None, None


def format_number(num):
    """Format số lớn thành dạng dễ đọc với dấu phẩy"""
    if num is None:
        return "N/A"
    return f"{num:,}"


def print_model_stats(model, dataset='ntu', num_frames=32, stage='fusion', 
                     device='cuda', img_size=299, verbose=True):
    """
    In ra thông tin chi tiết về model (Parameters và FLOPs)
    
    Args:
        model: PyTorch model
        dataset: 'ntu' hoặc 'utd'
        num_frames: Số frame skeleton
        stage: 'skeleton', 'rgb', hoặc 'fusion'
        device: 'cuda' hoặc 'cpu'
        img_size: Kích thước ảnh RGB
        verbose: In chi tiết hay không
    
    Returns:
        dict: Thông tin thống kê
    """
    config = get_dataset_config(dataset)
    
    if verbose:
        print("\n" + "="*70)
        print(f"MODEL STATISTICS")
        print("="*70)
        print(f"Dataset:         {config['dataset_name']} ({dataset. upper()})")
        print(f"Stage:          {stage. upper()}")
        print(f"Num Classes:    {config['num_classes']}")
        print(f"Num Joints:     {config['num_joints']}")
        print(f"Num Frames:     {num_frames}")
        print(f"RGB Size:       {img_size}x{img_size}")
        print(f"Device:         {device. upper()}")
        print("-"*70)
    
    # 1. Đếm Parameters
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    
    if verbose:
        print(f"Total Parameters:       {format_number(total_params)}")
        print(f"Trainable Parameters:  {format_number(trainable_params)}")
        print(f"Frozen Parameters:     {format_number(non_trainable_params)}")
    
    # 2. Tính FLOPs
    flops, params_thop = calculate_flops(model, dataset, num_frames, stage, device, img_size)
    
    if verbose:
        if flops is not None and THOP_AVAILABLE:
            flops_formatted, params_formatted = clever_format([flops, params_thop], "%. 3f")
            print(f"FLOPs:                   {flops_formatted}")
            print(f"Params (from thop):    {params_formatted}")
        else:
            print(f"FLOPs:                   N/A (install thop)")
        
        print("="*70 + "\n")
    
    # Trả về dictionary để có thể lưu vào file
    stats = {
        'dataset': dataset,
        'dataset_name': config['dataset_name'],
        'stage': stage,
        'num_classes': config['num_classes'],
        'num_joints': config['num_joints'],
        'num_frames': num_frames,
        'img_size': img_size,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'flops': float(flops) if flops is not None else None,
        'device': device
    }
    
    return stats


def compare_stages(model_class, dataset='ntu', num_frames=32, device='cuda', 
                   edge_importance=False, dropout=0.0, **model_kwargs):
    """
    So sánh Parameters và FLOPs của 3 stage (skeleton, rgb, fusion)
    
    Args:
        model_class: Class của model (MMFF_Net_Advanced)
        dataset: 'ntu' hoặc 'utd'
        num_frames:  Số frame skeleton
        device: 'cuda' hoặc 'cpu'
        edge_importance:  Enable edge importance weighting
        dropout:  Dropout rate
        **model_kwargs: Các tham số khác của model
    
    Returns: 
        dict: Thống kê của cả 3 stage
    """
    config = get_dataset_config(dataset)
    stages = ['skeleton', 'rgb', 'fusion']
    results = {}
    
    print("\n" + "="*70)
    print(f"📊 COMPARING ALL STAGES - {config['dataset_name']} ({dataset.upper()})")
    print("="*70)
    
    for stage in stages:
        print(f"\n{'─'*70}")
        print(f"Stage: {stage.upper()}")
        print('─'*70)
        
        # Tạo model mới cho mỗi stage
        model = model_class(
            num_classes=config['num_classes'],
            dataset=dataset,
            edge_importance_weighting=edge_importance,
            stgcn_dropout=dropout,
            **model_kwargs
        )
        model.to(device)
        
        # Tính stats
        stats = print_model_stats(
            model, 
            dataset=dataset, 
            num_frames=num_frames, 
            stage=stage, 
            device=device,
            verbose=True
        )
        
        results[stage] = stats
        
        # Giải phóng memory
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print("="*70 + "\n")
    
    # In bảng so sánh
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Stage':<12} {'Total Params':<18} {'Trainable':<18} {'FLOPs':<15}")
    print("-"*70)
    
    for stage in stages:
        stats = results[stage]
        total = format_number(stats['total_params'])
        trainable = format_number(stats['trainable_params'])
        
        if stats['flops'] is not None and THOP_AVAILABLE: 
            flops_str, _ = clever_format([stats['flops'], 0], "%.3f")
        else:
            flops_str = "N/A"
        
        print(f"{stage. upper():<12} {total:<18} {trainable:<18} {flops_str:<15}")
    
    print("="*70 + "\n")
    
    return results


def save_stats_to_file(stats, filename='model_stats.txt'):
    """
    Lưu thống kê model vào file text
    
    Args:
        stats: Dictionary chứa thông tin stats
        filename: Tên file để lưu
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MODEL STATISTICS\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset:        {stats. get('dataset_name', 'N/A')} ({stats.get('dataset', 'N/A').upper()})\n")
        f.write(f"Stage:          {stats.get('stage', 'N/A').upper()}\n")
        f.write(f"Num Classes:    {stats.get('num_classes', 'N/A')}\n")
        f.write(f"Num Joints:     {stats.get('num_joints', 'N/A')}\n")
        f.write(f"Num Frames:     {stats.get('num_frames', 'N/A')}\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Parameters:      {format_number(stats.get('total_params'))}\n")
        f.write(f"Trainable Parameters:   {format_number(stats. get('trainable_params'))}\n")
        f.write(f"Frozen Parameters:     {format_number(stats.get('non_trainable_params'))}\n")
        
        if stats.get('flops') is not None and THOP_AVAILABLE:
            flops_str, _ = clever_format([stats['flops'], 0], "%.3f")
            f.write(f"FLOPs:                 {flops_str}\n")
        else:
            f.write(f"FLOPs:                 N/A\n")
        
        f.write("="*70 + "\n")
    
    print(f"Stats saved to: {filename}")