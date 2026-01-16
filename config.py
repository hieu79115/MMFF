"""
Central Configuration File for MMFF (Multi-Modal Fusion Framework)

This configuration provides comprehensive hyperparameters for model architecture,
training strategy, and dataset parameters to achieve 90%+ accuracy.
"""


class Config:
    """Central configuration class for MMFF model and training"""
    
    # ============================================================================
    # Model Architecture Parameters
    # ============================================================================
    
    # ST-GCN (Skeleton Stream) Parameters
    SKEL_DIM = 256  # Output dimension of skeleton encoder
    
    # RGB Stream Parameters
    RGB_DIM = 2048  # Output dimension of RGB encoder (Xception backbone)
    
    # Fusion Transformer Parameters (Upgraded for better capacity)
    EMBED_DIM = 512  # Increased from 256 for higher model capacity
    NUM_HEADS = 8     # Increased from 4 for better attention
    TRANSFORMER_LAYERS = 3  # Increased from 1 for deeper feature learning
    DROPOUT = 0.3    # Reduced from 0.5 to prevent over-regularization
    
    # ============================================================================
    # Training Parameters per Stage
    # ============================================================================
    
    # Number of epochs for each training stage
    EPOCHS = {
        'skeleton': 50,  # Stage 1: Train skeleton stream
        'rgb': 50,       # Stage 2: Train RGB stream
        'fusion': 70     # Stage 3: Train fusion head
    }
    
    # Learning rates for each training stage
    LR = {
        'skeleton': 1e-3,  # 0.001 for skeleton stage
        'rgb': 5e-4,       # 0.0005 for RGB stage (lower due to pretrained backbone)
        'fusion': 5e-4     # 0.0005 for fusion stage
    }
    
    # Training hyperparameters
    BATCH_SIZE = 16      # Batch size for training
    WEIGHT_DECAY = 1e-4  # Weight decay for AdamW optimizer (regularization)
    WARMUP_EPOCHS = 5    # Number of epochs for learning rate warmup
    
    # ============================================================================
    # Loss Function Parameters
    # ============================================================================
    
    # Label Smoothing for CrossEntropyLoss
    LABEL_SMOOTHING = 0.1  # Helps prevent overconfidence and improves generalization
    
    # Focal Loss Parameters (for handling class imbalance)
    USE_FOCAL_LOSS = False  # Toggle to use Focal Loss instead of CrossEntropy
    FOCAL_ALPHA = 0.25      # Focal loss alpha parameter (balancing factor)
    FOCAL_GAMMA = 2.0       # Focal loss gamma parameter (focusing parameter)
    
    # ============================================================================
    # Dataset Parameters
    # ============================================================================
    
    # Number of classes for each dataset
    NUM_CLASSES = {
        'ntu': 60,  # NTU RGB+D dataset (60 action classes)
        'utd': 27   # UTD-MHAD dataset (27 action classes)
    }
    
    # Data preprocessing parameters
    NUM_FRAMES = 32   # Number of frames to resample skeleton sequences to
    IMG_SIZE = 299    # Image size for RGB input (Xception expects 299x299)
    
    # ============================================================================
    # RGB Training Strategy Parameters
    # ============================================================================
    
    # Gradual unfreezing for RGB backbone
    RGB_UNFREEZE_EPOCH = 15  # Epoch at which to unfreeze RGB backbone
    RGB_UNFREEZE_LR_FACTOR = 0.1  # Factor to reduce LR when unfreezing
    
    # ============================================================================
    # Learning Rate Scheduler Parameters
    # ============================================================================
    
    # Cosine Annealing parameters
    LR_MIN = 1e-6  # Minimum learning rate for cosine annealing
    
    @classmethod
    def get_num_classes(cls, dataset: str) -> int:
        """Get number of classes for a given dataset"""
        return cls.NUM_CLASSES.get(dataset.lower(), 60)
    
    @classmethod
    def get_epochs(cls, stage: str) -> int:
        """Get default number of epochs for a given stage"""
        return cls.EPOCHS.get(stage.lower(), 50)
    
    @classmethod
    def get_lr(cls, stage: str) -> float:
        """Get default learning rate for a given stage"""
        return cls.LR.get(stage.lower(), 1e-3)
