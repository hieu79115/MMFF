"""
Advanced Loss Functions for MMFF

This module provides loss functions for handling class imbalance and improving
model generalization, including Focal Loss and configurable loss selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002
    
    The focal loss applies a modulating term to the cross entropy loss in order to
    focus learning on hard misclassified examples. It is particularly useful for
    handling class imbalance in datasets.
    
    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive/negative examples.
                      Default: 0.25
        gamma (float): Exponent of the modulating factor (1 - p_t)^gamma.
                      Higher gamma increases focus on hard examples. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output:
                        'none' | 'mean' | 'sum'. Default: 'mean'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) where N is batch size
                                   and C is number of classes
            targets (torch.Tensor): Ground truth labels of shape (N,)
        
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Multi-class focal loss (stable): use log_softmax + gather
        log_probs = F.log_softmax(inputs, dim=-1)  # (N, C)
        targets = targets.long()

        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return inputs.new_zeros(())

        log_pt = log_probs[valid].gather(1, targets[valid].unsqueeze(1)).squeeze(1)  # (Nv,)
        pt = log_pt.exp()
        ce_loss = -log_pt  # (Nv,)

        # alpha can be:
        # - scalar (float)
        # - per-class tensor/list of shape (C,)
        alpha = self.alpha
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=inputs.dtype, device=inputs.device)
        if torch.is_tensor(alpha):
            alpha_t = alpha.gather(0, targets[valid])
        else:
            alpha_t = inputs.new_full((ce_loss.shape[0],), float(alpha))

        focal_loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def get_criterion(use_focal=False, label_smoothing=0.1, focal_alpha=0.25, focal_gamma=2.0):
    """
    Get loss criterion based on configuration
    
    This helper function returns either Focal Loss or Cross Entropy Loss with label
    smoothing based on the provided parameters. This allows easy switching between
    loss functions without changing training code.
    
    Args:
        use_focal (bool): If True, use Focal Loss; otherwise use CrossEntropyLoss.
                         Default: False
        label_smoothing (float): Label smoothing factor for CrossEntropyLoss (0.0 to 1.0).
                                Only used when use_focal=False. Default: 0.1
        focal_alpha (float): Alpha parameter for Focal Loss. Only used when use_focal=True.
                            Default: 0.25
        focal_gamma (float): Gamma parameter for Focal Loss. Only used when use_focal=True.
                            Default: 2.0
    
    Returns:
        nn.Module: Loss criterion (either FocalLoss or CrossEntropyLoss)
    
    Example:
        >>> # Get cross entropy loss with label smoothing
        >>> criterion = get_criterion(use_focal=False, label_smoothing=0.1)
        >>> 
        >>> # Get focal loss
        >>> criterion = get_criterion(use_focal=True, focal_alpha=0.25, focal_gamma=2.0)
    """
    if use_focal:
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
