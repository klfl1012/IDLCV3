import torch
import torch.nn as nn
import torchmetrics as tm

class DiceCoefficient(tm.Metric):
    """
    Manual implementation of Dice coefficient (F1 score) for segmentation.
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # Ensure same shape
        if preds.shape != targets.shape:
            raise ValueError(f"Predictions shape {preds.shape} != targets shape {targets.shape}")
        
        # Flatten tensors
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()
        
        # Calculate intersection and union
        intersection = torch.sum(preds_flat * targets_flat)
        union = torch.sum(preds_flat) + torch.sum(targets_flat)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        self.dice_sum += dice
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.dice_sum / self.count if self.count > 0 else torch.tensor(0.0) # pyright: ignore[reportOperatorIssue]


class FocalLoss(nn.Module):
    """
    Focal Loss - Addresses class imbalance by down-weighting well-classified examples, focusing on hard negatives.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t), where p_t is the probability of the correct class.
    """

    def __init__(self, alpha: float=1.0, gamma: float=2.0, epsilon: float=1e-7, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.from_logits = from_logits

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).float()
        
        if self.from_logits:
            p = torch.sigmoid(y_pred)

        else:
            p = y_pred

        p_t = p * y_true + (1 - p) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)

        focal_weight = (1 - p_t).pow(self.gamma)
        bce = -torch.log(p_t + self.epsilon)
        focal_loss = alpha_t * focal_weight * bce

        return focal_loss.mean()
    
__all__ = [
    'FocalLoss',
    'DiceCoefficient',
]