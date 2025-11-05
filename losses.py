import torch
import torch.nn as nn


class IOULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss - Measures overlap between predicted and ground truth masks.
    Formula: Loss = 1 - IoU = 1 - (|y_true ∩ y_pred|) / (|y_true| + |y_pred| - |y_true ∩ y_pred|)
    """
    def __init__(self, from_logits: bool=True):
        super().__init__()
        self.from_logits = from_logits

    def forward(self, y_pred, y_true):
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-6)
        y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min() + 1e-6)
    
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = y_pred

        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection
        iou = intersection / (union + 1e-6)

        loss = 1 - iou  

        return loss

    
class DiceLoss(nn.Module):
    """
    Dice Loss (Sorensen–Dice Coefficient) - Measures similarity between two samples, effective for imbalanced datasets.
    Formula: Loss = 1 - Dice = 1 - (2*|y_true ∩ y_pred| + epsilon) / (|y_true| + |y_pred| + epsilon)
    """
    def __init__(self, from_logits: bool=True):
        super().__init__()
        self.from_logits = from_logits

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        else: 
            y_pred = y_pred

        intersection = (y_pred * y_true).sum()
        loss = 1 - (2 * intersection + 1) / ((y_pred.sum() + y_true.sum()) + 1)
        return loss


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
    


class LovaszLoss(nn.Module):
    """
    Lovasz-Softmax Loss - Convex surrogate for directly optimizing the Jaccard index (IoU).
    Formula: Lovasz(y_pred, y_true) = sum_{grad_i} * |y_true_i - y_pred_i|, where g_i are Lovasz gradient weights based on sorted errors.
    """

    def __init__(self, from_logits=True):
        super().__init__()
        self.from_logits = from_logits

    
    def _compute_lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union

        if len(gt_sorted) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]

        return jaccard


    def forward(self, y_pred, y_true):

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).float()

        if self.from_logits:
            p = torch.sigmoid(y_pred)

        else: 
            p = y_pred

        error = torch.abs(y_true - p)
        errors_sorted, perm = torch.sort(error, descending=True)
        y_true_sorted = y_true[perm]

        grad = self._compute_lovasz_grad(y_true_sorted)
        loss = torch.dot(torch.relu(errors_sorted), grad)

        return loss
    

class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge Loss - Alternative Lovasz formulation using hinge loss, works directly with raw logits.
    Formula: Lovasz-Hinge(y_pred, y_true) = sum{g_i} * max(0, 1 - y_pred_i * s_i), where s_i = 2*y_true_i - 1 in {-1, 1}.
    """

    def __init__(self, from_logits=True):
        super().__init__()
        self.from_logits = from_logits

    def _compute_lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:

        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union

        if len(gt_sorted) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]

        return jaccard


    def forward(self, y_pred, y_true):

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).float()

        if self.from_logits:
            p = y_pred.sigmoid()
        else:
            p = y_pred

        y_true_signed = 2 * y_true - 1  # Convert {0,1} to {-1,1}
        errors = 1 - p * y_true_signed
        errors_sorted, perm = torch.sort(errors, descending=True)
        y_true_sorted = y_true[perm]

        grad = self._compute_lovasz_grad(y_true_sorted)
        loss = torch.dot(torch.relu(errors_sorted), grad)

        return loss
    

class CombinedLoss(nn.Module):
        def __init__(self, alpha: float=0.5, from_logits=True):
            super().__init__()
            self.dice = DiceLoss(from_logits=from_logits)
            self.bce = nn.BCEWithLogitsLoss()
            self.alpha = alpha

        def forward(self, logits, targets):
            return self.alpha * self.dice(logits, targets) + (1 - self.alpha) * self.bce(logits, targets)
        


__all__ = [
    'IOULoss',
    'DiceLoss',
    'FocalLoss',
    'LovaszLoss',
    'LovaszHingeLoss',
    'CombinedLoss',
]