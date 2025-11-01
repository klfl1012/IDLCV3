import torch
import torch.nn as nn

# Not sure if we need to implement this from scratch, I'm implementing IOU and Dice loss anyway
# To verify my understanding on the loss function

class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # 0. Scale inputs to be in range of 0 to 1
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-6)
        y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min() + 1e-6)
    
        # 1. Flatten the inputs
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # 2. Calculations
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection
        iou = intersection / (union + 1e-6)  # Add epsilon to avoid division by zero

        loss = 1 - iou  

        return loss
    
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # 1. Flatten input
        y_pred = y_pred.view(-1) # Flatten
        y_true = y_true.view(-1) # Flatten
        
        # 2. Calculations
        intersection = (y_pred * y_true).sum()
        loss = 1- (2 * intersection + 1) / ((y_pred.sum() + y_true.sum()) + 1)
        return loss


