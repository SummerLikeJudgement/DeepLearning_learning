import torch.nn as nn
import torch

class WeightedMSELoss(nn.Module):
    """
    Calculates Mean Squared Error (MSE) with target-specific weights.
    
    The standard competition loss usually heavily weights Dry_Total_g and GDM_g.
    """
    def __init__(self, target_weights, device):
        super().__init__()
        self.weights = torch.tensor(target_weights, dtype=torch.float32).to(device)

    def forward(self, inputs, targets):
        # Calculate the squared error
        squared_error = (inputs - targets) ** 2
        
        # Apply the element-wise weights
        weighted_squared_error = squared_error * self.weights
        
        # Calculate the mean across the batch and targets
        loss = torch.mean(weighted_squared_error)
        
        return loss

class WeightedR2Loss(nn.Module):
    """
    Calculates Weighted R² (Coefficient of Determination) as a loss function.
    The loss is defined as 1 - R², so lower values indicate better performance.
    """
    def __init__(self, target_weights, device):
        super().__init__()
        self.weights = torch.tensor(target_weights, dtype=torch.float32).to(device)
        self.eps = 1e-8  # To prevent division by zero

    def forward(self, inputs, targets):
        # Calculate weighted mean of targets
        weighted_mean = torch.sum(targets * self.weights, dim=0) / torch.sum(self.weights)
        weighted_mean = weighted_mean.expand_as(targets)  # Expand to match targets shape
        
        # Calculate weighted residual sum of squares (RSS)
        rss = torch.sum(self.weights * (targets - inputs) ** 2)
        
        # Calculate weighted total sum of squares (TSS)
        tss = torch.sum(self.weights * (targets - weighted_mean) ** 2)
        
        # Prevent division by zero
        tss = torch.max(tss, torch.tensor(self.eps, device=targets.device))
        
        # Calculate R² and convert to loss (1 - R²)
        r2 = 1 - (rss / tss)
        loss = 1 - r2
        
        return loss
