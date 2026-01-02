import matplotlib.pyplot as plt
import numpy as np
import torch

def plot(output_path, train_losses, val_losses):
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
        plt.plot(val_losses, label='Val Loss', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def weighted_r2_score(y_true, y_pred, sample_weights=None):
    """
    计算加权R²系数
    
    Args:
        y_true: 真实值数组 (n_samples, n_targets) 或 (n_samples,)
        y_pred: 预测值数组 (n_samples, n_targets) 或 (n_samples,)
        sample_weights: 样本权重数组 (n_samples,)
    
    Returns:
        加权R²系数
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 如果是多目标情况，展平数组
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
    
    # 如果没有提供权重，使用等权重
    if sample_weights is None:
        sample_weights = np.ones_like(y_true)
    else:
        sample_weights = np.array(sample_weights)
        # 如果权重是按目标类型的数组，需要扩展到与样本数量匹配
        if sample_weights.ndim == 1 and len(sample_weights) < len(y_true):
            # 假设每个目标类型的权重重复相同次数
            n_repeats = len(y_true) // len(sample_weights)
            sample_weights = np.repeat(sample_weights, n_repeats)
    
    # 计算全局加权均值
    weighted_mean = np.sum(sample_weights * y_true) / np.sum(sample_weights)
    
    # 计算残差平方和 (RSS)
    residual = y_true - y_pred
    rss = np.sum(sample_weights * residual ** 2)
    
    # 计算总平方和 (TSS)
    total = y_true - weighted_mean
    tss = np.sum(sample_weights * total ** 2)
    
    # 避免除以零
    if tss == 0:
        return 0.0
    
    # 计算加权R²
    r2 = 1 - (rss / tss)
    
    return r2
