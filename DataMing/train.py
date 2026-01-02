import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import warnings
import utils as ut
import data_loader as dl
from config import Config
from model import MultiTaskBiomassModel
from loss import WeightedMSELoss

warnings.filterwarnings('ignore')

# 为mixup数据计算损失函数
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 训练单个epoch
def train_epoch(model, loader, criterion_biomass, criterion_aux_reg, criterion_aux_cls, optimizer):
    model.train()
    total_loss = 0 # 平均总损失
    biomass_loss_total = 0 # 平均生物量损失
    aux_loss_total = 0 # 平均辅助任务损失
    
    for images, biomass_targets, auxiliary_targets in loader:
        images = images.to(Config.DEVICE)
        biomass_targets = biomass_targets.to(Config.DEVICE)
        
        # 如果使用mixup数据增强
        if Config.USE_MIXUP and np.random.random() < 0.5:
            images, targets_a, targets_b, lam = dl.mixup_data(images, biomass_targets, Config.MIXUP_ALPHA)
        else:
            targets_a = biomass_targets
            targets_b = None
            lam = 1.0
        
        optimizer.zero_grad()

        # 多任务学习
        if Config.USE_MULTITASK:
            outputs = model(images, return_auxiliary=True)
            
            # 计算生物量损失
            if targets_b is not None:
                # mixup损失函数
                biomass_loss = mixup_criterion(criterion_biomass, outputs['biomass'], targets_a, targets_b, lam)
            else:
                biomass_loss = criterion_biomass(outputs['biomass'], targets_a)
            
            # 计算辅助任务损失（不对辅助任务使用mixup）
            aux_loss = 0
            # NDVI回归损失
            aux_loss += criterion_aux_reg(outputs['ndvi'], auxiliary_targets['ndvi'].to(Config.DEVICE))
            # 草高回归损失
            aux_loss += criterion_aux_reg(outputs['height'], auxiliary_targets['height'].to(Config.DEVICE))
            # 地区分类损失
            aux_loss += criterion_aux_cls(outputs['state'], auxiliary_targets['state'].to(Config.DEVICE).squeeze())
            # 草种分类损失
            aux_loss += criterion_aux_cls(outputs['species'], auxiliary_targets['species'].to(Config.DEVICE).squeeze())
            # 平均辅助损失
            aux_loss = aux_loss / 4

            # 总损失 = 生物量损失 + 权重系数 × 辅助损失
            loss = biomass_loss + Config.AUXILIARY_WEIGHT * aux_loss
            biomass_loss_total += biomass_loss.item()
            aux_loss_total += aux_loss.item()
        # 单任务学习
        else:
            outputs = model(images, return_auxiliary=False)
            
            if targets_b is not None:
                loss = mixup_criterion(criterion_biomass, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion_biomass(outputs, targets_a)
            biomass_loss_total += loss.item()
        
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    n = len(loader)
    return total_loss / n, biomass_loss_total / n, aux_loss_total / n


# 验证模型
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, biomass_targets, _ in loader:
            images = images.to(Config.DEVICE)
            biomass_targets = biomass_targets.to(Config.DEVICE)

            outputs = model(images, return_auxiliary=False) # 只预测生物量
            loss = criterion(outputs, biomass_targets)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(biomass_targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 均方根误差
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    # 平均绝对误差
    mae = mean_absolute_error(all_targets, all_preds)
    
    # 计算加权R²
    r2 = ut.weighted_r2_score(all_targets, all_preds, Config.TARGET_WEIGHTS)
    
    return total_loss / len(loader), rmse, mae, r2

# 训练函数
def train_model():

    # 设置种子
    ut.set_seed(Config.SEED)
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("CSIRO - Image2Biomass Training")
    print("="*60)
    # 打印相关信息
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.BACKBONE_MODEL}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Multi-task learning: {Config.USE_MULTITASK}")
    print(f"Mixup: {Config.USE_MIXUP}" + (f" (alpha={Config.MIXUP_ALPHA})" if Config.USE_MIXUP else ""))
    
    print("\n" + "="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    # 加载训练数据
    train_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'train.csv'))
    print(f"Raw train shape: {train_df.shape}")
    # 将数据从长格式转换为宽格式
    train_wide = dl.pivot_train_data(train_df)
    print(f"Wide train shape: {train_wide.shape}")
    # 使用GroupKFold进行交叉验证（按图像ID分组）
    gkf = GroupKFold(n_splits=Config.NUM_FOLDS)
    groups = train_wide['image_id'].values
    

    folds_to_train = range(Config.NUM_FOLDS) if Config.TRAIN_FOLD == -1 else [Config.TRAIN_FOLD]
    # 遍历每个折进行训练
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_wide, groups=groups)):
        if fold_idx not in folds_to_train:
            continue
        
        print("\n" + "="*60)
        print(f"Training Fold {fold_idx + 1}/{Config.NUM_FOLDS}")
        print("="*60)
        
        train_data_raw = train_wide.iloc[train_idx].reset_index(drop=True)
        val_data_raw = train_wide.iloc[val_idx].reset_index(drop=True)
        

        # 对数据进行拟合和转换（包括标准化、编码等）
        train_data, val_data, scalers, label_encoders = dl.fit_and_transform_fold_data(
            train_data_raw, val_data_raw
        )

        # 创建预处理信息字典
        preprocessing_info = {
            'scalers': scalers,  # 标准化器（用于逆变换）
            'label_encoders': label_encoders,  # 标签编码器
            'num_states': len(label_encoders['State'].classes_) + 1,  # 地区类别数
            'num_species': len(label_encoders['Species'].classes_) + 1,  # 草种类别数
            'target_names': Config.TARGET_NAMES  # 目标名称
        }
        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
        # 保存预处理信息到文件
        with open(os.path.join(Config.OUTPUT_DIR, f'preprocessing_info_fold{fold_idx}.pkl'), 'wb') as f:
            pickle.dump(preprocessing_info, f)
        
        # 创建数据集
        train_dataset = dl.BiomassDataset(train_data, Config.DATA_DIR, dl.get_transforms(True))  # 训练集使用数据增强
        val_dataset = dl.BiomassDataset(val_data, Config.DATA_DIR, dl.get_transforms(False))  # 验证集不使用数据增强
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                                 shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                               shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        # 初始化模型
        model = MultiTaskBiomassModel(
            Config.BACKBONE_MODEL,
            num_biomass_targets=len(Config.TARGET_NAMES),
            num_states=preprocessing_info['num_states'],
            num_species=preprocessing_info['num_species'],
            use_multitask=Config.USE_MULTITASK
        ).to(Config.DEVICE)
        
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 定义损失函数
        criterion_biomass = WeightedMSELoss(Config.TARGET_WEIGHTS, Config.DEVICE) 
        criterion_aux_reg = nn.MSELoss()
        criterion_aux_cls = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 训练循环初始化
        best_val_loss = float('inf')
        patience_counter = 0 # 早停计数器
        train_losses = []
        val_losses = []
        
        print("\nStarting training...\n")
        
        for epoch in range(Config.NUM_EPOCHS):
            train_loss, biomass_loss, aux_loss = train_epoch(
                model, train_loader, criterion_biomass, criterion_aux_reg,
                criterion_aux_cls, optimizer
            )
            val_loss, val_rmse, val_mae, val_r2 = validate(model, val_loader, criterion_biomass)
            
            train_losses.append(biomass_loss) # only report the biomass loss for fair comparison with the val_losses
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1:02d}/{Config.NUM_EPOCHS}")
            if Config.USE_MULTITASK:
                print(f"  Train - Total: {train_loss:.4f} | Biomass: {biomass_loss:.4f} | Aux: {aux_loss:.4f}")
            else:
                print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # 保存模型检查点
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'model_name': Config.BACKBONE_MODEL,
                        'num_biomass_targets': len(Config.TARGET_NAMES),
                        'num_states': preprocessing_info['num_states'],
                        'num_species': preprocessing_info['num_species'],
                        'use_multitask': Config.USE_MULTITASK
                    },
                    'epoch': epoch,
                    'val_loss': val_loss,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2
                }
                # 保存最佳模型
                torch.save(checkpoint, os.path.join(Config.OUTPUT_DIR, f'best_model_fold{fold_idx}.pth'))
                print("Best model saved!")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        # 绘制训练曲线图
        ut.plot(os.path.join(Config.OUTPUT_DIR, f'train_graph_fold{fold_idx}.png'), train_losses, val_losses)
        print(f"\nBest validation loss for fold {fold_idx}: {best_val_loss:.4f}")
        print(f"Model saved to: {Config.OUTPUT_DIR}/best_model{fold_idx}.pth")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    train_model()
