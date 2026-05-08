import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
from models import MaskedTrafficAutoencoder, TrafficTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def check_and_clean_data(data):
    """检查并清理数据中的 NaN 和 Inf"""
    # 替换 NaN 为 0
    data = np.nan_to_num(data, nan=0.0)
    # 替换 Inf 为合理值 (用 mean 或 clip)
    data = np.clip(data, -100, 100)
    return data

def traffic_augmentation(x):
    """
    语义保留的流量数据增强策略
    策略1: 随机裁剪并填充 (模拟丢包/包顺序变化)
    策略2: 特征缩放抖动 (模拟网络波动)
    """
    b, f = x.shape  # (batch_size, features)
    
    x_aug = x.clone()
    
    # --- 策略 1: 随机裁剪与填充 ---
    crop_ratio = 0.1  # 裁剪 10% 的特征
    crop_len = int(f * crop_ratio)
    if crop_len > 0:
        start_idx = np.random.randint(0, f - crop_len)
        # 裁剪
        cropped = x_aug[:, start_idx:start_idx+crop_len]
        # 填充回原长度 (用均值填充更符合流量数据特征)
        mean_val = x_aug.mean(dim=1, keepdim=True).repeat(1, crop_len)
        x_aug[:, start_idx:start_idx+crop_len] = mean_val
    
    # --- 策略 2: 特征缩放抖动 (模拟网络波动) ---
    # 生成随机缩放因子 (0.9 ~ 1.1)，更温和
    scale = torch.rand(b, 1, device=x.device) * 0.1 + 0.95
    x_aug = x_aug * scale
    
    # 防止数值过大或过小
    x_aug = torch.clamp(x_aug, -50, 50)
    
    return x_aug

def contrastive_loss(projections1, projections2, temperature=0.1):
    projections1 = nn.functional.normalize(projections1, dim=1)
    projections2 = nn.functional.normalize(projections2, dim=1)
    similarity_matrix = torch.matmul(projections1, projections2.T)
    batch_size = projections1.shape[0]
    labels = torch.arange(batch_size, device=DEVICE)
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0]
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
    return loss

def train_mae(data_path, epochs=50, batch_size=1024, lr=0.001, mask_ratio=0.4):
    print(f"\n{'='*60}")
    print(f"训练真正的掩码自编码器 (MAE) - 掩码比例: {mask_ratio*100}%")
    print(f"{'='*60}")

    # 1. 加载和预处理数据
    data = np.load(data_path)
    print(f"原始数据维度: {data.shape}")
    
    # 2. 数据检查与清理
    data = check_and_clean_data(data)
    data = data.astype(np.float32)
    
    # 3. 创建数据集
    input_dim = data.shape[1]
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. 初始化模型
    model = MaskedTrafficAutoencoder(input_dim, mask_ratio=mask_ratio).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_data in loader:
            inputs = batch_data[0].to(DEVICE)
            optimizer.zero_grad()

            # --- 【核心修改 1】：实施随机掩码 ---
            # 这里我们让模型内部处理掩码 (MaskedTrafficAutoencoder 已有 mask_input 方法)
            # 如果需要手动控制，也可以在这里生成 mask
            decoded, mask = model(inputs)
            
            # --- 【核心修改 2】：只计算被掩码位置的损失 ---
            # 只对被 mask 的位置计算 loss，模型需要"猜"被遮住的特征
            # mask 矩阵：1 表示被遮住，0 表示保留
            masked_loss = criterion(decoded * mask, inputs * mask)
            
            masked_loss.backward()
            optimizer.step()
            total_loss += masked_loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], 掩码重构 Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"mae_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  → 保存: {ckpt_path}")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'loss_history': loss_history,
    }, os.path.join(CHECKPOINT_DIR, "mae_pretrain.pth"))

    np.save("results/data/loss_mae.npy", np.array(loss_history))
    print("✓ 真正的 MAE 预训练完成")
    return loss_history

def train_transformer(data_path, epochs=100, batch_size=512, lr=0.0001):
    print(f"\n{'='*60}")
    print(f"训练 Transformer 编码器 - 语义对比学习")
    print(f"{'='*60}")

    # 1. 加载和预处理数据
    data = np.load(data_path)
    print(f"原始数据维度: {data.shape}")
    
    # 2. 数据检查与清理
    data = check_and_clean_data(data)
    data = data.astype(np.float32)
    
    # 3. 验证数据质量
    if np.isnan(data).any() or np.isinf(data).any():
        print("错误：输入数据仍包含 NaN 或 Inf！")
        return

    input_dim = data.shape[1]
    print(f"处理后数据维度: {input_dim}")

    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TrafficTransformer(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data in loader:
            inputs = batch_data[0].to(DEVICE)
            
            # --- 【核心修改】：使用语义增强替代高斯噪声 ---
            augmented_inputs = traffic_augmentation(inputs)

            _, projections1 = model(inputs)
            _, projections2 = model(augmented_inputs)

            loss = contrastive_loss(projections1, projections2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], 对比学习 Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss_history': loss_history,
            }, ckpt_path)
            print(f"  → 保存: {ckpt_path}")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'loss_history': loss_history,
    }, os.path.join(CHECKPOINT_DIR, "transformer_pretrain.pth"))

    np.save("results/data/loss_transformer.npy", np.array(loss_history))
    print("✓ Transformer 语义对比预训练完成")
    return loss_history

def build_adjacency_matrix_for_features(input_dim, batch_size, k=5):
    """
    为流量数据构建邻接矩阵
    对于流量数据，每个特征维度是一个节点
    邻接矩阵反映特征之间的相关性（这里用随机初始化然后归一化）
    """
    adj = torch.eye(input_dim)
    for i in range(input_dim):
        distances = torch.rand(input_dim)
        _, indices = torch.topk(distances, min(k, input_dim))
        adj[i, indices] = 1.0
    adj = adj + adj.T
    adj = (adj > 0).float()
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1
    adj = adj / row_sum
    adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
    return adj


def main():
    parser = argparse.ArgumentParser(description='预训练模型 - 重构版')
    parser.add_argument('--model', type=str, default='all',
                        choices=['mae', 'transformer', 'all'],
                        help='选择预训练模型: mae, transformer, all')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--mask_ratio', type=float, default=0.4,
                        help='MAE 掩码比例 (default: 0.4)')
    args = parser.parse_args()

    data_path = "data/processed/unsw_X.npy"
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据 {data_path}")
        return

    os.makedirs("results/data", exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"网络流量预训练 - 重构版 - 设备: {DEVICE}")
    print(f"{'#'*60}")

    if args.model == 'mae':
        train_mae(data_path, 
                 epochs=args.epochs or 50, 
                 batch_size=args.batch_size or 1024,
                 mask_ratio=args.mask_ratio)
    elif args.model == 'transformer':
        train_transformer(data_path, 
                         epochs=args.epochs or 100, 
                         batch_size=args.batch_size or 512)
    elif args.model == 'all':
        print("\n>>> 训练所有预训练模型 <<<")
        train_mae(data_path, 
                 epochs=args.epochs or 50, 
                 batch_size=args.batch_size or 1024,
                 mask_ratio=args.mask_ratio)
        print("\n" + "#" * 60)
        train_transformer(data_path, 
                         epochs=args.epochs or 100, 
                         batch_size=args.batch_size or 512)
        print("\n" + "#" * 60)
        print("\n✓✓✓ 所有预训练模型训练完成 ✓✓✓")

if __name__ == "__main__":
    main()
