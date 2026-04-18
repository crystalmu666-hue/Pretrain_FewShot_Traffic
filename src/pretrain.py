import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# ==========================================
# 1. 配置参数 (符合任务书实验规范)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 0.001
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==========================================
# 2. 定义模型架构 (自编码器用于无监督特征学习)
# ==========================================
class TrafficAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TrafficAutoencoder, self).__init__()
        # 编码器：将高维流量压缩到隐含空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # 最终提取的 32 维通用特征
        )
        # 解码器：尝试还原原始输入
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==========================================
# 3. 训练函数
# ==========================================
def train_pretrain():
    print(f"--- 启动预训练阶段 (使用设备: {DEVICE}) ---")

    # 加载第一阶段生成的 .npy 文件
    data_path = "data/processed/unsw_pretrain.npy"
    if not os.path.exists(data_path):
        print(f"错误: 找不到预处理数据 {data_path}，请先运行 preprocess.py")
        return

    data = np.load(data_path)
    input_dim = data.shape[1]
    print(f"加载数据成功，特征维度: {input_dim}")

    # 准备 DataLoader
    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = TrafficAutoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 开始训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_data in loader:
            inputs = batch_data[0].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # 无监督：目标就是还原输入本身
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(loader)

        # 打印日志 (第4周实验记录素材)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

        # 保存 Checkpoint (任务书核心要求)
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"pretrain_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"√ 已保存权重至: {ckpt_path}")

    print("\n--- 预训练任务圆满完成 ---")


if __name__ == "__main__":
    train_pretrain()