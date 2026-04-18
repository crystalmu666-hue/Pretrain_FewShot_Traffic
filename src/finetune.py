import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score
import time


def get_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_name = time.strftime("%Y%m%d_%H%M%S") + "_finetune.log"
    log_path = os.path.join(log_dir, log_name)
    return log_path


def log_to_file(path, message):
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# ==========================================
# 1. 实验配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# --- [创新点超参数] ---
MARGIN = 1.0  # 强制隔离间隔
TRIPLET_WEIGHT = 0.1  # 三元组损失项的权重

PRETRAIN_PATH = "checkpoints/pretrain_epoch_50.pth"
#PRETRAIN_PATH = ""


SUBSET_PATH = "data/processed/cicids_1pct.npz"


# ==========================================
# 2. 定义模型
# ==========================================
class MetricNet(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super(MetricNet, self).__init__()
        self.feature_align = nn.Linear(input_dim, 40)
        self.encoder = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        # 可学习的原型矩阵
        self.prototypes = nn.Parameter(torch.randn(num_classes, 64))

    def forward(self, x):
        x = self.feature_align(x)
        features = self.encoder(x)

        # 计算欧氏距离
        features_exp = features.unsqueeze(1)
        prototypes_exp = self.prototypes.unsqueeze(0)
        distances = torch.norm(features_exp - prototypes_exp, p=2, dim=-1)

        return -distances, features


# ==========================================
# [核心创新实现] 混合损失函数
# ==========================================
def hybrid_triplet_prototype_loss(logits, targets, margin=1.0):
    """
    logits: 负距离矩阵 [-dist_1, -dist_2, ..., -dist_11]
    targets: 真实标签
    """
    # 1. 基础交叉熵损失 (让负距离大的类别概率大)
    ce_loss = nn.CrossEntropyLoss()(logits, targets)

    # 2. 三元组约束损失 (Triplet-style Margin Constraint)
    # 正例距离 (取负值的相反数)
    batch_range = torch.arange(logits.size(0)).to(DEVICE)
    pos_dist = -logits[batch_range, targets]

    # 找到最近的负例距离 (排除掉正确类别后的最大 logits)
    mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0.)
    # 排除掉正确类，取剩下的 logits 中的最大值（即最小距离的负例）
    nearest_neg_logits, _ = torch.max(logits * mask - (1 - mask) * 1e9, dim=1)
    nearest_neg_dist = -nearest_neg_logits

    # 损失公式: ReLU(d_pos - d_neg + margin)
    triplet_loss = torch.clamp(pos_dist - nearest_neg_dist + margin, min=0.0).mean()

    return ce_loss, triplet_loss


# ==========================================
# 3. 微调主函数
# ==========================================
def run_finetune():
    log_path = get_logger()
    print(f"--- 启动混合损失(Hybrid Loss)度量学习微调 ---")

    config_info = f"Config: LR={LEARNING_RATE}, MARGIN={MARGIN}, TRIPLET_W={TRIPLET_WEIGHT}"
    log_to_file(log_path, config_info)

    # A. 加载数据
    data = np.load(SUBSET_PATH)
    X, y = data['x'], data['y']
    input_dim = X.shape[1]
    num_classes = 11

    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # B. 初始化模型
    model = MetricNet(input_dim, num_classes).to(DEVICE)

    # 加载预训练权重
    if os.path.exists(PRETRAIN_PATH) and PRETRAIN_PATH != "":
        checkpoint = torch.load(PRETRAIN_PATH, map_location=DEVICE)

        # 1. 提取包含相关关键字的权重
        raw_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                    if 'encoder' in k or 'feature_align' in k}

        # 2. 【关键修改】过滤掉维度不匹配的层 (即 encoder.4)
        pretrained_dict = {}
        for k, v in raw_dict.items():
            # 检查当前模型中对应层的形状
            if k in model.state_dict():
                if v.shape == model.state_dict()[k].shape:
                    pretrained_dict[k] = v
                else:
                    print(
                        f"  [跳过] 层 {k} 维度不匹配 (预训练:{v.shape} -> 当前:{model.state_dict()[k].shape})，将随机初始化。")

        # 3. 加载过滤后的权重
        model.load_state_dict(pretrained_dict, strict=False)
        msg = f"√ 成功加载预训练权重（已跳过不匹配层）: {PRETRAIN_PATH}"
    else:
        msg = "! 警告：未发现预训练权重，模型将从随机初始化开始"
    print(msg);
    log_to_file(log_path, msg)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # D. 训练循环
    loss_history = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()

            logits, _ = model(batch_x)

            # --- 应用混合损失 ---
            ce_loss, triplet_loss = hybrid_triplet_prototype_loss(logits, batch_y, margin=MARGIN)
            combined_loss = ce_loss + TRIPLET_WEIGHT * triplet_loss

            combined_loss.backward()
            optimizer.step()
            total_loss += combined_loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        msg = f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f} (CE: {ce_loss:.4f}, Trip: {triplet_loss:.4f})"
        print(msg);
        log_to_file(log_path, msg)

    # E. 最终评估
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits, _ = model(batch_x.to(DEVICE))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    report = classification_report(all_labels, all_preds, digits=4)
    log_to_file(log_path, "\nFinal Evaluation Report:\n" + report)

    # 保存结果
    os.makedirs("results/data", exist_ok=True)
    mode = "hybrid_metric_with_pretrain" if (
                os.path.exists(PRETRAIN_PATH) and PRETRAIN_PATH != "") else "hybrid_metric_no_pretrain"
    np.save(f"results/data/loss_{mode}.npy", np.array(loss_history))
    np.save(f"results/data/preds_{mode}.npy", np.array(all_preds))
    np.save(f"results/data/labels_{mode}.npy", np.array(all_labels))
    print(f"√ 实验完成，模式: {mode}\n", report)


if __name__ == "__main__":
    run_finetune()