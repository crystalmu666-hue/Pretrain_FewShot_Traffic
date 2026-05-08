import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MaskedTrafficAutoencoder(nn.Module):
    """掩码自动编码器用于无监督特征学习"""
    def __init__(self, input_dim, mask_ratio=0.75, hidden_dim=128, latent_dim=32):
        super(MaskedTrafficAutoencoder, self).__init__()
        self.mask_ratio = mask_ratio
        self.input_dim = input_dim
        
        # 编码器：处理未被掩码的特征
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # 最终提取的 32 维通用特征
        )
        
        # 解码器：恢复被掩码的特征
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def mask_input(self, x):
        """生成掩码并应用到输入"""
        batch_size = x.shape[0]
        mask = torch.zeros((batch_size, self.input_dim), device=x.device)
        num_masked = int(self.mask_ratio * self.input_dim)
        
        for i in range(batch_size):
            mask_indices = torch.randperm(self.input_dim)[:num_masked]
            mask[i, mask_indices] = 1
        
        masked_x = x * (1 - mask)
        return masked_x, mask

    def forward(self, x):
        # 应用掩码
        masked_x, mask = self.mask_input(x)
        # 编码
        encoded = self.encoder(masked_x)
        # 解码
        decoded = self.decoder(encoded)
        return decoded, mask


class TrafficTransformer(nn.Module):
    """改进版：基于特征分组与简化架构的 Transformer"""

    def __init__(self, input_dim=40, hidden_dim=64, num_heads=2, num_layers=1, projection_dim=32):
        super(TrafficTransformer, self).__init__()

        # 确定分组逻辑：40 = 8个token * 每个token 5维特征
        self.num_tokens = 6
        self.token_dim = input_dim // self.num_tokens
        self.input_dim = input_dim

        # 1. 特征嵌入：处理分组后的特征
        self.embedding = nn.Linear(self.token_dim, hidden_dim)

        # 2. 位置编码：现在的序列长度是 8
        self.position_encoding = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim))

        # 3. 简化 Transformer 架构
        # 减少层数（2->1）和头数（4->2），降低在 1% 样本下的过拟合风险
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # 缩小中间层
            batch_first=True  # 使用 batch_first 简化维度变换
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 投影头保持不变
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, projection_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # 自动适配维度
        usable_dim = self.num_tokens * self.token_dim
        if x.shape[1] < usable_dim:
            raise ValueError(f"输入维度不足: {x.shape[1]} < {usable_dim}")

        x = x[:, :usable_dim]

        # reshape
        x = x.view(batch_size, self.num_tokens, self.token_dim)

        # embedding
        x = self.embedding(x)

        # position encoding
        x = x + self.position_encoding[:, :self.num_tokens, :]

        # transformer
        x = self.transformer(x)

        # pooling
        features = x.mean(dim=1)

        projections = self.projection(features)

        return features, projections



class MetricNet(nn.Module):
    """混合损失度量学习模型 (多架构动态适配版)"""

    def __init__(self, input_dim=40, num_classes=11, model_type='mae'):
        super(MetricNet, self).__init__()
        self.model_type = model_type

        # --- 核心改动：根据模型类型动态加载"灵魂（编码器）" ---
        # MAE 预训练模型: Linear(40,256) → ReLU → Linear(256,128) → ReLU → Linear(128,128) → ReLU → Linear(128,32)
        if model_type == 'mae':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 32)
            )
            self.latent_dim = 32

        # Transformer 预训练模型: embedding → transformer → pooling → projection(32维)
        elif model_type == 'transformer':
            self.encoder = TrafficTransformer(input_dim, hidden_dim=128)
            self.latent_dim = 32

        else:  # none (无预训练)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64)
            )
            self.latent_dim = 64

        # 原型矩阵动态匹配提取特征的输出维度
        self.prototypes = nn.Parameter(torch.randn(num_classes, self.latent_dim))

    def _build_adj_matrix(self, batch_size, input_dim, device):
        """为图卷积在微调阶段动态生成邻接矩阵"""
        adj = torch.eye(input_dim, device=device)
        for i in range(input_dim):
            distances = torch.rand(input_dim, device=device)
            _, indices = torch.topk(distances, min(5, input_dim))
            adj[i, indices] = 1.0
        adj = adj + adj.T
        adj = (adj > 0).float()
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1
        adj = adj / row_sum
        return adj.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x):
        # --- 根据模型类型，动态分配前向传播路线 ---
        if self.model_type == 'transformer':
            # TrafficTransformer 的返回值是 (features, projections)
            # 使用 projections (32维) 而不是 features (128维)
            _, features = self.encoder(x)

        elif self.model_type == 'graph':
            # Graph 需要输入邻接矩阵
            # 使用 projections (32维) 而不是 features
            batch_size, input_dim = x.shape
            adj_matrix = self._build_adj_matrix(batch_size, input_dim, x.device)
            _, features = self.encoder(x, adj_matrix)

        else:
            # MAE 只是纯 Sequential
            features = self.encoder(x)

        # 统一的距离计算逻辑（原型学习）
        features_exp = features.unsqueeze(1)
        prototypes_exp = self.prototypes.unsqueeze(0)
        distances = torch.norm(features_exp - prototypes_exp, p=2, dim=-1)

        return -distances, features


def load_pretrained_weights(model, checkpoint_path, model_type):
    """
    智能权重加载器：自动处理前缀不匹配、维度不匹配等问题
    """
    if not os.path.exists(checkpoint_path):
        print(f"  [警告] 权重文件不存在: {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # 自动处理不同格式的保存文件
    pretrained_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    model_dict = model.state_dict()
    new_state_dict = {}
    matched_layers = 0

    for k, v in pretrained_dict.items():
        # 核心逻辑：尝试为键名加上 'encoder.' 前缀来匹配 MetricNet 结构
        target_key = k if k in model_dict else f"encoder.{k}"

        if target_key in model_dict:
            # 只有当形状完全一致时才加载
            if v.shape == model_dict[target_key].shape:
                new_state_dict[target_key] = v
                matched_layers += 1

    model.load_state_dict(new_state_dict, strict=False)
    print(f"  ✓ {model_type.upper()} 成功匹配并加载了 {matched_layers} 层权重")
    return True