import os
import json
from dataclasses import dataclass, asdict

# ==========================================
# 配置类
# ==========================================
@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    checkpoint_dir: str = "checkpoints"
    
@dataclass
class DataConfig:
    """数据配置"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    unsw_file: str = "UNSW-NB15.csv"
    cicids_file: str = "CICIDS2017_all.csv"
    
@dataclass
class ModelConfig:
    """模型配置"""
    input_dim: int = 100  # 实际会根据数据自动调整
    hidden_dim: int = 64
    projection_dim: int = 32
    num_classes: int = 11
    
@dataclass
class ExperimentConfig:
    """实验配置"""
    k_values: list = None
    num_runs: int = 5
    sample_ratios: list = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 5, 10]
        if self.sample_ratios is None:
            self.sample_ratios = [1, 5, 10]

@dataclass
class Config:
    """总配置"""
    training: TrainingConfig = None
    data: DataConfig = None
    model: ModelConfig = None
    experiment: ExperimentConfig = None
    
    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()
    
    def to_dict(self):
        """转换为字典"""
        return {
            "training": asdict(self.training),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "experiment": asdict(self.experiment)
        }
    
    def save(self, path="config.json"):
        """保存配置到文件"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path="config.json"):
        """从文件加载配置"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(
                training=TrainingConfig(**data.get("training", {})),
                data=DataConfig(**data.get("data", {})),
                model=ModelConfig(**data.get("model", {})),
                experiment=ExperimentConfig(**data.get("experiment", {}))
            )
        else:
            return cls()

# ==========================================
# 全局配置实例
# ==========================================
config = Config()

# 确保目录存在
def ensure_directories():
    """确保所有必要的目录存在"""
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.data.raw_data_dir, exist_ok=True)
    os.makedirs(config.data.processed_data_dir, exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/k_shot", exist_ok=True)
    os.makedirs("results/ablation", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)
    os.makedirs("results/analysis/plots", exist_ok=True)
    os.makedirs("results/analysis/tables", exist_ok=True)
    os.makedirs("results/paper_figures/figures", exist_ok=True)
    os.makedirs("results/paper_figures/tables", exist_ok=True)

# 初始化配置
ensure_directories()
