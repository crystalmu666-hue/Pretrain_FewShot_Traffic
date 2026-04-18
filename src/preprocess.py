import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ==========================================
# 1. 实验配置 (符合任务书规范性要求)
# ==========================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def preprocess_pipeline(file_path, output_dir, is_pretrain=False):
    print(f"\n[开始处理] {os.path.basename(file_path)}")

    # 1. 加载数据
    df = pd.read_csv(file_path, low_memory=False)

    # 2. 标签锁定
    label_col = ' Label' if ' Label' in df.columns else ('Label' if 'Label' in df.columns else df.columns[-1])

    # 3. 核心：彻底清洗异常值 (解决 Infinity 报错)
    # 将所有列转为数值，无法转换的变 NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    X = df[numeric_cols].copy()

    # 将无穷大 (inf) 替换为 NaN，然后统一删除
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 合并标签一起删，确保行数对应
    temp_df = pd.concat([X, df[label_col]], axis=1)
    temp_df.dropna(inplace=True)  # 删除所有包含 NaN 或 inf 的行

    X = temp_df.iloc[:, :-1]
    y = temp_df.iloc[:, -1]

    print(f"确认标签列为: {label_col}, 包含类别数: {len(np.unique(y))}")
    print(f"数据清洗完成，剩余样本数: {len(X)}")

    # 4. 标准化 (现在不会报错了)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if is_pretrain:
        np.save(os.path.join(output_dir, "unsw_pretrain.npy"), X_scaled)
        print("√ UNSW 预训练数据已保存")
    else:
        # 5. 分层抽样 (必须有 2 类以上才能跑)
        y_encoded, _ = pd.factorize(y)
        unique_classes = np.unique(y_encoded)

        if len(unique_classes) < 2:
            print(f"!!! 警告：{os.path.basename(file_path)} 仅含单一类别，无法进行分类实验！")
            print(f"建议：请确认使用的是否为 Wednesday 或 Thursday 的数据，Monday 通常只有正常流量。")
            return  # 跳过保存，防止生成错误文件

        for r in [0.01, 0.05, 0.10]:
            _, X_sub, _, y_sub = train_test_split(
                X_scaled, y_encoded, test_size=r, random_state=42, stratify=y_encoded
            )
            np.savez(os.path.join(output_dir, f"cicids_{int(r * 100)}pct.npz"), x=X_sub, y=y_sub)
            print(f"v 已生成 {int(r * 100)}% 子集")
# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 获取当前工作目录，确保路径绝对化
    BASE_DIR = os.getcwd()
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

    # 自动创建输出文件夹
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("=== 网络流量小样本学习：数据工程阶段 ===")

    # 任务 1: 处理 UNSW-NB15 用于无监督预训练 (第3-4周)
    preprocess_pipeline(
        os.path.join(RAW_DATA_DIR, "UNSW-NB15.csv"),
        PROCESSED_DATA_DIR,
        is_pretrain=True
    )

    # 任务 2: 处理 CICIDS2017 用于小样本微调 (第5周)
    preprocess_pipeline(
        os.path.join(RAW_DATA_DIR, "CICIDS2017_all.csv"),
        PROCESSED_DATA_DIR,
        is_pretrain=False
    )

    print("\n[所有检查点已完成]")
    print(f"最终输出清单: {os.listdir(PROCESSED_DATA_DIR)}")