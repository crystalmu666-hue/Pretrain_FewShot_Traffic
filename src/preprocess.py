import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =========================================================
# UNSW 列名
# =========================================================
UNSW_COLUMNS = [
    "srcip","sport","dstip","dsport","proto","state","dur",
    "sbytes","dbytes","sttl","dttl","sloss","dloss","service",
    "Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb",
    "smeansz","dmeansz","trans_depth","res_bdy_len","Sjit","Djit",
    "Stime","Ltime","Sintpkt","Dintpkt","tcprtt","synack","ackdat",
    "is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login",
    "ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm",
    "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm",
    "attack_cat","label"
]

# =========================================================
# 通用特征处理
# =========================================================
def process_features(df, label_col):
    y = df[label_col].astype(str)
    y = y.replace(['nan', 'None', ''], np.nan)

    mask = ~y.isna()
    df = df[mask]
    y = y[mask]

    X = df.drop(columns=[label_col])

    # 删除高风险字段
    drop_cols = ['srcip', 'dstip', 'sport', 'dsport', 'Stime', 'Ltime']
    X = X.drop(columns=drop_cols, errors='ignore')

    # 分类特征 → Frequency Encoding（仅用训练集）
    cat_cols = X.select_dtypes(include=['object']).columns
    X_cat = pd.DataFrame(index=X.index)

    # 先划分临时训练集索引用于频率编码
    train_idx, _ = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    for col in cat_cols:
        freq = X[col].iloc[train_idx].value_counts(normalize=True)
        X[col] = X[col].map(freq).fillna(0)
        X_cat[col] = X[col]

    # 数值特征
    num_cols = X.select_dtypes(exclude=['object']).columns
    X_num = X[num_cols].apply(pd.to_numeric, errors='coerce')

    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
        q1 = X_num[col].quantile(0.01)
        q99 = X_num[col].quantile(0.99)
        X_num[col] = X_num[col].clip(q1, q99)

    # 合并
    X_all = pd.concat([X_num, X_cat], axis=1)
    return X_all, y

# =========================================================
# Few-shot 构造
# =========================================================
def create_fewshot(X, y, out_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.savez(os.path.join(out_dir, "test_set.npz"), x=X_test, y=y_test)

    # 按类别索引
    class_indices = {cls: np.where(y_train == cls)[0] for cls in np.unique(y_train)}

    ratios = [0.001, 0.01, 0.05, 0.1]
    seeds = [42, 52, 62]  # 可扩展多随机种子

    for r in ratios:
        for seed in seeds:
            np.random.seed(seed)
            X_sub_list, y_sub_list = [], []

            for cls, indices in class_indices.items():
                n_cls = min(len(indices), max(2, int(len(indices) * r)))
                selected = np.random.choice(indices, n_cls, replace=False)
                X_sub_list.append(X_train[selected])
                y_sub_list.append(y_train[selected])

            X_sub = np.vstack(X_sub_list)
            y_sub = np.concatenate(y_sub_list)

            perm = np.random.permutation(len(X_sub))
            X_sub = X_sub[perm]
            y_sub = y_sub[perm]

            print(f"Ratio {int(r*100)}% | Seed {seed} | 样本数: {len(X_sub)} | 类别数: {len(np.unique(y_sub))}")

            np.savez(os.path.join(out_dir, f"cicids_{int(r*100)}_seed{seed}.npz"), x=X_sub, y=y_sub)

# =========================================================
# UNSW 预训练
# =========================================================
def preprocess_unsw(path, out_dir):
    print("\n[UNSW] loading...")
    df = pd.read_csv(path, header=None, low_memory=False)
    df.columns = UNSW_COLUMNS
    label_col = "attack_cat"
    X, y = process_features(df, label_col)

    labels = sorted(y.unique())
    label_map = {l: i for i, l in enumerate(labels)}
    y_encoded = y.map(label_map).values

    with open(os.path.join(out_dir, "unsw_label_map.json"), "w") as f:
        json.dump(label_map, f)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("UNSW shape:", X_scaled.shape)
    print("UNSW classes:", len(label_map))

    np.save(os.path.join(out_dir, "unsw_X.npy"), X_scaled)
    np.save(os.path.join(out_dir, "unsw_y.npy"), y_encoded)
    print("[UNSW] done")

# =========================================================
# CICIDS 下游任务
# =========================================================
def preprocess_cicids(path, out_dir):
    print("\n[CICIDS]")
    df = pd.read_csv(path, low_memory=False)
    label_col = ' Label' if ' Label' in df.columns else 'Label'
    X, y = process_features(df, label_col)

    labels = sorted(y.unique())
    label_map = {l: i for i, l in enumerate(labels)}
    y_encoded = y.map(label_map).values

    with open(os.path.join(out_dir, "cicids_label_map.json"), "w") as f:
        json.dump(label_map, f)

    print("类别分布:", np.bincount(y_encoded))
    create_fewshot(X.values, y_encoded, out_dir)

# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    BASE = os.getcwd()
    RAW = os.path.join(BASE, "data", "raw")
    OUT = os.path.join(BASE, "data", "processed")
    os.makedirs(OUT, exist_ok=True)

    print("=== CLEAN PIPELINE START ===")
    preprocess_unsw(os.path.join(RAW, "UNSW-NB15.csv"), OUT)
    preprocess_cicids(os.path.join(RAW, "CICIDS2017_all.csv"), OUT)
    print("\n✅ DONE")