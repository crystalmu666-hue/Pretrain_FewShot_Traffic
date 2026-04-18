import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import time  # [新增] 用于获取时间戳

# ==========================================
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置信息
# ==========================================
CLASS_NAMES = [
    "Normal", "Botnet", "DDoS", "DoS-GoldenEye", "DoS-Hulk",
    "DoS-Slowloris", "FTP-Patator", "SSH-Patator", "Infiltration",
    "Web-BruteForce", "Web-XSS"
]

def plot_all_results():
    # 创建保存目录
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # [核心修改] 生成当前时间戳，格式为：20260410_163005
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"--- 正在读取实验数据 (时间戳: {timestamp}) ---")

    # ------------------------------------------
    # STEP 1: 加载数据 (.npy)
    # ------------------------------------------
    try:
        # 加载 Loss 曲线数据
        loss_pre = np.load("results/data/loss_hybrid_metric_with_pretrain.npy")
        loss_scratch = np.load("results/data/loss_hybrid_metric_no_pretrain.npy")

        # 加载预测结果与真实标签
        # 注意：这里的文件名需要与 finetune.py 中保存的 mode 对应
        y_true_pre = np.load("results/data/labels_hybrid_metric_with_pretrain.npy")
        y_pred_pre = np.load("results/data/preds_hybrid_metric_with_pretrain.npy")

        y_true_scr = np.load("results/data/labels_hybrid_metric_no_pretrain.npy")
        y_pred_scr = np.load("results/data/preds_hybrid_metric_no_pretrain.npy")

    except FileNotFoundError as e:
        print(f"\n[错误] 找不到数据文件: {e}")
        print("提示：请检查 results/data/ 下的文件名是否包含 'preds_' 和 'labels_' 前缀")
        return

    # ------------------------------------------
    # STEP 2: 绘制 Loss 对比折线图
    # ------------------------------------------
    plt.figure(figsize=(9, 6))
    epochs = range(1, len(loss_pre) + 1)

    plt.plot(epochs, loss_pre, label='Proposed (Hybrid Metric)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, loss_scratch, label='Baseline (Standard)', color='#d62728', linestyle='--', linewidth=2, marker='x', markersize=4)

    plt.title('模型收敛性能对比 (1% 样本规模)', fontsize=14)
    plt.xlabel('训练轮数 (Epochs)', fontsize=12)
    plt.ylabel('平均损失值 (Total Loss)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # [核心修改] 文件名加入时间戳
    loss_plot_path = f"results/plots/loss_comparison_{timestamp}.png"
    plt.savefig(loss_plot_path, dpi=300)
    print(f"√ Loss对比图已保存: {loss_plot_path}")

    # ------------------------------------------
    # STEP 3: 绘制混淆矩阵
    # ------------------------------------------
    cm = confusion_matrix(y_true_pre, y_pred_pre)
    plt.figure(figsize=(12, 10))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 9})

    plt.title(f'混淆矩阵可视化 (Hybrid Metric)\nGenerated at: {timestamp}', fontsize=14)
    plt.ylabel('实际类别 (Actual Label)', fontsize=12)
    plt.xlabel('预测类别 (Predicted Label)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # [核心修改] 文件名加入时间戳
    cm_plot_path = f"results/plots/confusion_matrix_{timestamp}.png"
    plt.savefig(cm_plot_path, dpi=300)
    print(f"√ 混淆矩阵图已保存: {cm_plot_path}")

    # ------------------------------------------
    # STEP 4: 导出分类报告对比
    # ------------------------------------------
    # 自动识别存在的类别，防止报错
    unique_labels = np.unique(y_true_pre)
    target_names = [CLASS_NAMES[i] for i in unique_labels]

    report_pre = classification_report(y_true_pre, y_pred_pre, target_names=target_names, digits=4)
    report_scr = classification_report(y_true_scr, y_pred_scr, target_names=target_names, digits=4)

    # [核心修改] 报告文件名也加入时间戳
    table_path = f"results/tables/metrics_report_{timestamp}.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"实验生成时间: {timestamp}\n")
        f.write("=" * 60 + "\n")
        f.write("实验组：混合损失度量学习 (Hybrid Metric - With Pre-train)\n")
        f.write("=" * 60 + "\n")
        f.write(report_pre)
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("对照组：普通度量学习 (No Pre-train)\n")
        f.write("=" * 60 + "\n")
        f.write(report_scr)

    print(f"√ 性能指标报告已保存: {table_path}")
    print("\n--- 可视化任务完成！ ---")
    plt.show()

if __name__ == "__main__":
    plot_all_results()