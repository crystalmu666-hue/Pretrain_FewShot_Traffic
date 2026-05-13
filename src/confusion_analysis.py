import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

from finetune import AdapterMetricNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEEDS = [42, 52, 62]
METHODS = ["RandomForest", "KNN", "MetricNet_MAE"]


def ratio_to_folder(ratio):
    return "0" if ratio < 1 else str(int(ratio))


def ratio_to_tag(ratio):
    return "0" if ratio < 1 else str(int(ratio))


def clean_label(label):
    return str(label).replace("\ufffd", "-").strip()


def load_class_names(data_dir):
    label_map_path = Path(data_dir) / "cicids_label_map.json"
    if not label_map_path.exists():
        return None

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    inverse = {int(v): clean_label(k) for k, v in label_map.items()}
    return [inverse[i] for i in sorted(inverse)]


def load_npz(path):
    data = np.load(path)
    return data["x"], data["y"]


def predict_traditional(method, x_train, y_train, x_test, seed):
    if method == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif method == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"unsupported traditional method: {method}")

    model.fit(x_train, y_train)
    return model.predict(x_test)


def load_mae_model(input_dim, num_classes, checkpoint_path):
    model = AdapterMetricNet(
        input_dim=input_dim,
        num_classes=num_classes,
        model_type="mae",
        adapter_dim=input_dim,
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_mae(x_train, y_train, x_test, checkpoint_path):
    model = load_mae_model(
        input_dim=x_train.shape[1],
        num_classes=len(np.unique(y_train)),
        checkpoint_path=checkpoint_path,
    )

    preds = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            batch = torch.FloatTensor(x_test[start : start + batch_size]).to(DEVICE)
            logits, _ = model(batch)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def collect_predictions(ratio, seeds, data_dir, checkpoint_dir):
    folder = ratio_to_folder(ratio)
    tag = ratio_to_tag(ratio)
    x_test, y_test = load_npz(Path(data_dir) / "test_set.npz")

    all_true = {method: [] for method in METHODS}
    all_pred = {method: [] for method in METHODS}
    runs = {method: [] for method in METHODS}

    for seed in seeds:
        train_path = Path(data_dir) / f"cicids_{folder}_seed{seed}.npz"
        if not train_path.exists():
            raise FileNotFoundError(f"missing few-shot data: {train_path}")

        x_train, y_train = load_npz(train_path)

        for method in ["RandomForest", "KNN"]:
            pred = predict_traditional(method, x_train, y_train, x_test, seed)
            all_true[method].append(y_test)
            all_pred[method].append(pred)
            runs[method].append({"seed": seed, "y_true": y_test, "y_pred": pred})

        ckpt_path = Path(checkpoint_dir) / f"finetune_mae_ratio{tag}_seed{seed}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing MAE checkpoint: {ckpt_path}")

        pred = predict_mae(x_train, y_train, x_test, ckpt_path)
        all_true["MetricNet_MAE"].append(y_test)
        all_pred["MetricNet_MAE"].append(pred)
        runs["MetricNet_MAE"].append({"seed": seed, "y_true": y_test, "y_pred": pred})

    return {
        method: {
            "y_true": np.concatenate(all_true[method]),
            "y_pred": np.concatenate(all_pred[method]),
            "runs": runs[method],
        }
        for method in METHODS
    }


def build_metrics(predictions, labels):
    rows = []
    labels = np.asarray(labels)

    for method, data in predictions.items():
        precision, recall, f1, support = precision_recall_fscore_support(
            data["y_true"],
            data["y_pred"],
            labels=labels,
            zero_division=0,
        )

        for idx, label in enumerate(labels):
            rows.append(
                {
                    "method": method,
                    "class_id": int(label),
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "support": int(support[idx]),
                }
            )

    return pd.DataFrame(rows)


def plot_confusion_matrices(predictions, labels, class_names, output_path, title):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(19, 5.8), constrained_layout=True)
    tick_labels = [str(i) for i in labels]

    for ax, method in zip(axes, METHODS):
        cm = confusion_matrix(
            predictions[method]["y_true"],
            predictions[method]["y_pred"],
            labels=labels,
            normalize="true",
        )

        sns.heatmap(
            cm,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar=method == METHODS[-1],
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            square=True,
            linewidths=0.2,
        )
        ax.set_title(method)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    legend_path = output_path.with_name(output_path.stem + "_class_legend.csv")
    pd.DataFrame({"class_id": labels, "class_name": class_names}).to_csv(legend_path, index=False)


def plot_per_class_bars(metrics_df, class_names, output_path, title):
    labels_df = pd.DataFrame(
        {"class_id": list(range(len(class_names))), "class_name": class_names}
    )
    plot_df = metrics_df.merge(labels_df, on="class_id", how="left")
    plot_df["class_label"] = plot_df["class_id"].astype(str) + ": " + plot_df["class_name"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)

    sns.barplot(
        data=plot_df,
        x="class_label",
        y="recall",
        hue="method",
        ax=axes[0],
        palette="Set2",
    )
    axes[0].set_title("Per-class recall")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="upper right")

    sns.barplot(
        data=plot_df,
        x="class_label",
        y="f1",
        hue="method",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("Per-class F1")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend_.remove()
    axes[1].tick_params(axis="x", rotation=35)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize(metrics_df, predictions, labels):
    summary = {}
    for method, group in metrics_df.groupby("method"):
        seed_macro_f1 = []
        seed_macro_recall = []
        for run in predictions[method]["runs"]:
            _, recall, f1, _ = precision_recall_fscore_support(
                run["y_true"],
                run["y_pred"],
                labels=labels,
                zero_division=0,
            )
            seed_macro_recall.append(float(np.mean(recall)))
            seed_macro_f1.append(float(np.mean(f1)))

        summary[method] = {
            "aggregate_macro_recall": float(group["recall"].mean()),
            "aggregate_macro_f1": float(group["f1"].mean()),
            "mean_seed_macro_recall": float(np.mean(seed_macro_recall)),
            "std_seed_macro_recall": float(np.std(seed_macro_recall)),
            "mean_seed_macro_f1": float(np.mean(seed_macro_f1)),
            "std_seed_macro_f1": float(np.std(seed_macro_f1)),
            "classes_with_nonzero_recall": int((group["recall"] > 0).sum()),
        }
    return summary


def build_seed_metrics(predictions, labels):
    rows = []
    for method in METHODS:
        for run in predictions[method]["runs"]:
            _, recall, f1, _ = precision_recall_fscore_support(
                run["y_true"],
                run["y_pred"],
                labels=labels,
                zero_division=0,
            )
            rows.append(
                {
                    "method": method,
                    "seed": int(run["seed"]),
                    "macro_recall": float(np.mean(recall)),
                    "macro_f1": float(np.mean(f1)),
                    "classes_with_nonzero_recall": int((recall > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate normalized confusion matrices and per-class metrics."
    )
    parser.add_argument("--ratio", type=float, default=0.1, help="few-shot sample ratio, e.g. 0.1")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="results/confusion2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.data_dir)
    if class_names is None:
        _, y_test = load_npz(Path(args.data_dir) / "test_set.npz")
        class_names = [f"class_{i}" for i in sorted(np.unique(y_test))]

    labels = np.arange(len(class_names))
    predictions = collect_predictions(args.ratio, args.seeds, args.data_dir, args.checkpoint_dir)
    metrics_df = build_metrics(predictions, labels)

    ratio_name = str(args.ratio).replace(".", "_")
    metrics_path = output_dir / f"per_class_metrics_ratio_{ratio_name}.csv"
    seed_metrics_path = output_dir / f"seed_metrics_ratio_{ratio_name}.csv"
    summary_path = output_dir / f"summary_ratio_{ratio_name}.json"
    confusion_path = output_dir / f"normalized_confusion_ratio_{ratio_name}.png"
    bars_path = output_dir / f"per_class_recall_f1_ratio_{ratio_name}.png"

    metrics_df.to_csv(metrics_path, index=False)
    seed_metrics_df = build_seed_metrics(predictions, labels)
    seed_metrics_df.to_csv(seed_metrics_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summarize(metrics_df, predictions, labels), f, indent=2)

    title = f"RF vs KNN vs MetricNet_MAE at {args.ratio}% samples"
    plot_confusion_matrices(predictions, labels, class_names, confusion_path, title)
    plot_per_class_bars(metrics_df, class_names, bars_path, title)

    print(f"saved metrics: {metrics_path}")
    print(f"saved seed metrics: {seed_metrics_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved confusion matrices: {confusion_path}")
    print(f"saved per-class bars: {bars_path}")


if __name__ == "__main__":
    main()
