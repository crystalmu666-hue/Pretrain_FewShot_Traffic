import json
import os

import matplotlib.pyplot as plt
import numpy as np


with open("results/comparison/comparison_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

save_dir = "results/figures3"
os.makedirs(save_dir, exist_ok=True)

PREFERRED_ORDER = [
    "SVM",
    "KNN",
    "RandomForest",
    "MLP",
    "MAE_Linear",
    "Adapter_MAE_Linear",
    "ProtoNet",
    "MetricNet",
    "MetricNet_MAE",
]
TRAINABLE_CURVE_METHODS = {"MLP", "MAE_Linear", "Adapter_MAE_Linear", "MetricNet", "MetricNet_MAE"}


def mean_std(values):
    return np.mean(values), np.std(values)


def smooth_curve(values, window=3):
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start : i + 1]))
    return smoothed


def average_history(histories):
    if not histories or not histories[0]:
        return [], [], [], [], []

    epochs = [h["epoch"] for h in histories[0]]
    avg_loss, std_loss = [], []
    avg_f1, std_f1 = [], []

    for i in range(len(epochs)):
        losses, f1s = [], []

        for seed_hist in histories:
            if i < len(seed_hist):
                if seed_hist[i]["loss"] is not None:
                    losses.append(seed_hist[i]["loss"])
                f1s.append(seed_hist[i]["macro_f1"])

        if losses:
            avg_loss.append(np.mean(losses))
            std_loss.append(np.std(losses))
        else:
            avg_loss.append(None)
            std_loss.append(None)

        avg_f1.append(np.mean(f1s))
        std_f1.append(np.std(f1s))

    return epochs, avg_loss, std_loss, avg_f1, std_f1


plt.figure(figsize=(8, 6))
ratios = sorted(results.keys(), key=lambda x: float(x))

all_methods = {}
for method in results[ratios[0]]["simple_dl"].keys():
    all_methods[method] = "simple_dl"
for method in results[ratios[0]]["traditional"].keys():
    all_methods[method] = "traditional"

ordered_methods = [m for m in PREFERRED_ORDER if m in all_methods]
ordered_methods.extend([m for m in all_methods if m not in ordered_methods])

for method in ordered_methods:
    group = all_methods[method]
    means = []
    stds = []

    for r in ratios:
        data = results[r][group][method]
        m, s = mean_std(data["macro_f1_list"])
        means.append(m)
        stds.append(s)

    plt.errorbar(ratios, means, yerr=stds, marker="o", capsize=4, label=method)

plt.xlabel("Sample Ratio (%)")
plt.ylabel("Macro-F1")
plt.title("Performance vs Sample Ratio")
plt.legend()
plt.grid()
plt.savefig(f"{save_dir}/f1_vs_ratio.png", dpi=300)
plt.close()


for r in ratios:
    plt.figure(figsize=(10, 6))
    methods = []
    scores = []

    method_data = {}
    for name, data in results[r]["traditional"].items():
        method_data[name] = data["macro_f1"]
    for name, data in results[r]["simple_dl"].items():
        method_data[name] = data["macro_f1"]

    ordered = [m for m in PREFERRED_ORDER if m in method_data]
    ordered.extend([m for m in method_data if m not in ordered])
    for name in ordered:
        methods.append(name)
        scores.append(method_data[name])

    x = np.arange(len(methods))
    plt.bar(x, scores)
    plt.xticks(x, methods, rotation=30)
    plt.ylabel("Macro-F1")
    plt.title(f"Model Comparison ({r}%)")

    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/bar_ratio_{r}.png", dpi=300)
    plt.close()


for r in ratios:
    plt.figure(figsize=(8, 6))
    has_curve = False

    for name, data in results[r]["simple_dl"].items():
        if name not in TRAINABLE_CURVE_METHODS:
            continue
        if "history_per_seed" not in data:
            continue

        epochs, _, _, f1, f1_std = average_history(data["history_per_seed"])
        if not epochs:
            continue

        plt.plot(epochs, f1, marker="o", label=name)
        plt.fill_between(
            epochs,
            np.array(f1) - np.array(f1_std),
            np.array(f1) + np.array(f1_std),
            alpha=0.2,
        )
        has_curve = True

    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"F1 Convergence ({r}%)")
    plt.grid()
    if has_curve:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No F1 history available", ha="center", va="center", transform=plt.gca().transAxes)

    plt.savefig(f"{save_dir}/f1_curve_ratio_{r}.png", dpi=300)
    plt.close()


for r in ratios:
    plt.figure(figsize=(8, 6))
    has_curve = False

    for name, data in results[r]["simple_dl"].items():
        if name not in TRAINABLE_CURVE_METHODS:
            continue
        if "history_per_seed" not in data:
            continue

        epochs, loss, loss_std, _, _ = average_history(data["history_per_seed"])
        if not epochs or all(l is None for l in loss):
            continue

        loss_smooth = smooth_curve(loss, window=3)
        plt.plot(epochs, loss_smooth, label=name)
        plt.fill_between(
            epochs,
            np.array(loss) - np.array(loss_std),
            np.array(loss) + np.array(loss_std),
            alpha=0.2,
        )
        has_curve = True

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Convergence ({r}%)")
    plt.grid()
    if has_curve:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No loss history available", ha="center", va="center", transform=plt.gca().transAxes)

    plt.savefig(f"{save_dir}/loss_curve_ratio_{r}.png", dpi=300)
    plt.close()


print("generated analysis figures:", save_dir)
