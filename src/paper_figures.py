"""
Generate paper-ready tables and figures for the final IR-SCB-Focal results.

The script focuses on the figures that directly support the paper claims:
  - main result tables for Macro-F1 / Balanced Accuracy / Rare Recall
  - grouped Macro-F1 bar charts for in-domain and cross-domain experiments
  - Macro-F1 gain over the strongest baseline
  - rho adaptation mechanism plot
  - Focal-family ablation bar chart for representative scenarios
  - optional seed-level stability point chart
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from finetune_methods import ABLATION_METHODS, MAIN_METHODS, METHOD_LABELS


RESULT_DIR = "results/ir_scb_focal_rhomin0_full"
OUTPUT_DIR = "results/paper_figures/ir_scb_focal_final"

MAIN_ORDER = ["ce", "wce", "focal", "balanced_softmax", "ir_scb_focal"]
ABLATION_ORDER = ["focal", "cb_focal", "scb_focal_rho025", "scb_focal_rho05", "ir_scb_focal"]

COLORS = {
    "ce": "#77c5ff",
    "wce": "#24d9ff",
    "focal": "#ffd354",
    "balanced_softmax": "#5df6c1",
    "cb_focal": "#ffab51",
    "scb_focal_rho025": "#ff8061",
    "scb_focal_rho05": "#ff557b",
    "ir_scb_focal": "#ff329b",
}

METRICS = [
    ("macro_f1", "Macro-F1"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("rare_recall", "Rare Recall"),
]

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


class PaperFigureBuilder:
    def __init__(self, result_dir=RESULT_DIR, output_dir=OUTPUT_DIR):
        self.result_dir = result_dir
        self.output_dir = output_dir
        self.figure_dir = os.path.join(output_dir, "figures")
        self.table_dir = os.path.join(output_dir, "tables")
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

    def generate_all(self):
        self._clean_output_dirs()

        main_data = {
            "indomain": self._load_required("indomain_main_results.json"),
            "crossdomain": self._load_required("crossdomain_main_results.json"),
        }
        ablation_data = {
            "indomain": self._load_required("indomain_ablation_results.json"),
            "crossdomain": self._load_required("crossdomain_ablation_results.json"),
        }

        for domain, data in main_data.items():
            self._ensure_summary(data)
            self._save_main_metric_tables(domain, data)
            self._plot_grouped_macro_f1(domain, data)

        self._save_gain_table_and_plot(main_data)
        self._plot_rho_mechanism(main_data["crossdomain"])

        for data in ablation_data.values():
            self._ensure_summary(data)
        self._save_ablation_tables(ablation_data)
        self._plot_ablation_heatmaps(ablation_data)
        self._plot_key_ablation_bars(ablation_data)
        self._plot_seed_stability(main_data)

        print(f"paper figures saved to: {self.figure_dir}")
        print(f"paper tables saved to:  {self.table_dir}")

    def _clean_output_dirs(self):
        for directory in [self.figure_dir, self.table_dir]:
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    os.remove(path)

    def _load_required(self, filename):
        path = os.path.join(self.result_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing result file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_csv(self, filename, rows, fieldnames):
        path = os.path.join(self.table_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved table: {path}")

    def _save_md(self, filename, rows, headers):
        path = os.path.join(self.table_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
        print(f"saved table: {path}")

    def _save_fig(self, fig, filename):
        path = os.path.join(self.figure_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure: {path}")

    @staticmethod
    def _ratio_label(ratio):
        return f"{float(ratio):g}%"

    @staticmethod
    def _ratio_key(ratio):
        return str(float(ratio))

    @staticmethod
    def _fmt(mean, std):
        return f"{mean:.4f} +/- {std:.4f}"

    @staticmethod
    def _mean_std(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return 0.0, 0.0
        return float(values.mean()), float(values.std(ddof=0))

    def _ratios(self, data):
        return sorted(data["summary"].keys(), key=lambda item: float(item))

    def _runs(self, data):
        runs = []
        for ratio_data in data.get("results", {}).values():
            for seed_data in ratio_data.values():
                runs.extend(seed_data.get("runs", []))
        return runs

    def _ensure_summary(self, data):
        runs = self._runs(data)
        if not runs:
            return

        grouped = defaultdict(list)
        for run in runs:
            grouped[(str(run["ratio"]), run["method"])].append(run)

        data.setdefault("summary", {})
        for (ratio, method), items in grouped.items():
            data["summary"].setdefault(ratio, {})
            summary = data["summary"][ratio].setdefault(
                method,
                {"runs": len(items), "method_label": METHOD_LABELS.get(method, method)},
            )
            for metric, _ in METRICS + [("accuracy", "Accuracy"), ("weighted_f1", "Weighted F1")]:
                values = [float(run["test"].get(metric, 0.0)) for run in items]
                summary[f"{metric}_mean"], summary[f"{metric}_std"] = self._mean_std(values)
                summary[f"{metric}_list"] = values

    def _method_order(self, data, preferred):
        methods = set()
        for ratio_summary in data["summary"].values():
            methods.update(ratio_summary.keys())
        ordered = [method for method in preferred if method in methods]
        ordered.extend(sorted(methods - set(ordered)))
        return ordered

    def _save_main_metric_tables(self, domain, data):
        rows_long = []
        for metric, label in METRICS:
            methods = self._method_order(data, MAIN_ORDER)
            ratios = self._ratios(data)
            headers = ["Method"] + [f"{self._ratio_label(r)} {label}" for r in ratios] + [f"Average {label}"]
            rows = []

            for method in methods:
                row = {"Method": METHOD_LABELS.get(method, method)}
                means = []
                for ratio in ratios:
                    item = data["summary"].get(ratio, {}).get(method)
                    if item is None:
                        row[f"{self._ratio_label(ratio)} {label}"] = ""
                        continue
                    mean = item[f"{metric}_mean"]
                    std = item[f"{metric}_std"]
                    row[f"{self._ratio_label(ratio)} {label}"] = self._fmt(mean, std)
                    means.append(mean)
                    rows_long.append(
                        {
                            "domain": domain,
                            "metric": label,
                            "ratio": self._ratio_label(ratio),
                            "method": METHOD_LABELS.get(method, method),
                            "mean": mean,
                            "std": std,
                        }
                    )
                row[f"Average {label}"] = f"{np.mean(means):.4f}" if means else ""
                rows.append(row)

            stem = f"{domain}_main_{metric}_table"
            self._save_csv(f"{stem}.csv", rows, headers)
            self._save_md(f"{stem}.md", rows, headers)

        self._save_csv(
            f"{domain}_main_all_metrics_long.csv",
            rows_long,
            ["domain", "metric", "ratio", "method", "mean", "std"],
        )

    def _plot_grouped_macro_f1(self, domain, data):
        methods = self._method_order(data, MAIN_ORDER)
        ratios = self._ratios(data)
        x = np.arange(len(ratios))
        width = 0.8 / max(len(methods), 1)

        fig, ax = plt.subplots(figsize=(10, 5.2))
        for idx, method in enumerate(methods):
            means = [data["summary"][ratio][method]["macro_f1_mean"] for ratio in ratios]
            stds = [data["summary"][ratio][method]["macro_f1_std"] for ratio in ratios]
            offset = (idx - (len(methods) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=METHOD_LABELS.get(method, method),
                color=COLORS.get(method),
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_xticks(x, [self._ratio_label(ratio) for ratio in ratios])
        ax.set_xlabel("Target labeled sample ratio")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0.0, min(1.0, self._max_metric(data, "macro_f1") + 0.12))
        ax.set_title(f"{self._domain_title(domain)}: Macro-F1 comparison")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        self._save_fig(fig, f"{domain}_main_macro_f1_grouped_bar.png")

    def _save_gain_table_and_plot(self, main_data):
        rows = []
        for domain, data in main_data.items():
            for ratio in self._ratios(data):
                target = data["summary"][ratio].get("ir_scb_focal")
                baselines = []
                for method in MAIN_ORDER:
                    if method == "ir_scb_focal":
                        continue
                    item = data["summary"][ratio].get(method)
                    if item:
                        baselines.append((method, item["macro_f1_mean"]))
                if not target or not baselines:
                    continue
                best_method, best_value = max(baselines, key=lambda item: item[1])
                ir_value = target["macro_f1_mean"]
                rows.append(
                    {
                        "domain": domain,
                        "ratio": self._ratio_label(ratio),
                        "best_baseline": METHOD_LABELS.get(best_method, best_method),
                        "best_baseline_macro_f1": best_value,
                        "ir_scb_focal_macro_f1": ir_value,
                        "delta_macro_f1": ir_value - best_value,
                    }
                )

        self._save_csv(
            "ir_scb_focal_gain_over_best_baseline.csv",
            rows,
            ["domain", "ratio", "best_baseline", "best_baseline_macro_f1", "ir_scb_focal_macro_f1", "delta_macro_f1"],
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
        for ax, domain in zip(axes, ["indomain", "crossdomain"]):
            domain_rows = [row for row in rows if row["domain"] == domain]
            x = np.arange(len(domain_rows))
            values = [row["delta_macro_f1"] for row in domain_rows]
            colors = ["#d62728" if value >= 0 else "#6b7280" for value in values]
            ax.bar(x, values, color=colors, width=0.62)
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_xticks(x, [row["ratio"] for row in domain_rows])
            ax.set_title(self._domain_title(domain))
            ax.set_xlabel("Target labeled sample ratio")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            for idx, value in enumerate(values):
                va = "bottom" if value >= 0 else "top"
                y = value + (0.004 if value >= 0 else -0.004)
                ax.text(idx, y, f"{value:+.4f}", ha="center", va=va, fontsize=9)
        axes[0].set_ylabel("Delta Macro-F1 over strongest baseline")
        fig.suptitle("IR-SCB-Focal gain over the strongest baseline", y=1.02)
        fig.tight_layout()
        self._save_fig(fig, "ir_scb_focal_gain_over_best_baseline.png")

    def _plot_rho_mechanism(self, data):
        runs = [run for run in self._runs(data) if run["method"] == "ir_scb_focal"]
        grouped = defaultdict(list)
        for run in runs:
            info = run.get("loss_info", {})
            grouped[str(run["ratio"])].append(info)

        rows = []
        ratios = sorted(grouped.keys(), key=lambda item: float(item))
        values = {key: [] for key in ["imbalance_ratio", "rho_ir", "ratio_decay", "rho"]}
        for ratio in ratios:
            infos = grouped[ratio]
            row = {"ratio": self._ratio_label(ratio)}
            for key in values:
                mean, std = self._mean_std([float(info.get(key, 0.0)) for info in infos])
                values[key].append(mean)
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
            rows.append(row)

        self._save_csv(
            "rho_adaptation_mechanism_values.csv",
            rows,
            [
                "ratio",
                "imbalance_ratio_mean",
                "imbalance_ratio_std",
                "rho_ir_mean",
                "rho_ir_std",
                "ratio_decay_mean",
                "ratio_decay_std",
                "rho_mean",
                "rho_std",
            ],
        )

        x = np.arange(len(ratios))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

        axes[0].bar(x, values["imbalance_ratio"], color="#9ca3af", width=0.58)
        axes[0].set_xticks(x, [self._ratio_label(ratio) for ratio in ratios])
        axes[0].set_xlabel("Target labeled sample ratio")
        axes[0].set_ylabel("Imbalance Ratio (IR)")
        axes[0].set_title("Class imbalance")
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

        axes[1].plot(x, values["rho_ir"], marker="o", linewidth=2.2, color="#1f77b4", label="rho_ir")
        axes[1].plot(x, values["ratio_decay"], marker="s", linewidth=2.2, color="#2ca02c", label="ratio_decay")
        axes[1].plot(x, values["rho"], marker="^", linewidth=2.4, color="#d62728", label="final rho")
        axes[1].set_xticks(x, [self._ratio_label(ratio) for ratio in ratios])
        axes[1].set_xlabel("Target labeled sample ratio")
        axes[1].set_ylabel("Value")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Adaptive rho mechanism")
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
        axes[1].legend(frameon=False)

        fig.suptitle("IR-SCB-Focal: imbalance-ratio and ratio-aware rho adaptation", y=1.02)
        fig.tight_layout()
        self._save_fig(fig, "rho_adaptation_mechanism.png")

    def _save_ablation_tables(self, ablation_data):
        rows = []
        for domain, data in ablation_data.items():
            for ratio in self._ratios(data):
                for method in self._method_order(data, ABLATION_ORDER):
                    item = data["summary"][ratio].get(method)
                    if not item:
                        continue
                    rows.append(
                        {
                            "domain": domain,
                            "ratio": self._ratio_label(ratio),
                            "method": METHOD_LABELS.get(method, method),
                            "macro_f1": self._fmt(item["macro_f1_mean"], item["macro_f1_std"]),
                            "balanced_accuracy": self._fmt(item["balanced_accuracy_mean"], item["balanced_accuracy_std"]),
                            "rare_recall": self._fmt(item["rare_recall_mean"], item["rare_recall_std"]),
                        }
                    )
        self._save_csv(
            "focal_family_ablation_all_metrics.csv",
            rows,
            ["domain", "ratio", "method", "macro_f1", "balanced_accuracy", "rare_recall"],
        )

    def _plot_key_ablation_bars(self, ablation_data):
        scenarios = [
            ("indomain", "1.0", "In-domain 1%"),
            ("crossdomain", "0.1", "Cross-domain 0.1%"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, (domain, ratio, title) in zip(axes, scenarios):
            data = ablation_data[domain]
            methods = self._method_order(data, ABLATION_ORDER)
            x = np.arange(len(methods))
            means = [data["summary"][ratio][method]["macro_f1_mean"] for method in methods]
            stds = [data["summary"][ratio][method]["macro_f1_std"] for method in methods]
            ax.bar(
                x,
                means,
                yerr=stds,
                capsize=4,
                color=[COLORS.get(method) for method in methods],
                width=0.66,
                edgecolor="white",
                linewidth=0.6,
            )
            ax.set_xticks(x, [METHOD_LABELS.get(method, method) for method in methods], rotation=25, ha="right")
            ax.set_xlabel("Focal-family variant")
            ax.set_title(title)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        axes[0].set_ylabel("Macro-F1")
        axes[0].set_ylim(0.0, 0.75)
        fig.suptitle("Focal-family ablation on representative scenarios", y=1.02)
        fig.tight_layout()
        self._save_fig(fig, "focal_family_ablation_key_scenarios.png")

    def _plot_ablation_heatmaps(self, ablation_data):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, constrained_layout=True)
        heatmap_values = []
        prepared = {}

        for domain, data in ablation_data.items():
            methods = self._method_order(data, ABLATION_ORDER)
            ratios = self._ratios(data)
            matrix = np.asarray(
                [
                    [data["summary"][ratio][method]["macro_f1_mean"] for ratio in ratios]
                    for method in methods
                ],
                dtype=np.float64,
            )
            prepared[domain] = (methods, ratios, matrix)
            heatmap_values.extend(matrix.reshape(-1).tolist())

        vmin = min(heatmap_values) if heatmap_values else 0.0
        vmax = max(heatmap_values) if heatmap_values else 1.0

        for ax, domain in zip(axes, ["indomain", "crossdomain"]):
            methods, ratios, matrix = prepared[domain]
            image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(ratios)), [self._ratio_label(ratio) for ratio in ratios])
            ax.set_yticks(np.arange(len(methods)), [METHOD_LABELS.get(method, method) for method in methods])
            ax.set_xlabel("Target labeled sample ratio")
            ax.set_title(self._domain_title(domain))

            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix[row, col]
                    text_color = "white" if value > (vmin + vmax) / 2.0 else "black"
                    ax.text(col, row, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)

        axes[0].set_ylabel("Focal-family variant")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, label="Macro-F1")
        fig.suptitle("Focal-family ablation Macro-F1 heatmap")
        self._save_fig(fig, "focal_family_ablation_macro_f1_heatmap.png")

    def _plot_seed_stability(self, main_data):
        scenarios = [
            ("indomain", "1.0", "In-domain 1%"),
            ("crossdomain", "0.1", "Cross-domain 0.1%"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
        for ax, (domain, ratio, title) in zip(axes, scenarios):
            data = main_data[domain]
            runs = self._runs(data)
            methods = self._method_order(data, MAIN_ORDER)
            for idx, method in enumerate(methods):
                vals = [
                    run["test"]["macro_f1"]
                    for run in runs
                    if run["method"] == method and self._ratio_key(run["ratio"]) == ratio
                ]
                if not vals:
                    continue
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
                ax.scatter(
                    np.full(len(vals), idx) + jitter,
                    vals,
                    color=COLORS.get(method),
                    s=36,
                    alpha=0.85,
                    zorder=3,
                )
                ax.hlines(np.mean(vals), idx - 0.22, idx + 0.22, color="black", linewidth=1.4)
            ax.set_xticks(np.arange(len(methods)), [METHOD_LABELS.get(m, m) for m in methods], rotation=25, ha="right")
            ax.set_ylabel("Macro-F1")
            ax.set_title(title)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig.suptitle("Seed-level Macro-F1 stability", y=1.02)
        fig.tight_layout()
        self._save_fig(fig, "seed_level_stability_key_scenarios.png")

    @staticmethod
    def _domain_title(domain):
        if domain == "indomain":
            return "In-domain"
        if domain == "crossdomain":
            return "Cross-domain"
        return domain

    @staticmethod
    def _max_metric(data, metric):
        max_value = 0.0
        for ratio_summary in data["summary"].values():
            for item in ratio_summary.values():
                max_value = max(max_value, float(item.get(f"{metric}_mean", 0.0)))
        return max_value


def main():
    parser = argparse.ArgumentParser(description="Generate final IR-SCB-Focal paper tables and figures")
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    PaperFigureBuilder(args.result_dir, args.output_dir).generate_all()


if __name__ == "__main__":
    main()
