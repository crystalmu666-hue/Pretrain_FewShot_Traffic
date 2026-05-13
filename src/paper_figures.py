"""
Generate paper-ready figures and tables for the MAE + AdapterMetricNet study.

Expected inputs:
  - results/k_shot/episodic/k_shot_results.json
  - results/ablation/ablation_results_all.json
  - results/cross_domain/cross_domain_results.json
  - results/cross_domain/cross_domain_results_0p1.json (optional)
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


class PaperFigures:
    def __init__(self, output_dir="results/paper_figures"):
        self.output_dir = output_dir
        self.figure_dir = os.path.join(output_dir, "figures2")
        self.table_dir = os.path.join(output_dir, "tables")
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

    def generate_all(self):
        self._generate_k_shot_curve()
        self._generate_k_shot_gain_heatmap()
        self._generate_ablation_core_bar()
        self._generate_ablation_tables()
        self._generate_cross_domain_figures()
        self._generate_cross_domain_table()
        print(f"figures saved to: {self.figure_dir}")
        print(f"tables saved to:  {self.table_dir}")

    @staticmethod
    def _load_json(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"result file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _mean_std(values):
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    def _save_csv(self, filename, rows, fieldnames):
        path = os.path.join(self.table_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved table: {path}")
        return path

    def _load_k_shot_summary(self):
        data = self._load_json("results/k_shot/episodic/k_shot_results.json")
        rows = []

        for ratio, ratio_data in data["results"].items():
            grouped = defaultdict(list)
            ci_values = defaultdict(list)

            for seed_data in ratio_data.values():
                for model_type, model_data in seed_data.items():
                    if model_type not in {"none", "mae"}:
                        continue
                    for k, metrics in model_data["k"].items():
                        key = (float(ratio), model_type, int(k))
                        grouped[key].append(metrics["macro_f1"]["mean"])
                        ci_values[key].append(metrics["macro_f1"].get("ci95", 0.0))

            for (ratio_value, model_type, k), values in grouped.items():
                mean, std = self._mean_std(values)
                rows.append(
                    {
                        "ratio": ratio_value,
                        "model_type": model_type,
                        "k": k,
                        "macro_f1_mean": mean,
                        "macro_f1_std_across_seeds": std,
                        "mean_episode_ci95": float(np.mean(ci_values[(ratio_value, model_type, k)])),
                        "seeds": len(values),
                    }
                )

        rows.sort(key=lambda r: (r["ratio"], r["model_type"], r["k"]))
        return rows

    def _generate_k_shot_curve(self):
        rows = self._load_k_shot_summary()
        self._save_csv(
            "k_shot_summary.csv",
            rows,
            [
                "ratio",
                "model_type",
                "k",
                "macro_f1_mean",
                "macro_f1_std_across_seeds",
                "mean_episode_ci95",
                "seeds",
            ],
        )

        ratios = sorted({row["ratio"] for row in rows})
        models = ["none", "mae"]
        colors = {"none": "#6b7280", "mae": "#d62728"}
        labels = {"none": "No pretrain", "mae": "MAE"}

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        axes = axes.flatten()

        for ax, ratio in zip(axes, ratios):
            for model in models:
                model_rows = [r for r in rows if r["ratio"] == ratio and r["model_type"] == model]
                if not model_rows:
                    continue
                model_rows.sort(key=lambda r: r["k"])
                x = [r["k"] for r in model_rows]
                y = [r["macro_f1_mean"] for r in model_rows]
                err = [r["macro_f1_std_across_seeds"] for r in model_rows]
                ax.errorbar(x, y, yerr=err, marker="o", linewidth=2, capsize=3, color=colors[model], label=labels[model])

            ax.set_title(f"{ratio:g}% labeled target data")
            ax.set_xlabel("K shots")
            ax.set_xticks([1, 2, 5, 10])
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        axes[0].set_ylabel("Episodic Macro-F1")
        axes[2].set_ylabel("Episodic Macro-F1")
        axes[0].legend(frameon=False, loc="lower right")
        fig.suptitle("K-shot episodic evaluation on CICIDS", y=0.98)
        fig.tight_layout()

        path = os.path.join(self.figure_dir, "k_shot_macro_f1_curves.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure: {path}")

    def _generate_k_shot_gain_heatmap(self):
        rows = self._load_k_shot_summary()
        ratios = sorted({row["ratio"] for row in rows})
        ks = sorted({row["k"] for row in rows})
        value_map = {(r["ratio"], r["model_type"], r["k"]): r["macro_f1_mean"] for r in rows}

        gains = np.zeros((len(ratios), len(ks)), dtype=np.float64)
        table_rows = []
        for i, ratio in enumerate(ratios):
            for j, k in enumerate(ks):
                gain = value_map[(ratio, "mae", k)] - value_map[(ratio, "none", k)]
                gains[i, j] = gain
                table_rows.append({"ratio": ratio, "k": k, "mae_minus_no_pretrain_macro_f1": gain})

        self._save_csv("k_shot_mae_gain_heatmap_values.csv", table_rows, ["ratio", "k", "mae_minus_no_pretrain_macro_f1"])

        fig, ax = plt.subplots(figsize=(7, 4.8))
        im = ax.imshow(gains, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_yticks(range(len(ratios)), [f"{ratio:g}%" for ratio in ratios])
        ax.set_xlabel("K shots")
        ax.set_ylabel("Labeled target ratio")
        ax.set_title("MAE pretraining gain in episodic Macro-F1")

        for i in range(len(ratios)):
            for j in range(len(ks)):
                ax.text(j, i, f"{gains[i, j]:+.3f}", ha="center", va="center", fontsize=10)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("MAE - no pretrain")
        fig.tight_layout()

        path = os.path.join(self.figure_dir, "k_shot_mae_gain_heatmap.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure: {path}")

    def _load_ablation_summary(self):
        data = self._load_json("results/ablation/ablation_results_all.json")
        rows = []
        display_names = {}

        for ratio, ratio_data in data.items():
            for variant in ratio_data["config"]["variants"]:
                display_names[variant["name"]] = variant.get("display_name", variant["name"])

            for key, values in ratio_data["summary"].items():
                _, variant = key.split("::", 1)
                rows.append(
                    {
                        "ratio": float(ratio),
                        "variant": variant,
                        "display_name": display_names.get(variant, variant),
                        "mean_macro_f1": values["mean_macro_f1"],
                        "std_macro_f1": values["std_macro_f1"],
                        "runs": values["runs"],
                    }
                )

        rows.sort(key=lambda r: (r["ratio"], r["variant"]))
        return rows

    def _generate_ablation_core_bar(self):
        rows = self._load_ablation_summary()
        self._save_csv(
            "ablation_full_summary.csv",
            rows,
            ["ratio", "variant", "display_name", "mean_macro_f1", "std_macro_f1", "runs"],
        )

        ratios_to_show = sorted({r["ratio"] for r in rows})[:4]
        variants = ["ce", "ce_fixed_pm", "cew_fixed_pm", "cew_csa_pm"]
        labels = ["CE", "CE +\nFixed PM", "CE^w +\nFixed PM", "CE^w +\nCSA-PM"]
        row_map = {(r["ratio"], r["variant"]): r for r in rows}

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        axes = axes.flatten()
        x = np.arange(len(variants))

        for ax, ratio in zip(axes, ratios_to_show):
            means = [row_map[(ratio, variant)]["mean_macro_f1"] for variant in variants]
            stds = [row_map[(ratio, variant)]["std_macro_f1"] for variant in variants]
            colors = ["#9ca3af", "#60a5fa", "#f59e0b", "#d62728"]
            ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.9)
            ax.set_title(f"{ratio:g}% labeled data")
            ax.set_xticks(x, labels)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            for idx, value in enumerate(means):
                ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)

        axes[0].set_ylabel("Test Macro-F1")
        axes[2].set_ylabel("Test Macro-F1")
        fig.suptitle("Loss ablation for MAE + AdapterMetricNet", y=0.98)
        fig.tight_layout()

        path = os.path.join(self.figure_dir, "ablation_loss_components.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure: {path}")

    def _generate_ablation_tables(self):
        rows = self._load_ablation_summary()
        row_map = {(r["ratio"], r["variant"]): r for r in rows}
        table_rows = []

        for ratio in sorted({r["ratio"] for r in rows}):
            ce = row_map[(ratio, "ce")]["mean_macro_f1"]
            ce_fixed = row_map[(ratio, "ce_fixed_pm")]["mean_macro_f1"]
            cew_fixed = row_map[(ratio, "cew_fixed_pm")]["mean_macro_f1"]
            cew_csa = row_map[(ratio, "cew_csa_pm")]["mean_macro_f1"]
            table_rows.append(
                {
                    "ratio": ratio,
                    "ce": ce,
                    "ce_fixed_pm": ce_fixed,
                    "cew_fixed_pm": cew_fixed,
                    "cew_csa_pm": cew_csa,
                    "fixed_pm_gain_vs_ce": ce_fixed - ce,
                    "class_weight_gain": cew_fixed - ce_fixed,
                    "csa_pm_gain_vs_fixed_pm": cew_csa - cew_fixed,
                    "csa_pm_gain_vs_ce": cew_csa - ce,
                }
            )

        self._save_csv(
            "ablation_core_deltas.csv",
            table_rows,
            [
                "ratio",
                "ce",
                "ce_fixed_pm",
                "cew_fixed_pm",
                "cew_csa_pm",
                "fixed_pm_gain_vs_ce",
                "class_weight_gain",
                "csa_pm_gain_vs_fixed_pm",
                "csa_pm_gain_vs_ce",
            ],
        )

    def _load_cross_domain_summary(self):
        paths = [
            "results/cross_domain/cross_domain_results.json",
            "results/cross_domain/cross_domain_results_0p1.json",
        ]
        merged = {}

        for path in paths:
            if not os.path.exists(path):
                continue

            data = self._load_json(path)
            for ratio, summary in data["summary"].items():
                for key, values in summary.items():
                    _, variant = key.split("::", 1)
                    ratio_value = float(ratio)
                    merged[(ratio_value, variant)] = {
                        "ratio": ratio_value,
                        "variant": variant,
                        "mean_macro_f1": values["mean_macro_f1"],
                        "std_macro_f1": values["std_macro_f1"],
                        "runs": values["runs"],
                        "source_file": path,
                    }

        if not merged:
            raise FileNotFoundError("no cross-domain result files found")

        rows = list(merged.values())
        rows.sort(key=lambda r: (r["ratio"], r["variant"]))
        return rows

    def _generate_cross_domain_figures(self):
        rows = self._load_cross_domain_summary()
        ratios = sorted({r["ratio"] for r in rows})
        variant_order = [
            "random",
            "mae_pretrained_finetune",
            "mae_pretrained_frozen",
            "mae_pretrained_coral",
            "mae_pretrained_mmd",
        ]
        label_map = {
            "random": "Random",
            "mae_pretrained_finetune": "MAE\nfinetune",
            "mae_pretrained_frozen": "MAE\nfrozen",
            "mae_pretrained_coral": "MAE\n+CORAL",
            "mae_pretrained_mmd": "MAE\n+MMD",
        }
        colors = {
            "random": "#6b7280",
            "mae_pretrained_finetune": "#2ca02c",
            "mae_pretrained_frozen": "#ff7f0e",
            "mae_pretrained_coral": "#d62728",
            "mae_pretrained_mmd": "#1f77b4",
        }
        row_map = {(r["ratio"], r["variant"]): r for r in rows}
        variants = [
            variant
            for variant in variant_order
            if any((ratio, variant) in row_map for ratio in ratios)
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(ratios))
        width = min(0.18, 0.8 / max(len(variants), 1))
        center_offset = (len(variants) - 1) / 2.0

        for idx, variant in enumerate(variants):
            means = [row_map.get((ratio, variant), {}).get("mean_macro_f1", np.nan) for ratio in ratios]
            stds = [row_map.get((ratio, variant), {}).get("std_macro_f1", 0.0) for ratio in ratios]
            ax.bar(
                x + (idx - center_offset) * width,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=colors[variant],
                label=label_map[variant],
            )

        ax.set_xticks(x, [f"{ratio:g}%" for ratio in ratios])
        ax.set_xlabel("Labeled target ratio")
        ax.set_ylabel("Test Macro-F1")
        ax.set_title("UNSW-NB15 to CICIDS transfer with MAE")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        fig.tight_layout()

        path = os.path.join(self.figure_dir, "cross_domain_mae_macro_f1_bars.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure: {path}")

    def _generate_cross_domain_table(self):
        rows = self._load_cross_domain_summary()
        row_map = {(r["ratio"], r["variant"]): r for r in rows}
        table_rows = []

        for ratio in sorted({r["ratio"] for r in rows}):
            random_row = row_map.get((ratio, "random"))
            random_mean = random_row["mean_macro_f1"] if random_row else np.nan
            for row in [r for r in rows if r["ratio"] == ratio]:
                table_rows.append(
                    {
                        "ratio": ratio,
                        "variant": row["variant"],
                        "mean_macro_f1": row["mean_macro_f1"],
                        "std_macro_f1": row["std_macro_f1"],
                        "runs": row["runs"],
                        "gain_vs_random": row["mean_macro_f1"] - random_mean,
                        "source_file": row.get("source_file", ""),
                    }
                )

        self._save_csv(
            "cross_domain_summary.csv",
            table_rows,
            [
                "ratio",
                "variant",
                "mean_macro_f1",
                "std_macro_f1",
                "runs",
                "gain_vs_random",
                "source_file",
            ],
        )


if __name__ == "__main__":
    PaperFigures().generate_all()
