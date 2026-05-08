"""
Generate paper-ready figures and tables from the current experiment outputs.

Inputs:
  - results/k_shot/episodic/k_shot_results.json
  - results/ablation/ablation_results_all.json
  - results/cross_domain/cross_domain_results.json

Outputs:
  - results/paper_figures/figures/*.png
  - results/paper_figures/tables/*.csv
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
        self.figure_dir = os.path.join(output_dir, "figures")
        self.table_dir = os.path.join(output_dir, "tables")
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

    def generate_all(self):
        print("Generating paper figures and tables...")
        self._generate_k_shot_curve()
        self._generate_k_shot_gain_heatmap()
        self._generate_ablation_core_bar()
        self._generate_ablation_tables()
        self._generate_cross_domain_figures()
        self._generate_cross_domain_table()
        print(f"Done. Figures: {self.figure_dir}")
        print(f"Done. Tables:  {self.table_dir}")

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

    # ------------------------------------------------------------------
    # K-shot
    # ------------------------------------------------------------------
    def _load_k_shot_summary(self):
        data = self._load_json("results/k_shot/episodic/k_shot_results.json")
        rows = []

        for ratio, ratio_data in data["results"].items():
            grouped = defaultdict(list)
            ci_values = defaultdict(list)

            for seed_data in ratio_data.values():
                for model_type, model_data in seed_data.items():
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
        models = ["none", "mae", "transformer"]
        colors = {"none": "#6b7280", "mae": "#d62728", "transformer": "#1f77b4"}
        labels = {"none": "No pretrain", "mae": "MAE", "transformer": "Transformer"}

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
                ax.errorbar(
                    x,
                    y,
                    yerr=err,
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    color=colors[model],
                    label=labels[model],
                )

            ax.set_title(f"{ratio:g}% labeled target data")
            ax.set_xlabel("K shots")
            ax.set_xticks([1, 2, 5, 10])
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        axes[0].set_ylabel("Episodic Macro-F1")
        axes[2].set_ylabel("Episodic Macro-F1")
        axes[0].legend(frameon=False, loc="lower right")
        fig.suptitle("K-shot episodic evaluation on the independent CICIDS test set", y=0.98)
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

        self._save_csv(
            "k_shot_mae_gain_heatmap_values.csv",
            table_rows,
            ["ratio", "k", "mae_minus_no_pretrain_macro_f1"],
        )

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

    # ------------------------------------------------------------------
    # Ablation
    # ------------------------------------------------------------------
    def _load_ablation_summary(self):
        data = self._load_json("results/ablation/ablation_results_all.json")
        rows = []
        for ratio, ratio_data in data.items():
            for key, values in ratio_data["summary"].items():
                model_type, variant = key.split("::", 1)
                rows.append(
                    {
                        "ratio": float(ratio),
                        "model_type": model_type,
                        "variant": variant,
                        "mean_macro_f1": values["mean_macro_f1"],
                        "std_macro_f1": values["std_macro_f1"],
                        "runs": values["runs"],
                    }
                )
        rows.sort(key=lambda r: (r["ratio"], r["model_type"], r["variant"]))
        return rows

    def _generate_ablation_core_bar(self):
        rows = self._load_ablation_summary()
        self._save_csv(
            "ablation_full_summary.csv",
            rows,
            ["ratio", "model_type", "variant", "mean_macro_f1", "std_macro_f1", "runs"],
        )

        ratios_to_show = [0.1, 1.0]
        variants = [
            "random_ce",
            "random_margin",
            "pretrained_ce",
            "pretrained_full",
            "pretrained_no_class_weights",
            "pretrained_frozen",
        ]
        variant_labels = [
            "Random\nCE",
            "Random\n+Margin",
            "Pretrain\nCE",
            "Pretrain\nFull",
            "Full\nno CW",
            "Pretrain\nFrozen",
        ]
        colors = {"mae": "#d62728", "transformer": "#1f77b4"}
        row_map = {(r["ratio"], r["model_type"], r["variant"]): r for r in rows}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
        x = np.arange(len(variants))
        width = 0.36

        for ax, ratio in zip(axes, ratios_to_show):
            for offset, model_type in [(-width / 2, "mae"), (width / 2, "transformer")]:
                means = [row_map[(ratio, model_type, v)]["mean_macro_f1"] for v in variants]
                stds = [row_map[(ratio, model_type, v)]["std_macro_f1"] for v in variants]
                ax.bar(
                    x + offset,
                    means,
                    width,
                    yerr=stds,
                    capsize=3,
                    color=colors[model_type],
                    alpha=0.85,
                    label=model_type.upper(),
                )

            ax.set_title(f"Ablation at {ratio:g}% labeled data")
            ax.set_xticks(x, variant_labels, rotation=0)
            ax.set_xlabel("Variant")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        axes[0].set_ylabel("Test Macro-F1")
        axes[1].legend(frameon=False, loc="upper left")
        fig.suptitle("Component ablation under low-label regimes", y=1.02)
        fig.tight_layout()

        path = os.path.join(self.figure_dir, "ablation_low_label_components.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure: {path}")

    def _generate_ablation_tables(self):
        rows = self._load_ablation_summary()
        row_map = {(r["ratio"], r["model_type"], r["variant"]): r for r in rows}
        table_rows = []

        for ratio in sorted({r["ratio"] for r in rows}):
            for model_type in ["mae", "transformer"]:
                random_ce = row_map[(ratio, model_type, "random_ce")]["mean_macro_f1"]
                pretrained_ce = row_map[(ratio, model_type, "pretrained_ce")]["mean_macro_f1"]
                pretrained_full = row_map[(ratio, model_type, "pretrained_full")]["mean_macro_f1"]
                pretrained_no_cw = row_map[(ratio, model_type, "pretrained_no_class_weights")]["mean_macro_f1"]
                table_rows.append(
                    {
                        "ratio": ratio,
                        "model_type": model_type,
                        "random_ce": random_ce,
                        "pretrained_ce": pretrained_ce,
                        "pretrained_full": pretrained_full,
                        "full_gain_vs_random_ce": pretrained_full - random_ce,
                        "margin_gain_vs_pretrained_ce": pretrained_full - pretrained_ce,
                        "class_weight_gain_vs_no_cw": pretrained_full - pretrained_no_cw,
                    }
                )

        self._save_csv(
            "ablation_core_deltas.csv",
            table_rows,
            [
                "ratio",
                "model_type",
                "random_ce",
                "pretrained_ce",
                "pretrained_full",
                "full_gain_vs_random_ce",
                "margin_gain_vs_pretrained_ce",
                "class_weight_gain_vs_no_cw",
            ],
        )

    # ------------------------------------------------------------------
    # Cross-domain transfer
    # ------------------------------------------------------------------
    def _load_cross_domain_summary(self):
        data = self._load_json("results/cross_domain/cross_domain_results.json")
        rows = []
        for ratio, summary in data["summary"].items():
            for key, values in summary.items():
                model_type, variant = key.split("::", 1)
                rows.append(
                    {
                        "ratio": float(ratio),
                        "model_type": model_type,
                        "variant": variant,
                        "mean_macro_f1": values["mean_macro_f1"],
                        "std_macro_f1": values["std_macro_f1"],
                        "runs": values["runs"],
                    }
                )
        rows.sort(key=lambda r: (r["ratio"], r["model_type"], r["variant"]))
        return rows

    def _load_cross_domain_pretrain_audit(self):
        data = self._load_json("results/cross_domain/cross_domain_results.json")
        audit = {}
        for ratio_data in data["results"].values():
            for seed_data in ratio_data.values():
                for run in seed_data["runs"]:
                    if run["pretrain"]["loaded"]:
                        key = (run["model_type"], run["variant"])
                        audit[key] = {
                            "matched_tensors": run["pretrain"]["matched_tensors"],
                            "source_tensors": run["pretrain"]["source_tensors"],
                            "path": run["pretrain"]["path"],
                        }
        return audit

    def _generate_cross_domain_figures(self):
        rows = self._load_cross_domain_summary()
        row_map = {(r["ratio"], r["model_type"], r["variant"]): r for r in rows}
        ratios = sorted({r["ratio"] for r in rows})
        variants = ["random", "pretrained_finetune", "pretrained_frozen"]
        labels = ["Random", "Pretrained\nfinetune", "Pretrained\nfrozen"]
        colors = {"random": "#6b7280", "pretrained_finetune": "#2ca02c", "pretrained_frozen": "#ff7f0e"}

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        x = np.arange(len(ratios))
        width = 0.25

        for ax, model_type in zip(axes, ["mae", "transformer"]):
            for idx, variant in enumerate(variants):
                means = [row_map[(ratio, model_type, variant)]["mean_macro_f1"] for ratio in ratios]
                stds = [row_map[(ratio, model_type, variant)]["std_macro_f1"] for ratio in ratios]
                ax.bar(
                    x + (idx - 1) * width,
                    means,
                    width,
                    yerr=stds,
                    capsize=3,
                    color=colors[variant],
                    label=labels[idx],
                )
            ax.set_title(f"{model_type.upper()} backbone")
            ax.set_xticks(x, [f"{ratio:g}%" for ratio in ratios])
            ax.set_xlabel("Labeled target ratio")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        axes[0].set_ylabel("Test Macro-F1")
        axes[1].legend(frameon=False, loc="lower right")
        fig.suptitle("UNSW-NB15 to CICIDS-2017 transfer performance", y=1.02)
        fig.tight_layout()

        path = os.path.join(self.figure_dir, "cross_domain_macro_f1_bars.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure: {path}")

        gain_rows = []
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for model_type, color in [("mae", "#d62728"), ("transformer", "#1f77b4")]:
            gains = []
            for ratio in ratios:
                random_f1 = row_map[(ratio, model_type, "random")]["mean_macro_f1"]
                pretrain_f1 = row_map[(ratio, model_type, "pretrained_finetune")]["mean_macro_f1"]
                gain = pretrain_f1 - random_f1
                gains.append(gain)
                gain_rows.append(
                    {
                        "ratio": ratio,
                        "model_type": model_type,
                        "pretrained_finetune_minus_random": gain,
                    }
                )
            ax.plot(ratios, gains, marker="o", linewidth=2, color=color, label=model_type.upper())

        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_xlabel("Labeled target ratio (%)")
        ax.set_ylabel("Macro-F1 gain")
        ax.set_title("Cross-domain pretraining gain over random initialization")
        ax.set_xticks(ratios, [f"{ratio:g}%" for ratio in ratios])
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        fig.tight_layout()

        path = os.path.join(self.figure_dir, "cross_domain_transfer_gain.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure: {path}")

        self._save_csv(
            "cross_domain_gain_values.csv",
            gain_rows,
            ["ratio", "model_type", "pretrained_finetune_minus_random"],
        )

    def _generate_cross_domain_table(self):
        rows = self._load_cross_domain_summary()
        row_map = {(r["ratio"], r["model_type"], r["variant"]): r for r in rows}
        audit = self._load_cross_domain_pretrain_audit()
        table_rows = []

        for ratio in sorted({r["ratio"] for r in rows}):
            for model_type in ["mae", "transformer"]:
                random_row = row_map[(ratio, model_type, "random")]
                finetune_row = row_map[(ratio, model_type, "pretrained_finetune")]
                frozen_row = row_map[(ratio, model_type, "pretrained_frozen")]
                audit_info = audit.get((model_type, "pretrained_finetune"), {})
                table_rows.append(
                    {
                        "ratio": ratio,
                        "model_type": model_type,
                        "random_mean_macro_f1": random_row["mean_macro_f1"],
                        "random_std_macro_f1": random_row["std_macro_f1"],
                        "pretrained_finetune_mean_macro_f1": finetune_row["mean_macro_f1"],
                        "pretrained_finetune_std_macro_f1": finetune_row["std_macro_f1"],
                        "pretrained_frozen_mean_macro_f1": frozen_row["mean_macro_f1"],
                        "pretrained_frozen_std_macro_f1": frozen_row["std_macro_f1"],
                        "gain_finetune_vs_random": finetune_row["mean_macro_f1"] - random_row["mean_macro_f1"],
                        "matched_tensors": audit_info.get("matched_tensors", ""),
                        "source_tensors": audit_info.get("source_tensors", ""),
                    }
                )

        self._save_csv(
            "cross_domain_summary.csv",
            table_rows,
            [
                "ratio",
                "model_type",
                "random_mean_macro_f1",
                "random_std_macro_f1",
                "pretrained_finetune_mean_macro_f1",
                "pretrained_finetune_std_macro_f1",
                "pretrained_frozen_mean_macro_f1",
                "pretrained_frozen_std_macro_f1",
                "gain_finetune_vs_random",
                "matched_tensors",
                "source_tensors",
            ],
        )


if __name__ == "__main__":
    PaperFigures().generate_all()
