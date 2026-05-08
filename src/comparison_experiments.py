import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEEDS = [42, 52, 62]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x), None


class ComparisonExperiments:
    def __init__(self, data_path, sample_ratio, output_dir="results/comparison"):
        self.data_path = data_path
        self.sample_ratio = sample_ratio
        self.output_dir = output_dir
        self.seed = self._parse_seed_from_path(data_path)
        os.makedirs(output_dir, exist_ok=True)
        self._set_training_epochs()
        self._load_data()

    def _set_training_epochs(self):
        self.epochs = 100 if self.sample_ratio <= 1 else 50
        print(f"sample ratio: {self.sample_ratio}%, training epochs: {self.epochs}")

    def _parse_seed_from_path(self, data_path):
        stem = os.path.splitext(os.path.basename(data_path))[0]
        for part in stem.split("_"):
            if part.startswith("seed"):
                return int(part.replace("seed", ""))
        return None

    def _load_data(self):
        data = np.load(self.data_path)
        self.X_train_full, self.y_train_full = data["x"], data["y"]

        test_data = np.load("data/processed/test_set.npz")
        self.X_test, self.y_test = test_data["x"], test_data["y"]
        self.num_classes = len(np.unique(self.y_train_full))

    def _analyze_rare_classes(self, report_dict):
        vals = []
        for k, v in report_dict.items():
            if k.isdigit() and v["support"] < 10:
                vals.append(v["recall"])
        return np.mean(vals) if vals else 0.0

    def _build_placeholder_history(self, macro_f1):
        return [{"epoch": 1, "loss": None, "macro_f1": float(macro_f1)}]

    def _load_model_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], checkpoint.get("history") or []
        return checkpoint, []

    def _run_simple_dl_methods(self):
        from finetune import AdapterMetricNet

        methods = {
            "MetricNet": "none",
            "MetricNet_MAE": "mae",
            "MetricNet_Transformer": "transformer",
        }
        results = {}

        for name, model_type in methods.items():
            if name not in results:
                results[name] = {
                    "macro_f1_list": [],
                    "accuracy_list": [],
                    "rare_recall_list": [],
                    "history_per_seed": [],
                }

            ratio_tag = "0" if self.sample_ratio < 1 else str(int(self.sample_ratio))
            ckpt_path = f"checkpoints/finetune_{model_type}_ratio{ratio_tag}_seed{self.seed}.pth"

            if not os.path.exists(ckpt_path):
                print(f"checkpoint not found: {ckpt_path}")
                continue

            model = AdapterMetricNet(
                self.X_train_full.shape[1],
                self.num_classes,
                model_type=model_type,
                adapter_dim=self.X_train_full.shape[1],
            ).to(DEVICE)

            state_dict, history = self._load_model_checkpoint(ckpt_path)
            model.load_state_dict(state_dict)
            model.eval()

            with torch.no_grad():
                logits, _ = model(torch.FloatTensor(self.X_test).to(DEVICE))
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()

            rep = classification_report(
                self.y_test,
                y_pred,
                output_dict=True,
                digits=4,
                zero_division=0,
            )
            macro_f1 = rep["macro avg"]["f1-score"]

            results[name]["macro_f1_list"].append(macro_f1)
            results[name]["accuracy_list"].append(accuracy_score(self.y_test, y_pred))
            results[name]["rare_recall_list"].append(self._analyze_rare_classes(rep))

            if history:
                results[name]["history_per_seed"].append(
                    [
                        {
                            "epoch": int(item["epoch"]),
                            "loss": None if item.get("loss") is None else float(item["loss"]),
                            "macro_f1": float(item["macro_f1"]),
                        }
                        for item in history
                    ]
                )
            else:
                results[name]["history_per_seed"].append(self._build_placeholder_history(macro_f1))

        for name in results:
            results[name]["macro_f1"] = float(np.mean(results[name]["macro_f1_list"]))
            results[name]["accuracy"] = float(np.mean(results[name]["accuracy_list"]))
            results[name]["rare_recall"] = float(np.mean(results[name]["rare_recall_list"]))

        return results

    def _run_traditional_methods(self):
        methods = {
            "SVM": SVC(kernel="rbf", C=1.0),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }

        results = {}
        for name, model in methods.items():
            model.fit(self.X_train_full, self.y_train_full)
            y_pred = model.predict(self.X_test)
            rep = classification_report(self.y_test, y_pred, output_dict=True, digits=4, zero_division=0)
            macro_f1 = rep["macro avg"]["f1-score"]

            results[name] = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "macro_f1": macro_f1,
                "rare_recall": self._analyze_rare_classes(rep),
                "history_per_seed": [self._build_placeholder_history(macro_f1)],
            }

        return results

    def run_all_experiments(self):
        results = {
            "traditional": self._run_traditional_methods(),
            "simple_dl": self._run_simple_dl_methods(),
        }

        mae = results["simple_dl"].get("MetricNet_MAE", {}).get("macro_f1", 0)
        base = results["simple_dl"].get("MetricNet", {}).get("macro_f1", 0)
        results["pretrain_gain"] = float(mae - base)
        return results


def run_comparison_experiments():
    print("=== start few-shot comparison experiments ===")
    all_ratios_results = {}

    for ratio in [0.1, 1, 5, 10]:
        folder = "0" if ratio < 1 else str(int(ratio))
        print(f"\n--- sample ratio {ratio}% ---")

        ratio_results = []
        for seed in RANDOM_SEEDS:
            data_path = f"data/processed/cicids_{folder}_seed{seed}.npz"
            if not os.path.exists(data_path):
                print(f"data file not found: {data_path}")
                continue

            exp = ComparisonExperiments(data_path, ratio)
            ratio_results.append(exp.run_all_experiments())

        if not ratio_results:
            continue

        combined_result = {"traditional": {}, "simple_dl": {}, "pretrain_gain_list": []}

        traditional_methods = ratio_results[0]["traditional"].keys()
        for name in traditional_methods:
            macro_f1_list = [r["traditional"][name]["macro_f1"] for r in ratio_results]
            acc_list = [r["traditional"][name]["accuracy"] for r in ratio_results]
            history_per_seed = [r["traditional"][name]["history_per_seed"][0] for r in ratio_results]

            combined_result["traditional"][name] = {
                "macro_f1_list": macro_f1_list,
                "macro_f1": float(np.mean(macro_f1_list)),
                "accuracy_list": acc_list,
                "accuracy": float(np.mean(acc_list)),
                "history_per_seed": history_per_seed,
            }

        dl_methods = ratio_results[0]["simple_dl"].keys()
        for name in dl_methods:
            macro_f1_list = []
            acc_list = []
            history_per_seed = []

            for r in ratio_results:
                data = r["simple_dl"].get(name, {})
                macro_f1_list.extend(data.get("macro_f1_list", []))
                acc_list.extend(data.get("accuracy_list", []))
                history_per_seed.extend(data.get("history_per_seed", []))

            combined_result["simple_dl"][name] = {
                "macro_f1_list": macro_f1_list,
                "macro_f1": float(np.mean(macro_f1_list)),
                "accuracy_list": acc_list,
                "accuracy": float(np.mean(acc_list)),
                "history_per_seed": history_per_seed,
            }

        combined_result["pretrain_gain_list"] = [r.get("pretrain_gain", 0) for r in ratio_results]
        combined_result["pretrain_gain"] = float(np.mean(combined_result["pretrain_gain_list"]))
        all_ratios_results[str(ratio)] = combined_result

    os.makedirs("results/comparison", exist_ok=True)
    json_path = "results/comparison/comparison_results.json"

    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_ratios_results, f, default=convert, indent=2)

    print(f"saved comparison results: {json_path}")
    return all_ratios_results


if __name__ == "__main__":
    run_comparison_experiments()
