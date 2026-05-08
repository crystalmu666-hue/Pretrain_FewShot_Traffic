import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from finetune import AdapterMetricNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/processed"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results/k_shot/episodic"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ratio_token(ratio):
    return str(int(ratio))


def load_fewshot_set(ratio, seed):
    path = os.path.join(DATA_DIR, f"cicids_{ratio_token(ratio)}_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"few-shot file not found: {path}")
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.int64), path


def load_test_set():
    path = os.path.join(DATA_DIR, "test_set.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"test set not found: {path}")
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.int64), path


def indices_by_class(y):
    grouped = defaultdict(list)
    for idx, label in enumerate(y):
        grouped[int(label)].append(idx)
    return {label: np.asarray(indices, dtype=np.int64) for label, indices in grouped.items()}


def build_model(input_dim, num_classes, model_type, checkpoint_path=None):
    model = AdapterMetricNet(
        input_dim=input_dim,
        num_classes=num_classes,
        model_type=model_type,
        adapter_dim=input_dim,
    ).to(DEVICE)

    loaded = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        current = model.state_dict()
        matched = {
            key: value
            for key, value in state_dict.items()
            if key in current and value.shape == current[key].shape
        }
        model.load_state_dict(matched, strict=False)
        loaded = len(matched)

    model.eval()
    return model, loaded


class EpisodicKShotEvaluator:
    def __init__(
        self,
        model,
        support_x,
        support_y,
        query_x,
        query_y,
        k,
        q_queries=20,
        n_way=None,
    ):
        self.model = model
        self.support_x = support_x
        self.support_y = support_y
        self.query_x = query_x
        self.query_y = query_y
        self.k = k
        self.q_queries = q_queries
        self.n_way = n_way
        self.support_by_class = indices_by_class(support_y)
        self.query_by_class = indices_by_class(query_y)

    def valid_classes(self):
        classes = []
        for label, support_indices in self.support_by_class.items():
            query_indices = self.query_by_class.get(label, [])
            if len(support_indices) >= self.k and len(query_indices) >= 1:
                classes.append(label)
        return sorted(classes)

    def sample_episode(self):
        valid = self.valid_classes()
        if len(valid) < 2:
            raise ValueError(f"not enough valid classes for {self.k}-shot evaluation")

        n_way = self.n_way or len(valid)
        n_way = min(n_way, len(valid))
        episode_classes = sorted(np.random.choice(valid, size=n_way, replace=False).tolist())

        support_indices = []
        query_indices = []
        query_labels = []

        for label in episode_classes:
            support_pool = self.support_by_class[label]
            query_pool = self.query_by_class[label]
            support_indices.extend(np.random.choice(support_pool, size=self.k, replace=False).tolist())

            q = min(self.q_queries, len(query_pool))
            sampled_query = np.random.choice(query_pool, size=q, replace=False).tolist()
            query_indices.extend(sampled_query)
            query_labels.extend([label] * q)

        return episode_classes, support_indices, query_indices, np.asarray(query_labels, dtype=np.int64)

    def evaluate_once(self):
        episode_classes, support_indices, query_indices, query_labels = self.sample_episode()

        support_x = torch.from_numpy(self.support_x[support_indices]).float().to(DEVICE)
        support_y = torch.from_numpy(self.support_y[support_indices]).long().to(DEVICE)
        query_x = torch.from_numpy(self.query_x[query_indices]).float().to(DEVICE)

        with torch.no_grad():
            _, support_features = self.model(support_x)
            _, query_features = self.model(query_x)

            prototypes = []
            for label in episode_classes:
                label_tensor = torch.tensor(label, device=DEVICE)
                prototypes.append(support_features[support_y == label_tensor].mean(dim=0))
            prototypes = torch.stack(prototypes, dim=0)

            distances = torch.cdist(query_features, prototypes)
            pred_proto_indices = torch.argmin(distances, dim=1).cpu().numpy()

        pred_labels = np.asarray([episode_classes[idx] for idx in pred_proto_indices], dtype=np.int64)
        return {
            "accuracy": float(accuracy_score(query_labels, pred_labels)),
            "balanced_accuracy": float(balanced_accuracy_score(query_labels, pred_labels)),
            "macro_f1": float(f1_score(query_labels, pred_labels, average="macro", zero_division=0)),
            "num_classes": int(len(episode_classes)),
            "num_queries": int(len(query_labels)),
        }

    def evaluate(self, episodes):
        episode_results = [self.evaluate_once() for _ in range(episodes)]
        summary = {}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
            values = np.asarray([result[metric] for result in episode_results], dtype=np.float64)
            summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "ci95": float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
            }
        summary["episodes"] = int(episodes)
        summary["valid_classes"] = [int(label) for label in self.valid_classes()]
        summary["mean_num_queries"] = float(np.mean([r["num_queries"] for r in episode_results]))
        return summary


def run_k_shot_evaluation(
    ratios=(0.1, 1, 5, 10),
    seeds=(42, 52, 62),
    model_types=("none", "mae", "transformer"),
    k_values=(1, 2, 5, 10),
    episodes=100,
    q_queries=20,
    n_way=None,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_x, test_y, test_path = load_test_set()
    num_classes = int(max(test_y.max(), 0) + 1)
    all_results = {
        "config": {
            "ratios": list(ratios),
            "seeds": list(seeds),
            "model_types": list(model_types),
            "k_values": list(k_values),
            "episodes": episodes,
            "q_queries": q_queries,
            "n_way": n_way,
            "query_source": test_path,
        },
        "results": {},
    }

    for ratio in ratios:
        ratio_key = str(ratio)
        all_results["results"][ratio_key] = {}
        for seed in seeds:
            set_seed(seed)
            support_x, support_y, support_path = load_fewshot_set(ratio, seed)
            input_dim = support_x.shape[1]
            seed_key = str(seed)
            all_results["results"][ratio_key][seed_key] = {}

            for model_type in model_types:
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"finetune_{model_type}_ratio{ratio_token(ratio)}_seed{seed}.pth",
                )
                model, loaded_tensors = build_model(input_dim, num_classes, model_type, ckpt_path)
                model_result = {
                    "checkpoint": ckpt_path,
                    "loaded_tensors": int(loaded_tensors),
                    "support_source": support_path,
                    "k": {},
                }

                for k in k_values:
                    evaluator = EpisodicKShotEvaluator(
                        model=model,
                        support_x=support_x,
                        support_y=support_y,
                        query_x=test_x,
                        query_y=test_y,
                        k=k,
                        q_queries=q_queries,
                        n_way=n_way,
                    )
                    model_result["k"][str(k)] = evaluator.evaluate(episodes)

                all_results["results"][ratio_key][seed_key][model_type] = model_result

    output_path = os.path.join(OUTPUT_DIR, "k_shot_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"k-shot episodic results saved to {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Episodic k-shot evaluation on an independent test set")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--models", nargs="+", default=["none", "mae", "transformer"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--q_queries", type=int, default=20)
    parser.add_argument("--n_way", type=int, default=None)
    args = parser.parse_args()

    run_k_shot_evaluation(
        ratios=tuple(args.ratios),
        seeds=tuple(args.seeds),
        model_types=tuple(args.models),
        k_values=tuple(args.k_values),
        episodes=args.episodes,
        q_queries=args.q_queries,
        n_way=args.n_way,
    )


if __name__ == "__main__":
    main()
