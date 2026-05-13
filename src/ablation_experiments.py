"""
Loss ablation for MAE-pretrained AdapterMetricNet.

Variants:
  - CE: cross entropy only
  - CE + Fixed PM: cross entropy plus fixed prototype margin
  - CEw + Fixed PM: class-weighted cross entropy plus fixed prototype margin
  - CEw + CSA-PM: class-weighted cross entropy plus clipped class-scarcity-aware adaptive prototype margin
"""

import argparse
import copy
import json
import os
import random
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from finetune import AdapterMetricNet, compute_adaptive_margins, load_pretrained_weights_into_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/processed"
OUTPUT_DIR = "results/ablation"

BATCH_SIZE = 64
EPOCHS = 30
VAL_SIZE = 0.2
FIXED_MARGIN = 1.0
CSA_BASE_MARGIN = 1.0
CSA_GAMMA = 0.2
CSA_MARGIN_MIN = 1.0
CSA_MARGIN_MAX = 2.0
PM_WEIGHT = 0.03
LR_ADAPTER = 1e-3
LR_ENCODER = 1e-5
LR_TOP = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ratio_token(ratio):
    return str(int(ratio))


def ratio_dir_name(ratio):
    return "0pct" if ratio < 1 else f"{int(ratio)}pct"


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


def stratified_train_val_split(X, y, val_size=VAL_SIZE, seed=42):
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for label in sorted(np.unique(y)):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.extend(indices.tolist())
            continue

        n_val = max(1, int(round(len(indices) * val_size)))
        n_val = min(n_val, len(indices) - 1)
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = len(y) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def make_loader(X, y, shuffle):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def build_variants():
    return [
        {
            "name": "ce",
            "display_name": "CE",
            "use_class_weights": False,
            "margin_type": "none",
            "description": "cross entropy only",
        },
        {
            "name": "ce_fixed_pm",
            "display_name": "CE + Fixed PM",
            "use_class_weights": False,
            "margin_type": "fixed",
            "description": "cross entropy plus fixed prototype margin",
        },
        {
            "name": "cew_fixed_pm",
            "display_name": "CE^w + Fixed PM",
            "use_class_weights": True,
            "margin_type": "fixed",
            "description": "class-weighted cross entropy plus fixed prototype margin",
        },
        {
            "name": "cew_csa_pm",
            "display_name": "CE^w + CSA-PM",
            "use_class_weights": True,
            "margin_type": "csa",
            "description": "class-weighted cross entropy plus clipped class-scarcity-aware adaptive prototype margin",
        },
    ]


def prototype_margin(features, prototypes, targets, margins):
    distances = torch.cdist(features, prototypes)
    batch_indices = torch.arange(distances.size(0), device=distances.device)
    positive_dist = distances[batch_indices, targets]

    negative_distances = distances.clone()
    negative_distances[batch_indices, targets] = float("inf")
    nearest_negative_dist = negative_distances.min(dim=1).values

    sample_margins = margins.to(DEVICE)[targets]
    loss_values = torch.clamp(positive_dist - nearest_negative_dist + sample_margins, min=0.0)
    return loss_values.mean()


def compute_loss(logits, features, prototypes, targets, variant, class_weights, fixed_margins, adaptive_margins):
    ce_weight = class_weights if variant["use_class_weights"] else None
    ce = nn.CrossEntropyLoss(weight=ce_weight)(logits, targets)

    if variant["margin_type"] == "none":
        return ce, {"ce": float(ce.detach().cpu()), "pm": 0.0}

    if variant["margin_type"] == "fixed":
        pm = prototype_margin(features, prototypes, targets, fixed_margins)
    elif variant["margin_type"] == "csa":
        pm = prototype_margin(features, prototypes, targets, adaptive_margins)
    else:
        raise ValueError(f"unsupported margin type: {variant['margin_type']}")

    loss = ce + PM_WEIGHT * pm
    return loss, {"ce": float(ce.detach().cpu()), "pm": float(pm.detach().cpu())}


def evaluate(model, loader, num_classes):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits, _ = model(batch_x.to(DEVICE))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            labels.extend(batch_y.numpy().tolist())

    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    class_labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=class_labels, zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", labels=class_labels, zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", labels=class_labels, zero_division=0)),
        "report": classification_report(labels, preds, labels=class_labels, output_dict=True, zero_division=0),
    }


def build_model(input_dim, num_classes):
    model = AdapterMetricNet(
        input_dim=input_dim,
        num_classes=num_classes,
        model_type="mae",
        adapter_dim=input_dim,
    ).to(DEVICE)
    loaded = load_pretrained_weights_into_model(model, "mae")
    if loaded <= 0:
        raise RuntimeError("MAE pretrained checkpoint did not match AdapterMetricNet")
    return model, loaded


def train_variant(model, variant, train_loader, val_loader, num_classes, class_weights, fixed_margins, adaptive_margins):
    optimizer = optim.Adam(
        [
            {"params": model.adapter.parameters(), "lr": LR_ADAPTER},
            {"params": model.prototypes, "lr": LR_TOP},
            {"params": [model.temperature], "lr": LR_TOP},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": LR_ENCODER},
        ]
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_pm = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits, features = model(batch_x)
            loss, parts = compute_loss(
                logits,
                features,
                model.prototypes,
                batch_y,
                variant,
                class_weights,
                fixed_margins,
                adaptive_margins,
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_ce += parts["ce"]
            total_pm += parts["pm"]
            batches += 1

        val_metrics = evaluate(model, val_loader, num_classes)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(batches, 1),
                "ce": total_ce / max(batches, 1),
                "prototype_margin": total_pm / max(batches, 1),
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_macro_f1),
        "history": history,
    }


def run_single_variant(
    variant,
    input_dim,
    num_classes,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    fixed_margins,
    adaptive_margins,
):
    model, loaded_tensors = build_model(input_dim, num_classes)
    train_info = train_variant(
        model,
        variant,
        train_loader,
        val_loader,
        num_classes,
        class_weights,
        fixed_margins,
        adaptive_margins,
    )
    test_metrics = evaluate(model, test_loader, num_classes)
    return {
        "model_type": "mae",
        "variant": variant["name"],
        "display_name": variant["display_name"],
        "settings": variant,
        "pretrain": {"path": "checkpoints/mae_pretrain.pth", "loaded_tensors": int(loaded_tensors)},
        "training": train_info,
        "test": test_metrics,
    }


def summarize_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        key = f"mae::{run['variant']}"
        grouped[key].append(run["test"]["macro_f1"])

    summary = {}
    for key, values in grouped.items():
        values = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean_macro_f1": float(values.mean()),
            "std_macro_f1": float(values.std(ddof=0)),
            "runs": int(len(values)),
        }
    return summary


def run_ablation_experiments(ratios=(0.1, 1, 5, 10), seeds=(42, 52, 62)):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_X, test_y, test_path = load_test_set()
    num_classes = int(max(test_y.max(), 0) + 1)
    test_loader = make_loader(test_X, test_y, shuffle=False)

    all_results = {}
    variants = build_variants()

    for ratio in ratios:
        ratio_result = {
            "config": {
                "ratio": ratio,
                "seeds": list(seeds),
                "model_type": "mae",
                "target_test": test_path,
                "epochs": EPOCHS,
                "validation_size": VAL_SIZE,
                "pm_weight": PM_WEIGHT,
                "fixed_margin": FIXED_MARGIN,
                "csa_base_margin": CSA_BASE_MARGIN,
                "csa_gamma": CSA_GAMMA,
                "csa_margin_min": CSA_MARGIN_MIN,
                "csa_margin_max": CSA_MARGIN_MAX,
                "prototype_margin_weighted": False,
                "variants": variants,
            },
            "runs": {},
            "summary": {},
        }
        summary_runs = []

        for seed in seeds:
            set_seed(seed)
            X, y, support_path = load_fewshot_set(ratio, seed)
            train_X, val_X, train_y, val_y = stratified_train_val_split(X, y, VAL_SIZE, seed)

            train_loader = make_loader(train_X, train_y, shuffle=True)
            val_loader = make_loader(val_X, val_y, shuffle=False)
            class_weights = compute_class_weights(train_y, num_classes)
            fixed_margins = torch.full((num_classes,), FIXED_MARGIN, dtype=torch.float32, device=DEVICE)
            adaptive_margins, margin_info = compute_adaptive_margins(
                train_y,
                num_classes,
                base_margin=CSA_BASE_MARGIN,
                gamma=CSA_GAMMA,
                margin_min=CSA_MARGIN_MIN,
                margin_max=CSA_MARGIN_MAX,
            )
            adaptive_margins = adaptive_margins.to(DEVICE)

            seed_runs = []
            for variant in variants:
                run = run_single_variant(
                    variant=variant,
                    input_dim=int(X.shape[1]),
                    num_classes=num_classes,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    class_weights=class_weights,
                    fixed_margins=fixed_margins,
                    adaptive_margins=adaptive_margins,
                )
                seed_runs.append(run)
                summary_runs.append(run)
                print(
                    f"ratio={ratio} seed={seed} {variant['display_name']} "
                    f"macro_f1={run['test']['macro_f1']:.4f} "
                    f"best_epoch={run['training']['best_epoch']}"
                )

            ratio_result["runs"][str(seed)] = {
                "support_source": support_path,
                "train_size": int(len(train_X)),
                "val_size": int(len(val_X)),
                "class_counts_train": np.bincount(train_y, minlength=num_classes).astype(int).tolist(),
                "class_counts_val": np.bincount(val_y, minlength=num_classes).astype(int).tolist(),
                "adaptive_margin_info": margin_info,
                "results": seed_runs,
            }

        ratio_result["summary"] = summarize_runs(summary_runs)
        all_results[str(ratio)] = ratio_result

        output_dir = os.path.join(OUTPUT_DIR, ratio_dir_name(ratio))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ablation_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ratio_result, f, indent=2)
        print(f"ablation results saved to {output_path}")

    combined_path = os.path.join(OUTPUT_DIR, "ablation_results_all.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"combined ablation results saved to {combined_path}")
    return all_results


def main():
    global EPOCHS

    parser = argparse.ArgumentParser(description="Loss ablation for MAE + AdapterMetricNet")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    EPOCHS = args.epochs
    run_ablation_experiments(ratios=tuple(args.ratios), seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
