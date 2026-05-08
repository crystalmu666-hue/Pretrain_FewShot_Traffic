"""
Ablation experiments for CICIDS few-shot classification.

Training/validation data come from each CICIDS few-shot seed file.
The final test metrics are always computed on data/processed/test_set.npz.
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

from models import MaskedTrafficAutoencoder, TrafficTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/processed"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results/ablation"

BATCH_SIZE = 64
EPOCHS = 30
VAL_SIZE = 0.2
MARGIN_WEIGHT = 0.1
MARGIN = 1.0
LR_ADAPTER = 1e-3
LR_ENCODER = 1e-4
LR_HEAD = 1e-3

PRETRAIN_MODELS = {
    "mae": os.path.join(CHECKPOINT_DIR, "mae_pretrain.pth"),
    "transformer": os.path.join(CHECKPOINT_DIR, "transformer_pretrain.pth"),
}


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


def load_source_dim():
    path = os.path.join(DATA_DIR, "unsw_X.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"source feature file not found: {path}")
    return int(np.load(path, mmap_mode="r").shape[1])


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


class AblationClassifier(nn.Module):
    def __init__(self, target_dim, encoder_dim, num_classes, model_type, use_adapter=True):
        super().__init__()
        self.model_type = model_type
        self.use_adapter = use_adapter

        if use_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(target_dim, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.ReLU(),
            )
        else:
            self.adapter = nn.Identity()
            encoder_dim = target_dim

        if model_type == "mae":
            self.encoder = MaskedTrafficAutoencoder(
                input_dim=encoder_dim,
                mask_ratio=0.4,
                hidden_dim=128,
                latent_dim=32,
            ).encoder
            latent_dim = 32
        elif model_type == "transformer":
            self.encoder = TrafficTransformer(
                input_dim=encoder_dim,
                hidden_dim=64,
                num_layers=1,
                projection_dim=32,
            )
            latent_dim = 32
        else:
            raise ValueError(f"unsupported model_type: {model_type}")

        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        x = self.adapter(x)
        if self.model_type == "transformer":
            _, features = self.encoder(x)
        else:
            features = self.encoder(x)
        return self.classifier(features), features


def load_pretrained_encoder(model, model_type):
    path = PRETRAIN_MODELS[model_type]
    if not os.path.exists(path):
        return {"path": path, "loaded": False, "matched_tensors": 0, "source_tensors": 0}

    checkpoint = torch.load(path, map_location=DEVICE)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = model.encoder.state_dict()
    matched = {}

    for source_key, value in source_state.items():
        candidates = [
            source_key,
            source_key.removeprefix("encoder."),
            f"encoder.{source_key}",
        ]
        for candidate in candidates:
            if candidate in encoder_state and value.shape == encoder_state[candidate].shape:
                matched[candidate] = value
                break

    if matched:
        model.encoder.load_state_dict(matched, strict=False)

    return {
        "path": path,
        "loaded": bool(matched),
        "matched_tensors": int(len(matched)),
        "source_tensors": int(len(source_state)),
    }


def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False


def margin_loss(logits, targets, margin=MARGIN):
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    positive = logits[batch_indices, targets]
    negative_mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0.0)
    hardest_negative, _ = torch.max(logits * negative_mask - (1.0 - negative_mask) * 1e9, dim=1)
    return torch.clamp(hardest_negative - positive + margin, min=0.0).mean()


def compute_loss(logits, targets, class_weights, use_margin, use_class_weights):
    weight = class_weights if use_class_weights else None
    ce = nn.CrossEntropyLoss(weight=weight)(logits, targets)
    if not use_margin:
        return ce
    return ce + MARGIN_WEIGHT * margin_loss(logits, targets)


def make_loader(X, y, shuffle):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


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


def train_model(model, train_loader, val_loader, num_classes, class_weights, use_margin, use_class_weights):
    param_groups = [
        {"params": list(model.adapter.parameters()) if not isinstance(model.adapter, nn.Identity) else [], "lr": LR_ADAPTER},
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": LR_ENCODER},
        {"params": list(model.classifier.parameters()), "lr": LR_HEAD},
    ]
    optimizer = optim.Adam([group for group in param_groups if group["params"]])

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = compute_loss(logits, batch_y, class_weights, use_margin, use_class_weights)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        val_metrics = evaluate(model, val_loader, num_classes)
        avg_loss = total_loss / max(batches, 1)
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
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


def build_variants():
    return [
        {
            "name": "random_ce",
            "use_pretrain": False,
            "use_adapter": True,
            "use_margin": False,
            "use_class_weights": True,
            "freeze": False,
        },
        {
            "name": "random_margin",
            "use_pretrain": False,
            "use_adapter": True,
            "use_margin": True,
            "use_class_weights": True,
            "freeze": False,
        },
        {
            "name": "pretrained_ce",
            "use_pretrain": True,
            "use_adapter": True,
            "use_margin": False,
            "use_class_weights": True,
            "freeze": False,
        },
        {
            "name": "pretrained_full",
            "use_pretrain": True,
            "use_adapter": True,
            "use_margin": True,
            "use_class_weights": True,
            "freeze": False,
        },
        {
            "name": "pretrained_no_class_weights",
            "use_pretrain": True,
            "use_adapter": True,
            "use_margin": True,
            "use_class_weights": False,
            "freeze": False,
        },
        {
            "name": "pretrained_frozen",
            "use_pretrain": True,
            "use_adapter": True,
            "use_margin": True,
            "use_class_weights": True,
            "freeze": True,
        },
        {
            "name": "target_native_no_adapter",
            "use_pretrain": False,
            "use_adapter": False,
            "use_margin": True,
            "use_class_weights": True,
            "freeze": False,
        },
    ]


def run_single_variant(
    variant,
    model_type,
    target_dim,
    source_dim,
    num_classes,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
):
    encoder_dim = source_dim if variant["use_adapter"] else target_dim
    model = AblationClassifier(
        target_dim=target_dim,
        encoder_dim=encoder_dim,
        num_classes=num_classes,
        model_type=model_type,
        use_adapter=variant["use_adapter"],
    ).to(DEVICE)

    pretrain_info = {"path": None, "loaded": False, "matched_tensors": 0, "source_tensors": 0}
    if variant["use_pretrain"]:
        pretrain_info = load_pretrained_encoder(model, model_type)
        if not pretrain_info["loaded"]:
            raise RuntimeError(
                f"{model_type}/{variant['name']} expected pretrained weights, "
                f"but no tensors matched {pretrain_info['path']}"
            )

    if variant["freeze"]:
        freeze_encoder(model)

    train_info = train_model(
        model,
        train_loader,
        val_loader,
        num_classes,
        class_weights,
        use_margin=variant["use_margin"],
        use_class_weights=variant["use_class_weights"],
    )
    test_metrics = evaluate(model, test_loader, num_classes)
    return {
        "model_type": model_type,
        "variant": variant["name"],
        "settings": variant,
        "pretrain": pretrain_info,
        "training": train_info,
        "test": test_metrics,
    }


def summarize_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        key = f"{run['model_type']}::{run['variant']}"
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


def run_ablation_experiments(ratios=(0.1, 1, 5, 10), seeds=(42, 52, 62), model_types=("mae", "transformer")):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    source_dim = load_source_dim()
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
                "model_types": list(model_types),
                "source_dim": source_dim,
                "target_test": test_path,
                "epochs": EPOCHS,
                "validation_size": VAL_SIZE,
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

            seed_runs = []
            for model_type in model_types:
                for variant in variants:
                    run = run_single_variant(
                        variant=variant,
                        model_type=model_type,
                        target_dim=int(X.shape[1]),
                        source_dim=source_dim,
                        num_classes=num_classes,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        class_weights=class_weights,
                    )
                    seed_runs.append(run)
                    summary_runs.append(run)
                    print(
                        f"ratio={ratio} seed={seed} {model_type}/{variant['name']} "
                        f"macro_f1={run['test']['macro_f1']:.4f} "
                        f"best_epoch={run['training']['best_epoch']}"
                    )

            ratio_result["runs"][str(seed)] = {
                "support_source": support_path,
                "train_size": int(len(train_X)),
                "val_size": int(len(val_X)),
                "class_counts_train": np.bincount(train_y, minlength=num_classes).astype(int).tolist(),
                "class_counts_val": np.bincount(val_y, minlength=num_classes).astype(int).tolist(),
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

    parser = argparse.ArgumentParser(description="Component ablations for few-shot traffic classification")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--models", nargs="+", default=["mae", "transformer"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    EPOCHS = args.epochs
    run_ablation_experiments(
        ratios=tuple(args.ratios),
        seeds=tuple(args.seeds),
        model_types=tuple(args.models),
    )


if __name__ == "__main__":
    main()
