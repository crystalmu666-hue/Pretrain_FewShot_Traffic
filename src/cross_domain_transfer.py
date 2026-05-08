"""
Cross-domain transfer experiment.

Source domain: UNSW-NB15 self-supervised pretraining.
Target domain: CICIDS-2017 few-shot supervised fine-tuning.

The experiment uses the same target architecture for random-initialized and
pretrained variants, selects the best epoch on a validation split, and evaluates
the selected model once on the fixed CICIDS test set.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models import MaskedTrafficAutoencoder, TrafficTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/processed"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results/cross_domain"

BATCH_SIZE = 64
EPOCHS = 50
LR_ADAPTER = 1e-3
LR_ENCODER = 1e-4
LR_HEAD = 1e-3
VAL_SIZE = 0.2

SOURCE_DATA_PATH = os.path.join(DATA_DIR, "unsw_X.npy")
TEST_SET_PATH = os.path.join(DATA_DIR, "test_set.npz")
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


def load_fewshot_set(ratio, seed):
    path = os.path.join(DATA_DIR, f"cicids_{ratio_token(ratio)}_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"target few-shot file not found: {path}")
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.int64), path


def load_test_set():
    if not os.path.exists(TEST_SET_PATH):
        raise FileNotFoundError(f"target test set not found: {TEST_SET_PATH}")
    data = np.load(TEST_SET_PATH)
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def stratified_train_val_split(X, y, val_size=0.2, seed=42):
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


class DomainTransferClassifier(nn.Module):
    def __init__(self, target_dim, source_dim, num_classes, model_type):
        super().__init__()
        self.model_type = model_type
        self.adapter = nn.Sequential(
            nn.Linear(target_dim, source_dim),
            nn.LayerNorm(source_dim),
            nn.ReLU(),
        )

        if model_type == "mae":
            self.encoder = MaskedTrafficAutoencoder(
                input_dim=source_dim,
                mask_ratio=0.4,
                hidden_dim=128,
                latent_dim=32,
            ).encoder
            latent_dim = 32
        elif model_type == "transformer":
            self.encoder = TrafficTransformer(
                input_dim=source_dim,
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
        logits = self.classifier(features)
        return logits, features


def load_pretrained_encoder(model, model_type):
    path = PRETRAIN_MODELS[model_type]
    if not os.path.exists(path):
        return {"path": path, "matched_tensors": 0, "source_tensors": 0, "loaded": False}

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
        "matched_tensors": len(matched),
        "source_tensors": len(source_state),
        "loaded": bool(matched),
    }


def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=list(range(num_classes)), zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", labels=list(range(num_classes)), zero_division=0)),
        "report": classification_report(
            labels,
            preds,
            labels=list(range(num_classes)),
            output_dict=True,
            zero_division=0,
        ),
    }


def train_one_model(model, train_loader, val_loader, num_classes, class_weights):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    param_groups = [
        {"params": list(model.adapter.parameters()), "lr": LR_ADAPTER},
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": LR_ENCODER},
        {"params": list(model.classifier.parameters()), "lr": LR_HEAD},
    ]
    optimizer = optim.Adam([group for group in param_groups if group["params"]])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
            loss = criterion(logits, batch_y)
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
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_macro_f1),
        "history": history,
        "trainable_parameters": int(sum(p.numel() for p in trainable_params)),
    }


def run_variant(
    model_type,
    variant,
    target_dim,
    source_dim,
    num_classes,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
):
    model = DomainTransferClassifier(target_dim, source_dim, num_classes, model_type).to(DEVICE)
    load_info = {"path": None, "matched_tensors": 0, "source_tensors": 0, "loaded": False}

    if variant in {"pretrained_finetune", "pretrained_frozen"}:
        load_info = load_pretrained_encoder(model, model_type)
        if not load_info["loaded"]:
            raise RuntimeError(
                f"{model_type} pretrained encoder did not match target architecture; "
                f"checkpoint={load_info['path']}"
            )

    if variant == "pretrained_frozen":
        freeze_encoder(model)

    train_info = train_one_model(model, train_loader, val_loader, num_classes, class_weights)
    test_metrics = evaluate(model, test_loader, num_classes)
    return {
        "model_type": model_type,
        "variant": variant,
        "pretrain": load_info,
        "training": train_info,
        "test": test_metrics,
    }


def summarize_seed_results(seed_results):
    grouped = defaultdict(list)
    for item in seed_results:
        key = f"{item['model_type']}::{item['variant']}"
        grouped[key].append(item["test"]["macro_f1"])

    summary = {}
    for key, values in grouped.items():
        values = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean_macro_f1": float(values.mean()),
            "std_macro_f1": float(values.std(ddof=0)),
            "runs": int(len(values)),
        }
    return summary


def run_cross_domain_experiment(ratios=(1, 5, 10), seeds=(42, 52, 62)):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(f"source data not found: {SOURCE_DATA_PATH}")

    source_X = np.load(SOURCE_DATA_PATH)
    source_dim = int(source_X.shape[1])
    test_X, test_y = load_test_set()
    num_classes = int(max(test_y.max(), 0) + 1)
    test_loader = make_loader(test_X, test_y, shuffle=False)

    all_results = {
        "config": {
            "source_data": SOURCE_DATA_PATH,
            "target_test": TEST_SET_PATH,
            "source_dim": source_dim,
            "target_test_dim": int(test_X.shape[1]),
            "num_classes": num_classes,
            "ratios": list(ratios),
            "seeds": list(seeds),
            "epochs": EPOCHS,
            "validation_size": VAL_SIZE,
        },
        "results": {},
        "summary": {},
    }

    for ratio in ratios:
        ratio_key = str(ratio)
        all_results["results"][ratio_key] = {}
        seed_level = []

        for seed in seeds:
            set_seed(seed)
            target_X, target_y, target_path = load_fewshot_set(ratio, seed)
            train_X, val_X, train_y, val_y = stratified_train_val_split(target_X, target_y, VAL_SIZE, seed)

            train_loader = make_loader(train_X, train_y, shuffle=True)
            val_loader = make_loader(val_X, val_y, shuffle=False)
            class_weights = compute_class_weights(train_y, num_classes)
            target_dim = int(target_X.shape[1])

            variants = []
            for model_type in ["mae", "transformer"]:
                for variant in ["random", "pretrained_finetune", "pretrained_frozen"]:
                    variants.append((model_type, variant))

            seed_results = []
            for model_type, variant in variants:
                result = run_variant(
                    model_type=model_type,
                    variant=variant,
                    target_dim=target_dim,
                    source_dim=source_dim,
                    num_classes=num_classes,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    class_weights=class_weights,
                )
                seed_results.append(result)
                seed_level.append(result)
                print(
                    f"ratio={ratio} seed={seed} {model_type}/{variant} "
                    f"test_macro_f1={result['test']['macro_f1']:.4f} "
                    f"best_epoch={result['training']['best_epoch']}"
                )

            all_results["results"][ratio_key][str(seed)] = {
                "target_support": target_path,
                "train_size": int(len(train_X)),
                "val_size": int(len(val_X)),
                "class_counts_train": np.bincount(train_y, minlength=num_classes).astype(int).tolist(),
                "class_counts_val": np.bincount(val_y, minlength=num_classes).astype(int).tolist(),
                "runs": seed_results,
            }

        all_results["summary"][ratio_key] = summarize_seed_results(seed_level)

    output_path = os.path.join(OUTPUT_DIR, "cross_domain_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"cross-domain results saved to {output_path}")
    return all_results


def main():
    global EPOCHS

    parser = argparse.ArgumentParser(description="Cross-domain transfer from UNSW-NB15 to CICIDS-2017")
    parser.add_argument("--ratios", nargs="+", type=float, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    EPOCHS = args.epochs

    run_cross_domain_experiment(ratios=tuple(args.ratios), seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
