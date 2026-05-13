"""
MAE-only cross-domain transfer experiment with optional domain alignment.

Source domain: UNSW-NB15 MAE self-supervised pretraining.
Target domain: CICIDS-2017 few-shot supervised fine-tuning.
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

from models import MaskedTrafficAutoencoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/processed"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results/cross_domain"

BATCH_SIZE = 64
SOURCE_BATCH_SIZE = 256
EPOCHS = 50
LR_ADAPTER = 1e-3
LR_ENCODER = 1e-4
LR_HEAD = 1e-3
VAL_SIZE = 0.2
ALIGNMENT_WEIGHT = 0.05
ALIGNMENT_REFERENCE_RATIO = 5.0
ALIGNMENT_MIN_SCALE = 0.5
ALIGNMENT_MAX_SCALE = 2.0
MMD_KERNEL_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)

BASE_MARGIN = 1.0
FIXED_MARGIN = 1.0
SCARCITY_GAMMA = 0.2
ADAPTIVE_MARGIN_MIN = 1.0
ADAPTIVE_MARGIN_MAX = 2.0
PROTOTYPE_MARGIN_WEIGHT = 0.03
CSA_PM_MAX_RATIO = 0.1

SOURCE_DATA_PATH = os.path.join(DATA_DIR, "unsw_X.npy")
TEST_SET_PATH = os.path.join(DATA_DIR, "test_set.npz")
MAE_PRETRAIN_PATH = os.path.join(CHECKPOINT_DIR, "mae_pretrain.pth")


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


def compute_adaptive_margins(
    y,
    num_classes,
    base_margin=BASE_MARGIN,
    gamma=SCARCITY_GAMMA,
    margin_min=ADAPTIVE_MARGIN_MIN,
    margin_max=ADAPTIVE_MARGIN_MAX,
):
    class_counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total_samples = float(len(y))
    safe_counts = np.maximum(class_counts, 1.0)

    scarcity = total_samples / (num_classes * safe_counts)
    normalized_scarcity = scarcity / np.mean(scarcity)
    raw_margins = base_margin * (1.0 + gamma * normalized_scarcity)
    margins = np.clip(raw_margins, margin_min, margin_max)

    return (
        torch.tensor(margins, dtype=torch.float32, device=DEVICE),
        {
            "class_counts": class_counts.astype(int).tolist(),
            "scarcity": scarcity.tolist(),
            "normalized_scarcity": normalized_scarcity.tolist(),
            "raw_adaptive_margins": raw_margins.tolist(),
            "adaptive_margins": margins.tolist(),
            "base_margin": float(base_margin),
            "gamma": float(gamma),
            "margin_min": float(margin_min),
            "margin_max": float(margin_max),
            "prototype_margin_weighted": False,
        },
    )


def prototype_margin_loss(features, prototypes, targets, margins):
    distances = torch.cdist(features, prototypes)
    batch_range = torch.arange(distances.size(0), device=distances.device)
    positive_dist = distances[batch_range, targets]

    negative_distances = distances.clone()
    negative_distances[batch_range, targets] = float("inf")
    nearest_negative_dist = negative_distances.min(dim=1).values

    sample_margins = margins.to(features.device)[targets]
    return torch.clamp(positive_dist - nearest_negative_dist + sample_margins, min=0.0).mean()


def select_loss_policy(ratio):
    if ratio <= CSA_PM_MAX_RATIO:
        return {
            "name": "cew_csa_pm",
            "display_name": "CE^w + CSA-PM",
            "use_class_weights": True,
            "margin_type": "csa",
        }
    return {
        "name": "ce_fixed_pm",
        "display_name": "CE + Fixed PM",
        "use_class_weights": False,
        "margin_type": "fixed",
    }


def compute_policy_loss(
    logits,
    features,
    prototypes,
    targets,
    policy,
    class_weights,
    fixed_margins,
    adaptive_margins,
):
    ce_weight = class_weights.to(features.device) if policy["use_class_weights"] else None
    ce_loss = nn.CrossEntropyLoss(weight=ce_weight)(logits, targets)

    if policy["margin_type"] == "csa":
        pm_loss = prototype_margin_loss(features, prototypes, targets, adaptive_margins)
    elif policy["margin_type"] == "fixed":
        pm_loss = prototype_margin_loss(features, prototypes, targets, fixed_margins)
    else:
        raise ValueError(f"unsupported margin type: {policy['margin_type']}")

    return ce_loss + PROTOTYPE_MARGIN_WEIGHT * pm_loss, ce_loss, pm_loss


def compute_adaptive_alignment_weight(ratio, base_weight=ALIGNMENT_WEIGHT):
    safe_ratio = max(float(ratio), 1e-6)
    raw_scale = np.sqrt(ALIGNMENT_REFERENCE_RATIO / safe_ratio)
    scale = float(np.clip(raw_scale, ALIGNMENT_MIN_SCALE, ALIGNMENT_MAX_SCALE))
    effective_weight = float(base_weight * scale)
    return effective_weight, {
        "base_weight": float(base_weight),
        "effective_weight": effective_weight,
        "target_ratio": float(ratio),
        "reference_ratio": float(ALIGNMENT_REFERENCE_RATIO),
        "raw_scale": float(raw_scale),
        "scale": scale,
        "min_scale": float(ALIGNMENT_MIN_SCALE),
        "max_scale": float(ALIGNMENT_MAX_SCALE),
    }


class MAEDomainTransferClassifier(nn.Module):
    def __init__(self, target_dim, source_dim, num_classes):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(target_dim, source_dim),
            nn.LayerNorm(source_dim),
            nn.ReLU(),
        )
        self.encoder = MaskedTrafficAutoencoder(
            input_dim=source_dim,
            mask_ratio=0.4,
            hidden_dim=128,
            latent_dim=32,
        ).encoder
        self.prototypes = nn.Parameter(torch.randn(num_classes, 32))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def encode_target(self, x):
        return self.encoder(self.adapter(x))

    def encode_source(self, x):
        return self.encoder(x)

    def forward(self, x):
        features = self.encode_target(x)
        distances = torch.cdist(features, self.prototypes)
        logits = -(distances / self.temperature)
        return logits, features


def load_mae_encoder(model):
    if not os.path.exists(MAE_PRETRAIN_PATH):
        return {"path": MAE_PRETRAIN_PATH, "matched_tensors": 0, "source_tensors": 0, "loaded": False}

    checkpoint = torch.load(MAE_PRETRAIN_PATH, map_location=DEVICE)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    encoder_state = model.encoder.state_dict()
    matched = {}

    for source_key, value in source_state.items():
        candidates = [source_key, source_key.removeprefix("encoder."), f"encoder.{source_key}"]
        for candidate in candidates:
            if candidate in encoder_state and value.shape == encoder_state[candidate].shape:
                matched[candidate] = value
                break

    if matched:
        model.encoder.load_state_dict(matched, strict=False)

    return {
        "path": MAE_PRETRAIN_PATH,
        "matched_tensors": int(len(matched)),
        "source_tensors": int(len(source_state)),
        "loaded": bool(matched),
    }


def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_source_loader(source_X):
    dataset = TensorDataset(torch.from_numpy(source_X.astype(np.float32)).float())
    return DataLoader(dataset, batch_size=SOURCE_BATCH_SIZE, shuffle=True, drop_last=True)


def covariance(features):
    centered = features - features.mean(dim=0, keepdim=True)
    denom = max(features.size(0) - 1, 1)
    return centered.T.matmul(centered) / denom


def coral_loss(source_features, target_features):
    source_cov = covariance(source_features)
    target_cov = covariance(target_features)
    dim = source_features.size(1)
    return (source_cov - target_cov).pow(2).sum() / (4.0 * dim * dim)


def mmd_loss(source_features, target_features, kernel_multipliers=MMD_KERNEL_MULTIPLIERS):
    combined = torch.cat([source_features, target_features], dim=0)
    pairwise_sq_dist = torch.cdist(combined, combined).pow(2)

    with torch.no_grad():
        positive_dist = pairwise_sq_dist[pairwise_sq_dist > 0]
        base_bandwidth = positive_dist.median() if positive_dist.numel() else torch.tensor(1.0, device=combined.device)
        base_bandwidth = torch.clamp(base_bandwidth, min=1e-6)

    kernels = 0.0
    for multiplier in kernel_multipliers:
        bandwidth = base_bandwidth * multiplier
        kernels = kernels + torch.exp(-pairwise_sq_dist / bandwidth)

    n_source = source_features.size(0)
    source_kernel = kernels[:n_source, :n_source]
    target_kernel = kernels[n_source:, n_source:]
    cross_kernel = kernels[:n_source, n_source:]
    return source_kernel.mean() + target_kernel.mean() - 2.0 * cross_kernel.mean()


def compute_alignment_loss(source_features, target_features, alignment):
    if alignment == "none":
        return source_features.new_tensor(0.0)
    if alignment == "coral":
        return coral_loss(source_features, target_features)
    if alignment == "mmd":
        return mmd_loss(source_features, target_features)
    if alignment == "coral_mmd":
        return coral_loss(source_features, target_features) + mmd_loss(source_features, target_features)
    raise ValueError(f"unsupported alignment: {alignment}")


def next_source_batch(source_iter, source_loader):
    try:
        (source_x,) = next(source_iter)
    except StopIteration:
        source_iter = iter(source_loader)
        (source_x,) = next(source_iter)
    return source_x, source_iter


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
        "report": classification_report(labels, preds, labels=class_labels, output_dict=True, zero_division=0),
    }


def train_one_model(
    model,
    train_loader,
    val_loader,
    num_classes,
    class_weights,
    loss_policy,
    fixed_margins,
    adaptive_margins,
    source_loader=None,
    alignment="none",
    lambda_align=ALIGNMENT_WEIGHT,
):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    param_groups = [
        {"params": list(model.adapter.parameters()), "lr": LR_ADAPTER},
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": LR_ENCODER},
        {"params": [model.prototypes, model.temperature], "lr": LR_HEAD},
    ]
    optimizer = optim.Adam([group for group in param_groups if group["params"]])

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_macro_f1 = -1.0
    history = []
    use_alignment = alignment != "none" and source_loader is not None and lambda_align > 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_pm_loss = 0.0
        total_align_loss = 0.0
        batches = 0
        source_iter = iter(source_loader) if use_alignment else None

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits, target_features = model(batch_x)
            task_loss, ce_loss, pm_loss = compute_policy_loss(
                logits,
                target_features,
                model.prototypes,
                batch_y,
                loss_policy,
                class_weights,
                fixed_margins,
                adaptive_margins,
            )

            align_loss = target_features.new_tensor(0.0)
            if use_alignment:
                source_x, source_iter = next_source_batch(source_iter, source_loader)
                source_x = source_x.to(DEVICE)
                source_features = model.encode_source(source_x)
                align_loss = compute_alignment_loss(source_features, target_features, alignment)

            loss = task_loss + lambda_align * align_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_ce_loss += float(ce_loss.item())
            total_pm_loss += float(pm_loss.item())
            total_align_loss += float(align_loss.item())
            batches += 1

        val_metrics = evaluate(model, val_loader, num_classes)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(batches, 1),
                "ce_loss": total_ce_loss / max(batches, 1),
                "prototype_margin_loss": total_pm_loss / max(batches, 1),
                "alignment_loss": total_align_loss / max(batches, 1),
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
        "trainable_parameters": int(sum(p.numel() for p in trainable_params)),
        "loss_policy": loss_policy,
        "prototype_margin_weight": float(PROTOTYPE_MARGIN_WEIGHT),
        "alignment": alignment,
        "lambda_align": float(lambda_align),
    }


def variant_settings(variant):
    settings = {
        "random": {"load_pretrain": False, "freeze_encoder": False, "alignment": "none"},
        "mae_pretrained_finetune": {"load_pretrain": True, "freeze_encoder": False, "alignment": "none"},
        "mae_pretrained_frozen": {"load_pretrain": True, "freeze_encoder": True, "alignment": "none"},
        "mae_pretrained_coral": {"load_pretrain": True, "freeze_encoder": False, "alignment": "coral"},
        "mae_pretrained_mmd": {"load_pretrain": True, "freeze_encoder": False, "alignment": "mmd"},
    }
    if variant not in settings:
        raise ValueError(f"unsupported variant: {variant}")
    return settings[variant]


def run_variant(
    variant,
    target_dim,
    source_dim,
    num_classes,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    loss_policy,
    fixed_margins,
    adaptive_margins,
    source_loader,
    lambda_align,
    lambda_align_info,
):
    settings = variant_settings(variant)
    model = MAEDomainTransferClassifier(target_dim, source_dim, num_classes).to(DEVICE)
    load_info = {"path": None, "matched_tensors": 0, "source_tensors": 0, "loaded": False}

    if settings["load_pretrain"]:
        load_info = load_mae_encoder(model)
        if not load_info["loaded"]:
            raise RuntimeError(f"MAE pretrained encoder did not match target architecture: {load_info['path']}")

    if settings["freeze_encoder"]:
        freeze_encoder(model)

    train_info = train_one_model(
        model,
        train_loader,
        val_loader,
        num_classes,
        class_weights,
        loss_policy,
        fixed_margins,
        adaptive_margins,
        source_loader=source_loader,
        alignment=settings["alignment"],
        lambda_align=lambda_align,
    )
    test_metrics = evaluate(model, test_loader, num_classes)
    return {
        "model_type": "mae",
        "variant": variant,
        "settings": settings,
        "pretrain": load_info,
        "training": train_info,
        "alignment_weight_info": lambda_align_info,
        "test": test_metrics,
    }


def summarize_seed_results(seed_results):
    grouped = defaultdict(list)
    for item in seed_results:
        key = f"mae::{item['variant']}"
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


def run_cross_domain_experiment(
    ratios=(1, 5, 10),
    seeds=(42, 52, 62),
    variants=None,
    lambda_align=ALIGNMENT_WEIGHT,
    output_path=None,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(f"source data not found: {SOURCE_DATA_PATH}")

    source_X = np.load(SOURCE_DATA_PATH).astype(np.float32)
    source_dim = int(source_X.shape[1])
    source_loader = make_source_loader(source_X)
    test_X, test_y = load_test_set()
    num_classes = int(max(test_y.max(), 0) + 1)
    test_loader = make_loader(test_X, test_y, shuffle=False)

    if variants is None:
        variants = [
            "random",
            "mae_pretrained_finetune",
            "mae_pretrained_frozen",
            "mae_pretrained_coral",
            "mae_pretrained_mmd",
        ]

    all_results = {
        "config": {
            "source_data": SOURCE_DATA_PATH,
            "target_test": TEST_SET_PATH,
            "source_dim": source_dim,
            "target_test_dim": int(test_X.shape[1]),
            "num_classes": num_classes,
            "ratios": list(ratios),
            "seeds": list(seeds),
            "variants": list(variants),
            "epochs": EPOCHS,
            "validation_size": VAL_SIZE,
            "source_batch_size": SOURCE_BATCH_SIZE,
            "lambda_align_base": float(lambda_align),
            "alignment_reference_ratio": float(ALIGNMENT_REFERENCE_RATIO),
            "alignment_min_scale": float(ALIGNMENT_MIN_SCALE),
            "alignment_max_scale": float(ALIGNMENT_MAX_SCALE),
            "prototype_margin_weight": float(PROTOTYPE_MARGIN_WEIGHT),
            "fixed_margin": float(FIXED_MARGIN),
            "csa_pm_max_ratio": float(CSA_PM_MAX_RATIO),
            "alignment_losses": ["coral", "mmd"],
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
            loss_policy = select_loss_policy(ratio)
            adaptive_margins, margin_info = compute_adaptive_margins(train_y, num_classes)
            fixed_margins = torch.full((num_classes,), FIXED_MARGIN, dtype=torch.float32, device=DEVICE)
            effective_lambda_align, lambda_align_info = compute_adaptive_alignment_weight(ratio, lambda_align)
            target_dim = int(target_X.shape[1])

            seed_results = []
            for variant in variants:
                settings = variant_settings(variant)
                variant_lambda_align = effective_lambda_align if settings["alignment"] != "none" else 0.0
                variant_lambda_info = dict(lambda_align_info)
                variant_lambda_info["effective_weight"] = float(variant_lambda_align)
                result = run_variant(
                    variant=variant,
                    target_dim=target_dim,
                    source_dim=source_dim,
                    num_classes=num_classes,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    class_weights=class_weights,
                    loss_policy=loss_policy,
                    fixed_margins=fixed_margins,
                    adaptive_margins=adaptive_margins,
                    source_loader=source_loader,
                    lambda_align=variant_lambda_align,
                    lambda_align_info=variant_lambda_info,
                )
                result["loss_policy"] = loss_policy
                result["margin_info"] = margin_info if loss_policy["margin_type"] == "csa" else None
                seed_results.append(result)
                seed_level.append(result)
                print(
                    f"ratio={ratio} seed={seed} {variant} "
                    f"test_macro_f1={result['test']['macro_f1']:.4f} "
                    f"best_epoch={result['training']['best_epoch']} "
                    f"loss={loss_policy['name']} "
                    f"align={result['training']['alignment']} "
                    f"lambda_align={variant_lambda_align:.4f}"
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

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "cross_domain_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"cross-domain results saved to {output_path}")
    return all_results


def main():
    global EPOCHS

    parser = argparse.ArgumentParser(description="MAE-only cross-domain transfer from UNSW-NB15 to CICIDS-2017")
    parser.add_argument("--ratios", nargs="+", type=float, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lambda-align", type=float, default=ALIGNMENT_WEIGHT)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=[
            "random",
            "mae_pretrained_finetune",
            "mae_pretrained_frozen",
            "mae_pretrained_coral",
            "mae_pretrained_mmd",
        ],
    )
    args = parser.parse_args()

    EPOCHS = args.epochs
    run_cross_domain_experiment(
        ratios=tuple(args.ratios),
        seeds=tuple(args.seeds),
        variants=args.variants,
        lambda_align=args.lambda_align,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
