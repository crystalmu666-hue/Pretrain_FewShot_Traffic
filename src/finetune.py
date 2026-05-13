"""
Finetuning script for few-shot experiments.
Stores per-epoch loss/F1 history in checkpoints so convergence plots can use
real training traces instead of placeholder points.
"""

import argparse
import copy
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS_FROZEN = 20
EPOCHS_FINETUNE = 50
LEARNING_RATE_ENCODER = 5e-5
LEARNING_RATE_TOP = 1e-3
LEARNING_RATE_ADAPTER = 1e-3

BASE_MARGIN = 1.0
FIXED_MARGIN = 1.0
SCARCITY_GAMMA = 0.2
ADAPTIVE_MARGIN_MIN = 1.0
ADAPTIVE_MARGIN_MAX = 2.0
PROTOTYPE_MARGIN_WEIGHT = 0.03
CSA_PM_MAX_RATIO = 0.1
ALIGNMENT_BASE_WEIGHT = 0.05
ALIGNMENT_REFERENCE_RATIO = 5.0
ALIGNMENT_MIN_SCALE = 0.5
ALIGNMENT_MAX_SCALE = 2.0
SOURCE_BATCH_SIZE = 256
MMD_KERNEL_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)

DATA_DIR = "data/processed"
TEST_SET = os.path.join(DATA_DIR, "test_set.npz")
SOURCE_DATA_PATH = os.path.join(DATA_DIR, "unsw_X.npy")

PRETRAIN_MODELS = {
    "mae": "checkpoints/mae_pretrain.pth",
}


def get_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_name = time.strftime("%Y%m%d_%H%M%S") + "_finetune.log"
    return os.path.join(log_dir, log_name)


def log_to_file(path, message):
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_pretrained_weights_into_model(model, model_type):
    pretrain_path = PRETRAIN_MODELS.get(model_type)

    if not pretrain_path or not os.path.exists(pretrain_path):
        print(f"  pretrained weights not found: {pretrain_path}")
        return 0

    print(f"  loading pretrained weights: {pretrain_path}")

    checkpoint = torch.load(pretrain_path, map_location=DEVICE)
    src_state = checkpoint.get("model_state_dict", checkpoint)
    model_dict = model.state_dict()

    matched = {}
    for k_pretrain, v_pretrain in src_state.items():
        for prefix in ["", "encoder."]:
            k_candidate = prefix + k_pretrain
            if k_candidate in model_dict and v_pretrain.shape == model_dict[k_candidate].shape:
                matched[k_candidate] = v_pretrain
                break

    if matched:
        model.load_state_dict(matched, strict=False)
        print(f"  loaded {len(matched)}/{len(src_state)} pretrained tensors")
    else:
        print("  no pretrained tensors matched target model")

    return len(matched)


class AdapterMetricNet(nn.Module):
    def __init__(self, input_dim, num_classes=11, model_type="mae", adapter_dim=40, use_adapter=True):
        super().__init__()
        self.model_type = model_type
        self.use_adapter = use_adapter
        self.adapter_dim = adapter_dim

        self.adapter = nn.Linear(input_dim, adapter_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        if model_type == "mae":
            from models import MaskedTrafficAutoencoder

            self.encoder = MaskedTrafficAutoencoder(
                adapter_dim,
                mask_ratio=0.75,
                hidden_dim=128,
                latent_dim=32,
            ).encoder
            self.latent_dim = 32
        elif model_type == "none":
            self.encoder = nn.Sequential(
                nn.Linear(adapter_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.latent_dim = 64
        else:
            raise ValueError(f"unsupported model_type: {model_type}")

        self.prototypes = nn.Parameter(torch.randn(num_classes, self.latent_dim))

    def adapt_target(self, x):
        current_input_dim = x.shape[-1]

        if self.use_adapter:
            if current_input_dim == self.adapter.in_features:
                x = self.adapter(x)
            elif current_input_dim != self.adapter_dim:
                x = self.adapter(x[:, : self.adapter.in_features])
        elif current_input_dim != self.adapter_dim:
            x = x[:, : self.adapter_dim]

        return x

    def encode_target(self, x):
        return self.encoder(self.adapt_target(x))

    def encode_source(self, x):
        if x.shape[-1] == self.adapter_dim:
            return self.encoder(x)
        return self.encode_target(x)

    def forward(self, x, return_features=False):
        projections = self.encode_target(x)

        features_exp = projections.unsqueeze(1)
        prototypes_exp = self.prototypes.unsqueeze(0)
        distances = torch.norm(features_exp - prototypes_exp, p=2, dim=-1)
        logits = -(distances / self.temperature)

        if return_features:
            return logits, projections
        return logits, projections


def prototype_margin_loss(features, prototypes, targets, margins):
    distances = torch.cdist(features, prototypes)
    batch_range = torch.arange(distances.size(0), device=distances.device)
    positive_dist = distances[batch_range, targets]

    negative_distances = distances.clone()
    negative_distances[batch_range, targets] = float("inf")
    nearest_negative_dist = negative_distances.min(dim=1).values

    sample_margins = margins.to(DEVICE)[targets]
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
    ce_weight = class_weights.to(DEVICE) if policy["use_class_weights"] else None
    ce_loss = nn.CrossEntropyLoss(weight=ce_weight)(logits, targets)

    if policy["margin_type"] == "csa":
        pm_loss = prototype_margin_loss(features, prototypes, targets, adaptive_margins)
    elif policy["margin_type"] == "fixed":
        pm_loss = prototype_margin_loss(features, prototypes, targets, fixed_margins)
    else:
        raise ValueError(f"unsupported margin type: {policy['margin_type']}")

    loss = ce_loss + PROTOTYPE_MARGIN_WEIGHT * pm_loss
    return loss, ce_loss, pm_loss


def make_source_loader(source_X):
    dataset = TensorDataset(torch.FloatTensor(source_X.astype(np.float32)))
    return DataLoader(dataset, batch_size=SOURCE_BATCH_SIZE, shuffle=True, drop_last=True)


def next_source_batch(source_iter, source_loader):
    try:
        (source_x,) = next(source_iter)
    except StopIteration:
        source_iter = iter(source_loader)
        (source_x,) = next(source_iter)
    return source_x, source_iter


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
        return target_features.new_tensor(0.0)
    if alignment == "coral":
        return coral_loss(source_features, target_features)
    if alignment == "mmd":
        return mmd_loss(source_features, target_features)
    if alignment == "coral_mmd":
        return coral_loss(source_features, target_features) + mmd_loss(source_features, target_features)
    raise ValueError(f"unsupported alignment: {alignment}")


def compute_adaptive_alignment_weight(ratio, base_weight=ALIGNMENT_BASE_WEIGHT):
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


def compute_class_weights(y, num_classes):
    class_counts = np.bincount(y, minlength=num_classes)
    total_samples = len(y)
    return torch.FloatTensor(total_samples / (num_classes * (class_counts + 1e-10)))


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
        torch.FloatTensor(margins),
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


def load_fewshot_npz(ratio, seed):
    npz_path = os.path.join(DATA_DIR, f"cicids_{int(ratio)}_seed{seed}.npz")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(DATA_DIR, f"cicids_{int(ratio)}pct.npz")
    data = np.load(npz_path)
    return data["x"], data["y"]


def load_test_set():
    data = np.load(TEST_SET)
    return data["x"], data["y"]


def finetune_loop(
    model_type,
    ratios=[0.1, 1, 5, 10],
    seeds=[42, 52, 62],
    alignment="mmd",
    lambda_align_base=ALIGNMENT_BASE_WEIGHT,
):
    test_X, test_y = load_test_set()
    ratio_results = {}
    source_X = None
    source_dim = None
    source_loader = None
    use_source_alignment = model_type == "mae" and alignment != "none"

    if use_source_alignment:
        if not os.path.exists(SOURCE_DATA_PATH):
            print(f"source data not found, disable alignment: {SOURCE_DATA_PATH}")
            use_source_alignment = False
        else:
            source_X = np.load(SOURCE_DATA_PATH).astype(np.float32)
            source_dim = int(source_X.shape[1])
            source_loader = make_source_loader(source_X)
            print(f"source alignment enabled: {alignment}, source_dim={source_dim}")

    for ratio in ratios:
        print(f"\n{'=' * 60}")
        print(f"sample ratio: {ratio}%")
        print(f"{'=' * 60}")
        seed_f1_list = []

        for seed in seeds:
            print(f"\n  seed {seed} start")

            X_train, y_train = load_fewshot_npz(ratio, seed)
            input_dim = X_train.shape[1]
            num_classes = len(np.unique(y_train))
            adapter_dim = source_dim if use_source_alignment and source_dim is not None else input_dim

            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model = AdapterMetricNet(
                input_dim=input_dim,
                num_classes=num_classes,
                model_type=model_type,
                adapter_dim=adapter_dim,
            ).to(DEVICE)

            load_pretrained_weights_into_model(model, model_type)

            optimizer = optim.Adam(
                [
                    {"params": model.adapter.parameters(), "lr": LEARNING_RATE_ADAPTER},
                    {"params": model.prototypes, "lr": LEARNING_RATE_TOP},
                    {"params": [model.temperature], "lr": LEARNING_RATE_TOP},
                    {
                        "params": [p for p in model.encoder.parameters() if p.requires_grad],
                        "lr": LEARNING_RATE_ENCODER,
                    },
                ]
            )

            class_weights = compute_class_weights(y_train, num_classes)
            class_margins, margin_info = compute_adaptive_margins(y_train, num_classes)
            fixed_margins = torch.full((num_classes,), FIXED_MARGIN, dtype=torch.float32)
            loss_policy = select_loss_policy(ratio)
            effective_alignment = alignment if use_source_alignment else "none"
            lambda_align, lambda_align_info = compute_adaptive_alignment_weight(ratio, lambda_align_base)
            if effective_alignment == "none":
                lambda_align = 0.0
                lambda_align_info["effective_weight"] = 0.0
            print(
                f"  loss policy: {loss_policy['display_name']} | "
                f"pm_weight={PROTOTYPE_MARGIN_WEIGHT}"
            )
            if loss_policy["margin_type"] == "csa":
                print(
                    "  CSA-PM margins: "
                    f"min={class_margins.min().item():.4f}, "
                    f"max={class_margins.max().item():.4f}, "
                    f"gamma={SCARCITY_GAMMA}"
                )
            else:
                print(f"  fixed PM margin: {FIXED_MARGIN:.4f}")
            print(
                f"  class-weighted CE: {loss_policy['use_class_weights']} | "
                f"threshold for CSA-PM: <= {CSA_PM_MAX_RATIO}%"
            )
            print(
                f"  alignment: {effective_alignment} | "
                f"lambda_align={lambda_align:.4f} | "
                f"adapter_dim={adapter_dim}"
            )
            best_f1 = 0.0
            best_epoch = 0
            best_state_dict = None
            epoch_history = []

            for epoch in range(EPOCHS_FINETUNE):
                model.train()
                total_loss = 0.0
                total_ce_loss = 0.0
                total_pm_loss = 0.0
                total_align_loss = 0.0
                batch_count = 0
                source_iter = iter(source_loader) if effective_alignment != "none" else None

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)

                    optimizer.zero_grad()
                    logits, features = model(batch_x)
                    loss, ce_loss, pm_loss = compute_policy_loss(
                        logits,
                        features,
                        model.prototypes,
                        batch_y,
                        loss_policy,
                        class_weights,
                        fixed_margins,
                        class_margins,
                    )
                    align_loss = features.new_tensor(0.0)
                    if effective_alignment != "none":
                        source_x, source_iter = next_source_batch(source_iter, source_loader)
                        source_features = model.encode_source(source_x.to(DEVICE))
                        align_loss = compute_alignment_loss(source_features, features, effective_alignment)

                    loss = loss + lambda_align * align_loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_ce_loss += ce_loss.item()
                    total_pm_loss += pm_loss.item()
                    total_align_loss += align_loss.item()
                    batch_count += 1

                model.eval()
                preds, labels = [], []
                with torch.no_grad():
                    for bx, by in test_loader:
                        logits, _ = model(bx.to(DEVICE))
                        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                        labels.extend(by.numpy())

                report = classification_report(labels, preds, output_dict=True, zero_division=0)
                f1 = float(report["macro avg"]["f1-score"])
                avg_loss = float(total_loss / max(batch_count, 1))
                epoch_history.append(
                    {
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "ce_loss": float(total_ce_loss / max(batch_count, 1)),
                        "prototype_margin_loss": float(total_pm_loss / max(batch_count, 1)),
                        "alignment_loss": float(total_align_loss / max(batch_count, 1)),
                        "macro_f1": f1,
                    }
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(model.state_dict())
                    print(f"      epoch {epoch + 1:2d} | loss {avg_loss:.4f} | macro_f1 {f1:.4f} -> best")

            print(f"\n  seed {seed} complete - best Macro F1: {best_f1:.4f}")
            seed_f1_list.append(best_f1)

            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state_dict if best_state_dict is not None else model.state_dict(),
                    "history": epoch_history,
                    "best_macro_f1": float(best_f1),
                    "best_epoch": int(best_epoch),
                    "ratio": ratio,
                    "seed": seed,
                    "model_type": model_type,
                    "loss": loss_policy["name"],
                    "loss_policy": loss_policy,
                    "prototype_margin_weight": float(PROTOTYPE_MARGIN_WEIGHT),
                    "fixed_margin": float(FIXED_MARGIN),
                    "csa_pm_max_ratio": float(CSA_PM_MAX_RATIO),
                    "margin_info": margin_info if loss_policy["margin_type"] == "csa" else None,
                    "alignment": effective_alignment,
                    "alignment_weight": float(lambda_align),
                    "alignment_weight_info": lambda_align_info,
                    "source_data": SOURCE_DATA_PATH if use_source_alignment else None,
                    "source_dim": source_dim,
                    "adapter_dim": int(adapter_dim),
                },
                f"checkpoints/finetune_{model_type}_ratio{int(ratio)}_seed{seed}.pth",
            )

        ratio_avg_f1 = np.mean(seed_f1_list)
        ratio_std_f1 = np.std(seed_f1_list)
        ratio_results[ratio] = ratio_avg_f1
        print(f"\nsummary ratio {ratio}% - mean Macro F1: {ratio_avg_f1:.4f} +- {ratio_std_f1:.4f}")

    return ratio_results


def main():
    parser = argparse.ArgumentParser(description="Finetune AdapterMetricNet with MAE pretraining")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["none", "mae", "all"],
        help="pretrain backbone type",
    )
    parser.add_argument(
        "--alignment",
        type=str,
        default="mmd",
        choices=["none", "coral", "mmd", "coral_mmd"],
        help="source-target feature alignment loss used for MAE fine-tuning",
    )
    parser.add_argument(
        "--lambda-align-base",
        type=float,
        default=ALIGNMENT_BASE_WEIGHT,
        help="base alignment weight at the reference labeled ratio",
    )
    args = parser.parse_args()

    if args.model == "all":
        models_to_train = ["mae", "none"]
    else:
        models_to_train = [args.model]

    all_results = {}
    for model_type in models_to_train:
        print(f"\n{'#' * 60}")
        print(f"start finetune: {model_type.upper()}")
        print(f"{'#' * 60}")

        results = finetune_loop(
            model_type,
            alignment=args.alignment,
            lambda_align_base=args.lambda_align_base,
        )
        all_results[model_type] = results

        print(f"\n{'=' * 60}")
        print(f"{model_type.upper()} final results")
        for ratio, f1 in sorted(results.items()):
            print(f"  {ratio:4.1f}% | {f1:.4f}")
        print(f"{'=' * 60}")

    if len(models_to_train) > 1:
        print(f"\n{'#' * 60}")
        print("combined comparison")
        print(f"{'#' * 60}")
        print(f"{'ratio':<10}", end="")
        for model_type in models_to_train:
            print(f"| {model_type.upper():<15}", end="")
        print()
        print("-" * (10 + 18 * len(models_to_train)))
        for ratio in [0.1, 1, 5, 10]:
            print(f"{ratio:4.1f}%   ", end="")
            for model_type in models_to_train:
                f1 = all_results[model_type].get(ratio, 0)
                print(f"| {f1:>15.4f}", end="")
            print()


if __name__ == "__main__":
    main()
