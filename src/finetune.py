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
EPOCHS_FINETUNE = 30
LEARNING_RATE_ENCODER = 1e-5
LEARNING_RATE_TOP = 1e-3
LEARNING_RATE_ADAPTER = 1e-3

MARGIN = 1.0
TRIPLET_WEIGHT = 0.1
MMD_WEIGHT = 0.1

DATA_DIR = "data/processed"
TEST_SET = os.path.join(DATA_DIR, "test_set.npz")
SOURCE_DATA_PATH = os.path.join(DATA_DIR, "unsw_X.npy")

PRETRAIN_MODELS = {
    "mae": "checkpoints/mae_pretrain.pth",
    "transformer": "checkpoints/transformer_pretrain.pth",
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
        elif model_type == "transformer":
            from models import TrafficTransformer

            self.encoder = TrafficTransformer(adapter_dim, hidden_dim=64, projection_dim=32)
            self.latent_dim = 32
        else:
            self.encoder = nn.Sequential(
                nn.Linear(adapter_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.latent_dim = 64

        self.prototypes = nn.Parameter(torch.randn(num_classes, self.latent_dim))

    def forward(self, x, return_features=False):
        current_input_dim = x.shape[-1]

        if self.use_adapter:
            if current_input_dim == self.adapter.in_features:
                x = self.adapter(x)
            elif current_input_dim != self.adapter_dim:
                x = self.adapter(x[:, : self.adapter.in_features])
        elif current_input_dim != self.adapter_dim:
            x = x[:, : self.adapter_dim]

        if self.model_type == "transformer":
            _, projections = self.encoder(x)
        else:
            projections = self.encoder(x)

        features_exp = projections.unsqueeze(1)
        prototypes_exp = self.prototypes.unsqueeze(0)
        distances = torch.norm(features_exp - prototypes_exp, p=2, dim=-1)
        logits = -(distances / self.temperature)

        if return_features:
            return logits, projections
        return logits, projections


def hybrid_triplet_prototype_loss(logits, targets, margin=1.0, class_weights=None):
    if class_weights is not None:
        ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))(logits, targets)
    else:
        ce_loss = nn.CrossEntropyLoss()(logits, targets)

    batch_range = torch.arange(logits.size(0)).to(DEVICE)
    pos_dist = -logits[batch_range, targets]
    mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0.0)
    nearest_neg_logits, _ = torch.max(logits * mask - (1 - mask) * 1e9, dim=1)
    nearest_neg_dist = -nearest_neg_logits
    triplet_loss = torch.clamp(pos_dist - nearest_neg_dist + margin, min=0.0).mean()
    return ce_loss, triplet_loss


def compute_class_weights(y, num_classes):
    class_counts = np.bincount(y, minlength=num_classes)
    total_samples = len(y)
    return torch.FloatTensor(total_samples / (num_classes * (class_counts + 1e-10)))


def load_fewshot_npz(ratio, seed):
    npz_path = os.path.join(DATA_DIR, f"cicids_{int(ratio)}_seed{seed}.npz")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(DATA_DIR, f"cicids_{int(ratio)}pct.npz")
    data = np.load(npz_path)
    return data["x"], data["y"]


def load_test_set():
    data = np.load(TEST_SET)
    return data["x"], data["y"]


def finetune_loop(model_type, ratios=[0.1, 1, 5, 10], seeds=[42, 52, 62]):
    test_X, test_y = load_test_set()
    ratio_results = {}

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

            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model = AdapterMetricNet(
                input_dim=input_dim,
                num_classes=num_classes,
                model_type=model_type,
                adapter_dim=input_dim,
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
            best_f1 = 0.0
            best_epoch = 0
            best_state_dict = None
            epoch_history = []

            for epoch in range(EPOCHS_FINETUNE):
                model.train()
                total_loss = 0.0
                batch_count = 0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)

                    optimizer.zero_grad()
                    logits, _ = model(batch_x)
                    ce_loss, triplet_loss = hybrid_triplet_prototype_loss(
                        logits,
                        batch_y,
                        class_weights=class_weights,
                    )
                    loss = ce_loss + TRIPLET_WEIGHT * triplet_loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
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
                epoch_history.append({"epoch": epoch + 1, "loss": avg_loss, "macro_f1": f1})

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
                },
                f"checkpoints/finetune_{model_type}_ratio{int(ratio)}_seed{seed}.pth",
            )

        ratio_avg_f1 = np.mean(seed_f1_list)
        ratio_std_f1 = np.std(seed_f1_list)
        ratio_results[ratio] = ratio_avg_f1
        print(f"\nsummary ratio {ratio}% - mean Macro F1: {ratio_avg_f1:.4f} +- {ratio_std_f1:.4f}")

    return ratio_results


def main():
    parser = argparse.ArgumentParser(description="Finetune script with MAE/Transformer support")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["none", "mae", "transformer", "all"],
        help="pretrain backbone type",
    )
    args = parser.parse_args()

    if args.model == "all":
        models_to_train = ["mae", "transformer", "none"]
    else:
        models_to_train = [args.model]

    all_results = {}
    for model_type in models_to_train:
        print(f"\n{'#' * 60}")
        print(f"start finetune: {model_type.upper()}")
        print(f"{'#' * 60}")

        results = finetune_loop(model_type)
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
