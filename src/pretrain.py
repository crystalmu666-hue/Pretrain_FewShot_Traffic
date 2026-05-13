import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import MaskedTrafficAutoencoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def check_and_clean_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=100.0, neginf=-100.0)
    return np.clip(data, -100, 100)


def train_mae(data_path, epochs=50, batch_size=1024, lr=0.001, mask_ratio=0.4):
    print(f"\n{'=' * 60}")
    print(f"Training MAE pretrain model, mask ratio={mask_ratio:.2f}")
    print(f"{'=' * 60}")

    data = np.load(data_path)
    print(f"source data shape: {data.shape}")
    data = check_and_clean_data(data).astype(np.float32)

    input_dim = data.shape[1]
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MaskedTrafficAutoencoder(input_dim, mask_ratio=mask_ratio).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs("results/data", exist_ok=True)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for (inputs,) in loader:
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()

            decoded, mask = model(inputs)
            loss = criterion(decoded * mask, inputs * mask)

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        loss_history.append(avg_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch [{epoch + 1}/{epochs}] masked reconstruction loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"mae_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            print(f"saved: {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "mae_pretrain.pth")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "loss_history": loss_history,
        },
        final_path,
    )
    np.save("results/data/loss_mae.npy", np.asarray(loss_history, dtype=np.float32))
    print(f"MAE pretraining complete: {final_path}")
    return loss_history


def main():
    parser = argparse.ArgumentParser(description="MAE pretraining for traffic features")
    parser.add_argument("--model", type=str, default="mae", choices=["mae"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    args = parser.parse_args()

    data_path = "data/processed/unsw_X.npy"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"source pretrain data not found: {data_path}")

    print(f"device: {DEVICE}")
    train_mae(
        data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
    )


if __name__ == "__main__":
    main()
