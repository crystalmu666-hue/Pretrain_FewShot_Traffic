import os

import torch
import torch.nn as nn


class MaskedTrafficAutoencoder(nn.Module):
    """Masked autoencoder for self-supervised traffic feature pretraining."""

    def __init__(self, input_dim, mask_ratio=0.75, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def mask_input(self, x):
        batch_size = x.shape[0]
        mask = torch.zeros((batch_size, self.input_dim), device=x.device)
        num_masked = int(self.mask_ratio * self.input_dim)

        for i in range(batch_size):
            mask_indices = torch.randperm(self.input_dim, device=x.device)[:num_masked]
            mask[i, mask_indices] = 1.0

        return x * (1.0 - mask), mask

    def forward(self, x):
        masked_x, mask = self.mask_input(x)
        encoded = self.encoder(masked_x)
        decoded = self.decoder(encoded)
        return decoded, mask


def load_pretrained_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"pretrained checkpoint not found: {checkpoint_path}")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    matched = {}

    for source_key, value in source_state.items():
        for target_key in (source_key, f"encoder.{source_key}"):
            if target_key in target_state and value.shape == target_state[target_key].shape:
                matched[target_key] = value
                break

    model.load_state_dict(matched, strict=False)
    return len(matched)
