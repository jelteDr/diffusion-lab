"""Smoke-Test für das U-Net: stimmen die Formen, läuft es auf MPS, wie groß ist es?"""

import torch

from ddpm.model import UNet, count_params, timestep_embedding

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 1) Zeit-Embedding: Form und Wertebereich
t = torch.tensor([0, 1, 10, 100, 999])
emb = timestep_embedding(t, dim=128)
assert emb.shape == (5, 128), emb.shape
assert emb.abs().max() <= 1.0, "sin/cos müssen in [-1, 1] liegen"
assert not torch.allclose(emb[0], emb[1]), "t=0 und t=1 müssen sich unterscheiden"
print("Zeit-Embedding ok:", tuple(emb.shape))

# 2) U-Net: Ausgabe hat die Form der Eingabe
model = UNet().to(device)
x = torch.randn(8, 1, 32, 32, device=device)
t = torch.randint(0, 1000, (8,), device=device)
with torch.no_grad():
    eps_hat = model(x, t)
assert eps_hat.shape == x.shape, eps_hat.shape
print(f"U-Net ok auf {device}: {tuple(x.shape)} -> {tuple(eps_hat.shape)}")
print(f"Parameter: {count_params(model):,}")
