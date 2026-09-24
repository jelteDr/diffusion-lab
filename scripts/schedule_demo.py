"""Vergleich linearer vs. Cosinus-Schedule: alpha_bar-Kurve und Vorwärtsprozess."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from ddpm.data import mnist_transform
from ddpm.schedule import SCHEDULES, q_sample

T = 1000
timesteps = [0, 50, 100, 200, 400, 600, 800, 999]
schedules = {name: fn(T) for name, fn in SCHEDULES.items()}

# 1) alpha_bar über t
plt.figure(figsize=(6, 3.4))
for name, sch in schedules.items():
    plt.plot(sch.alphas_cumprod, label=name)
plt.xlabel("Zeitschritt t"); plt.ylabel("ᾱ_t (Anteil Originalsignal)")
plt.title("Noise-Schedules im Vergleich"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("runs/schedules_alpha_bar.png", dpi=120)

# 2) Vorwärtsprozess beider Schedules auf derselben Ziffer mit demselben Rauschen
x0, label = datasets.MNIST("data", train=True, download=True, transform=mnist_transform())[7]
x0 = x0.unsqueeze(0)
torch.manual_seed(0)
noise = torch.randn_like(x0)
fig, axes = plt.subplots(len(schedules), len(timesteps), figsize=(2 * len(timesteps), 2.3 * len(schedules)))
for row, (name, sch) in zip(axes, schedules.items()):
    for ax, t in zip(row, timesteps):
        xt = q_sample(sch, x0, torch.tensor([t]), noise)
        ax.imshow(xt[0, 0], cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t={t}  ᾱ={sch.alphas_cumprod[t]:.2f}", fontsize=8); ax.axis("off")
    row[0].text(-6, 16, name, fontsize=11, ha="right", va="center")
fig.tight_layout()
fig.savefig("runs/schedules_forward.png", dpi=120)
print("gespeichert: runs/schedules_alpha_bar.png, runs/schedules_forward.png")
