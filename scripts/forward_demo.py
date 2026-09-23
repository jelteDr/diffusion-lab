"""Zeigt den Vorwärtsprozess: eine MNIST-Ziffer wird über die Zeit verrauscht."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from ddpm.schedule import linear_schedule, q_sample

# Ein einzelnes MNIST-Bild laden, auf [-1, 1] skalieren (Standard bei DDPM)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST("data", train=True, download=True, transform=tf)
x0, label = mnist[7]
x0 = x0.unsqueeze(0)  # (1, 1, 28, 28)

schedule = linear_schedule(num_steps=1000)
timesteps = [0, 50, 100, 200, 400, 600, 800, 999]

torch.manual_seed(0)
noise = torch.randn_like(x0)

fig, axes = plt.subplots(1, len(timesteps), figsize=(2 * len(timesteps), 2.4))
for ax, t in zip(axes, timesteps):
    xt = q_sample(schedule, x0, torch.tensor([t]), noise)
    ax.imshow(xt[0, 0], cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"t={t}\nᾱ={schedule.alphas_cumprod[t]:.3f}", fontsize=9)
    ax.axis("off")
fig.suptitle(f"Vorwärtsprozess, Ziffer {label}")
fig.tight_layout()
fig.savefig("runs/forward_process.png", dpi=120)
print("gespeichert: runs/forward_process.png")
