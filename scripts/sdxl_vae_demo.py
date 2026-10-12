"""Was der SDXL-VAE mit einem Bild macht: kodieren, Latent anschauen, zurückrechnen.

Aufruf:  uv run python scripts/sdxl_vae_demo.py runs/sdxl/fox/euler_w5.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image

from sdxl.download import VAE_ID
from sdxl.pipeline import device_and_dtype

path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/sdxl/fox/euler_w5.png")
device, _ = device_and_dtype()
vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float32).to(device).eval()
scale = vae.config.scaling_factor

img = Image.open(path).convert("RGB")
x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)[None] / 127.5 - 1.0  # [-1, 1], (1,3,H,W)
x = x.to(device)

with torch.no_grad():
    dist = vae.encode(x).latent_dist          # Encoder liefert eine Gaußverteilung pro Latent-Wert
    z = dist.mean                              # deterministisch: Mittelwert statt Stichprobe
    z_scaled = z * scale                       # so sieht der Latent für das U-Net aus
    x_rec = vae.decode(z).sample.clamp(-1, 1)

n_pix, n_lat = x.numel(), z.numel()
mse = torch.mean((x_rec - x) ** 2).item()
psnr = 10 * np.log10(4.0 / mse)  # Wertebereich [-1, 1] hat Spannweite 2 -> Spitze² = 4
print(f"Bild  {tuple(x.shape[1:])} = {n_pix:,} Werte")
print(f"Latent {tuple(z.shape[1:])} = {n_lat:,} Werte  -> Kompression {n_pix / n_lat:.0f}x")
print(f"Latent roh:       Mittel {z.mean():+.3f}, Std {z.std():.3f}")
print(f"Latent skaliert:  Mittel {z_scaled.mean():+.3f}, Std {z_scaled.std():.3f}  (scaling_factor {scale})")
print(f"Rekonstruktion: PSNR {psnr:.1f} dB, mittlere Abweichung {torch.mean(torch.abs(x_rec - x)).item() * 127.5:.1f} von 255")

to_np = lambda t: ((t[0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
fig, axes = plt.subplots(1, 7, figsize=(21, 3.4))
axes[0].imshow(to_np(x)); axes[0].set_title("Original 1024×1024×3")
for c in range(4):
    axes[1 + c].imshow(z_scaled[0, c].cpu().numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axes[1 + c].set_title(f"Latent-Kanal {c} (128×128)")
axes[5].imshow(to_np(x_rec)); axes[5].set_title(f"Rekonstruktion, PSNR {psnr:.1f} dB")
diff = (x_rec - x).abs().mean(1)[0].cpu().numpy()
axes[6].imshow(diff, cmap="magma", vmin=0, vmax=0.3); axes[6].set_title("Abweichung (hell = groß)")
for ax in axes: ax.axis("off")
fig.tight_layout()
out = Path("docs/img/sdxl_vae_latent.png"); fig.savefig(out, dpi=100)
print("gespeichert:", out)
