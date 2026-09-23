"""Zeigt das Sinus-Zeit-Embedding als Heatmap über alle Zeitschritte.
Aufruf:  uv run python scripts/time_embedding_demo.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ddpm.model import timestep_embedding

DIM = 128
NUM_STEPS = 1000

# Embedding für jeden Zeitschritt 0 .. 999 berechnen -> Form (1000, 128)
emb = timestep_embedding(torch.arange(NUM_STEPS), dim=DIM)

# Transponiert zeichnen: x-Achse = Zeit, y-Achse = Embedding-Dimension
plt.figure(figsize=(9, 3.2))
plt.imshow(emb.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
plt.xlabel("Zeitschritt t")
plt.ylabel(f"Embedding-Dimension (0-{DIM // 2 - 1} sin, {DIM // 2}-{DIM - 1} cos)")
plt.colorbar(label="Wert")
plt.title(f"Sinus-Zeit-Embedding, dim={DIM}")
plt.tight_layout()
plt.savefig("runs/time_embedding.png", dpi=120)
print("gespeichert: runs/time_embedding.png")
