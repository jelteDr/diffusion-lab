"""Loss-Kurve eines Laufs (neu) zeichnen:  uv run python scripts/plot_loss.py mnist_base"""

import sys
from pathlib import Path

from ddpm.plotting import plot_loss

run = sys.argv[1] if len(sys.argv) > 1 else "mnist_base"
print("gespeichert:", plot_loss(Path("runs") / run))
