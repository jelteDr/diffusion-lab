"""Diagramme rund ums Training."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(run_dir: Path) -> Path:
    """Zeichnet die Loss-Kurve eines Laufs aus loss.csv nach loss.png.

    Zwei Linien: die rohe Loss alle 50 Schritte (verrauscht, weil jeder Batch
    andere t zieht) und der Epochen-Mittelwert (glatt, der eigentliche Trend).
    """
    rows = list(csv.DictReader(open(run_dir / "loss.csv")))
    if not rows:
        return run_dir / "loss.png"
    steps = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]

    # Epochen-Mittelwert aus den geloggten Punkten
    per_epoch: dict[int, list[float]] = {}
    for r in rows:
        per_epoch.setdefault(int(r["epoch"]), []).append(float(r["loss"]))
    ep_steps = [max(int(r["step"]) for r in rows if int(r["epoch"]) == e) for e in per_epoch]
    ep_mean = [sum(v) / len(v) for v in per_epoch.values()]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(steps, loss, lw=0.7, alpha=0.5, label="Loss (alle 50 Schritte)")
    ax.plot(ep_steps, ep_mean, marker="o", lw=1.8, label="Mittel pro Epoche")
    ax.set_yscale("log")
    ax.set_xlabel("Optimierungsschritt")
    ax.set_ylabel("MSE(ε, ε̂)")
    ax.set_title(f"Trainings-Loss, Lauf {run_dir.name}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = run_dir / "loss.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
