"""Diagramme rund ums Training."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


def plot_loss(run_dir: Path) -> Path:
    """Zeichnet die Loss-Kurve eines Laufs aus loss.csv nach loss.png.

    Zwei Linien: die rohe Loss alle 50 Schritte (verrauscht, weil jeder Batch
    andere t zieht) und der Epochen-Mittelwert (glatt, der eigentliche Trend).
    """
    rows = list(csv.DictReader(open(run_dir / "loss.csv")))
    if not rows:
        return run_dir / "loss.png"
    # Schritte -> Epochen (mit Bruchteilen), damit die x-Achse "Epoche" heißt
    config_path = run_dir / "config.json"
    steps_per_epoch = json.loads(config_path.read_text()).get("steps_per_epoch") if config_path.exists() else None
    if not steps_per_epoch:  # Fallback für alte Läufe ohne den Eintrag
        steps_per_epoch = max(int(r["step"]) for r in rows) / max(int(r["epoch"]) for r in rows)
    x = [int(r["step"]) / steps_per_epoch for r in rows]
    loss = [float(r["loss"]) for r in rows]

    # Epochen-Mittelwert aus den geloggten Punkten, gesetzt ans Epochen-Ende
    per_epoch: dict[int, list[float]] = {}
    for r in rows:
        per_epoch.setdefault(int(r["epoch"]), []).append(float(r["loss"]))
    ep_x = list(per_epoch.keys())
    ep_mean = [sum(v) / len(v) for v in per_epoch.values()]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(x, loss, lw=0.7, alpha=0.5, label="Loss (alle 50 Batches)")
    ax.plot(ep_x, ep_mean, marker="o", lw=1.8, label="Mittel pro Epoche")
    ax.set_xlim(0, max(ep_x) + 0.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    decimal = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(decimal)
    ax.yaxis.set_minor_formatter(decimal)
    ax.set_xlabel("Epoche")
    ax.set_ylabel("MSE(ε, ε̂)")
    ax.set_title(f"Trainings-Loss, Lauf {run_dir.name}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = run_dir / "loss.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
