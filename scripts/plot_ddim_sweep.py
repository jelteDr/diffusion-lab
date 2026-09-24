"""Trade-off-Kurve Sampling-Schritte vs. Qualität aus den eval_*.json eines Laufs.

Aufruf:  uv run python scripts/plot_ddim_sweep.py mnist_base
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1] if len(sys.argv) > 1 else "mnist_base"
run_dir = Path("runs") / run


def load(tag: str) -> dict | None:
    p = run_dir / f"eval_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def series(prefix: str, suffix: str = "") -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    steps, fid, fid_sd, is_, is_sd = [], [], [], [], []
    for k in (10, 20, 50, 100, 250, 1000):
        d = load(f"{prefix}{k}{suffix}")
        if d is None:
            continue
        steps.append(k)
        fid.append(d["fid"]["mean"]); fid_sd.append(d["fid"]["std"])
        is_.append(d["inception_score"]["mean"]); is_sd.append(d["inception_score"]["std"])
    return steps, fid, fid_sd, is_, is_sd


# DDPM-Referenz (1000 Schritte): Mittel über die vorhandenen Einzel-Seed-Dateien
ddpm = [d for d in (load("ddpm"), load("ddpm_seed2")) if d]
ddpm_fid = sum(d["fid"] for d in ddpm) / len(ddpm)
ddpm_is = sum(d["inception_score"] for d in ddpm) / len(ddpm)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, key, ref, label in ((axes[0], 1, ddpm_fid, "FID ↓"), (axes[1], 3, ddpm_is, "Inception-Score ↑")):
    for prefix, suffix, name, marker in (("ddim", "", "DDIM η=0 (deterministisch)", "o"),
                                         ("ddim", "_eta1", "DDIM η=1 (stochastisch)", "s")):
        s = series(prefix, suffix)
        if s[0]:
            ax.errorbar(s[0], s[key], yerr=s[key + 1], marker=marker, capsize=3, label=name)
    ax.axhline(ref, color="gray", ls="--", label=f"DDPM 1000 Schritte ({ref:.1f})")
    ax.set_xscale("log"); ax.set_xticks([10, 20, 50, 100, 250, 1000]); ax.set_xticklabels([10, 20, 50, 100, 250, 1000])
    ax.set_xlabel("Sampling-Schritte"); ax.set_ylabel(label); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.suptitle(f"Sampling-Schritte vs. Qualität, Lauf {run} (3 Seeds, n=1000)")
fig.tight_layout()
out = Path("docs/img") / "ddim_tradeoff.png"
fig.savefig(out, dpi=120)
print("gespeichert:", out)
