"""Exp 4: Classifier-free Guidance — Stärke w vs. FID, IS und Label-Treffer.
Aufruf:  uv run python scripts/plot_guidance.py mnist_cond"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1] if len(sys.argv) > 1 else "mnist_cond"
run_dir = Path("runs") / run
rows = []
for p in sorted(run_dir.glob("eval_ddim100_w*.json")):
    d = json.loads(p.read_text())
    rows.append((d["guidance"], d["fid"]["mean"], d["fid"]["std"], d["inception_score"]["mean"],
                 d["inception_score"]["std"], d["label_accuracy"]["mean"]))
rows.sort()
w = [r[0] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].errorbar(w, [r[1] for r in rows], yerr=[r[2] for r in rows], marker="o", capsize=3)
axes[0].set_ylabel("FID ↓")
axes[1].errorbar(w, [r[3] for r in rows], yerr=[r[4] for r in rows], marker="o", capsize=3)
axes[1].set_ylabel("Inception-Score ↑"); axes[1].axhline(10, color="gray", ls=":", lw=1)
axes[2].plot(w, [100 * r[5] for r in rows], marker="o")
axes[2].set_ylabel("Label-Treffer [%] ↑"); axes[2].set_ylim(0, 102)
for ax in axes:
    ax.set_xlabel("Guidance-Stärke w"); ax.grid(alpha=0.3); ax.set_xticks(w)
    ax.axvline(1, color="gray", ls="--", lw=1)
fig.suptitle(f"Classifier-free Guidance, Lauf {run} (DDIM η=1, 100 Schritte, n=2000, 2 Seeds)")
fig.tight_layout()
out = Path("docs/img/guidance.png"); fig.savefig(out, dpi=120); print("gespeichert:", out)
