"""Exp 3: Modellgröße vs. Qualität und Sampling-Kosten.  uv run python scripts/plot_model_size.py"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

RUNS = [("mnist_cosine_b16", 16, 0.67), ("mnist_cosine", 32, 2.17), ("mnist_cosine_b64", 64, 8.10)]
TAG = "ddim100_eta1_n5000"

params, fid, fid_sd, is_, is_sd = [], [], [], [], []
for run, base, mparams in RUNS:
    d = json.loads((Path("runs") / run / f"eval_{TAG}.json").read_text())
    params.append(mparams)
    fid.append(d["fid"]["mean"]); fid_sd.append(d["fid"]["std"])
    is_.append(d["inception_score"]["mean"]); is_sd.append(d["inception_score"]["std"])
real = json.loads(Path("runs/eval_real.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, y, sd, ref, label in ((axes[0], fid, fid_sd, real["fid"], "FID ↓"),
                              (axes[1], is_, is_sd, real["inception_score"], "Inception-Score ↑")):
    ax.errorbar(params, y, yerr=sd, marker="o", capsize=3, label="DDIM η=1, 100 Schritte")
    ax.axhline(ref, color="gray", ls="--", label=f"echte Ziffern ({ref:.1f})")
    for p_, v, base in zip(params, y, (16, 32, 64)):
        ax.annotate(f"base={base}", (p_, v), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("log"); ax.set_xticks(params); ax.set_xticklabels([f"{p:.1f} M" for p in params])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Parameter"); ax.set_ylabel(label); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.suptitle("Modellgröße vs. Qualität (Cosinus-Schedule, 20 Epochen, n=5000, 3 Seeds)")
fig.tight_layout()
out = Path("docs/img/model_size.png"); fig.savefig(out, dpi=120); print("gespeichert:", out)
