"""Exp 5: CLIP-Score, Vielfalt und Sättigung über der Guidance-Stärke.

Aufruf:  uv run python scripts/plot_clip_eval.py exp5
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

name = sys.argv[1] if len(sys.argv) > 1 else "exp5"
run_dir = Path("runs/sdxl") / name
evals = {p.stem.removeprefix("eval_clip_"): json.loads(p.read_text())
         for p in sorted(run_dir.glob("eval_clip_*.json"))}
if not evals:
    raise SystemExit(f"keine eval_clip_*.json in {run_dir}")

primary = "L14" if "L14" in evals else next(iter(evals))
rows = evals[primary]["results"]
prompts = sorted({r["prompt_key"] for r in rows})
scheds = sorted({r["scheduler"] for r in rows})
colors = dict(zip(prompts, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
styles = {"L14": "-", "B32-laion": "--"}


def series(rs, pk, sched, key):
    sel = sorted((r for r in rs if r["prompt_key"] == pk and r["scheduler"] == sched and r.get(key) is not None),
                 key=lambda r: r["guidance"])
    return [r["guidance"] for r in sel], [r[key] for r in sel], [r.get(key + "_std", 0) for r in sel]


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for sched in scheds:
    for pk in prompts:
        for jk, ev in evals.items():
            w, y, sd = series(ev["results"], pk, sched, "clip_score")
            axes[0].errorbar(w, y, yerr=sd, color=colors[pk], ls=styles.get(jk, ":"), marker="o", ms=4, capsize=2,
                             label=f"{pk} ({jk})" if sched == scheds[0] else None)
        w, y, _ = series(rows, pk, sched, "diversity")
        axes[1].plot(w, y, color=colors[pk], marker="o", ms=4, label=pk if sched == scheds[0] else None)
        w, y, _ = series(rows, pk, sched, "saturation")
        axes[2].plot(w, y, color=colors[pk], marker="o", ms=4, label=pk if sched == scheds[0] else None)

axes[0].set_ylabel("CLIP-Score ↑ (Prompt-Treue)")
axes[1].set_ylabel(f"Vielfalt ↑ (1 − cos, Richter {primary})")
axes[2].set_ylabel("mittlere Sättigung (HSV)")
for ax in axes:
    ax.set_xlabel("Guidance-Stärke w"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_xticks(sorted({r["guidance"] for r in rows}))
n_seeds = max(r["n_seeds"] for r in rows)
fig.suptitle(f"SDXL Exp 5 ({name}): Treue vs. Vielfalt vs. Sättigung, {', '.join(scheds)}, bis {n_seeds} Seeds")
fig.tight_layout()
out = Path("docs/img") / f"sdxl_{name}_clip.png"
fig.savefig(out, dpi=110)
print("gespeichert:", out)
