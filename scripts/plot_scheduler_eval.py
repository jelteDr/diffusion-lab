"""Exp 6: Scheduler-Vergleich bei fester Guidance — Balkendiagramm, Bildraster, gepaarter Vergleich.

Aufruf:  uv run python scripts/plot_scheduler_eval.py exp6 [--reference euler] [--guidance 5]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from sdxl.clip_eval import paired_comparison
from sdxl.schedulers import SCHEDULERS, STOCHASTIC

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="exp6")
parser.add_argument("--reference", default="euler", help="Referenz-Scheduler für den gepaarten Vergleich")
parser.add_argument("--guidance", type=float, default=None, help="nötig, falls der Lauf mehrere w enthält")
args = parser.parse_args()

run_dir = Path("runs/sdxl") / args.name
evals = {p.stem.removeprefix("eval_clip_"): json.loads(p.read_text())["results"]
         for p in sorted(run_dir.glob("eval_clip_*.json"))}
if not evals:
    raise SystemExit(f"keine eval_clip_*.json in {run_dir}: zuerst `uv run python -m sdxl.clip_eval --name {args.name}`")
ws = sorted({r["guidance"] for rows in evals.values() for r in rows})
w = args.guidance if args.guidance is not None else ws[0]
if args.guidance is None and len(ws) > 1:
    raise SystemExit(f"Lauf enthält mehrere w {ws}, bitte --guidance angeben")
evals = {jk: [r for r in rows if r["guidance"] == w] for jk, rows in evals.items()}

primary = "L14" if "L14" in evals else next(iter(evals))
rows = evals[primary]
order = [s for s in SCHEDULERS if any(r["scheduler"] == s for r in rows)]   # feste Reihenfolge
prompts = sorted({r["prompt_key"] for r in rows})
get = lambda rs, s, p: next(r for r in rs if r["scheduler"] == s and r["prompt_key"] == p)

# ---------- 1) Balkendiagramm ----------
panels = [(jk, "clip_score", f"CLIP-Score ↑ (Richter {jk})") for jk in evals] + \
         [(primary, "diversity", "Vielfalt ↑"), (primary, "saturation", "Sättigung"),
          (primary, "seconds_mean", "Sekunden pro Bild ↓")]
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
x = np.arange(len(order))
width = 0.8 / len(prompts)
for ax, (jk, key, label) in zip(axes, panels):
    for i, pk in enumerate(prompts):
        vals = [get(evals[jk], s, pk)[key] or 0 for s in order]
        has_std = key + "_std" in get(evals[jk], order[0], pk)
        err = [get(evals[jk], s, pk)[key + "_std"] for s in order] if has_std else None
        bars = ax.bar(x + (i - (len(prompts) - 1) / 2) * width, vals, width, yerr=err, capsize=2, label=pk)
        for b, s in zip(bars, order):
            if s in STOCHASTIC:
                b.set_hatch("//")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(label); ax.grid(alpha=0.3, axis="y")
    if key == "clip_score":
        lo = min(get(evals[jk], s, p)[key] for s in order for p in prompts)
        ax.set_ylim(max(0, lo - 6), None)
axes[0].legend(fontsize=7)
fig.suptitle(f"SDXL Exp 6 ({args.name}): Scheduler bei w={w:g}, schraffiert = stochastisch")
fig.tight_layout()
out_bars = Path("docs/img") / f"sdxl_{args.name}_schedulers.png"
fig.savefig(out_bars, dpi=110)
print("gespeichert:", out_bars)

# ---------- 2) Bildraster: Zeile = Scheduler, Spalte = Prompt, Seed 0 ----------
thumb, pad_left, pad_top = 256, 130, 24
grid = Image.new("RGB", (pad_left + len(prompts) * thumb, pad_top + len(order) * thumb), "white")
draw = ImageDraw.Draw(grid)
for c, pk in enumerate(prompts):
    draw.text((pad_left + c * thumb + 8, 6), pk, fill="black")
for r_i, s in enumerate(order):
    draw.text((8, pad_top + r_i * thumb + thumb // 2), s + (" *" if s in STOCHASTIC else ""), fill="black")
    for c, pk in enumerate(prompts):
        f = run_dir / pk / f"{s}_w{w:g}_s0.png"
        if f.exists():
            grid.paste(Image.open(f).convert("RGB").resize((thumb, thumb), Image.LANCZOS),
                       (pad_left + c * thumb, pad_top + r_i * thumb))
out_grid = Path("docs/img") / f"sdxl_{args.name}_grid.png"
grid.save(out_grid)
print("gespeichert:", out_grid)

# ---------- 3) Gepaarter Vergleich gegen die Referenz ----------
if args.reference in order and len(order) > 1:
    for jk, rs in evals.items():
        ref = [v for pk in prompts for v in get(rs, args.reference, pk)["clip_per_seed"]]
        print(f"\nGepaart gegen {args.reference}, Richter {jk} ({len(ref)} Paare aus Prompt × Seed):")
        print(f"  {'Scheduler':14s} {'Δ CLIP':>7s}   besser/gleich/schlechter")
        for s in order:
            if s == args.reference:
                continue
            sc = [v for pk in prompts for v in get(rs, s, pk)["clip_per_seed"]]
            res = paired_comparison(sc, ref)
            print(f"  {s:14s} {res['mean_diff']:+7.2f}   {res['wins']:>2d} / {res['ties']:>2d} / {res['losses']:>2d}")
