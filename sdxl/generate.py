"""Bilder mit SDXL lokal erzeugen, optional als Vergleichsraster über Scheduler und Guidance.

Beispiele:
  uv run python -m sdxl.generate --prompt "a red fox in a snowy forest, photo" --name fox
  uv run python -m sdxl.generate --prompt "..." --name sweep --guidance 1 3 5 7.5 12 --scheduler euler euler_a

Ausgabe unter runs/sdxl/<name>/: ein PNG pro Kombination, grid.png (Zeilen = Scheduler,
Spalten = Guidance), meta.json mit Zeit, Speicher und allen Parametern.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sdxl.pipeline import MemoryTracker, device_and_dtype, load_pipeline, param_counts
from sdxl.schedulers import SCHEDULERS, STOCHASTIC, make_scheduler


def make_grid(images: list[list[Image.Image]], row_labels: list[str], col_labels: list[str],
              thumb: int = 384) -> Image.Image:
    """Raster mit Beschriftung: images[zeile][spalte]."""
    pad_left, pad_top = 150, 30
    rows, cols = len(images), len(images[0])
    grid = Image.new("RGB", (pad_left + cols * thumb, pad_top + rows * thumb), "white")
    draw = ImageDraw.Draw(grid)
    for c, label in enumerate(col_labels):
        draw.text((pad_left + c * thumb + 8, 8), label, fill="black")
    for r, label in enumerate(row_labels):
        draw.text((8, pad_top + r * thumb + thumb // 2), label, fill="black")
        for c, img in enumerate(images[r]):
            grid.paste(img.resize((thumb, thumb), Image.LANCZOS), (pad_left + c * thumb, pad_top + r * thumb))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="SDXL lokal: Bilder erzeugen und vergleichen")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="", help="Negativ-Prompt (was NICHT im Bild sein soll)")
    parser.add_argument("--name", default="test", help="Ordner unter runs/sdxl/")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, nargs="+", default=[5.0], help="CFG-Stärke w, mehrere möglich")
    parser.add_argument("--scheduler", nargs="+", choices=list(SCHEDULERS), default=["euler"])
    parser.add_argument("--eta", type=float, default=0.0, help="nur DDIM: 0 = deterministisch, 1 = stochastisch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=1024, help="Kantenlänge, SDXL ist auf 1024 trainiert")
    parser.add_argument("--attention-slicing", action="store_true", help="weniger Speicher, langsamer")
    args = parser.parse_args()

    out_dir = Path("runs/sdxl") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = device_and_dtype()

    t0 = time.perf_counter()
    pipe = load_pipeline(device, dtype, args.attention_slicing)
    load_s = time.perf_counter() - t0
    params = param_counts(pipe)
    print(f"Pipeline geladen in {load_s:.0f} s auf {device} ({dtype}), Parameter [M]: {params}")
    base_config = pipe.scheduler.config
    tracker = MemoryTracker(device)

    results, grid_rows = [], []
    for sched in args.scheduler:
        pipe.scheduler = make_scheduler(sched, base_config)
        row = []
        for w in args.guidance:
            # CPU-Generator: auf MPS reproduzierbar und geräteunabhängig (Empfehlung von diffusers)
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            tracker.reset()
            t1 = time.perf_counter()
            image = pipe(
                prompt=args.prompt, negative_prompt=args.negative or None,
                num_inference_steps=args.steps, guidance_scale=w,
                height=args.size, width=args.size, generator=generator, eta=args.eta,
                callback_on_step_end=tracker,
            ).images[0]
            seconds = time.perf_counter() - t1

            pixels = np.asarray(image)
            if pixels.max() == 0:
                print("  WARNUNG: schwarzes Bild (NaN in der VAE?)")
            fname = f"{sched}_w{w:g}.png"
            image.save(out_dir / fname)
            row.append(image)
            step_s = float(np.median(tracker.step_times[1:])) if len(tracker.step_times) > 1 else None
            r = {"scheduler": sched, "stochastic": sched in STOCHASTIC, "guidance": w,
                 "seconds": round(seconds, 1), "s_per_step": round(step_s, 2) if step_s else None,
                 "peak_gb": round(tracker.peak_bytes / 1e9, 2), "file": fname}
            results.append(r)
            print(f"  {sched:13s} w={w:<5g} {seconds:6.1f} s  ({r['s_per_step']} s/Schritt)  "
                  f"Spitze {r['peak_gb']} GB  -> {out_dir / fname}")
        grid_rows.append(row)

    if len(results) > 1:
        grid = make_grid(grid_rows, args.scheduler, [f"w = {w:g}" for w in args.guidance])
        grid.save(out_dir / "grid.png")
        print("Raster:", out_dir / "grid.png")

    meta = {"prompt": args.prompt, "negative": args.negative, "steps": args.steps, "seed": args.seed,
            "size": args.size, "eta": args.eta, "device": device, "dtype": str(dtype),
            "load_seconds": round(load_s, 1), "params_m": params, "results": results}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
