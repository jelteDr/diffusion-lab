"""Exp 5: SDXL über mehrere Prompts, Guidance-Stärken und Seeds — fortsetzbar für Nachtläufe.

Aufruf:  uv run python -m sdxl.experiment --name exp5
         (abbrechen jederzeit möglich; erneuter Aufruf überspringt fertige Bilder)

Ablage:  runs/sdxl/<name>/<prompt_key>/<scheduler>_w<w>_s<seed>.png
         runs/sdxl/<name>/meta.jsonl   (eine Zeile pro Bild: Zeit, Speicher, Parameter)
         runs/sdxl/<name>/config.json  (Prompts und Einstellungen)

Schleifenreihenfolge: Seed ganz außen. Nach jedem Seed-Durchlauf liegt für jede Kombination
aus Prompt, Scheduler und Guidance mindestens ein Bild vor; ein abgebrochener Lauf ist
also schon auswertbar, nur mit weniger Seeds.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from sdxl.pipeline import MemoryTracker, device_and_dtype, load_pipeline
from sdxl.schedulers import SCHEDULERS, make_scheduler

# Drei bewusst verschiedene Prompts:
#  fox    — Foto, ein Motiv, mehrere Attribute (Schnee, Gegenlicht)
#  cabin  — Szene + Malstil (prüft, ob der Stil getroffen wird)
#  shapes — kompositionell: Farben, Formen, räumliche Beziehung (schwer für Diffusionsmodelle)
PROMPTS = {
    "fox": "a red fox sitting in a snowy forest, golden hour, photo",
    "cabin": "a cozy wooden cabin by a lake at dusk, warm light in the windows, oil painting",
    "shapes": "a red cube on top of a blue sphere, next to a green pyramid, studio lighting",
}


def image_path(run_dir: Path, prompt_key: str, sched: str, w: float, seed: int) -> Path:
    return run_dir / prompt_key / f"{sched}_w{w:g}_s{seed}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 5: Guidance-Sweep über Prompts und Seeds")
    parser.add_argument("--name", default="exp5")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS))
    parser.add_argument("--guidance", type=float, nargs="+", default=[1, 3, 5, 7.5, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--scheduler", nargs="+", choices=list(SCHEDULERS), default=["euler"])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--size", type=int, default=768)
    args = parser.parse_args()

    run_dir = Path("runs/sdxl") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"prompts": {k: PROMPTS[k] for k in args.prompts}, "guidance": args.guidance,
              "seeds": args.seeds, "schedulers": args.scheduler, "steps": args.steps, "size": args.size}
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f"{config_path} existiert mit anderen Einstellungen. Anderen --name wählen.")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    jobs = [(seed, pk, sched, w) for seed in args.seeds for pk in args.prompts
            for sched in args.scheduler for w in args.guidance]
    todo = [j for j in jobs if not image_path(run_dir, j[1], j[2], j[3], j[0]).exists()]
    print(f"{len(jobs)} Bilder geplant, {len(jobs) - len(todo)} schon fertig, {len(todo)} offen")
    if not todo:
        return

    device, dtype = device_and_dtype()
    pipe = load_pipeline(device, dtype)
    base_config = pipe.scheduler.config
    tracker = MemoryTracker(device)
    current_sched = None
    t_start = time.perf_counter()

    with open(run_dir / "meta.jsonl", "a") as meta:
        for i, (seed, pk, sched, w) in enumerate(todo, start=1):
            if sched != current_sched:
                pipe.scheduler = make_scheduler(sched, base_config)
                current_sched = sched
            generator = torch.Generator(device="cpu").manual_seed(seed)
            tracker.reset()
            t1 = time.perf_counter()
            image = pipe(prompt=PROMPTS[pk], num_inference_steps=args.steps, guidance_scale=w,
                         height=args.size, width=args.size, generator=generator,
                         callback_on_step_end=tracker).images[0]
            seconds = time.perf_counter() - t1

            path = image_path(run_dir, pk, sched, w, seed)
            path.parent.mkdir(exist_ok=True)
            image.save(path)
            meta.write(json.dumps({"prompt_key": pk, "scheduler": sched, "guidance": w, "seed": seed,
                                   "seconds": round(seconds, 1), "peak_gb": round(tracker.peak_bytes / 1e9, 2),
                                   "file": str(path.relative_to(run_dir))}) + "\n")
            meta.flush()

            elapsed = time.perf_counter() - t_start
            eta_min = elapsed / i * (len(todo) - i) / 60
            print(f"[{i}/{len(todo)}] {pk:6s} {sched} w={w:<4g} seed {seed}: {seconds:5.0f} s   "
                  f"Rest ca. {eta_min:.0f} min", flush=True)


if __name__ == "__main__":
    main()
