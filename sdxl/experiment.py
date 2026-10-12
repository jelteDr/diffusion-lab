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
import shutil
import time
from pathlib import Path

import torch
from tqdm import tqdm

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


def reuse_images(src: Path, run_dir: Path, todo: list, args) -> list:
    """Übernimmt Bilder aus einem früheren Lauf, wenn Prompt-Text, Scheduler, w, Seed,
    Schritte und Größe identisch sind. Bei festem Seed rechnet SDXL deterministisch
    (CPU-Generator), ein Neuberechnen ergäbe dasselbe Bild. Gibt die verbleibenden Jobs zurück."""
    src_cfg = json.loads((src / "config.json").read_text())
    if src_cfg["steps"] != args.steps or src_cfg["size"] != args.size:
        print(f"Übernahme aus {src} nicht möglich: andere Schritte/Größe")
        return todo
    src_meta = {}
    for line in (src / "meta.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            src_meta[r["file"]] = r
    remaining, reused = [], 0
    with open(run_dir / "meta.jsonl", "a") as meta:
        for seed, pk, sched, w in todo:
            src_path = image_path(src, pk, sched, w, seed)
            same_prompt = src_cfg["prompts"].get(pk) == PROMPTS[pk]
            rel = str(src_path.relative_to(src))
            if same_prompt and src_path.exists() and rel in src_meta:
                reused += 1
                if not args.dry_run:
                    dst = image_path(run_dir, pk, sched, w, seed)
                    dst.parent.mkdir(exist_ok=True)
                    shutil.copy2(src_path, dst)
                    meta.write(json.dumps(src_meta[rel] | {"reused_from": src.name}) + "\n")
            else:
                remaining.append((seed, pk, sched, w))
    print(f"{'würde übernehmen' if args.dry_run else 'übernommen'}: {reused} Bilder aus {src.name}")
    return remaining


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 5: Guidance-Sweep über Prompts und Seeds")
    parser.add_argument("--name", default="exp5")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS))
    parser.add_argument("--guidance", type=float, nargs="+", default=[1, 3, 5, 7.5, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--scheduler", nargs="+", choices=list(SCHEDULERS), default=["euler"])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--reuse-from", default=None,
                        help="fertige Bilder mit identischen Einstellungen aus diesem Lauf übernehmen statt neu rechnen")
    parser.add_argument("--dry-run", action="store_true", help="nur planen und anzeigen, nichts erzeugen")
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

    if args.reuse_from:
        todo = reuse_images(Path("runs/sdxl") / args.reuse_from, run_dir, todo, args)

    print(f"{len(jobs)} Bilder geplant, {len(jobs) - len(todo)} schon fertig, {len(todo)} offen")
    if not todo or args.dry_run:
        return

    device, dtype = device_and_dtype()
    pipe = load_pipeline(device, dtype)
    # Schrittbalken von diffusers in die zweite Zeile, unter den Gesamtbalken
    pipe.set_progress_bar_config(position=1, leave=False, desc="  Schritte", ncols=90)
    base_config = pipe.scheduler.config
    tracker = MemoryTracker(device)
    current_sched = None
    t_start = time.perf_counter()

    # Gesamtbalken über ALLE Bilder des Experiments; bei Fortsetzung startet er beim erreichten Stand
    overall = tqdm(total=len(jobs), initial=len(jobs) - len(todo), desc="Gesamt", unit="Bild",
                   position=0, ncols=90, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]")

    with open(run_dir / "meta.jsonl", "a") as meta:
        for i, (seed, pk, sched, w) in enumerate(todo, start=1):
            overall.set_description(f"Gesamt ({pk}, w={w:g}, Seed {seed})")
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
            # tqdm.write statt print: schreibt oberhalb der Balken, ohne sie zu zerreißen
            tqdm.write(f"[{i}/{len(todo)}] {pk:6s} {sched} w={w:<4g} seed {seed}: {seconds:5.0f} s   "
                       f"Rest ca. {eta_min:.0f} min")
            overall.update(1)
    overall.close()
    print(f"fertig: {len(jobs)} Bilder unter {run_dir}")


if __name__ == "__main__":
    main()
