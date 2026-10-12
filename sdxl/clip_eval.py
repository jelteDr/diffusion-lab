"""CLIP-Auswertung für Exp 5: Prompt-Treue, Vielfalt und Sättigung pro Guidance-Stärke.

Aufruf:  uv run python -m sdxl.clip_eval --name exp5

Kennzahlen pro Gruppe (Prompt × Scheduler × Guidance, Mittel über die Seeds):
- CLIP-Score: 100 · max(cos(Bild, Text), 0) — wie gut passt das Bild zum Prompt
  (Definition wie torchmetrics / Hessel et al. 2021). Höher = treuer.
- Vielfalt: wie unterschiedlich die Bilder verschiedener Seeds im CLIP-Bildraum sind.
  Höher = vielfältiger. Gegenstück zum FID-Vielfaltsaspekt aus Exp 4.
- Sättigung: mittlere HSV-Sättigung der Pixel in [0, 1]. Misst das "Übersättigen"
  bei hoher Guidance ohne neuronalen Richter.

Zwei Richter, weil der SDXL-Text-Encoder selbst ein CLIP ViT-L/14 ist: Der Standard-Richter
ist mit dem Generator verwandt. Der LAION-Richter hat andere Gewichte und Trainingsdaten.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from sdxl.pipeline import device_and_dtype

JUDGES = {
    "L14": "openai/clip-vit-large-patch14",           # Standard in der Literatur, 1,7 GB
    "B32-laion": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",  # unabhängige Gegenprobe, 0,6 GB
}


class ClipJudge:
    def __init__(self, repo: str, device: str):
        self.device = device
        self.model = CLIPModel.from_pretrained(repo, use_safetensors=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(repo)

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        # je nach transformers-Version Tensor oder Ausgabe-Objekt mit pooler_output
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image], batch: int = 16) -> torch.Tensor:
        """L2-normierte Bild-Embeddings, Form (N, D), auf der CPU."""
        outs = []
        for i in range(0, len(images), batch):
            inputs = self.processor(images=images[i : i + batch], return_tensors="pt").to(self.device)
            outs.append(self._as_tensor(self.model.get_image_features(**inputs)).float().cpu())
        emb = torch.cat(outs)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """L2-normierte Text-Embeddings, Form (N, D), auf der CPU."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        emb = self._as_tensor(self.model.get_text_features(**inputs)).float().cpu()
        return emb / emb.norm(dim=-1, keepdim=True)


def clip_score(img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
    """Pro Bild: 100 · max(cos, 0). img_emb (N, D), txt_emb (D,) — beide normiert."""
    return 100 * (img_emb @ txt_emb).clamp_min(0)


def diversity(img_emb: torch.Tensor) -> float:
    """Mittlere paarweise Kosinus-Distanz zwischen den Bildern einer Gruppe.

    Args:
        img_emb: L2-normierte Bild-Embeddings der Seeds einer Gruppe, Form (N, D), N >= 2

    Returns:
        Mittelwert von (1 - cos) über alle Paare i != j. 0 = alle Bilder identisch.
    """
    n = img_emb.shape[0]
    sim = img_emb @ img_emb.T
    mask = ~torch.eye(n, dtype=torch.bool)
    return float((1.0 - sim[mask]).mean())


def saturation(image: Image.Image) -> float:
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    return float(hsv[..., 1].mean())


def evaluate(run_dir: Path, judge_key: str, device: str) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    records = [json.loads(l) for l in (run_dir / "meta.jsonl").read_text().splitlines() if l.strip()]
    # Doppelte Einträge (z. B. nach Neustart) auf die letzte Zeile pro Datei reduzieren
    records = list({r["file"]: r for r in records}.values())
    judge = ClipJudge(JUDGES[judge_key], device)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["prompt_key"], r["scheduler"], r["guidance"])].append(r)

    text_emb = {pk: judge.embed_texts([p])[0] for pk, p in config["prompts"].items()}
    results = []
    for (pk, sched, w), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        images = [Image.open(run_dir / r["file"]).convert("RGB") for r in rs]
        emb = judge.embed_images(images)
        scores = clip_score(emb, text_emb[pk])
        results.append({
            "prompt_key": pk, "scheduler": sched, "guidance": w, "n_seeds": len(rs),
            "clip_score": round(float(scores.mean()), 2),
            "clip_score_std": round(float(scores.std()), 2) if len(rs) > 1 else 0.0,
            "clip_per_seed": [round(float(s), 2) for s in scores],
            "diversity": round(diversity(emb), 4) if len(rs) > 1 else None,
            "saturation": round(float(np.mean([saturation(im) for im in images])), 4),
            "seconds_mean": round(float(np.mean([r["seconds"] for r in rs])), 1),
        })
        print(f"  {pk:6s} {sched:7s} w={w:<4g} n={len(rs)}  CLIP {results[-1]['clip_score']:5.2f}  "
              f"Vielfalt {results[-1]['diversity']}  Sättigung {results[-1]['saturation']:.3f}")
    return {"judge": judge_key, "judge_repo": JUDGES[judge_key], "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP-Auswertung eines SDXL-Experiments")
    parser.add_argument("--name", default="exp5")
    parser.add_argument("--judges", nargs="+", choices=list(JUDGES), default=list(JUDGES))
    args = parser.parse_args()
    run_dir = Path("runs/sdxl") / args.name
    device, _ = device_and_dtype()
    for jk in args.judges:
        print(f"Richter {jk} ({JUDGES[jk]})")
        out = evaluate(run_dir, jk, device)
        path = run_dir / f"eval_clip_{jk}.json"
        path.write_text(json.dumps(out, indent=2))
        print("gespeichert:", path)


if __name__ == "__main__":
    main()
