"""Bewertet generierte Bilder eines Laufs mit dem MNIST-Klassifikator.

Aufruf:  uv run python -m ddpm.evaluate --run mnist_base --n 1000

Zwei Kennzahlen, beide Analogien zu den Standardmaßen der Bildgenerierung:

- IS (Inception Score, hier "Classifier Score"): exp( E_x[ KL( p(y|x) || p(y) ) ] ).
  Hoch, wenn jedes Bild eindeutig einer Ziffer zugeordnet wird (p(y|x) spitz) UND
  alle Ziffern vorkommen (p(y) flach). Maximum = 10 bei zehn Klassen.
- FID (Fréchet-Distanz): Abstand zwischen der Merkmalsverteilung generierter und
  echter Bilder (Mittelwert + Kovarianz der 128 Merkmale). Niedrig = besser, 0 = identisch.

Dazu: Klassen-Histogramm, mittlere Konfidenz und Sampling-Zeit.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ddpm.classifier import DigitClassifier, load_classifier
from ddpm.data import mnist_loader
from ddpm.sample import ddim_sample, load_run, sample

REAL_STATS_PATH = Path("runs") / "real_stats.pt"


@torch.no_grad()
def classifier_outputs(clf: DigitClassifier, x: torch.Tensor, batch: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    """Gibt (Wahrscheinlichkeiten (N, 10), Merkmale (N, 128)) zurück, alles auf CPU."""
    probs, feats = [], []
    for i in range(0, x.shape[0], batch):
        xb = x[i : i + batch]
        f = clf.features(xb)
        probs.append(F.softmax(clf.fc_out(f), dim=1).cpu())
        feats.append(f.cpu())
    return torch.cat(probs), torch.cat(feats)


def inception_score(probs: torch.Tensor) -> float:
    p_y = probs.mean(0, keepdim=True)
    kl = (probs * (probs.clamp_min(1e-12).log() - p_y.clamp_min(1e-12).log())).sum(1)
    return float(kl.mean().exp())


def frechet_distance(mu1: torch.Tensor, s1: torch.Tensor, mu2: torch.Tensor, s2: torch.Tensor) -> float:
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 * sqrt(S1 S2)), in float64 auf der CPU."""
    diff = (mu1 - mu2).double()
    s1, s2 = s1.double(), s2.double()
    # Tr(sqrt(S1 S2)) = Summe der Wurzeln der Eigenwerte von S1 S2 (reell und >= 0 für PSD-Matrizen)
    eig = torch.linalg.eigvals(s1 @ s2).real.clamp_min(0)
    return float(diff.dot(diff) + torch.trace(s1) + torch.trace(s2) - 2 * eig.sqrt().sum())


def feature_stats(feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return feats.mean(0), torch.cov(feats.T)


def real_stats(clf: DigitClassifier, device: str, n: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
    """Merkmalsstatistik echter Test-Ziffern, einmal berechnet und gecacht."""
    if REAL_STATS_PATH.exists():
        d = torch.load(REAL_STATS_PATH)
        return d["mu"], d["sigma"]
    xs = []
    for x, _ in mnist_loader(1000, train=False):
        xs.append(x)
        if sum(t.shape[0] for t in xs) >= n:
            break
    x = torch.cat(xs)[:n].to(device)
    _, feats = classifier_outputs(clf, x)
    mu, sigma = feature_stats(feats)
    torch.save({"mu": mu, "sigma": sigma, "n": n}, REAL_STATS_PATH)
    return mu, sigma


def evaluate_images(x: torch.Tensor, clf: DigitClassifier, device: str, y: torch.Tensor | None = None) -> dict:
    probs, feats = classifier_outputs(clf, x)
    mu_r, sigma_r = real_stats(clf, device)
    mu_g, sigma_g = feature_stats(feats)
    pred = probs.argmax(1)
    hist = torch.bincount(pred, minlength=10)
    result = {
        "n": int(x.shape[0]),
        "inception_score": round(inception_score(probs), 3),
        "fid": round(frechet_distance(mu_g, sigma_g, mu_r, sigma_r), 3),
        "mean_confidence": round(float(probs.max(1).values.mean()), 4),
        "class_histogram": hist.tolist(),
    }
    if y is not None:
        # Konditionierung: Wie oft erkennt der Richter die angeforderte Ziffer?
        result["label_accuracy"] = round(float((pred == y.cpu()).float().mean()), 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generierte Bilder eines Laufs bewerten")
    parser.add_argument("--run", default="mnist_base")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=500, help="Bilder pro Sampling-Durchgang")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="mehrere Seeds, z. B. --seeds 0 1 2 -> Mittelwert ± Streuung")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG-Stärke w bei konditioniertem Modell")
    parser.add_argument("--uncond", action="store_true", help="konditioniertes Modell OHNE Labels sampeln")
    parser.add_argument("--tag", default=None, help="Name der Auswertung (Dateiname eval_<tag>.json)")
    parser.add_argument("--real", action="store_true",
                        help="Referenz: echte Trainingsbilder statt Samples bewerten (Bestwert der Metriken)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clf = load_classifier(device)

    if args.real:
        torch.manual_seed(args.seed)
        x = torch.cat([xb for xb, _ in mnist_loader(args.n, train=True)][:1])[: args.n].to(device)
        result = {"run": "real_mnist", "sampler": "none"} | evaluate_images(x, clf, device)
        out = Path("runs") / "eval_real.json"
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print("gespeichert:", out)
        return

    model, schedule, config = load_run(args.run, device)
    seeds = args.seeds if args.seeds else [args.seed]
    conditional = model.num_classes is not None and not args.uncond
    tag = args.tag or (f"ddim{args.ddim_steps}" if args.sampler == "ddim" else "ddpm")
    if model.num_classes is not None and not args.tag:
        tag += "_uncond" if args.uncond else f"_w{args.guidance:g}"
    # Konditioniert: Labels gleichverteilt 0..9 (i-tes Bild bekommt Ziffer i mod 10)
    y_all = torch.arange(args.n, device=device) % 10 if conditional else None

    def generate(seed: int) -> tuple[torch.Tensor, float]:
        t0 = time.time()
        xs = []
        for i in range(0, args.n, args.batch):
            n_b = min(args.batch, args.n - i)
            y_b = y_all[i : i + n_b] if conditional else None
            if args.sampler == "ddim":
                x, _ = ddim_sample(model, schedule, n_b, device, num_steps=args.ddim_steps,
                                   eta=args.eta, seed=seed * 100_000 + i, y=y_b, guidance_scale=args.guidance)
            else:
                x, _ = sample(model, schedule, n_b, device, seed=seed * 100_000 + i, y=y_b,
                              guidance_scale=args.guidance)
            xs.append(x)
        return torch.cat(xs), time.time() - t0

    per_seed = []
    for seed in seeds:
        x, seconds = generate(seed)
        r = {"seed": seed, "seconds": round(seconds, 1), "ms_per_image": round(1000 * seconds / args.n, 1)}
        r |= evaluate_images(x, clf, device, y_all)
        per_seed.append(r)
        acc = f"  Label-Treffer {r['label_accuracy']:.3f}" if "label_accuracy" in r else ""
        print(f"Seed {seed}: IS {r['inception_score']:.3f}  FID {r['fid']:.2f}{acc}  ({r['ms_per_image']} ms/Bild)")

    def mean_std(key: str) -> dict:
        vals = torch.tensor([r[key] for r in per_seed], dtype=torch.float64)
        return {"mean": round(float(vals.mean()), 3), "std": round(float(vals.std()) if len(vals) > 1 else 0.0, 3)}

    result = {"run": args.run, "schedule": config.get("schedule", "linear"), "sampler": args.sampler,
              "steps": args.ddim_steps if args.sampler == "ddim" else schedule.num_steps,
              "eta": args.eta if args.sampler == "ddim" else None, "n": args.n, "seeds": seeds,
              "inception_score": mean_std("inception_score"), "fid": mean_std("fid"),
              "mean_confidence": mean_std("mean_confidence"), "ms_per_image": mean_std("ms_per_image"),
              "conditional": conditional, "guidance": args.guidance if conditional else None,
              "label_accuracy": mean_std("label_accuracy") if conditional else None,
              "per_seed": per_seed}
    out = Path("runs") / args.run / f"eval_{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("gespeichert:", out)


if __name__ == "__main__":
    main()
