"""Training des DDPM auf MNIST.

Aufruf:  uv run python -m ddpm.train --epochs 20 --run mnist_base

Trainingsziel (Ho et al. 2020, "simple loss"): Für jedes Bild x_0 ein zufälliges t
und Rauschen ε ziehen, x_t bilden, dem Netz (x_t, t) geben und den mittleren
quadratischen Fehler zwischen ε und der Vorhersage ε̂ minimieren.
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ddpm.data import mnist_loader
from ddpm.model import UNet, count_params
from ddpm.plotting import plot_loss
from ddpm.schedule import SCHEDULES, NoiseSchedule, make_schedule, q_sample


def training_step(model: UNet, schedule: NoiseSchedule, x0: torch.Tensor) -> torch.Tensor:
    """Ein DDPM-Trainingsschritt. Gibt die Loss (Skalar) zurück.

    Args:
        model:    das U-Net, sagt aus (x_t, t) das Rauschen voraus
        schedule: Noise-Schedule, liegt bereits auf dem Device von x0
        x0:       Batch sauberer Bilder, Form (B, 1, 32, 32), Werte in [-1, 1]
    """
    # 1. t: für jedes Bild ein zufälliger Zeitschritt in [0, schedule.num_steps - 1],
    #    Form (B,), auf x0.device
    # 2. noise: Standard-Gauß-Rauschen in der Form von x0
    # 3. x_t über q_sample bilden
    # 4. Vorhersage des Modells holen und die MSE-Loss zwischen noise und Vorhersage zurückgeben
    batch_size = x0.shape[0]
    t = torch.randint(0, schedule.num_steps, (batch_size,), device=x0.device)  # zufälliges t pro Bild
    noise = torch.randn_like(x0)
    x_t = q_sample(schedule, x0, t, noise)
    noise_pred = model(x_t, t)
    return F.mse_loss(noise_pred, noise)


class EMA:
    """Exponentiell gleitender Durchschnitt der Modellgewichte.

    DDPMs werden fast immer mit den EMA-Gewichten gesampelt: Sie glätten das
    Zittern der letzten Optimierungsschritte und liefern sichtbar sauberere Bilder.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for p_ema, p in zip(self.shadow.parameters(), model.parameters()):
            p_ema.lerp_(p, 1.0 - self.decay)  # p_ema = decay * p_ema + (1 - decay) * p


def save_checkpoint(path: Path, model: UNet, ema: EMA, config: dict, epoch: int) -> None:
    torch.save(
        {"model": model.state_dict(), "ema": ema.shadow.state_dict(), "config": config, "epoch": epoch},
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DDPM auf MNIST trainieren")
    parser.add_argument("--run", default="mnist_base", help="Name des Laufs (Ordner unter runs/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=1000, help="T, Anzahl Diffusionsschritte")
    parser.add_argument("--schedule", choices=list(SCHEDULES), default="linear", help="Noise-Schedule")
    parser.add_argument("--base", type=int, default=32, help="Basis-Kanalzahl des U-Nets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="vorhandenen Lauf gleichen Namens überschreiben")
    args = parser.parse_args()

    run_dir = Path("runs") / args.run
    if (run_dir / "ckpt.pt").exists() and not args.overwrite:
        raise SystemExit(f"Lauf '{args.run}' existiert schon ({run_dir}). "
                         f"Anderen Namen mit --run wählen oder --overwrite setzen.")

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_dir.mkdir(parents=True, exist_ok=True)
    loader = mnist_loader(args.batch_size)
    config = vars(args) | {"device": device, "steps_per_epoch": len(loader)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    schedule = make_schedule(args.schedule, args.steps).to(device)
    model = UNet(base=args.base).to(device)
    ema = EMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Lauf {args.run} auf {device}: {count_params(model):,} Parameter, "
          f"{len(loader)} Batches/Epoche, T={args.steps}, Schedule {args.schedule}")

    loss_log = open(run_dir / "loss.csv", "w", newline="")
    writer = csv.writer(loss_log)
    writer.writerow(["epoch", "step", "loss"])

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoche {epoch}/{args.epochs}", leave=False, unit="batch")
        for i, (x0, _label) in enumerate(pbar, start=1):  # Labels werden (noch) nicht gebraucht
            x0 = x0.to(device)
            loss = training_step(model, schedule, x0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update(model)

            step += 1
            running += loss.item()
            if step % 50 == 0:
                writer.writerow([epoch, step, loss.item()])
            if i % 10 == 0:
                pbar.set_postfix(loss=f"{running / i:.4f}")  # laufender Mittelwert der Epoche

        avg = running / len(loader)
        print(f"Epoche {epoch:3d}/{args.epochs}  Loss {avg:.4f}  ({time.time() - t0:.0f} s)")
        save_checkpoint(run_dir / "ckpt.pt", model, ema, config, epoch)
        loss_log.flush()
        plot_loss(run_dir)  # Kurve nach jeder Epoche aktualisieren

    loss_log.close()
    print(f"fertig, Checkpoint: {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
