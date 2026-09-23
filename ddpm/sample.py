"""Sampling (Rückwärtsprozess) eines trainierten DDPM.

Aufruf:  uv run python -m ddpm.sample --run mnist_base --n 64

Wir starten bei reinem Rauschen x_T ~ N(0, I) und gehen T Schritte rückwärts.
In jedem Schritt schätzt das Netz das Rauschen ε̂ in x_t, daraus berechnen wir den
Erwartungswert von x_{t-1} und addieren (außer im letzten Schritt) wieder etwas
frisches Rauschen. Das ist "ancestral sampling" aus Ho et al. 2020, Algorithmus 2:

    x_{t-1} = 1/sqrt(alpha_t) * ( x_t - beta_t / sqrt(1 - alpha_bar_t) * ε̂ ) + sigma_t * z
    mit z ~ N(0, I) für t > 0, z = 0 für t = 0, und sigma_t = sqrt(beta_t).
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from ddpm.data import to_image
from ddpm.model import UNet
from ddpm.schedule import NoiseSchedule, _gather, linear_schedule


@torch.no_grad()
def p_sample_step(model: UNet, schedule: NoiseSchedule, x_t: torch.Tensor, t: int) -> torch.Tensor:
    """Ein Rückwärtsschritt: aus x_t wird x_{t-1}. t ist für den ganzen Batch gleich.

    Args:
        model:    trainiertes U-Net (im eval-Modus)
        schedule: Noise-Schedule auf dem Device von x_t
        x_t:      aktueller Batch, Form (B, 1, 32, 32)
        t:        aktueller Zeitschritt als int, T-1 .. 0

    Returns:
        x_{t-1}, gleiche Form wie x_t
    """
    batch_size = x_t.shape[0]
    t_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
    beta_t = _gather(schedule.betas, t_batch, x_t.shape)
    alpha_t = _gather(schedule.alphas, t_batch, x_t.shape)
    alpha_bar_t = _gather(schedule.alphas_cumprod, t_batch, x_t.shape)

    eps_hat = model(x_t, t_batch)
    mean = (x_t - beta_t / (1.0 - alpha_bar_t).sqrt() * eps_hat) / alpha_t.sqrt()
    if t == 0:
        return mean
    noise = torch.randn_like(x_t)
    return mean + beta_t.sqrt() * noise


@torch.no_grad()
def sample(
    model: UNet,
    schedule: NoiseSchedule,
    n: int,
    device: str,
    keep_every: int | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Erzeugt n Bilder aus reinem Rauschen.

    Returns:
        (x_0, trajectory): fertige Bilder in [-1, 1] und, falls keep_every gesetzt,
        Zwischenstände alle keep_every Schritte (Liste von (n, 1, 32, 32)-Tensoren).
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    x = torch.randn(n, 1, 32, 32, device=device, generator=generator)

    trajectory: list[torch.Tensor] = []
    steps = list(range(schedule.num_steps - 1, -1, -1))  # T-1, ..., 0
    for t in tqdm(steps, desc="Sampling", unit="Schritt", leave=False):
        if keep_every and t % keep_every == 0:
            trajectory.append(x.cpu())
        x = p_sample_step(model, schedule, x, t)
    trajectory.append(x.cpu())
    return x, trajectory


def load_run(run: str, device: str, use_ema: bool = True) -> tuple[UNet, NoiseSchedule, dict]:
    """Lädt Modell (EMA- oder Roh-Gewichte) und Schedule eines Laufs."""
    run_dir = Path("runs") / run
    ckpt = torch.load(run_dir / "ckpt.pt", map_location=device)
    config = ckpt["config"]
    model = UNet(base=config["base"]).to(device)
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.eval()
    schedule = linear_schedule(config["steps"]).to(device)
    return model, schedule, config


def save_trajectory_grid(trajectory: list[torch.Tensor], path: Path, n_show: int = 8) -> None:
    """Zeilen = einzelne Bilder, Spalten = Zeitpunkte von x_T (links) bis x_0 (rechts)."""
    frames = torch.stack([tr[:n_show] for tr in trajectory], dim=1)  # (n_show, n_frames, 1, 32, 32)
    grid = make_grid(to_image(frames.flatten(0, 1)), nrow=frames.shape[1], padding=1)
    save_image(grid, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bilder aus einem trainierten DDPM sampeln")
    parser.add_argument("--run", default="mnist_base")
    parser.add_argument("--n", type=int, default=64, help="Anzahl Bilder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ema", action="store_true", help="Roh- statt EMA-Gewichte nutzen")
    parser.add_argument("--keep-every", type=int, default=100, help="Zwischenstand alle k Schritte")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, schedule, config = load_run(args.run, device, use_ema=not args.no_ema)
    print(f"Lauf {args.run} (Epoche {config.get('epochs')}, T={schedule.num_steps}) auf {device}, "
          f"{'EMA' if not args.no_ema else 'Roh'}-Gewichte")

    x0, trajectory = sample(model, schedule, args.n, device, keep_every=args.keep_every, seed=args.seed)

    run_dir = Path("runs") / args.run
    suffix = "" if not args.no_ema else "_raw"
    grid = make_grid(to_image(x0), nrow=int(args.n ** 0.5), padding=1)
    save_image(grid, run_dir / f"samples{suffix}.png")
    save_trajectory_grid(trajectory, run_dir / f"trajectory{suffix}.png")
    print(f"gespeichert: {run_dir / f'samples{suffix}.png'}, {run_dir / f'trajectory{suffix}.png'}")


if __name__ == "__main__":
    main()
