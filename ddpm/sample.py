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
from torch._dynamo.polyfills import torch_c_nn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from ddpm.data import to_image
from ddpm.model import UNet
from ddpm.schedule import NoiseSchedule, _gather, make_schedule


@torch.no_grad()
def predict_eps(
    model: UNet, x_t: torch.Tensor, t_batch: torch.Tensor,
    y: torch.Tensor | None = None, guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Rauschschätzung ε̂, optional mit Classifier-free Guidance (Ho & Salimans 2022).

    Ohne Label (y=None) oder bei guidance_scale == 1: ein normaler Modellaufruf.
    Sonst zwei Vorhersagen, konditioniert (mit y) und unkonditioniert (Null-Label), die
    linear extrapoliert werden:

        ε̂ = ε̂_uncond + w * (ε̂_cond - ε̂_uncond)

    w = 1 ergibt die reine konditionierte Vorhersage, w > 1 verstärkt die Richtung
    "weg vom Unkonditionierten, hin zur Klasse" (schärfere, klassentypischere Bilder,
    bei zu großem w weniger Vielfalt und Artefakte). w = 0 ist unkonditioniert.
    """
    if y is None or guidance_scale == 1.0:
        return model(x_t, t_batch, y)

    # 1. y_null = Null-Label für den ganzen Batch: torch.full_like(y, model.num_classes)
    # 2. Beide Vorhersagen in EINEM Modellaufruf: x_t, t_batch und (y, y_null) jeweils
    #    mit torch.cat entlang dim=0 verdoppeln, dann model(...) aufrufen und das
    #    Ergebnis mit .chunk(2) in eps_cond, eps_uncond aufteilen
    # 3. Formel aus dem Docstring zurückgeben
    y_null = torch.full_like(y,model.num_classes)
    eps_both = model(torch.cat([x_t, x_t]), torch.cat([t_batch, t_batch]), torch.cat([y, y_null]))
    eps_cond, eps_uncond = eps_both.chunk(2)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)



@torch.no_grad()
def p_sample_step(
    model: UNet, schedule: NoiseSchedule, x_t: torch.Tensor, t: int,
    y: torch.Tensor | None = None, guidance_scale: float = 1.0,
) -> torch.Tensor:
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

    eps_hat = predict_eps(model, x_t, t_batch, y, guidance_scale)
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
    y: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Erzeugt n Bilder aus reinem Rauschen (optional klassenkonditioniert mit Labels y).

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
        x = p_sample_step(model, schedule, x, t, y, guidance_scale)
    trajectory.append(x.cpu())
    return x, trajectory


def ddim_timesteps(num_train_steps: int, num_sample_steps: int) -> list[int]:
    """Teilfolge der Trainings-Zeitschritte, absteigend, z. B. T=1000, 50 Schritte:
    [999, 979, 959, ..., 19]. Der Sprung geht jeweils zum nächsten Eintrag, zuletzt zu -1 (= x_0)."""
    stride = num_train_steps // num_sample_steps
    return list(range(num_train_steps - 1, -1, -stride))[:num_sample_steps]


@torch.no_grad()
def ddim_step(
    model: UNet, schedule: NoiseSchedule, x_t: torch.Tensor, t: int, t_prev: int, eta: float = 0.0,
    y: torch.Tensor | None = None, guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Ein DDIM-Schritt (Song et al. 2021) von t nach t_prev, wobei t_prev < t beliebig weit
    entfernt sein darf. t_prev = -1 bedeutet: direkt nach x_0 (alpha_bar_prev = 1).

    Idee: Aus ε̂ zuerst das saubere Bild x̂_0 schätzen, dann x_{t_prev} so konstruieren,
    als hätte man x̂_0 mit dem Vorwärtsprozess bis t_prev verrauscht, aber mit ε̂ statt
    frischem Rauschen (deterministisch bei eta = 0):

        x̂_0      = (x_t - sqrt(1 - ᾱ_t) * ε̂) / sqrt(ᾱ_t)
        σ        = eta * sqrt((1 - ᾱ_prev) / (1 - ᾱ_t)) * sqrt(1 - ᾱ_t / ᾱ_prev)
        x_{prev} = sqrt(ᾱ_prev) * x̂_0 + sqrt(1 - ᾱ_prev - σ²) * ε̂ + σ * z

    eta = 0: deterministisch (DDIM), eta = 1: entspricht DDPM-Rauschen.
    """
    batch_size = x_t.shape[0]
    t_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
    alpha_bar_t = _gather(schedule.alphas_cumprod, t_batch, x_t.shape)
    if t_prev >= 0:
        t_prev_batch = torch.full((batch_size,), t_prev, device=x_t.device, dtype=torch.long)
        alpha_bar_prev = _gather(schedule.alphas_cumprod, t_prev_batch, x_t.shape)
    else:
        alpha_bar_prev = torch.ones_like(alpha_bar_t)
    # 1. eps_hat = model(x_t, t_batch)
    # 2. x0_pred aus x_t und eps_hat (optional .clamp(-1, 1), stabilisiert frühe Schritte)
    # 3. sigma nach der Formel (bei eta = 0 ist sigma = 0)
    # 4. x_prev = sqrt(alpha_bar_prev) * x0_pred + sqrt(1 - alpha_bar_prev - sigma**2) * eps_hat
    #    und, falls eta > 0, + sigma * randn_like(x_t)
    eps_hat = predict_eps(model, x_t, t_batch, y, guidance_scale)
    x0_pred = ((x_t - (1.0 - alpha_bar_t).sqrt() * eps_hat) / alpha_bar_t.sqrt()).clamp(-1, 1)
    # Nach dem Clamp ε̂ aus dem geclampten x̂_0 zurückrechnen, damit x̂_0 und ε̂ konsistent
    # bleiben (wie diffusers DDIMScheduler). Ohne das zerfällt DDIM bei Guidance w > 1.
    eps_hat = (x_t - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt()
    sigma = eta * ((1.0 -alpha_bar_prev)/(1.0 - alpha_bar_t)).sqrt() * (1.0 -alpha_bar_t / alpha_bar_prev).sqrt()
    direction = (1.0-alpha_bar_prev -sigma**2).sqrt()*eps_hat
    x_prev = alpha_bar_prev.sqrt() * x0_pred + direction
    if eta > 0:
        x_prev = x_prev + sigma * torch.randn_like(x_t)
    return x_prev


@torch.no_grad()
def ddim_sample(
    model: UNet,
    schedule: NoiseSchedule,
    n: int,
    device: str,
    num_steps: int = 50,
    eta: float = 0.0,
    keep_every: int | None = None,
    seed: int | None = None,
    y: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Erzeugt n Bilder mit DDIM in num_steps Schritten (statt T), optional konditioniert."""
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    x = torch.randn(n, 1, 32, 32, device=device, generator=generator)

    steps = ddim_timesteps(schedule.num_steps, num_steps)
    trajectory: list[torch.Tensor] = []
    for i, t in enumerate(tqdm(steps, desc=f"DDIM {num_steps}", unit="Schritt", leave=False)):
        if keep_every and i % keep_every == 0:
            trajectory.append(x.cpu())
        t_prev = steps[i + 1] if i + 1 < len(steps) else -1
        x = ddim_step(model, schedule, x, t, t_prev, eta, y, guidance_scale)
    trajectory.append(x.cpu())
    return x, trajectory


def load_run(run: str, device: str, use_ema: bool = True) -> tuple[UNet, NoiseSchedule, dict]:
    """Lädt Modell (EMA- oder Roh-Gewichte) und Schedule eines Laufs."""
    run_dir = Path("runs") / run
    ckpt = torch.load(run_dir / "ckpt.pt", map_location=device)
    config = ckpt["config"]
    model = UNet(base=config["base"], num_classes=10 if config.get("cond") else None).to(device)
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.eval()
    schedule = make_schedule(config.get("schedule", "linear"), config["steps"]).to(device)
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
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM-Stochastik, 0 = deterministisch")
    parser.add_argument("--label", default=None,
                        help="Ziffer 0-9 für alle Bilder, oder 'all' = jede Zeile eine Ziffer (nur konditioniertes Modell)")
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG-Stärke w (1 = aus)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, schedule, config = load_run(args.run, device, use_ema=not args.no_ema)
    print(f"Lauf {args.run} (Epoche {config.get('epochs')}, T={schedule.num_steps}) auf {device}, "
          f"{'EMA' if not args.no_ema else 'Roh'}-Gewichte")

    y = None
    nrow = int(args.n ** 0.5)
    if args.label is not None:
        if model.num_classes is None:
            raise SystemExit("Dieser Lauf ist nicht konditioniert trainiert (--cond fehlt beim Training).")
        if args.label == "all":  # 10 Zeilen à nrow Bilder, Zeile k = Ziffer k
            nrow = max(1, args.n // 10)
            args.n = nrow * 10
            y = torch.arange(10, device=device).repeat_interleave(nrow)
        else:
            y = torch.full((args.n,), int(args.label), device=device, dtype=torch.long)

    if args.sampler == "ddim":
        keep = max(1, args.ddim_steps // 10)  # ~10 Zwischenstände wie beim DDPM-Sampler
        x0, trajectory = ddim_sample(model, schedule, args.n, device, num_steps=args.ddim_steps,
                                     eta=args.eta, keep_every=keep, seed=args.seed, y=y, guidance_scale=args.guidance)
    else:
        x0, trajectory = sample(model, schedule, args.n, device, keep_every=args.keep_every, seed=args.seed,
                                y=y, guidance_scale=args.guidance)

    run_dir = Path("runs") / args.run
    suffix = "" if not args.no_ema else "_raw"
    if args.sampler == "ddim":
        suffix += f"_ddim{args.ddim_steps}"
    if args.label is not None:
        suffix += f"_y{args.label}_w{args.guidance:g}"
    grid = make_grid(to_image(x0), nrow=nrow, padding=1)
    save_image(grid, run_dir / f"samples{suffix}.png")
    save_trajectory_grid(trajectory, run_dir / f"trajectory{suffix}.png")
    print(f"gespeichert: {run_dir / f'samples{suffix}.png'}, {run_dir / f'trajectory{suffix}.png'}")


if __name__ == "__main__":
    main()
