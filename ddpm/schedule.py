"""Noise-Schedule und Vorwärtsprozess (Forward Diffusion) eines DDPM.

Der Vorwärtsprozess zerstört ein Bild x_0 schrittweise mit Gauß-Rauschen:

    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)

Der Trick, der Training überhaupt praktikabel macht: Man muss die T Schritte
nicht nacheinander ausführen. Mit alpha_t = 1 - beta_t und dem Produkt
alpha_bar_t = alpha_1 * ... * alpha_t gilt in geschlossener Form:

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

Man kann also für jedes beliebige t direkt von x_0 nach x_t springen.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoiseSchedule:
    """Alle vorberechneten Größen für T Diffusionsschritte (Index 0 .. T-1)."""

    betas: torch.Tensor          # beta_t, Rauschvarianz pro Schritt
    alphas: torch.Tensor         # 1 - beta_t
    alphas_cumprod: torch.Tensor # alpha_bar_t, kumulatives Produkt

    @property
    def num_steps(self) -> int:
        return int(self.betas.shape[0])

    def to(self, device: torch.device | str) -> "NoiseSchedule":
        return NoiseSchedule(
            self.betas.to(device), self.alphas.to(device), self.alphas_cumprod.to(device)
        )


def linear_schedule(num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> NoiseSchedule:
    """Linearer Schedule aus dem DDPM-Paper (Ho et al. 2020)."""
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return NoiseSchedule(betas, alphas, alphas_cumprod)


def cosine_schedule(num_steps: int = 1000, s: float = 0.008, max_beta: float = 0.999) -> NoiseSchedule:
    """Cosinus-Schedule aus "Improved DDPM" (Nichol & Dhariwal 2021).

    Statt beta_t direkt festzulegen, wird alpha_bar_t als Kurve vorgegeben:

        f(t) = cos( (t/T + s) / (1 + s) * pi/2 )^2        für t = 0 .. T
        alpha_bar_t = f(t) / f(0)

    und beta_t daraus zurückgerechnet:  beta_t = 1 - alpha_bar_t / alpha_bar_{t-1},
    nach oben auf max_beta begrenzt, damit die letzten Schritte nicht explodieren.
    Der kleine Offset s verhindert, dass beta_t ganz am Anfang praktisch 0 ist.
    """
    # TODO(human): Berechne betas nach der Formel oben und gib einen NoiseSchedule zurück.
    # Skizze:
    #   steps = torch.arange(num_steps + 1, dtype=torch.float32)     T+1 Stützstellen: t = 0 .. T
    #   f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
    #   alpha_bar = f / f[0]
    #   betas = 1 - alpha_bar[1:] / alpha_bar[:-1]                    Form (T,)
    #   betas = betas.clamp(max=max_beta)
    #   alphas, alphas_cumprod wie im linearen Schedule aus betas ableiten
    steps = torch.arange(num_steps+1, dtype=torch.float32)
    f = torch.cos(((steps / num_steps) + s) / (1+s) * math.pi / 2) **2
    alpha_bar = f / f[0]
    betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=max_beta)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return NoiseSchedule(betas, alphas, alphas_cumprod)


SCHEDULES = {"linear": linear_schedule, "cosine": cosine_schedule}


def make_schedule(name: str, num_steps: int = 1000) -> NoiseSchedule:
    """Schedule per Name bauen ("linear" oder "cosine")."""
    if name not in SCHEDULES:
        raise ValueError(f"Unbekannter Schedule '{name}', erlaubt: {list(SCHEDULES)}")
    return SCHEDULES[name](num_steps)


def _gather(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Holt values[t] für einen Batch von Zeitschritten und bringt das Ergebnis
    auf die Form (B, 1, 1, 1), damit es per Broadcasting mit Bildern (B, C, H, W)
    multipliziert werden kann."""
    out = values.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def q_sample(schedule: NoiseSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Vorwärtsprozess in geschlossener Form: erzeugt x_t direkt aus x_0.

    Args:
        schedule: vorberechnete Größen (bereits auf dem Device von x0)
        x0:    saubere Bilder, Form (B, C, H, W), Werte in [-1, 1]
        t:     Zeitschritte pro Bild, Form (B,), ganzzahlig in [0, T-1]
        noise: Standard-Gauß-Rauschen epsilon, gleiche Form wie x0

    Returns:
        x_t, gleiche Form wie x0
    """
    alpha_bar = _gather(schedule.alphas_cumprod, t, x0.shape)  # (B, 1, 1, 1)
    return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise
