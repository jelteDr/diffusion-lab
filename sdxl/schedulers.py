"""Scheduler (Sampler) für SDXL, per Name auswählbar.

In diffusers heißt der Sampler "Scheduler": Er legt fest, welche Zeitschritte besucht werden
und wie aus der Rauschschätzung ε̂ des U-Nets der nächste, weniger verrauschte Latent entsteht.
Das ist genau die Rolle von `p_sample_step` und `ddim_step` aus Stufe 1.

Alle Scheduler werden aus der Konfiguration des Modells gebaut (`from_config`), damit
Noise-Schedule, Anzahl Trainingsschritte und Vorhersage-Typ zum trainierten U-Net passen.
"""

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin

# Name -> Kurzbeschreibung, für --help und die Doku
SCHEDULERS = {
    "ddim": "DDIM (Song et al. 2021), deterministisch bei eta=0 — wie ddim_step in Stufe 1",
    "euler": "Euler (Karras et al. 2022), deterministisch",
    "euler_a": "Euler Ancestral, stochastisch (frisches Rauschen pro Schritt)",
    "dpmpp_2m": "DPM-Solver++ 2M (Lu et al. 2022), deterministisch, Mehrschritt-Verfahren 2. Ordnung",
    "dpmpp_2m_sde": "DPM-Solver++ 2M SDE, stochastische Variante",
}

# Welche Scheduler in jedem Schritt frisches Rauschen ziehen (Analogon zu eta=1 aus Exp 2)
STOCHASTIC = {"euler_a", "dpmpp_2m_sde"}


def make_scheduler(name: str, config: dict) -> SchedulerMixin:
    """Baut den Scheduler `name` aus der Scheduler-Konfiguration des Modells.

    Args:
        name:   Schlüssel aus SCHEDULERS
        config: pipe.scheduler.config des geladenen Modells

    Returns:
        eine Scheduler-Instanz, die man der Pipeline zuweist: pipe.scheduler = ...
    """
    match name:
        case "ddim":
            return DDIMScheduler.from_config(config)
        case "euler":
            return EulerDiscreteScheduler.from_config(config)
        case "euler_a":
            return EulerAncestralDiscreteScheduler.from_config(config)
        case "dpmpp_2m":
            return DPMSolverMultistepScheduler.from_config(config,algorithm_type="dpmsolver++")
        case "dpmpp_2m_sde":
            return DPMSolverMultistepScheduler.from_config(config,algorithm_type="sde-dpmsolver++")
        case _:
            raise ValueError(f"Unbekannter Scheduler '{name}', erlaubt: {list(SCHEDULERS)}")


