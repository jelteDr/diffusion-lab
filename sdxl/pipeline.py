"""Laden der SDXL-Pipeline und Messung von Zeit und Speicher auf Apple Silicon."""

import time

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

from sdxl.download import MODEL_ID, VAE_ID


def device_and_dtype() -> tuple[str, torch.dtype]:
    """MPS (Apple) oder CUDA mit fp16, sonst CPU mit fp32 (CPU kann kein schnelles fp16)."""
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_pipeline(device: str, dtype: torch.dtype, attention_slicing: bool = False) -> StableDiffusionXLPipeline:
    """SDXL Base 1.0 mit der fp16-tauglichen VAE.

    Bestandteile (vgl. Stufe 1):
      - unet:           das Rauschschätzer-Netz ε̂(x_t, t, Text), 2,6 Mrd. Parameter
      - text_encoder(_2): zwei CLIP-Encoder, liefern die Konditionierung (statt Klassen-Label)
      - vae:            komprimiert 1024x1024x3 Pixel <-> 128x128x4 Latents (Diffusion läuft dort)
      - scheduler:      der Sampler (Pendant zu p_sample_step / ddim_step)
    """
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    pipe.to(device)
    if attention_slicing:
        # Attention in Häppchen rechnen: weniger Spitzen-Speicher, etwas langsamer
        pipe.enable_attention_slicing()
    return pipe


class MemoryTracker:
    """Misst über callback_on_step_end den höchsten vom GPU-Treiber belegten Speicher."""

    def __init__(self, device: str):
        self.device = device
        self.peak_bytes = 0
        self.step_times: list[float] = []
        self._last = time.perf_counter()

    def current_bytes(self) -> int:
        if self.device == "mps":
            return torch.mps.driver_allocated_memory()
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated()
        return 0

    def __call__(self, pipe, step: int, timestep, callback_kwargs: dict) -> dict:
        now = time.perf_counter()
        self.step_times.append(now - self._last)
        self._last = now
        self.peak_bytes = max(self.peak_bytes, self.current_bytes())
        return callback_kwargs

    def reset(self) -> None:
        self.peak_bytes = self.current_bytes()
        self.step_times = []
        self._last = time.perf_counter()


def param_counts(pipe: StableDiffusionXLPipeline) -> dict[str, float]:
    """Parameter je Bestandteil in Millionen."""
    parts = {"unet": pipe.unet, "text_encoder": pipe.text_encoder,
             "text_encoder_2": pipe.text_encoder_2, "vae": pipe.vae}
    return {k: round(sum(p.numel() for p in m.parameters()) / 1e6, 1) for k, m in parts.items()}
