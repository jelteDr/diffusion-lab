"""Lädt nur die benötigten fp16-Gewichte von SDXL Base 1.0 (~7 GB) in den Hugging-Face-Cache.

Aufruf:  uv run python -m sdxl.download

Das vollständige Modell-Repo enthält zusätzlich fp32-, ONNX- und OpenVINO-Varianten
(zusammen > 70 GB). allow_patterns beschränkt den Download auf das, was die Pipeline in
fp16 tatsächlich lädt. Ein abgebrochener Download wird beim nächsten Aufruf fortgesetzt.
"""

from huggingface_hub import snapshot_download

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# VAE mit fp16-Fix: Die Original-SDXL-VAE läuft in fp16 über (NaN -> schwarze Bilder).
# madebyollin hat sie so nachtrainiert, dass die Aktivierungen in den fp16-Bereich passen.
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

BASE_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",      # CLIP ViT-L, 0,25 GB
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",    # OpenCLIP ViT-bigG, 1,39 GB
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",  # U-Net, 5,14 GB
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",   # Original-VAE, 0,17 GB (Fallback)
]
VAE_PATTERNS = ["config.json", "diffusion_pytorch_model.safetensors"]


def main() -> None:
    print(f"Lade {MODEL_ID} (fp16, ~6,8 GB) ...")
    path = snapshot_download(MODEL_ID, allow_patterns=BASE_PATTERNS)
    print("  ->", path)
    print(f"Lade {VAE_ID} (~0,33 GB) ...")
    path = snapshot_download(VAE_ID, allow_patterns=VAE_PATTERNS)
    print("  ->", path)
    print("fertig")


if __name__ == "__main__":
    main()
