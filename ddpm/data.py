"""MNIST-Datenpipeline für das DDPM."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist_transform() -> transforms.Compose:
    """28x28 -> 32x32 (Rand 2 Pixel), Werte von [0, 1] auf [-1, 1].

    32x32 lässt sich sauber zweimal halbieren (32 -> 16 -> 8), und [-1, 1]
    passt zur Annahme des Vorwärtsprozesses, dass x_0 etwa Einheitsvarianz hat.
    """
    return transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def mnist_loader(batch_size: int = 128, root: str = "data", train: bool = True) -> DataLoader:
    ds = datasets.MNIST(root, train=train, download=True, transform=mnist_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=0)


def to_image(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] zum Anzeigen/Speichern."""
    return ((x.clamp(-1, 1) + 1) / 2)
