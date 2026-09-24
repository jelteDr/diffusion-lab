"""Kleiner MNIST-Klassifikator als "Richter" für generierte Bilder.

Aufruf:  uv run python -m ddpm.classifier        (trainiert und speichert runs/classifier.pt)

Er wird auf echten Ziffern trainiert und später auf generierte angewendet:
- seine Klassen-Wahrscheinlichkeiten liefern ein Inception-Score-Analogon,
- seine vorletzte Schicht (128 Merkmale) den Raum für ein FID-Analogon.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ddpm.data import mnist_loader

CLASSIFIER_PATH = Path("runs") / "classifier.pt"


class DigitClassifier(nn.Module):
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.fc_feat = nn.Linear(128 * 4 * 4, feat_dim)
        self.fc_out = nn.Linear(feat_dim, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Vorletzte Schicht, Form (B, feat_dim). Eingabe in [-1, 1]."""
        return F.relu(self.fc_feat(self.conv(x).flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.features(x))


def load_classifier(device: str) -> DigitClassifier:
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"{CLASSIFIER_PATH} fehlt, erst `uv run python -m ddpm.classifier` ausführen")
    clf = DigitClassifier().to(device)
    clf.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    return clf.eval()


def main(epochs: int = 5) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    train_loader = mnist_loader(128, train=True)
    test_loader = mnist_loader(512, train=False)
    clf = DigitClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        clf.train()
        for x, y in tqdm(train_loader, desc=f"Epoche {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (clf(x).argmax(1) == y).sum().item()
                total += y.numel()
        print(f"Epoche {epoch}: Test-Genauigkeit {correct / total:.4f}")

    CLASSIFIER_PATH.parent.mkdir(exist_ok=True)
    torch.save(clf.state_dict(), CLASSIFIER_PATH)
    print(f"gespeichert: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
