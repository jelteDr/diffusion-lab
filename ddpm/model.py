"""Kleines U-Net für DDPM: schätzt aus (x_t, t) das Rauschen epsilon.

Aufbau (für 32x32-Bilder, base=32, ch_mults=(1, 2, 4)):

    x_t (1x32x32) ──stem──► 32x32x32 ─ResBlock─► ▼  (Down)
                                     64x16x16 ─ResBlock─► ▼
                                              128x8x8 ─ResBlock─► Mitte ─► ▲ (Up, + Skip)
                                     ...  ◄────────────────────────────┘
    ε̂ (1x32x32) ◄──out-Conv──

Die Zeit t wird als Sinus-Embedding kodiert, durch ein MLP geschickt und in
jedem ResBlock auf die Feature-Maps addiert.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusförmiges Zeit-Embedding wie im Transformer-Paper ("Attention is all you need").

    Args:
        t:   Zeitschritte, Form (B,), ganzzahlig oder float
        dim: Dimension des Embeddings (gerade Zahl)
        max_period: steuert die längste Periode der Frequenzen

    Returns:
        Tensor der Form (B, dim). Erste Hälfte sin, zweite Hälfte cos.
    """
    # Skizze: half = dim // 2 Frequenzen, geometrisch fallend von 1 bis 1/max_period:
    #   freqs = exp(-ln(max_period) * arange(half) / half)        Form (half,)
    #   args  = t[:, None].float() * freqs[None, :]                Form (B, half)
    #   return cat([sin(args), cos(args)], dim=-1)                 Form (B, dim)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Residual-Block mit GroupNorm, SiLU und Zeit-Einspeisung."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)  # Zeit-Vektor -> ein Bias pro Kanal
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # Skip-Pfad: 1x1-Conv, falls sich die Kanalzahl ändert
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]  # (B, C) -> (B, C, 1, 1), Broadcast über H, W
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base: int = 32,
        ch_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 128,
        num_classes: int | None = None,
    ):
        """num_classes: falls gesetzt, wird das Netz klassenkonditioniert. Index num_classes
        (also z. B. 10 bei zehn Ziffern) ist das "Null-Label" für unkonditionierte Vorhersage."""
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes
        # Klassen-Embedding: ein lernbarer Vektor pro Klasse + einer fürs Null-Label.
        # Er wird einfach auf das Zeit-Embedding addiert und erreicht so jeden ResBlock.
        self.class_emb = nn.Embedding(num_classes + 1, time_dim) if num_classes else None
        # Sinus-Embedding -> MLP, damit das Netz die Zeit-Information verformen kann
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim)
        )

        chs = [base * m for m in ch_mults]  # z. B. [32, 64, 128]
        self.stem = nn.Conv2d(in_ch, chs[0], kernel_size=3, padding=1)

        # Down-Pfad: pro Stufe ein ResBlock, danach (außer zuletzt) Halbierung der Auflösung
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = chs[0]
        for i, ch in enumerate(chs):
            self.down_blocks.append(ResBlock(prev, ch, time_dim))
            is_last = i == len(chs) - 1
            self.downsamples.append(
                nn.Identity() if is_last else nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
            )
            prev = ch

        # Mitte: zwei ResBlocks auf der kleinsten Auflösung
        self.mid1 = ResBlock(chs[-1], chs[-1], time_dim)
        self.mid2 = ResBlock(chs[-1], chs[-1], time_dim)

        # Up-Pfad: Skip-Feature-Maps anhängen (Kanäle verdoppeln), ResBlock, dann Verdopplung der Auflösung
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            self.up_blocks.append(ResBlock(prev + ch, ch, time_dim))
            is_first_level = i == 0
            self.upsamples.append(
                nn.Identity() if is_first_level else nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
            )
            prev = ch

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """y: Klassen-Labels (B,), nur bei konditioniertem Modell; Wert num_classes = Null-Label."""
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        if self.class_emb is not None:
            if y is None:  # kein Label übergeben -> unkonditioniert
                y = torch.full_like(t, self.num_classes)
            t_emb = t_emb + self.class_emb(y)

        h = self.stem(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for block, up in zip(self.up_blocks, self.upsamples):
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, t_emb)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
