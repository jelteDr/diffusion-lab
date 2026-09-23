# Fahrplan: diffusion-lab

Ziel: Verstehen, wie Diffusionsmodelle funktionieren, indem ich zuerst ein kleines
DDPM von Grund auf baue (Stufe 1) und danach ein großes, vortrainiertes Modell
(SDXL) per DreamBooth-LoRA anpasse (Stufe 2). Portfolio-Stück mit dem gleichen
Anspruch wie book-RAG: saubere Experimente, ehrliche Befunde, README als Mini-Thesis.

Hardware: Apple M5, 24 GB Unified Memory (MPS). Training in Stufe 2 auf Kaggle/Colab.

Legende: ✅ fertig · 🚧 in Arbeit · ⬜ offen

---

## Stufe 1: Mini-DDPM auf MNIST (alles lokal)

| # | Meilenstein | Inhalt | Lernziel | Status |
|---|-------------|--------|----------|--------|
| M0 | Grundgerüst | uv-Projekt (Python 3.12, torch/MPS), Paket `ddpm/`, Git + GitHub | Projektaufbau | ✅ |
| M0 | Vorwärtsprozess | `linear_schedule`, `q_sample`, Demo `scripts/forward_demo.py` | Wie Rauschen ins Bild kommt, geschlossene Form q(x_t \| x_0) | ✅ |
| M1 | Modell | Sinus-Zeit-Embedding, kleines U-Net (Down/Up-Pfad, Skip-Verbindungen, ResBlocks), Formen-Smoke-Test | Wie das Netz t „sieht“; warum U-Net | ✅ |
| M2 | Training | MNIST-Loader, Loss = MSE zwischen echtem und vorhergesagtem Rauschen, Trainingsschleife auf MPS, Checkpoints, Loss-Kurve | Das eigentliche DDPM-Trainingsziel (ε-Prediction) | ⬜ |
| M3 | Sampling | Rückwärtsprozess Schritt für Schritt (x_T → x_0), Bild-Grid, Trajektorie-Visualisierung | Wie aus Rauschen ein Bild wird | ⬜ |
| M4 | Experimente | (a) linear vs. cosine Schedule, (b) DDIM: weniger Sampling-Schritte, (c) Modellgröße; Metrik: Klassifikator-basierter Score auf generierten Ziffern + Loss | Kontrollierte Ablationen, Trade-off Qualität vs. Geschwindigkeit | ⬜ |
| M5 | Konditionierung (optional) | Klassen-Label als Bedingung, Classifier-free Guidance („erzeuge eine 7“) | Brücke zu Text-Konditionierung in SDXL | ⬜ |
| M6 | Write-up | README als Mini-Thesis: Theorie kurz, Bilder aus `docs/`, Befunde aus M4 | Erklären können, was ein Diffusionsmodell ist | ⬜ |

**Definition of Done Stufe 1:** Das Modell erzeugt erkennbare Ziffern, die Ablationen sind
gemessen und dokumentiert, README erklärt den Ablauf mit eigenen Bildern.

---

## Stufe 2: SDXL + DreamBooth-LoRA (Inferenz lokal, Training in der Cloud)

| # | Meilenstein | Inhalt | Lernziel | Status |
|---|-------------|--------|----------|--------|
| S1 | SDXL-Inferenz lokal | diffusers-Pipeline auf MPS (fp16), Base + Refiner, erste Bilder | Latent Diffusion, VAE, Text-Encoder, Scheduler-Auswahl | ⬜ |
| S2 | Datensatz | 5 bis 10 eigene Fotos, Captions mit Trigger-Wort, Regularisierungsbilder | DreamBooth-Idee: neues Konzept in ein Modell bringen | ⬜ |
| S3 | LoRA-Training | diffusers-Skript `train_dreambooth_lora_sdxl.py` auf Kaggle; Parameter Rank, Lernrate, Prior Preservation bewusst wählen | LoRA: kleine Adapter statt Vollmodell | ⬜ |
| S4 | Adapter lokal | LoRA-Gewichte laden, Prompts vergleichen, Base vs. Base+LoRA | Adapter-Mechanik in der Pipeline | ⬜ |
| S5 | Evaluation | CLIP-Score (Prompt-Treue), Ähnlichkeit zum Trainingsmotiv, Rank-Ablation | Bildgenerierung messbar machen | ⬜ |
| S6 | Frontend (optional) | kleine FastAPI + Oberfläche im Apple-Stil: Prompt, Adapter-Auswahl, Vergleich | Serving eines Diffusionsmodells | ⬜ |
| S7 | Write-up | README-Abschnitt Stufe 2, Bilder, Befunde | | ⬜ |

**Scope-Grenze:** Stufe 1 komplett ist ein vorzeigbares Projekt. Stufe 2 baut darauf auf,
S6 ist Bonus.

---

## Konventionen

- Code-Kommentare und Doku auf Deutsch, Commits auf Deutsch.
- `data/` und `runs/` werden nie committet; Bilder für die README kommen nach `docs/`.
- Jedes Experiment: Hypothese vorher notieren, dann messen, Befund ehrlich festhalten
  (auch negative Ergebnisse).
