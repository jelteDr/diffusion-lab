# Ergebnisse der Experimente

Alle Zahlen: MNIST 32x32, U-Net mit 2,17 M Parametern (base=32), 20 Epochen, Batch 128,
AdamW lr 2e-4, EMA 0.999, Seed 0, Apple M5 (MPS). Bewertung mit dem Klassifikator-Richter
(`ddpm/classifier.py`, 99,0 % Test-Genauigkeit) auf 1000 generierten Bildern.

**Referenz (echte MNIST-Bilder, n=1000):** IS 9,71 · FID 5,24 · Konfidenz 0,991.
Der FID ist auch zwischen echten Bildern nicht 0, weil 1000 Bilder die Verteilung nur
schätzen. **Achtung:** für generierte Bilder ist die Streuung deutlich größer (siehe Exp 1:
±3 FID-Punkte zwischen zwei Sampling-Seeds desselben Modells).

---

## Exp 1: Linearer vs. Cosinus-Schedule

**Hypothese (vorab):** Der Cosinus-Schedule (Nichol & Dhariwal 2021) verteilt die
Bildentstehung gleichmäßiger über die Zeitschritte (beim linearen Schedule ist ab t≈400
kaum Signal übrig, siehe `docs/img/schedules_alpha_bar.png`). Erwartung: bessere
Bildqualität (niedrigerer FID, höherer IS) bei gleichem Trainingsbudget.

**Aufbau:** identisch bis auf `--schedule`. Sampling: DDPM ancestral, 1000 Schritte,
σ_t = √β_t.

| Lauf | Schedule | Trainings-Loss (Ep. 20) | IS ↑ (Seed 0 / 2) | FID ↓ (Seed 0 / 2) | Konfidenz |
|------|----------|------------------------:|------------------:|-------------------:|----------:|
| mnist_base   | linear | 0,018 | 8,90 / 8,85 → **8,88** | 14,0 / 19,9 → **17,0** | 0,959 |
| mnist_cosine | cosine | 0,031 | 8,72 / 8,82 → **8,77** | 20,0 / 16,6 → **18,3** | 0,956 |

**Befund: kein belastbarer Unterschied.** Nach dem ersten Seed sah der Cosinus-Schedule
mit FID 20,0 gegen 14,0 klar schlechter aus. Ein zweiter Sampling-Seed zeigte jedoch, dass
**derselbe** lineare Lauf zwischen 14,0 und 19,9 schwankt: Die FID-Streuung bei n=1000 ist
so groß wie der vermeintliche Effekt. Der IS ist stabiler (±0,05 beim linearen Modell)
und zeigt einen kleinen Vorsprung für linear (8,88 vs. 8,77), der aber ebenfalls innerhalb
der Cosinus-Streuung (8,72–8,82) liegt. Die Hypothese „Cosinus ist besser" wird damit nicht
bestätigt; „Cosinus ist schlechter" lässt sich aber auch nicht sagen.

**Was sich sicher sagen lässt:** Die Trajektorien zeigen den erwarteten qualitativen Effekt
(Ziffern werden beim Cosinus-Modell schon ab t≈800 sichtbar statt ab t≈400), und die
Trainings-Loss ist zwischen Schedules nicht vergleichbar (beim linearen Schedule ist die
ε-Vorhersage für die Hälfte aller t trivial, weil x_t ≈ ε).

**Methodische Lehre:** Die Streuung einer Metrik muss bekannt sein, bevor man Unterschiede
interpretiert. Der Referenzvergleich „echt gegen echt" (FID 5,2) unterschätzt die Streuung
generierter Bilder deutlich. Der FID ist außerdem bekannt dafür, bei kleinem n nach oben
verzerrt zu sein (128×128-Kovarianz aus 1000 Bildern). Abhilfe: n ≥ 5000 und mehrere Seeds
mit Mittelwert ± Streuung. Beides wird erst mit DDIM (Exp 2) bezahlbar, weil 1000-Schritt-
Sampling ~6 min pro 1000 Bilder kostet.

**Interpretation, warum kein Gewinn (Hypothesen, nicht gemessen):**
- Das Paper (Nichol & Dhariwal 2021) führt den Cosinus-Schedule zusammen mit gelernter
  Varianz und Hybrid-Loss ein; hier ist nur der Schedule getauscht, mit fester Varianz β_t.
- MNIST-Ziffern sind spärliche, einfache Bilder; die Feinstruktur bei kleinem t entscheidet
  über die Erkennbarkeit, und dort investiert der lineare Schedule mehr Trainingsbeispiele.

**Offen:** Wiederholung mit n=5000 und 3 Seeds via DDIM; Sampling mit σ_t² = β̃_t
(posterior variance) als Gegenprobe.

Bilder: `docs/img/schedules_alpha_bar.png`, `docs/img/trajectory_linear.png`,
`docs/img/trajectory_cosine.png`.

---

## Exp 2: DDIM-Sampling — Schritte vs. Qualität

**Hypothese (vorab):** DDIM (Song et al. 2021) mit 50 Schritten ist ~20× schneller als
DDPM mit 1000 Schritten bei nur geringem Qualitätsverlust (FID im Bereich des Basismodells).

**Aufbau:** Basismodell `mnist_base` (linear), kein Neutraining. Sampler DDIM mit
Teilfolge `range(999, -1, -T/k)`, x̂_0 auf [-1, 1] geclampt. η=0 (deterministisch) und
η=1 (stochastisch, DDPM-artige Varianz). Je 3 Sampling-Seeds, n=1000, Mittel ± Streuung.

| Sampler | Schritte | IS ↑ | FID ↓ | ms/Bild | Speedup |
|---------|---------:|-----:|------:|--------:|--------:|
| DDIM η=0 | 10  | 8,04 ± 0,05 | 64,6 ± 2,3 | 2,7 | 142× |
| DDIM η=0 | 20  | 8,19 ± 0,11 | 42,8 ± 3,5 | 5,5 | 70× |
| DDIM η=0 | 50  | 8,29 ± 0,13 | 36,1 ± 3,7 | 13,6 | 28× |
| DDIM η=0 | 100 | 8,32 ± 0,12 | 37,6 ± 4,1 | 28,0 | 14× |
| DDIM η=0 | 250 | 7,97 ± 0,07 | 50,4 ± 5,1 | 79,4 | 5× |
| DDIM η=1 | 20  | 8,57 ± 0,12 | 30,1 ± 2,4 | 7,0 | 55× |
| DDIM η=1 | 50  | 8,70 ± 0,03 | 21,0 ± 4,7 | 16,8 | 23× |
| DDIM η=1 | 100 | 8,74 ± 0,08 | 19,0 ± 2,9 | 35,2 | 11× |
| DDIM η=1 | 250 | 8,70 ± 0,04 | **17,1 ± 0,9** | 81,8 | 4,7× |
| DDPM (Referenz) | 1000 | 8,88 (2 Seeds) | 17,0 (2 Seeds) | 384 | 1× |

![Trade-off](img/ddim_tradeoff.png)

**Befund 1 — Hypothese in der ursprünglichen Form NICHT bestätigt:** Deterministisches
DDIM-50 verdoppelt den FID (36 vs. 17) und senkt den IS (8,29 vs. 8,88). Der Unterschied
liegt weit außerhalb der Streuung.

**Befund 2 — deterministisches DDIM wird ab 100 Schritten nicht besser, bei 250 sogar
schlechter** (FID 50,4). Mehr Schritte ≠ mehr Qualität.

**Befund 3 — Stochastik ist der entscheidende Faktor, nicht die Schrittzahl:** Mit η=1
verbessert sich die Kurve monoton mit den Schritten und erreicht bei 250 Schritten
den DDPM-Wert (17,1 vs. 17,0) bei 4,7× Geschwindigkeit; bei 50 Schritten (23×) fehlen
nur 4 FID-Punkte. Die praktische Empfehlung für dieses Modell: **DDIM η=1, 50–100 Schritte**.

**Interpretation (Hypothese, nicht gemessen):** Bei η=0 folgt der Sampler einer
deterministischen Bahn, die vollständig vom ε-Schätzer des Netzes bestimmt wird. Ein
kleines, kurz trainiertes Netz schätzt ε (und damit x̂_0) systematisch fehlerhaft; mehr
Schritte folgen dieser fehlerhaften Bahn nur genauer. Frisches Rauschen in jedem Schritt
löscht einen Teil des angesammelten Fehlers und erlaubt spätere Korrektur. Das deckt sich
mit Karras et al. (2022, „Elucidating the Design Space"): stochastisches Sampling hilft
bei unvollkommenen Modellen, deterministisches lohnt sich erst bei starken Modellen (→ SDXL).

**Nebenbefund zur Methodik:** Bei η=1/250 ist die FID-Streuung mit ±0,9 deutlich kleiner
als bei den anderen Punkten (±3–5). Der FID-Rauschanteil hängt also selbst vom Sampler ab.

**Offen:** (a) Exp 1 (Schedules) mit DDIM η=1/100 und n=5000 nachmessen, jetzt bezahlbar
(~3 min statt 30). (b) Einfluss des x̂_0-Clamps bei η=0 prüfen. (c) η zwischen 0 und 1.
