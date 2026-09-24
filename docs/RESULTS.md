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
