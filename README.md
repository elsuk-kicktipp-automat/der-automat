# Der Automat

Ein selbstlernender KI-Tipper für die Kicktipp-Runde. Konzept: siehe [concept.md](concept.md).
Website: **<https://elsuk-kicktipp-automat.github.io/der-automat/>**

**Stand: Phase 2** – Statistik-Engine (OpenLigaDB + ELO, Dixon-Coles-Modell,
Kicktipp-Punkteoptimierung, Backtesting) plus Astro-Website, Hash-Versiegelung
und GitHub-Actions-Automatisierung. Die Pipeline läuft im Test-/Härtungsbetrieb
mit der **WM 2026** (bis 19.07.2026), danach wird per `config.yaml` auf die
Bundesliga 2026/27 umgestellt. Eine Kicktipp-Runde ist dafür nicht nötig –
die Punkte rechnet die Engine selbst ab; die Abgabe bei kicktipp.de ist Phase 4.

## Wie es funktioniert

1. **Spieldaten:** [OpenLigaDB](https://api.openligadb.de) (kostenlos, kein API-Key).
   Die WM 2026 ist dort auf zwei Ligen verteilt: `wm2026` (Gruppenphase) und `mb`
   (K.o.-Runde) – die Engine führt sie zusammen. Teams werden über normalisierte
   Namen identifiziert, weil die Community-Ligen keine stabilen Team-IDs haben.
2. **ELO-Ratings:** gemeinsame Schnittstelle mit zwei Adaptern (`team_type` in
   config.yaml): [clubelo.com](http://clubelo.com/API) für Vereine (mit historischen
   Ständen pro Stichtag) und [eloratings.net](https://eloratings.net) für
   Nationalteams (nur aktueller Stand). Namens-Zuordnung: `data/mappings/`.
3. **Modell:** Dixon-Coles-Poisson. Erwartete Tore pro Team aus Angriffs-/
   Abwehrstärke (exponentiell abklingend gewichtete Form), Heimvorteil (bei der
   WM per `neutral_venue: true` abgeschaltet) und ELO-Differenz. Der ELO-Koeffizient
   wird mitgeschätzt und zum Prior regularisiert – so trägt ELO die Prognose,
   solange wenig Spieldaten da sind (WM-Gruppenphase), und die gefitteten
   Teamstärken übernehmen mit wachsender Datenmenge. Output pro Spiel: die
   vollständige Wahrscheinlichkeitsmatrix aller Ergebnisse von 0:0 bis 6:6.
4. **Punkteoptimierung:** Getippt wird nicht das wahrscheinlichste Ergebnis,
   sondern der Tipp mit dem höchsten **Punkte-Erwartungswert** unter dem
   Kicktipp-Schema der Runde (config.yaml, Default 4/3/2). Kicktipp-Standard:
   bei Unentschieden gibt es keine Tordifferenz-Punkte, nur exakt oder Tendenz.
5. **K.o.-Spiele:** Gewertet wird das Ergebnis nach 90 Minuten (OpenLigaDB
   resultTypeID 2, bei der WM „Endergebniss (o.E.)" = ohne Elfmeterschießen) –
   ein Unentschieden ist ein gültiger und tippbarer Ausgang.

## Setup

Voraussetzung: Python ≥ 3.11

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Benutzung

```bash
# Tests (laufen ohne Netzwerk)
python -m pytest tests/

# Tipps für die nächste anstehende Runde -> data/predictions/ (gitignored!)
python -m engine.cli predict

# Tipps versiegeln: Hash öffentlich, Klartext verschlüsselt (braucht SEAL_SECRET)
python -m engine.cli seal

# Tipps nach Anstoß enthüllen
python -m engine.cli unseal

# Abrechnung der enthüllten Tipps gegen die realen Ergebnisse -> data/results/
python -m engine.cli evaluate

# Backtests -> data/backtests/  (--mode club | national | all)
python -m engine.cli backtest
```

Alle erzeugten Daten sind menschenlesbares JSON im Repo (`data/`) – das Repo
ist die Datenbank. API-Antworten werden unter `data/cache/` (gitignored) gecacht.

## Fairness-Mechanismus & Automatisierung

Klartext-Tipps liegen **nie** vor Anstoß im öffentlichen Repo
(`data/predictions/` ist gitignored). Stattdessen (concept.md §5):

1. **Versiegeln:** Pro Spiel wird nur der SHA-256-Hash von
   `(Teams, Anstoß, Tipp, Begründung, Salt)` veröffentlicht
   (`data/matchdays/`); der Klartext liegt Fernet-verschlüsselt in
   `data/sealed/*.enc`. Schlüssel: `SEAL_SECRET` (GitHub Actions Secret /
   lokale `.env`). Der Commit-Zeitstempel beweist den Zeitpunkt.
2. **Entsiegeln:** Ab 5 Minuten nach Anstoß wird der Klartext samt Salt in die
   Spieltags-Datei geschrieben – jeder kann den Hash nachrechnen (Anleitung
   auf der Website unter „Wie ich denke").

GitHub Actions übernimmt den Betrieb (`.github/workflows/`):

| Workflow | Zeitplan | Aufgabe |
| --- | --- | --- |
| `spieltag.yml` | täglich 06:00 UTC | predict → seal → evaluate → Commit |
| `unseal.yml` | alle 30 min | fällige Tipps enthüllen + abrechnen (früher Abbruch ohne fällige Spiele) |
| `deploy-site.yml` | bei Daten-/Site-Änderungen | Astro-Build → GitHub Pages |

K.o.-Pläne mit Platzhaltern („Sieger SF 12") werden unterstützt: Sobald
Nachzügler-Paarungen feststehen, versiegelt der nächste Lauf sie als weiteren
Batch derselben Runde.

## Website

Astro-Site unter `site/`, deployed auf
<https://elsuk-kicktipp-automat.github.io/der-automat/>:
**Spieltag** (versiegelte/enthüllte Tipps), **Archiv**, **Bilanz**
(Live-Punkte + Backtests), **Wie ich denke** (Modell & Hash-Verifikation).

```bash
cd site && npm install && npm run dev   # lokale Vorschau
```

## Backtesting

- **club:** Rollierend über die letzten 3 Bundesliga-Saisons. Vor jedem Spieltag
  wird nur auf bis dahin gespielten Partien gefittet (plus 2 Vorsaisons als
  Warmup); die ELO-Stände kommen historisch korrekt vom jeweiligen Stichtag.
- **national:** WM 2026 out-of-sample – Gruppenphase + bisherige K.o.-Spiele,
  Runde für Runde nur mit den davor gespielten Partien. Einschränkung:
  eloratings.net bietet keine historischen Stände, die Retro-Zahlen tragen
  dadurch einen leichten Lookahead-Bias (für *künftige* Spiele irrelevant).
- Verglichen wird gegen zwei Baselines: **(a)** immer 2:1 für den ELO-Favoriten,
  **(b)** immer 1:1. Reports inkl. Punkten pro Spieltag/Runde und Trefferquoten
  (exakt/Differenz/Tendenz): `data/backtests/club.json` bzw. `national.json`.

### Ergebnisse (Lauf vom 03.07.2026, Schema 4/3/2)

| Backtest | Spiele | Punkte | Ø/Spiel | ELO-Favorit 2:1 | immer 1:1 |
| --- | --- | --- | --- | --- | --- |
| Bundesliga 2023/24 | 306 | 408 | 1,333 | 395 | 244 |
| Bundesliga 2024/25 | 306 | 383 | 1,252 | 379 | 206 |
| Bundesliga 2025/26 | 306 | 418 | 1,366 | 437 | 218 |
| **Bundesliga gesamt** | **918** | **1209** | **1,317** | **1211** | **668** |
| **WM 2026 (out-of-sample)** | **82** | **138** | **1,683** | **123** | **72** |

Befunde:

- Bei der WM schlägt das Modell beide Baselines klar. Voraussetzung war die
  starke L2-Regularisierung für Nationalteams: mit dem Club-Wert (0.2) über-
  erklären die Team-Parameter die 3–7 Turnierspiele pro Team und übertönen den
  ELO-Term (107 statt 138 Punkte).
- In der Bundesliga liegt das Modell gleichauf mit der ELO-Favorit-2:1-Baseline
  (1209 vs. 1211 – bei 918 Spielen Rauschen). Diese Baseline ist unter dem
  4/3/2-Schema sehr stark; Mehrwert gegenüber ihr soll v.a. die Quoten-Schicht
  (Phase 3+, siehe concept.md) bringen.

## Konfiguration (`config.yaml`)

- `competition` / `leagues` / `season` – aktiver Wettbewerb (WM 2026: zwei Ligen)
- `team_type` – `club` (clubelo.com) oder `national` (eloratings.net)
- `neutral_venue` – Heimvorteil abschalten (WM)
- `kicktipp.points` – Punkteschema der Runde; `advance` = Zusatzfrage
  „Wer kommt weiter?" bei K.o.-Spielen (separat ausgewiesen, 0 = deaktiviert)
- `model.*` – Zeitgewichtung, Regularisierung, Tor-Raster, ELO-Prior
- `backtest.*` – Parameter der beiden Backtest-Modi

Umstieg auf Bundesliga 2026/27: Kommentarblock am Kopf der config.yaml.

## Projektstruktur

```text
engine/                Python-Engine
  cli.py               Einstiegspunkt: predict / seal / unseal / evaluate / backtest
  predict.py           Prognose der nächsten Runde -> data/predictions/ (gitignored)
  seal.py              Hash-Versiegelung + Entsiegelung nach Anstoß
  evaluate.py          Punkteabrechnung -> data/results/
  backtest.py          Backtests (club + national) -> data/backtests/
  model.py             Dixon-Coles-Poisson mit ELO-Term
  optimizer.py         Kicktipp-Punktelogik + EV-Optimierung + Baselines
  teams.py             Team-Identität über normalisierte Namen
  sources/
    openligadb.py      Spielplan/Ergebnisse mit Cache
    elo.py             ELO-Adapter (clubelo.com | eloratings.net)
tests/                 pytest-Suite (ohne Netzwerkzugriff lauffähig)
data/                  JSON-„Datenbank" (cache/ ist gitignored)
  matchdays/           öffentliche Spieltags-Dateien (Hashes bzw. Enthülltes)
  sealed/              verschlüsselte Klartext-Tipps bis zum Anstoß
  mappings/            Namens-Zuordnung OpenLigaDB -> ELO-Quellen
site/                  Astro-Website (GitHub Pages)
.github/workflows/     GitHub Actions (Spieltag, Entsiegeln, Site-Deploy)
config.yaml            Wettbewerb, Punkteschema, Modell- und Backtest-Parameter
.env.example           Vorlage für Secrets späterer Phasen (Phase 1 braucht keine)
```

## Roadmap

- [x] **Phase 1:** Engine (OpenLigaDB + ELO), Modell, Punkteoptimierung,
      Backtesting, WM-2026-Testbetrieb
- [x] **Phase 2:** Astro-Site, Hash-Versiegelung, GitHub-Actions-Betrieb,
      Deployment auf Pages
- [ ] **Phase 3:** LLM-Schicht (Dossier, Adjustierung, Begründungen)
- [ ] **Phase 4:** Kicktipp-Bot (Playwright-Abgabe)
- [ ] **Phase 5:** Selbstlernen, Schattentipper, Dashboard
