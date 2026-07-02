# Der Automat

Ein selbstlernender KI-Tipper für die Kicktipp-Runde. Konzept: siehe [concept.md](concept.md).

**Stand: Phase 1** – Statistik-Engine mit OpenLigaDB-Anbindung, Dixon-Coles-Poisson-Modell,
Kicktipp-Punkteoptimierung und Backtesting-Harness.

## Wie es funktioniert

1. **Daten:** [OpenLigaDB](https://api.openligadb.de) liefert alle Bundesliga-Spiele
   (kostenlos, kein API-Key). Antworten werden unter `data/cache/` gecacht.
2. **Modell:** Ein Dixon-Coles-Poisson-Modell schätzt pro Team Angriffs- und
   Abwehrstärke plus Heimvorteil und die Korrektur für torarme Ergebnisse (rho).
   Ältere Spiele klingen exponentiell ab; L2-Regularisierung hält Aufsteiger ohne
   Historie beim Ligaschnitt. Ergebnis pro Spiel: eine vollständige
   Wahrscheinlichkeitsmatrix aller Ergebnisse (0:0, 1:0, 2:1, …).
3. **Punkteoptimierung:** Getippt wird nicht das wahrscheinlichste Ergebnis, sondern
   der Tipp mit dem höchsten **Punkte-Erwartungswert** unter dem Kicktipp-Schema der
   Runde (konfigurierbar in `config.yaml`, Default 4/3/2 für Ergebnis/Tordifferenz/Tendenz).

## Setup

Voraussetzung: Python ≥ 3.11

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Benutzung

```bash
# Tests
python -m pytest tests/

# Backtesting über die letzten 3 Saisons (siehe config.yaml)
python -m engine.backtest
```

Das Backtesting simuliert einen echten Tippbetrieb: Vor jedem Spieltag wird das Modell
ausschließlich auf bis dahin gespielten Partien gefittet (plus zwei Vorsaisons als
Warmup), dann pro Spiel der EV-optimale Tipp abgegeben und gegen das reale Ergebnis
abgerechnet. Der Report landet in `data/backtests/latest.json`.

### Ergebnis (Lauf vom 02.07.2026, Schema 4/3/2)

| Saison | Punkte | Ø/Spiel | Exakt | Differenz | Tendenz | Trefferquote |
|---|---|---|---|---|---|---|
| 2023/24 | 407 | 1,330 | 30 | 31 | 97 | 51,6 % |
| 2024/25 | 395 | 1,291 | 23 | 43 | 87 | 50,0 % |
| 2025/26 | 414 | 1,353 | 25 | 42 | 94 | 52,6 % |
| **Gesamt** | **1216** | **1,325** | | | | |

Die naive Baseline „tippe das wahrscheinlichste Ergebnis" holt im selben Zeitraum nur
**1039 Punkte** – die Erwartungswert-Optimierung bringt **+17 %**.

## Konfiguration (`config.yaml`)

- `kicktipp.points` – Punkteschema der Runde (exact/goal_diff/tendency)
- `model.time_decay_xi` – Abklingrate der Zeitgewichtung (pro Tag)
- `model.l2_penalty` – Regularisierung der Team-Parameter
- `backtest.seasons` – Backtest-Saisons (OpenLigaDB-Schlüssel = Jahr des Saisonstarts)

## Projektstruktur

```
engine/            Python-Engine
  openligadb.py    API-Client mit Cache
  model.py         Dixon-Coles-Poisson-Modell
  kicktipp.py      Punktelogik + Erwartungswert-Optimierung
  backtest.py      Backtesting-Harness (python -m engine.backtest)
  config.py        Konfigurations-Loader
tests/             pytest-Suite (ohne Netzwerkzugriff lauffähig)
data/              JSON-„Datenbank" (cache/ ist gitignored, backtests/ versioniert)
site/              Astro-Website (Phase 2, noch leer)
.github/workflows/ GitHub Actions (Phase 2+, noch leer)
config.yaml        Punkteschema, Modell- und Backtest-Parameter
.env.example       Vorlage für Secrets späterer Phasen (Phase 1 braucht keine)
```

## Roadmap

- [x] **Phase 1:** Engine, Modell, Punkteoptimierung, Backtesting
- [ ] **Phase 2:** Astro-Site, Hash-Versiegelung, Deployment
- [ ] **Phase 3:** LLM-Schicht (Dossier, Adjustierung, Begründungen)
- [ ] **Phase 4:** Kicktipp-Bot (Playwright-Abgabe)
- [ ] **Phase 5:** Selbstlernen, Schattentipper, Dashboard
