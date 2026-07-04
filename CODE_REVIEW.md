# Code Review: Elsuk-kicktipp-automat

**Datum:** 04.07.2026  
**Reviewer:** Cline (automatisiert)  
**Scope:** Gesamtes Repository (`engine/`, `tests/`, `site/`, `.github/`, `config.yaml`)  
**Getestet:** `python -m pytest tests/` → **143 passed in 0.88s** (ohne Netzwerkzugriff)

---

## 1. Zusammenfassung

Das Projekt ist **beeindruckend durchdacht** für ein Hobby-/Lernprojekt. Die Architektur
ist klar geschichtet, das Konzept (`concept.md`) und die Umsetzung (`README.md`) sind
hervorragend dokumentiert. Die Fairness-Mechanismus (Hash-Versiegelung) und die Graceful-
Degradation (Best-Effort bei ELO, Quoten, LLM) zeigen Reife.

| Aspekt | Bewertung |
|---|---|
| Architektur & Strukturierung | ⭐⭐⭐⭐⭐ |
| Dokumentation | ⭐⭐⭐⭐⭐ |
| Testabdeckung & -qualität | ⭐⭐⭐⭐☆ |
| Robustheit/Fehlerbehandlung | ⭐⭐⭐☆☆ |
| Sicherheit | ⭐⭐⭐☆☆ |
| Code-Stil & Konsistenz | ⭐⭐⭐⭐☆ |

**Empfohlene Priorität der Behebungen:**

1. 🔴 **Hoch:** `odds.py` potenzieller `KeyError`, `model.py` fehlende Konvergenzprüfung
2. 🟡 **Mittel:** Grammatikfehler in Begründungstexten, Logging statt `print()`
3. 🟢 **Niedrig:** Pinning in `requirements.txt`, Typ-Annotationen, Ineffizienzen

---

## 2. Stärken

### 2.1 Architektur
- **Schichtentrennung** ist exzellent: Sources → Model → Optimizer → Market → LLM → Seal → Evaluate.
  Jede Schicht hat klare Verantwortlichkeiten und saubere Schnittstellen.
- **Datenbank-als-Repo**-Pattern (versionierte JSON-Dateien) ist für das Fairness-Konzept
  ideal: Jede Prognose ist nachvollziehbar, jeder Modellstand dokumentiert.
- **GitHub-Actions als "Server"** mit `concurrency.group: data-updates` verhindert Race
  Conditions zwischen `spieltag.yml` und `unseal.yml`.

### 2.2 Graceful Degradation
Das System ist durchgängig resilient: Jede externe Abhängigkeit (ELO, Quoten, LLM, News)
fällt bei Fehler sauber auf einen Default zurück, ohne die Pipeline zu brechen:

```python
# engine/predict.py - exemplarisch für alle Quellen
def load_elo(config, team_type, on_date=None):
    if not config["model"]["elo"]["enabled"]:
        return None
    try:
        return make_elo_source(team_type).ratings(on_date)
    except requests.RequestException as exc:
        print(f"ELO-Ratings nicht verfügbar ({team_type}): {exc}")
        return None  # ← Modell läuft ohne ELO weiter
```

### 2.3 Fairness-Mechanismus
- SHA-256-Hash der kanonischen Payload + Fernet-Verschlüsselung bis Anstoß +5min.
- Test `test_revealed_hash_is_verifiable` beweist, dass der Hash nachrechenbar ist.
- Salt pro Spiel verhindert Regenbomb-Attacken auf die Hashes.

### 2.4 Testqualität
- 143 Tests, alle ohne Netzwerk lauffähig (Mocking/Synthetic Data).
- Modell-Tests decken Edge Cases ab (Warmstart, leeres Training, neutrales Venue, ELO-Term).
- Optimizer-Tests prüfen EV-Optimierung gegen naive "wahrscheinlichstes Ergebnis"-Strategie.

### 2.5 Schatten-Tipper als Benchmarking
Konsequente Mitführung von Vergleichsstrategien (`most_probable`, `elo_favorite`,
`always_draw`, `llm_adjusted`) erlaubt empirische Bewertung jeder Schicht.

---

## 3. Gefundene Probleme

### 3.1 🔴 Bugs & Fehler

#### 3.1.1 `engine/sources/odds.py` — Potenzieller `KeyError`
**Datei:** `engine/sources/odds.py`, Zeilen 86–88  
**Severity:** Hoch (kann Pipeline brechen)

```python
# Aktuell:
home_p = devigged.get(event["home_team"])   # ← KeyError wenn "home_team" fehlt
away_p = devigged.get(event["away_team"])   # ← KeyError wenn "away_team" fehlt
draw_p = devigged.get("Draw")
```

Die äußere Schleife prüft mit `.get()` auf `None`, aber innerorts wird mit `event["home_team"]`
direkt zugegriffen. Die Odds API kann (selten) Events ohne `home_team` liefern.

**Empfehlung:**
```python
home_team = event.get("home_team")
away_team = event.get("away_team")
if home_team is None or away_team is None:
    continue
home_p = devigged.get(home_team)
away_p = devigged.get(away_team)
```

#### 3.1.2 `engine/model.py` — Fehlende Konvergenzprüfung
**Datei:** `engine/model.py`, Zeile 149  
**Severity:** Mittel (stille Fehlprognosen)

```python
result = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
theta = result.x  # ← result.success wird nie geprüft
```

Bei Nicht-Konvergenz (`result.success == False`) werden trotzdem Parameter verwendet.
Bei Daten mit pathologischer Verteilung (z.B. nur ein Team mit Toren) kann das zu
ungültigen Prognosen führen, ohne dass es bemerkt wird.

**Empfehlung:**
```python
if not result.success:
    print(f"⚠️ Dixon-Coles fit konvergierte nicht: {result.message}")
    # Fallback auf Priors oder letztes params
```

#### 3.1.3 `engine/predict.py` & `engine/llm.py` — Grammatikfehler in Begründung
**Datei:** `engine/predict.py` Zeilen 118–120, `engine/llm.py` Zeile 60  
**Severity:** Niedrig (User-Facing Text)

```python
f"Von {llm_adjustment['news_count']} geprüften aktuellen Schlagzeilen lieferte eine "
f"einen möglichen Grund für eine Anpassung auf {llm_adjustment['tip'][0]}:"
```

„lieferte **eine einen**" → sollte „**einen**" heißen (oder „eine der Schlagzeilen lieferte
einen").

#### 3.1.4 `engine/evaluate.py` — Trefferquoten-Zuordnung fehleranfällig
**Datei:** `engine/evaluate.py`, Zeilen 116–119  
**Severity:** Mittel (bei benutzerdefinierten Schemas falsche Statistik)

```python
key = next(
    (k for k in ("exact", "goal_diff", "tendency") if points == scheme[k] and points > 0),
    "miss",
)
```

Wenn zwei Kategorien denselben Punktwert haben (z.B. `tendency: 3, goal_diff: 3`),
wird immer `goal_diff` zurückgegeben, auch wenn es `tendency` war.

**Empfehlung:** Stattdessen `match_points` so erweitern, dass es die Kategorie zurückgibt,
oder die Trefferkategorie direkt im Optimizer berechnen.

#### 3.1.5 `engine/kicktipp_bot.py` — Selektor `.col1/.col2` positionsbasiert
**Datei:** `engine/kicktipp_bot.py`, Zeilen 120–121  
**Severity:** Mittel (Bot bricht bei Layoutänderung)

```python
home_el = row.locator(".col1").first
away_el = row.locator(".col2").first
```

`.col1`/`.col2` sind CSS-Klassen, die Kicktipp jederzeit umbenennen kann. Robuster wäre,
nach Text-inhalt zu matchen oder strukturellere Selektoren zu verwenden.

**Dokumentiert**, aber erwähnenswert für die Langzeitstabilität.

#### 3.1.6 `config.yaml` — Live-Schaltung des Kicktipp-Bots
**Datei:** `config.yaml`, Zeile 87  
**Severity:** Hoch (potenziell schädlich)

```yaml
kicktipp_submission:
  enabled: true
  dry_run: false  # ← LIVE! Scharf geschaltet seit 04.07.2026
```

Der Bot gibt jetzt echte Tipps ab. Bei einem Bug im Bot oder einer Layoutänderung
bei Kicktipp können falsche Tipps abgegeben werden, ohne dass es ein einfaches
Rollback gibt.

**Empfehlung:** Zumindest ein automatischer Notifications-Mechanismus bei Fehlern
(`unmatched`-Liste nicht leer, `filled` != erwartete Anzahl).

### 3.2 🟡 Design-Verbesserungen

#### 3.2.1 Überall `print()` statt `logging`
**Dateien:** Alle `engine/*.py`  
**Severity:** Niedrig, aber produktionsrelevant

```python
print(f"ELO-Ratings nicht verfügbar ({team_type}): {exc}")
print("ODDS_API_KEY fehlt, laufe ohne Quoten-Prior.")
```

In GitHub Actions ist `print` in Ordnung, aber für späteres Debugging (Phase 5:
Selbstlernen) wäre `logging` mit Logleveln (`INFO`/`WARNING`/`ERROR`) sauberer und
würde es erlauben, Diagnose-Output zu filtern.

#### 3.2.2 `requirements.txt` ohne Versionen
**Datei:** `requirements.txt`  
**Severity:** Mittel (Reproduzierbarkeit)

```
requests
numpy
scipy
PyYAML
pytest
cryptography
playwright
```

Kein Pinning. Ein `numpy>=2.0` oder ein API-Break in `scipy` kann die CI ohne Vorwarnung
brechen. Für ein Forschungsprojekt akzeptabel, aber für den Live-Betrieb riskant.

**Empfehlung:** `pip-compile` oder `uv pip compile` nutzen, um ein `requirements.lock`
zu erzeugen.

#### 3.2.3 `engine/predict.py` — `trained_on_matches` pro Spiel neu berechnet
**Datei:** `engine/predict.py`, Zeilen 266 & 305  
**Severity:** Niedrig (Performance)

```python
"trained_on_matches": len([t for t in train if t.has_result]),
```

Diese List-Comprehension wird pro Spiel in der Schleife ausgeführt, obwohl `train` sich
nicht ändert. Bei 9 Spielen und 1000 Trainingsmatches harmlos, aber unnötig.

**Empfehlung:** Vor der Schleife einmal berechnen:
```python
trained_n = len([t for t in train if t.has_result])
```

#### 3.2.4 `engine/backtest.py` — Asymmetrisches `force_refresh`
**Datei:** `engine/backtest.py`, Zeile 141 vs. 92  
**Severity:** Niedrig

`backtest_national` nutzt `force_refresh=True`, `backtest_club` nicht. Das führt zu:
- National-Backtest zieht immer frisch (Netzwerk nötig)
- Club-Backtest läuft mit potenziell veraltetem Cache

Ist im `CLAUDE_HANDOFF.md` dokumentiert, aber inkonsistent.

#### 3.2.5 `engine/sources/openligadb.py` — `_extract_final_score` Fallback
**Datei:** `engine/sources/openligadb.py`, Zeilen 62–69  
**Severity:** Mittel (falsche Ergebnisse möglich)

```python
final = next(
    (r for r in results if r.get("resultTypeID") == FINAL_RESULT_TYPE_ID),
    results[-1] if results else None,  # ← Fallback auf letztes Ergebnis
)
```

Wenn `resultTypeID == 2` fehlt, wird das letzte Ergebnis verwendet. Das könnte ein
Halbzeitstand oder ein "nach Elfmeterschießen"-Ergebnis sein. Für Backtests kritisch,
denn es kann die Modellqualität verfälschen.

**Empfehlung:** Bei Fallback warnen oder `None` zurückgeben, um das Match zu überspringen.

#### 3.2.6 `engine/seal.py` — Fernet-Key ohne KDF
**Datei:** `engine/seal.py`, Zeilen 41–43  
**Severity:** Mittel (Sicherheit)

```python
def _fernet(secret: str) -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key)
```

Ein einfacher SHA-256-Hash ohne Salt/Iteration ist anfällig für Wörterbuchattacken, falls
`SEAL_SECRET` schwach ist. Für ein Hobby-Projekt mit zufälligem Secret akzeptabel, aber
eine PBKDF2/HKDF-basierte Ableitung wäre sicherer.

### 3.3 🟢 Code-Stil & Kleinigkeiten

#### 3.3.1 Inkonsistente Typ-Annotationen
- `engine/model.py`: Vollständig annotiert.
- `engine/sources/openligadb.py`: Kaum Annotationen (z.B. `_extract_final_score`).
- `engine/market.py`: Teilweise.

Einheitliche Annotationen würden Lesbarkeit und Tooling (Mypy, IDE-Support) verbessern.

#### 3.3.2 `engine/sources/news.py` — Sortierung mit `None`
**Datei:** `engine/sources/news.py`, Zeile 88  
**Severity:** Niedrig

```python
relevant.sort(key=lambda i: i["published"] or cutoff, reverse=True)
```

Wenn `published` None ist, wird `cutoff` als Sortierschlüssel verwendet. Semantisch
unklar (ein Feed ohne Datum gilt als "genau so aktuell wie der cutoff"). Funktioniert,
aber ein Datum-Default wie `datetime.min` wäre klarer.

#### 3.3.3 `engine/predict.py` — `_covered_pairings` ohne Schema-Validierung
**Datei:** `engine/predict.py`, Zeile 368  
**Severity:** Niedrig

```python
covered |= {(m["home"], m["away"]) for m in data["matches"]}
```

Wenn eine JSON-Datei beschädigt ist (kein `matches`-Schlüssel), bricht die Pipeline
mit `KeyError`. Ein `data.get("matches", [])` wäre robuster.

#### 3.3.4 `.env.example` — Inkonsistenz Kommentierung
**Datei:** `.env.example`, Zeilen 19–21  
**Severity:** Sehr niedrig

```
# Kicktipp-Bot (Phase 4) – eigener Bot-Account, nie der persönliche:
# KICKTIPP_EMAIL=
# KICKTIPP_PASSWORD=
# KICKTIPP_RUNDE=
```

Die Kicktipp-Variablen sind auskommentiert, obwohl der Bot in `config.yaml` schon
scharf geschaltet ist (`dry_run: false`). Entweder die Variablen aktivieren oder
klarstellen, dass sie nur optional benötigt werden.

#### 3.3.5 `engine/llm.py` — Kein Rate-Limit-Handling
**Datei:** `engine/llm.py`, Zeile 99  
**Severity:** Niedrig

```python
except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
    print(f"Groq-LLM nicht verfügbar: {exc}")
    return None
```

Bei HTTP 429 (Rate Limit) gibt es keinen Retry. Für Free-Tier mit niedrigem Limit
könnte ein exponential-backoff helfen. Aktuell ist es als "best-effort" akzeptabel.

### 3.4 Architektur-Beobachtungen

#### 3.4.1 Keine Daten-Schema-Validierung
**Severity:** Mittel

JSON-Dateien in `data/matchdays/`, `data/results/`, `data/predictions/` haben kein
formales Schema. Ein kaputtes JSON (z.B. durch einen halb fertigen Commit) würde
die Astro-Site kaputt machen. `pydantic` oder JSON-Schema würden helfen.

#### 3.4.2 Keine Metriken-/Monitoring-Schicht
**Severity:** Niedrig (für aktuelle Phase)

Phase 5 plant Selbstlernen mit Lernkurve. Aktuell gibt es keine zentrale
Metriken-Sammlung (Brier-Score, Punkte/Spiel) über alle Saisons hinweg. Die
Backtests schreiben JSON, aber es gibt keine Aggregation/Visualisierung der
Modellentwicklung über die Zeit.

#### 3.4.3 Abhängigkeit von `unseal.yml` alle 30 Minuten
**Severity:** Niedrig

Der `unseal`-Workflow läuft alle 30 Minuten, auch wenn nichts fällig ist. Das
vorangestellte `jq`-Check (`steps.check`) verhindert unnötige Python-Ausführung,
aber der Checkout passiert trotzdem. Für GitHub Free (2000 min/Monat) verschmerzbar.

---

## 4. Sicherheits-Review

### 4.1 🔴 Secrets-Handling
- `.env` ist korrekt gitignored ✓
- Secrets kommen nur aus Environment/GitHub Actions Secrets ✓
- `load_dotenv` nutzt `setdefault`, sodass ENV-Variablen Vorrang haben ✓

### 4.2 🟡 Live-Tippabgabe ohne Fallback-Notification
**Datei:** `engine/kicktipp_bot.py`, `spieltag.yml`  
**Severity:** Mittel

Der `concept.md` §6 empfiehlt: „Fallback, falls die Abgabe scheitert: Benachrichtigung
per E-Mail/ntfy.sh". Aktuell gibt es nur `print()` im Log, keine aktive Benachrichtigung
bei:
- `log["unmatched"]` nicht leer
- `log["filled"]` entspricht nicht der erwarteten Anzahl
- `RuntimeError` beim Login

**Empfehlung:** Schritt im `spieltag.yml` mit `if: failure()` einen ntfy.sh-Webhook
oder eine GitHub-Issue anlegen.

### 4.3 🟢 Fairness-Mechanismus
- SHA-256 der kanonischen Payload ist kryptografisch solide ✓
- Salt pro Spiel (16 Bytes `secrets.token_hex`) ✓
- Fernet (AES-128-CBC + HMAC) ist angemessen ✓
- Commit-Zeitstempel als zusätzlicher Beweis ✓

---

## 5. Test-Review

### 5.1 Abdeckung
- **143 Tests**, alle ohne Netzwerk ✓
- Modell, Optimizer, Seal, Predict, ELO, Evaluate, Kicktipp-Bot, LLM, Market, News,
  OpenLigaDB, Odds sind abgedeckt.
- Edge Cases: Warmstart, leeres Training, neutrales Venue, K.o.-Spiele ✓

### 5.2 Lücken
- **`engine/sources/odds.py`**: `parse_probabilities` hat keinen Test mit
  fehlendem `home_team`-Feld (siehe Bug 3.1.1).
- **`engine/model.py`**: Kein Test für Nicht-Konvergenz (`result.success == False`).
- **`engine/evaluate.py`**: `load_manual_results` wird getestet, aber
  `_advance_sides` mit `None`-Advancers nicht vollständig.
- **Site-Komponenten** (`site/src/`): Keine Tests. Astro-Build-Check in `test.yml`
  reicht für Struktur, aber nicht für Logik in `data.mjs`.

### 5.3 Empfehlung für neue Tests
```python
# tests/test_odds.py
def test_parse_probabilities_skips_event_without_home_team():
    raw = [{"away_team": "Team B", "bookmakers": [...]}]  # home_team fehlt
    assert parse_probabilities(raw) == {}

# tests/test_model.py
def test_fit_handles_non_convergence_gracefully(monkeypatch):
    # minimize.success = False simulieren, prüfen dass Priors bleiben
```

---

## 6. Astro-Site (`site/`)

### 6.1 Stärken
- Statischer Build, keine Runtime-Logik ✓
- `data.mjs` liest JSON zur Build-Zeit ✓
- Saubere Trennung von Layout, Pages, Components ✓
- `CLAUDE_HANDOFF.md` dokumentiert die „Build-Falle" in `bilanz.astro` ✓

### 6.2 Beobachtungen
- `site/src/lib/data.mjs` hat keine Fehlerbehandlung für nicht-JSON-Dateien
  (`.filter(f => f.endsWith('.json'))` reicht nicht, wenn JSON malformed ist).
- `MatchCard.astro` nutzt `factors.probabilities` direkt — falls ein altes Match
  ohne `factors`-Struktur geladen wird, fällt es auf `{}` zurück. Gut.
- Keine CSP-Headers oder Meta-Tags für Sicherheit.

---

## 7. Priorisierte Empfehlungen

### Sofort (Live-Betrieb)
1. **`odds.py` KeyError fixen** (Bug 3.1.1) — 5 Minuten.
2. **Fehler-Notification für Kicktipp-Bot** einrichten (ntfy.sh-Webhook) — 30 Minuten.
3. **Konvergenz-Warnung in `model.py`** (Bug 3.1.2) — 10 Minuten.

### Bald
4. **`requirements.txt` pinning** mit `uv pip compile` — 15 Minuten.
5. **`logging` statt `print()`** — 1–2 Stunden für alle Module.
6. **`evaluate.py` Trefferquoten-Zuordnung** korrigieren (Bug 3.1.4) — 30 Minuten.
7. **Grammatikfehler** in Begründungstexten (Bug 3.1.3) — 5 Minuten.

### Langfristig
8. **Daten-Schema-Validierung** (z.B. pydantic für Prediction/Matchday) — 2–3 Stunden.
9. **Metriken-Aggregation** für Phase 5 (Selbstlernen) —konzeptieren.
10. **Type-Annotationen** vereinheitlichen — 1 Stunde.

---

## 8. Fazit

„Der Automat" ist ein **beeindruckend reifes Hobby-Projekt** mit durchdachter
Architektur, exzellenter Dokumentation und solider Testabdeckung. Das Fairness-Konzept
und die Graceful-Degradation sind herausragend.

Die gefundenen Probleme sind überwiegend **klein bis mittel** — kein kritischer
Architekturfehler, keine sicherheitskritische Lücke. Die wichtigste Behebung ist der
`KeyError` in `odds.py` und die fehlende Fehler-Notification beim (jetzt live
geschalteten) Kicktipp-Bot.

**Gesamtnote: B+ (sehr gut für ein Projekt in dieser Phase)**

Mit den oben genannten Behebungen und `logging` statt `print` wäre es A- bis A.