"""LLM-Schicht: Begründungstexte + Anpassungsvorschlag (concept.md Schicht 3,
Groq Free Tier).

Begründung: ersetzt die Template-Begründung durch einen vom LLM formulierten
Analysetext, der dieselben Modellzahlen in flüssigerer Sprache einordnet.

Anpassungsvorschlag: Mit News-Schnipseln (engine/sources/news.py) darf das LLM
einen Tipp innerhalb von ±1 Tor vorschlagen – aber nur mit konkretem Grund
(Verletzung, Sperre, Rotation), nicht auf Basis von nichts. Läuft aktuell im
Schatten-Modus: der Vorschlag wird nur geloggt und als eigener Schattentipper
bewertet (siehe evaluate.py), er ändert NICHT den echten/versiegelten Tipp.
Erst wenn Phase 5 belegt, dass er über mehrere Spieltage Punkte bringt, wird
er scharf geschaltet (LLM-Vertrauensregler, engine/learn.py).

Fällt das LLM aus (kein Key, Netzwerkfehler, Rate-Limit) oder gibt es keine
News-Schnipsel, bleibt die Template-Begründung bzw. bleibt die Anpassung aus –
das System bleibt immer funktionsfähig.
"""

import json
import re

import requests

GROQ_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def build_prompt(match_context: dict) -> str:
    """Kurzes Dossier für die LLM-Analyse – Modellzahlen und Quoten, keine News."""
    home, away = match_context["home"], match_context["away"]
    probs = match_context["probabilities"]
    lam, mu = match_context["expected_goals"]
    tip = match_context["tip"]
    lines = [
        f"Fußballspiel: {home} (Heim) gegen {away} (Auswärts), {match_context['stage']}.",
        "Statistisches Modell (Dixon-Coles-Poisson mit ELO-Prior):",
        f"- Heimsieg {probs['home']:.0%}, Remis {probs['draw']:.0%}, Auswärtssieg {probs['away']:.0%}",
        f"- Erwartete Tore: {home} {lam:.2f} : {mu:.2f} {away}",
    ]
    if match_context.get("market_probabilities"):
        m = match_context["market_probabilities"]
        lines.append(
            f"- Buchmacherquoten (entvigt): Heimsieg {m['home']:.0%}, Remis {m['draw']:.0%}, "
            f"Auswärtssieg {m['away']:.0%}"
        )
    lines.append(f"- Punkte-Erwartungswert-optimaler Tipp fürs Kicktipp-Schema: {tip[0]}:{tip[1]}")
    lines.append(
        "Schreibe 3-4 knappe, sachliche Sätze auf Deutsch, die diesen Tipp für Laien "
        "einordnen. Erkläre kurz, warum der Tipp den Punkte-Erwartungswert maximiert "
        "(nicht zwingend das wahrscheinlichste Einzelergebnis ist). Keine Anrede, keine "
        "Überschrift, nur der Fließtext."
    )
    return "\n".join(lines)


def call_groq(
    prompt: str, api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.4, max_tokens: int = 300
) -> str | None:
    """Best-effort Chat-Completion; None bei jedem Fehler (Fallback greift dann)."""
    try:
        resp = requests.post(
            f"{GROQ_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text or None
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"Groq-LLM nicht verfügbar: {exc}")
        return None


def generate_begruendung(
    match_context: dict, api_key: str | None, model: str = DEFAULT_MODEL
) -> tuple[str | None, str]:
    """(text, quelle) – quelle ist "llm" oder "template". text ist None, wenn der
    Aufrufer auf die Template-Begründung zurückfallen soll."""
    if not api_key:
        return None, "template"
    text = call_groq(build_prompt(match_context), api_key, model)
    return (text, "llm") if text else (None, "template")


def build_adjustment_prompt(match_context: dict, news: list[dict]) -> str:
    home, away = match_context["home"], match_context["away"]
    tip = match_context["tip"]
    lines = [
        f"Fußballspiel: {home} (Heim) gegen {away} (Auswärts).",
        f"Statistischer Tipp (Punkte-Erwartungswert-optimal): {tip[0]}:{tip[1]}.",
        "Aktuelle Schlagzeilen (unsortiert, nicht alle relevant):",
    ]
    for item in news:
        lines.append(f"- [{item['source']}] {item['title']}: {item['description']}")
    lines.append(
        "Gibt es unter diesen Schlagzeilen einen KONKRETEN harten Grund (Verletzung/Sperre "
        "eines Schlüsselspielers, Trainerwechsel kurz vor dem Spiel, angekündigte Schonung "
        "vor einem wichtigeren Spiel), der im statistischen Modell nicht steckt und eine "
        "Anpassung um höchstens 1 Tor pro Team rechtfertigt? Wenn nein, oder wenn die "
        "Schlagzeilen nur allgemeine Spielberichte/Analysen ohne harten Fakt sind, antworte "
        "mit adjust=false. Antworte NUR mit einem einzeiligen JSON-Objekt, keine Erklärung "
        "davor oder danach, exakt in diesem Format: "
        '{"adjust": true oder false, "home_delta": -1/0/1, "away_delta": -1/0/1, "grund": "kurzer Satz"}'
    )
    return "\n".join(lines)


def parse_adjustment_response(text: str) -> dict | None:
    """Extrahiert und validiert das JSON-Objekt; None bei jedem Parse-/Schema-
    fehler oder wenn adjust=false (dann gibt es nichts anzuwenden)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or not data.get("adjust"):
        return None

    def clamp(value) -> int | None:
        try:
            return max(-1, min(1, int(value)))
        except (TypeError, ValueError):
            return None

    home_delta, away_delta = clamp(data.get("home_delta")), clamp(data.get("away_delta"))
    if home_delta is None or away_delta is None or (home_delta == 0 and away_delta == 0):
        return None

    return {
        "home_delta": home_delta,
        "away_delta": away_delta,
        "grund": str(data.get("grund", ""))[:300],
    }


def propose_adjustment(
    match_context: dict, news: list[dict], api_key: str | None, model: str = DEFAULT_MODEL
) -> dict | None:
    """Schattentipp-Vorschlag (siehe Modul-Docstring) oder None, wenn keine
    News vorliegen, das LLM ausfällt oder kein harter Grund gefunden wurde."""
    if not api_key or not news:
        return None
    text = call_groq(build_adjustment_prompt(match_context, news), api_key, model, temperature=0.2, max_tokens=200)
    if not text:
        return None
    return parse_adjustment_response(text)
