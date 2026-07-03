"""LLM-Begründungsschicht (concept.md Schicht 3, Groq Free Tier).

Ersetzt die Template-Begründung durch einen vom LLM formulierten Analysetext,
der dieselben Modellzahlen in flüssigerer Sprache einordnet.

Passt (noch) NICHT den Tipp an: eine echte Sanity-Check-Anpassung bräuchte
Kontext wie Verletzungen/Sperren (News-Dossier), den es in dieser Ausbaustufe
noch nicht gibt – ohne solche Fakten wäre eine LLM-Anpassung nur geraten statt
begründet. Das ist Schicht 3 aus concept.md nur teilweise umgesetzt (Begründung
ja, Anpassung erst mit News-Anbindung).

Fällt das LLM aus (kein Key, Netzwerkfehler, Rate-Limit), bleibt die
Template-Begründung aktiv – das System bleibt immer funktionsfähig.
"""

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


def call_groq(prompt: str, api_key: str, model: str = DEFAULT_MODEL) -> str | None:
    """Best-effort Chat-Completion; None bei jedem Fehler (Fallback greift dann)."""
    try:
        resp = requests.post(
            f"{GROQ_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 300,
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text or None
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"Groq-LLM nicht verfügbar, falle auf Template-Begründung zurück: {exc}")
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
