"""Punkteabrechnung: enthüllte Tipps gegen die realen Ergebnisse.

Liest die öffentlichen Spieltags-Dateien (data/matchdays/) des aktiven
Wettbewerbs, wertet alle enthüllten Tipps mit vorliegendem Ergebnis und
schreibt die Abrechnung nach data/results/. Versiegelte Tipps können und
müssen nicht gewertet werden – ihr Spiel hat noch nicht stattgefunden.
"""

import json

from .config import MATCHDAYS_DIR, PROJECT_ROOT, RESULTS_DIR
from .optimizer import match_points
from .sources.openligadb import fetch_competition
from .teams import normalize


def evaluate_matchday(matchday: dict, results_by_pairing: dict, scheme: dict) -> dict:
    """Rechnet eine Spieltags-Datei ab; Spiele ohne Ergebnis/Tipp bleiben offen."""
    matches, total, scored = [], 0, 0
    counts = {"exact": 0, "goal_diff": 0, "tendency": 0, "miss": 0}
    for m in matchday["matches"]:
        entry = {k: m[k] for k in ("home", "away", "kickoff_utc", "status")}
        result = results_by_pairing.get((normalize(m["home"]), normalize(m["away"])))
        if m["status"] == "revealed" and result is not None:
            points = match_points(tuple(m["tip"]), result, scheme)
            entry.update(tip=m["tip"], result=list(result), points=points)
            total += points
            scored += 1
            key = next(
                (k for k in ("exact", "goal_diff", "tendency") if points == scheme[k] and points > 0),
                "miss",
            )
            counts[key] += 1
        matches.append(entry)

    return {
        "competition": matchday["competition"],
        "season": matchday["season"],
        "matchday": matchday["matchday"],
        "stage": matchday.get("stage"),
        "model_version": matchday.get("model_version"),
        "points_total": total,
        "matches_scored": scored,
        "matches_open": len(matches) - scored,
        "hits": counts,
        "matches": matches,
    }


def main(config: dict) -> None:
    prefix = f"{config['competition']}_{config['season']}_"
    matchday_files = sorted(MATCHDAYS_DIR.glob(f"{prefix}*.json"))
    if not matchday_files:
        print(f"Keine Spieltags-Dateien für {config['competition']} in data/matchdays/ gefunden.")
        return

    scheme = config["kicktipp"]["points"]
    finished = [
        m for m in fetch_competition(config["leagues"], config["season"], force_refresh=True)
        if m.has_result
    ]
    results_by_pairing = {
        (m.home_key, m.away_key): (m.home_goals, m.away_goals) for m in finished
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for md_file in matchday_files:
        matchday = json.loads(md_file.read_text(encoding="utf-8"))
        report = evaluate_matchday(matchday, results_by_pairing, scheme)
        out = RESULTS_DIR / md_file.name
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{md_file.stem}: {report['points_total']} Punkte aus "
            f"{report['matches_scored']} gewerteten Spielen "
            f"({report['matches_open']} offen) -> {out.relative_to(PROJECT_ROOT)}"
        )
