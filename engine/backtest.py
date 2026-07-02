"""Backtesting des Dixon-Coles-Modells mit Kicktipp-Punkteoptimierung.

Simuliert für jede Backtest-Saison einen realen Tippbetrieb: Vor jedem
Spieltag wird das Modell ausschließlich auf bis dahin gespielten Partien
gefittet (inkl. Vorsaisons als Warmup), dann wird pro Spiel der Tipp mit dem
höchsten Punkte-Erwartungswert abgegeben und gegen das reale Ergebnis
abgerechnet. Als Vergleich läuft die naive Baseline "wahrscheinlichstes
Ergebnis" mit.

Aufruf:  python -m engine.backtest
"""

import json
from datetime import datetime, timezone

from .config import PROJECT_ROOT, load_config
from .kicktipp import best_tip, match_points, most_probable_score
from .model import DixonColes
from .openligadb import fetch_season

OUTPUT_FILE = PROJECT_ROOT / "data" / "backtests" / "latest.json"


def run_backtest(config: dict) -> dict:
    league = config["league"]
    scheme = config["kicktipp"]["points"]
    model_cfg = config["model"]
    seasons = config["backtest"]["seasons"]
    lookback = config["backtest"]["lookback_seasons"]

    all_seasons = range(min(seasons) - lookback, max(seasons) + 1)
    matches_by_season = {s: fetch_season(league, s) for s in all_seasons}

    season_reports = []
    all_matches_detail = []

    for season in seasons:
        print(f"\n=== Saison {season}/{str(season + 1)[-2:]} ===")
        history = [
            m
            for s in range(season - lookback, season)
            for m in matches_by_season[s]
            if m.has_result
        ]
        season_matches = [m for m in matches_by_season[season] if m.has_result]
        matchdays = sorted({m.matchday for m in season_matches})

        model = DixonColes(
            xi=model_cfg["time_decay_xi"],
            l2_penalty=model_cfg["l2_penalty"],
            max_goals=model_cfg["max_goals"],
        )

        totals = {"points": 0, "baseline_points": 0, "exact": 0, "goal_diff": 0, "tendency": 0}

        for matchday in matchdays:
            md_matches = [m for m in season_matches if m.matchday == matchday]
            train = history + [m for m in season_matches if m.matchday < matchday]
            ref_date = min(m.kickoff_utc for m in md_matches)
            model.fit(train, ref_date)

            for m in md_matches:
                matrix = model.score_matrix(m.home_id, m.away_id)
                tip, ev = best_tip(matrix, scheme, model_cfg["max_tip_goals"])
                baseline_tip = most_probable_score(matrix)
                result = (m.home_goals, m.away_goals)
                points = match_points(tip, result, scheme)
                baseline_points = match_points(baseline_tip, result, scheme)

                totals["points"] += points
                totals["baseline_points"] += baseline_points
                if points == scheme["exact"]:
                    totals["exact"] += 1
                elif points == scheme["goal_diff"]:
                    totals["goal_diff"] += 1
                elif points == scheme["tendency"]:
                    totals["tendency"] += 1

                all_matches_detail.append(
                    {
                        "season": season,
                        "matchday": matchday,
                        "home": m.home_name,
                        "away": m.away_name,
                        "tip": list(tip),
                        "expected_points": round(ev, 3),
                        "baseline_tip": list(baseline_tip),
                        "result": list(result),
                        "points": points,
                        "baseline_points": baseline_points,
                    }
                )

        n = len(season_matches)
        hits = totals["exact"] + totals["goal_diff"] + totals["tendency"]
        report = {
            "season": season,
            "matches": n,
            "points": totals["points"],
            "points_per_match": round(totals["points"] / n, 3),
            "baseline_points": totals["baseline_points"],
            "exact": totals["exact"],
            "goal_diff": totals["goal_diff"],
            "tendency": totals["tendency"],
            "hit_rate": round(hits / n, 3),
        }
        season_reports.append(report)
        print(
            f"Spiele: {n} | Punkte: {report['points']} (Ø {report['points_per_match']}/Spiel) | "
            f"Baseline (wahrscheinlichstes Ergebnis): {report['baseline_points']}"
        )
        print(
            f"Exakt: {report['exact']} | Tordifferenz: {report['goal_diff']} | "
            f"Tendenz: {report['tendency']} | Trefferquote: {report['hit_rate']:.1%}"
        )

    total_points = sum(r["points"] for r in season_reports)
    total_baseline = sum(r["baseline_points"] for r in season_reports)
    total_matches = sum(r["matches"] for r in season_reports)
    print(
        f"\n=== Gesamt: {total_points} Punkte in {total_matches} Spielen "
        f"(Ø {total_points / total_matches:.3f}/Spiel), Baseline: {total_baseline} ==="
    )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "league": league,
        "scheme": scheme,
        "model": model_cfg,
        "summary": {
            "points": total_points,
            "baseline_points": total_baseline,
            "matches": total_matches,
            "points_per_match": round(total_points / total_matches, 3),
        },
        "seasons": season_reports,
        "matches": all_matches_detail,
    }


def main():
    config = load_config()
    report = run_backtest(config)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Report gespeichert: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
