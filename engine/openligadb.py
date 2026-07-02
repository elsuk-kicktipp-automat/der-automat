"""OpenLigaDB-Client (api.openligadb.de, kein API-Key nötig).

Lädt komplette Saisons und cacht die Roh-JSON-Antworten unter data/cache/,
damit Backtests nicht bei jedem Lauf die API belasten.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import requests

from .config import PROJECT_ROOT

API_BASE = "https://api.openligadb.de"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# resultTypeID 2 = "Endergebnis" (nach 90 Minuten), 1 = Halbzeit
FINAL_RESULT_TYPE_ID = 2


@dataclass(frozen=True)
class Match:
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    home_goals: int | None
    away_goals: int | None
    kickoff_utc: datetime
    matchday: int
    finished: bool

    @property
    def has_result(self) -> bool:
        return self.finished and self.home_goals is not None and self.away_goals is not None


def _extract_final_score(match_json: dict) -> tuple[int | None, int | None]:
    results = match_json.get("matchResults") or []
    final = next(
        (r for r in results if r.get("resultTypeID") == FINAL_RESULT_TYPE_ID),
        results[-1] if results else None,
    )
    if final is None:
        return None, None
    return final.get("pointsTeam1"), final.get("pointsTeam2")


def parse_matches(raw: list[dict]) -> list[Match]:
    matches = []
    for m in raw:
        home_goals, away_goals = _extract_final_score(m)
        matches.append(
            Match(
                home_id=m["team1"]["teamId"],
                away_id=m["team2"]["teamId"],
                home_name=m["team1"]["teamName"],
                away_name=m["team2"]["teamName"],
                home_goals=home_goals,
                away_goals=away_goals,
                kickoff_utc=datetime.fromisoformat(
                    m["matchDateTimeUTC"].replace("Z", "+00:00")
                ),
                matchday=m["group"]["groupOrderID"],
                finished=bool(m.get("matchIsFinished")),
            )
        )
    return matches


def fetch_season(
    league: str,
    season: int,
    cache_dir: Path = CACHE_DIR,
    force_refresh: bool = False,
) -> list[Match]:
    """Alle Spiele einer Saison, aus dem Cache oder frisch von der API."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{league}_{season}.json"

    if cache_file.exists() and not force_refresh:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        resp = requests.get(f"{API_BASE}/getmatchdata/{league}/{season}", timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    return parse_matches(raw)
