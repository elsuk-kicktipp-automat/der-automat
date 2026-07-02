import json
from datetime import datetime, timezone

from engine.openligadb import fetch_season, parse_matches

SAMPLE_MATCH = {
    "matchID": 77561,
    "matchDateTimeUTC": "2026-05-16T13:30:00Z",
    "group": {"groupName": "34. Spieltag", "groupOrderID": 34, "groupID": 47644},
    "team1": {"teamId": 98, "teamName": "FC St. Pauli", "shortName": "St. Pauli"},
    "team2": {"teamId": 131, "teamName": "VfL Wolfsburg", "shortName": "Wolfsburg"},
    "matchIsFinished": True,
    "matchResults": [
        {"resultTypeID": 1, "resultName": "Halbzeit", "pointsTeam1": 0, "pointsTeam2": 1},
        {"resultTypeID": 2, "resultName": "Endergebnis", "pointsTeam1": 2, "pointsTeam2": 1},
    ],
}

UNFINISHED_MATCH = {
    "matchID": 99999,
    "matchDateTimeUTC": "2026-08-22T13:30:00Z",
    "group": {"groupName": "1. Spieltag", "groupOrderID": 1, "groupID": 50000},
    "team1": {"teamId": 40, "teamName": "FC Bayern München", "shortName": "Bayern"},
    "team2": {"teamId": 7, "teamName": "Borussia Dortmund", "shortName": "Dortmund"},
    "matchIsFinished": False,
    "matchResults": [],
}


class TestParseMatches:
    def test_parses_final_result_not_halftime(self):
        (m,) = parse_matches([SAMPLE_MATCH])
        assert (m.home_goals, m.away_goals) == (2, 1)

    def test_teams_and_matchday(self):
        (m,) = parse_matches([SAMPLE_MATCH])
        assert m.home_id == 98
        assert m.away_id == 131
        assert m.home_name == "FC St. Pauli"
        assert m.matchday == 34

    def test_kickoff_is_utc(self):
        (m,) = parse_matches([SAMPLE_MATCH])
        assert m.kickoff_utc == datetime(2026, 5, 16, 13, 30, tzinfo=timezone.utc)

    def test_finished_flag_and_has_result(self):
        finished, unfinished = parse_matches([SAMPLE_MATCH, UNFINISHED_MATCH])
        assert finished.has_result
        assert not unfinished.has_result
        assert unfinished.home_goals is None

    def test_fallback_to_last_result_entry(self):
        # Ältere Saisons haben teils keinen Eintrag mit resultTypeID 2
        match = dict(SAMPLE_MATCH)
        match["matchResults"] = [
            {"resultTypeID": 1, "resultName": "Halbzeit", "pointsTeam1": 1, "pointsTeam2": 0},
            {"resultTypeID": 3, "resultName": "n.V.", "pointsTeam1": 3, "pointsTeam2": 2},
        ]
        (m,) = parse_matches([match])
        assert (m.home_goals, m.away_goals) == (3, 2)


class TestFetchSeasonCache:
    def test_uses_cache_without_network(self, tmp_path):
        cache_file = tmp_path / "bl1_2025.json"
        cache_file.write_text(json.dumps([SAMPLE_MATCH]), encoding="utf-8")

        matches = fetch_season("bl1", 2025, cache_dir=tmp_path)
        assert len(matches) == 1
        assert matches[0].home_name == "FC St. Pauli"

    def test_writes_cache_on_fetch(self, tmp_path, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return [SAMPLE_MATCH]

        requested_urls = []

        def fake_get(url, timeout):
            requested_urls.append(url)
            return FakeResponse()

        monkeypatch.setattr("engine.openligadb.requests.get", fake_get)

        matches = fetch_season("bl1", 2024, cache_dir=tmp_path)
        assert len(matches) == 1
        assert requested_urls == ["https://api.openligadb.de/getmatchdata/bl1/2024"]
        assert (tmp_path / "bl1_2024.json").exists()

        # Zweiter Aufruf kommt aus dem Cache, kein weiterer Request
        fetch_season("bl1", 2024, cache_dir=tmp_path)
        assert len(requested_urls) == 1
