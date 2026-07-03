from engine.evaluate import evaluate_matchday

SCHEME = {"exact": 4, "goal_diff": 3, "tendency": 2}

MATCHDAY = {
    "competition": "wm2026",
    "season": 2026,
    "matchday": 5,
    "stage": "Achtelfinale",
    "model_version": "dixon-coles-elo-1",
    "matches": [
        {"home": "Kanada", "away": "Marokko", "kickoff_utc": "2026-07-04T17:00:00Z",
         "status": "revealed", "tip": [1, 1]},
        {"home": "Paraguay", "away": "Frankreich", "kickoff_utc": "2026-07-04T21:00:00Z",
         "status": "revealed", "tip": [0, 2]},
        {"home": "Brasilien", "away": "Norwegen", "kickoff_utc": "2026-07-05T20:00:00Z",
         "status": "sealed", "hash": "ab" * 32},
    ],
}

RESULTS = {
    ("kanada", "marokko"): (2, 2),      # Tipp 1:1 -> Tendenz (2), kein Differenz-Punkt
    ("paraguay", "frankreich"): (0, 2), # Tipp 0:2 -> exakt (4)
    ("brasilien", "norwegen"): (1, 0),  # versiegelt -> darf nicht gewertet werden
}


class TestEvaluateMatchday:
    def test_scores_revealed_matches(self):
        report = evaluate_matchday(MATCHDAY, RESULTS, SCHEME)
        assert report["points_total"] == 6
        assert report["matches_scored"] == 2
        assert report["hits"] == {"exact": 1, "goal_diff": 0, "tendency": 1, "miss": 0}

    def test_sealed_matches_stay_open_without_leaking_tip(self):
        report = evaluate_matchday(MATCHDAY, RESULTS, SCHEME)
        sealed_entry = report["matches"][2]
        assert report["matches_open"] == 1
        assert sealed_entry["status"] == "sealed"
        assert "tip" not in sealed_entry
        assert "points" not in sealed_entry

    def test_revealed_without_result_stays_open(self):
        results = {("paraguay", "frankreich"): (0, 2)}
        report = evaluate_matchday(MATCHDAY, results, SCHEME)
        assert report["matches_scored"] == 1
        assert report["matches_open"] == 2
