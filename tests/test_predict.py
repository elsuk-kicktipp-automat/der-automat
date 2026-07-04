from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from engine.predict import (
    build_begruendung,
    in_tip_window,
    build_model,
    load_elo,
    marginal_expected_goals,
    outcome_probabilities,
    resolve_l2_penalty,
)
from engine.sources.openligadb import Match

MODEL_CFG = {
    "time_decay_xi": 0.002,
    "l2_penalty": {"club": 0.2, "national": 5.0},
    "max_goals": 6,
    "max_tip_goals": 5,
    "elo": {"enabled": True, "beta_prior": 0.15, "beta_penalty": 50.0},
}


class TestOutcomeProbabilities:
    def test_sums_to_one_and_splits_correctly(self):
        matrix = np.zeros((3, 3))
        matrix[1, 0] = 0.5   # Heimsieg
        matrix[1, 1] = 0.3   # Remis
        matrix[0, 2] = 0.2   # Auswärtssieg
        probs = outcome_probabilities(matrix)
        assert probs["home"] == pytest.approx(0.5)
        assert probs["draw"] == pytest.approx(0.3)
        assert probs["away"] == pytest.approx(0.2)


class TestMarginalExpectedGoals:
    def test_matches_simple_distribution(self):
        matrix = np.zeros((3, 3))
        matrix[2, 0] = 0.5  # 2:0
        matrix[0, 1] = 0.5  # 0:1
        lam, mu = marginal_expected_goals(matrix)
        assert lam == pytest.approx(1.0)  # 0.5*2 + 0.5*0
        assert mu == pytest.approx(0.5)  # 0.5*0 + 0.5*1


class TestLoadEloResilience:
    def test_network_failure_returns_none_instead_of_raising(self, monkeypatch):
        import requests

        class FailingSource:
            def ratings(self, on_date):
                raise requests.ConnectionError("eloratings.net blockiert gerade")

        monkeypatch.setattr("engine.predict.make_elo_source", lambda team_type: FailingSource())
        config = {"model": {"elo": {"enabled": True}}}
        assert load_elo(config, "national") is None

    def test_disabled_returns_none_without_network_call(self, monkeypatch):
        def fail_if_called(team_type):
            raise AssertionError("make_elo_source darf bei enabled=False nicht aufgerufen werden")

        monkeypatch.setattr("engine.predict.make_elo_source", fail_if_called)
        config = {"model": {"elo": {"enabled": False}}}
        assert load_elo(config, "club") is None


class TestL2PerTeamType:
    def test_dict_resolves_per_team_type(self):
        assert resolve_l2_penalty(MODEL_CFG, "club") == 0.2
        assert resolve_l2_penalty(MODEL_CFG, "national") == 5.0

    def test_plain_float_applies_to_both(self):
        cfg = {**MODEL_CFG, "l2_penalty": 0.1}
        assert resolve_l2_penalty(cfg, "club") == 0.1
        assert resolve_l2_penalty(cfg, "national") == 0.1

    def test_build_model_uses_team_type(self):
        config = {"model": MODEL_CFG}
        assert build_model(config, True, "national").l2_penalty == 5.0
        assert build_model(config, False, "club").l2_penalty == 0.2
        assert build_model(config, True, "national").neutral_venue is True


def _match(home="Australien", away="Ägypten"):
    return Match(
        home_name=home,
        away_name=away,
        home_goals=None,
        away_goals=None,
        kickoff_utc=datetime(2026, 7, 3, 18, 0, tzinfo=timezone.utc),
        matchday=4,
        stage_name="Sechzehntelfinale",
        finished=False,
    )


class TestBuildBegruendungAdvanceTip:
    def test_appends_shootout_sentence_when_advance_tip_given(self):
        probs = {"home": 0.355, "draw": 0.371, "away": 0.274}
        advance_tip = {"pick": "Australien", "probability": 0.564}
        text = build_begruendung(_match(), 1.18, 1.02, probs, (1, 1), 1.088, advance_tip)
        assert "Elfmeterschießen" in text
        assert "Australien" in text
        assert "56%" in text or "56 %" in text

    def test_no_shootout_sentence_without_advance_tip(self):
        probs = {"home": 0.355, "draw": 0.371, "away": 0.274}
        text = build_begruendung(_match(), 1.18, 1.02, probs, (1, 1), 1.088)
        assert "Elfmeterschießen" not in text


class TestBuildBegruendungFactors:
    """Die Begründung soll Quellen laienverständlich einordnen."""

    PROBS = {"home": 0.49, "draw": 0.31, "away": 0.20}

    def test_mentions_elo_values_when_present(self):
        text = build_begruendung(
            _match(), 1.73, 1.09, self.PROBS, (2, 1), 1.28,
            elo={"home": 1683.0, "away": 1608.0},
        )
        assert "ELO-Zahlen" in text
        assert "Australien" in text

    def test_omits_elo_sentence_when_absent(self):
        text = build_begruendung(_match(), 1.73, 1.09, self.PROBS, (2, 1), 1.28, elo=None)
        assert "ELO-Bewertung" not in text

    def test_mentions_market_odds_with_weight(self):
        text = build_begruendung(
            _match(), 1.73, 1.09, self.PROBS, (2, 1), 1.28,
            market_probs={"home": 0.45, "draw": 0.30, "away": 0.25}, market_weight=0.7,
        )
        assert "Quoten" in text
        assert "Australien" in text

    def test_omits_market_sentence_when_weight_zero(self):
        text = build_begruendung(
            _match(), 1.73, 1.09, self.PROBS, (2, 1), 1.28,
            market_probs={"home": 0.45, "draw": 0.30, "away": 0.25}, market_weight=0.0,
        )
        assert "Buchmacherquoten" not in text

    def test_mentions_llm_adjustment_as_shadow_only(self):
        text = build_begruendung(
            _match(), 1.73, 1.09, self.PROBS, (2, 1), 1.28,
            llm_adjustment={"tip": [1, 1], "grund": "Stammtorwart fehlt", "news_count": 3},
        )
        assert "Stammtorwart fehlt" in text
        assert "Schattentipp" in text

    def test_mentions_news_checked_without_finding(self):
        text = build_begruendung(_match(), 1.73, 1.09, self.PROBS, (2, 1), 1.28, news_checked=3)
        assert "3 aktuelle Schlagzeile" in text

    def test_mentions_no_news_found(self):
        text = build_begruendung(_match(), 1.73, 1.09, self.PROBS, (2, 1), 1.28, news_checked=0)
        assert "keine relevante aktuelle Nachricht" in text


class TestTipWindow:
    """Fairness + Aktualität: getippt wird nur im Fenster (now+margin, now+window]."""

    NOW = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)
    WINDOW = timedelta(hours=4)
    MARGIN = timedelta(minutes=20)

    def _match_at(self, kickoff, finished=False, home="Kanada", away="Marokko"):
        return Match(
            home_name=home, away_name=away, home_goals=None, away_goals=None,
            kickoff_utc=kickoff, matchday=5, stage_name="Achtelfinale", finished=finished,
        )

    def test_match_within_window_is_tippable(self):
        m = self._match_at(datetime(2026, 7, 4, 15, 0, tzinfo=timezone.utc))  # in 3h
        assert in_tip_window(m, self.NOW, self.WINDOW, self.MARGIN)

    def test_match_too_far_in_future_is_not_tippable(self):
        m = self._match_at(datetime(2026, 7, 4, 17, 0, tzinfo=timezone.utc))  # in 5h
        assert not in_tip_window(m, self.NOW, self.WINDOW, self.MARGIN)

    def test_match_already_started_is_never_tippable(self):
        m = self._match_at(datetime(2026, 7, 4, 11, 0, tzinfo=timezone.utc))  # vor 1h
        assert not in_tip_window(m, self.NOW, self.WINDOW, self.MARGIN)

    def test_match_inside_safety_margin_is_not_tippable(self):
        m = self._match_at(datetime(2026, 7, 4, 12, 10, tzinfo=timezone.utc))  # in 10min < 20min Reserve
        assert not in_tip_window(m, self.NOW, self.WINDOW, self.MARGIN)

    def test_window_boundary_is_inclusive(self):
        m = self._match_at(datetime(2026, 7, 4, 16, 0, tzinfo=timezone.utc))  # exakt in 4h
        assert in_tip_window(m, self.NOW, self.WINDOW, self.MARGIN)

    def test_finished_or_placeholder_is_not_tippable(self):
        m = self._match_at(datetime(2026, 7, 4, 15, 0, tzinfo=timezone.utc), finished=True)
        assert not in_tip_window(m, self.NOW, self.WINDOW, self.MARGIN)
        placeholder = self._match_at(datetime(2026, 7, 4, 15, 0, tzinfo=timezone.utc), home="Sieger SF 12")
        assert not in_tip_window(placeholder, self.NOW, self.WINDOW, self.MARGIN)
