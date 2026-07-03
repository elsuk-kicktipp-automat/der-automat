from datetime import datetime, timedelta, timezone

import pytest

from engine.market import blend_with_market, outcome_probs
from engine.model import DixonColes
from engine.sources.openligadb import Match

REF_DATE = datetime(2026, 5, 1, tzinfo=timezone.utc)


def make_match(home, away, hg, ag, days_before_ref=100):
    return Match(
        home_name=home,
        away_name=away,
        home_goals=hg,
        away_goals=ag,
        kickoff_utc=REF_DATE - timedelta(days=days_before_ref),
        matchday=1,
        stage_name="1. Runde",
        finished=True,
    )


@pytest.fixture(scope="module")
def fitted():
    """Team A klar stärker als Team B, ohne Quoten fitten."""
    matches = [make_match("Team A", "Team B", 2, 0, d) for d in range(5)]
    matches += [make_match("Team B", "Team A", 0, 2, d + 5) for d in range(5)]
    model = DixonColes(xi=0.0, l2_penalty=0.5)
    model.fit(matches, REF_DATE)
    return model


class TestOutcomeProbs:
    def test_sums_to_one(self, fitted):
        matrix = fitted.score_matrix("teama", "teamb")
        probs = outcome_probs(matrix)
        assert sum(probs.values()) == pytest.approx(1.0)


class TestBlendWithMarket:
    def test_weight_zero_keeps_model_matrix(self, fitted):
        model_matrix = fitted.score_matrix("teama", "teamb")
        model_probs = outcome_probs(model_matrix)
        market_probs = {"home": 0.1, "draw": 0.2, "away": 0.7}  # starker Kontrast zum Modell

        blended = blend_with_market(fitted, "teama", "teamb", market_probs, weight=0.0)
        assert outcome_probs(blended)["home"] == pytest.approx(model_probs["home"], abs=0.01)

    def test_weight_one_matches_market_closely(self, fitted):
        market_probs = {"home": 0.2, "draw": 0.3, "away": 0.5}
        blended = blend_with_market(fitted, "teama", "teamb", market_probs, weight=1.0)
        result = outcome_probs(blended)
        assert result["home"] == pytest.approx(market_probs["home"], abs=0.02)
        assert result["draw"] == pytest.approx(market_probs["draw"], abs=0.02)
        assert result["away"] == pytest.approx(market_probs["away"], abs=0.02)

    def test_partial_weight_is_between(self, fitted):
        model_probs = outcome_probs(fitted.score_matrix("teama", "teamb"))
        market_probs = {"home": 0.1, "draw": 0.2, "away": 0.7}
        blended = outcome_probs(blend_with_market(fitted, "teama", "teamb", market_probs, weight=0.5))
        # Bei hälftigem Gewicht liegt das Ergebnis zwischen Modell- und Marktmeinung
        assert min(model_probs["home"], market_probs["home"]) - 0.02 <= blended["home"]
        assert blended["home"] <= max(model_probs["home"], market_probs["home"]) + 0.02

    def test_result_is_valid_probability_matrix(self, fitted):
        blended = blend_with_market(
            fitted, "teama", "teamb", {"home": 0.05, "draw": 0.15, "away": 0.80}, weight=1.0
        )
        assert blended.sum() == pytest.approx(1.0)
        assert (blended >= 0).all()
