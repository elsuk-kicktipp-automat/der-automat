from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from engine.model import DixonColes
from engine.openligadb import Match


def make_match(home_id, away_id, hg, ag, days_before_ref=100, matchday=1):
    kickoff = REF_DATE - timedelta(days=days_before_ref)
    return Match(
        home_id=home_id,
        away_id=away_id,
        home_name=f"Team {home_id}",
        away_name=f"Team {away_id}",
        home_goals=hg,
        away_goals=ag,
        kickoff_utc=kickoff,
        matchday=matchday,
        finished=True,
    )


REF_DATE = datetime(2026, 5, 1, tzinfo=timezone.utc)


def synthetic_season(seed=7):
    """Doppelrunde mit 6 Teams: Team 1 ist klar am stärksten, Team 6 am schwächsten."""
    rng = np.random.default_rng(seed)
    strength = {1: 0.5, 2: 0.25, 3: 0.1, 4: -0.1, 5: -0.25, 6: -0.5}
    matches = []
    day = 300
    for _ in range(4):  # 4 Durchgänge für stabilere Schätzung
        for home in strength:
            for away in strength:
                if home == away:
                    continue
                lam = np.exp(0.2 + 0.25 + strength[home] - strength[away] * 0.8)
                mu = np.exp(0.2 + strength[away] - strength[home] * 0.8)
                matches.append(
                    make_match(home, away, rng.poisson(lam), rng.poisson(mu), day)
                )
                day -= 0.1
    return matches


@pytest.fixture(scope="module")
def fitted():
    model = DixonColes(xi=0.0, l2_penalty=0.1, max_goals=8)
    model.fit(synthetic_season(), REF_DATE)
    return model


class TestDixonColesFit:
    def test_home_advantage_positive(self, fitted):
        assert fitted.params.home_adv > 0

    def test_recovers_team_order(self, fitted):
        attack = fitted.params.attack
        assert attack[1] > attack[6]
        assert attack[2] > attack[5]

    def test_expected_goals_favor_stronger_team(self, fitted):
        lam, mu = fitted.expected_goals(1, 6)
        assert lam > mu

    def test_unknown_team_uses_league_average(self, fitted):
        lam, mu = fitted.expected_goals(99, 98)
        assert 0.5 < lam < 3.0
        assert 0.5 < mu < 3.0
        assert lam > mu  # Heimvorteil bleibt


class TestScoreMatrix:
    def test_sums_to_one(self, fitted):
        matrix = fitted.score_matrix(1, 2)
        assert matrix.sum() == pytest.approx(1.0)
        assert (matrix >= 0).all()

    def test_shape(self, fitted):
        assert fitted.score_matrix(1, 2).shape == (9, 9)

    def test_home_win_more_likely_for_strong_home_team(self, fitted):
        matrix = fitted.score_matrix(1, 6)
        p_home = np.tril(matrix, -1).sum()  # Heimtore > Gasttore
        p_away = np.triu(matrix, 1).sum()
        assert p_home > p_away


class TestWeightingAndErrors:
    def test_time_decay_prefers_recent_form(self):
        # Team 1 war früher schwach (verlor gegen 2), zuletzt stark (gewann hoch).
        old = [make_match(1, 2, 0, 3, days_before_ref=700 + i) for i in range(10)]
        old += [make_match(2, 1, 3, 0, days_before_ref=750 + i) for i in range(10)]
        recent = [make_match(1, 2, 3, 0, days_before_ref=10 + i) for i in range(10)]
        recent += [make_match(2, 1, 0, 3, days_before_ref=30 + i) for i in range(10)]

        no_decay = DixonColes(xi=0.0, l2_penalty=0.1)
        no_decay.fit(old + recent, REF_DATE)
        strong_decay = DixonColes(xi=0.01, l2_penalty=0.1)
        strong_decay.fit(old + recent, REF_DATE)

        # Mit Abklinggewichtung dominiert die junge Form von Team 1
        assert (
            strong_decay.params.attack[1] - strong_decay.params.attack[2]
            > no_decay.params.attack[1] - no_decay.params.attack[2]
        )

    def test_fit_without_matches_raises(self):
        with pytest.raises(ValueError):
            DixonColes().fit([], REF_DATE)

    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError):
            DixonColes().score_matrix(1, 2)

    def test_rho_within_bounds(self):
        model = DixonColes(xi=0.0, l2_penalty=0.1)
        model.fit(synthetic_season(), REF_DATE)
        assert -0.3 <= model.params.rho <= 0.3
