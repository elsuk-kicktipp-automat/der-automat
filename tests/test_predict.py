import numpy as np
import pytest

from engine.predict import build_model, outcome_probabilities, resolve_l2_penalty

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
