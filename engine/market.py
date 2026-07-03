"""Quoten-Nachjustierung der Modell-Prognose (concept.md §3, Schicht 1).

Anders als die ELO-Differenz lassen sich Quoten nicht als Koeffizient aus
historischen Spielen fitten – The Odds API bietet keinen historischen
Endpunkt, es gibt also nie "Trainingsdaten" mit angehängter Vergangenheits-
quote. Der Markt-Einfluss wird deshalb erst zur Vorhersagezeit angewendet:
die Modell-Wahrscheinlichkeitsmatrix wird so verschoben, dass ihre
Heimsieg-/Remis-/Auswärtssieg-Randverteilung näher an einen Blend aus
Marktquote und Modellmeinung rückt – die relative Form (rho, wer wie
wahrscheinlich mit welcher Tordifferenz gewinnt) bleibt vom Modell bestimmt.
"""

import numpy as np
from scipy.optimize import minimize

from .model import DixonColes


def outcome_probs(matrix: np.ndarray) -> dict[str, float]:
    return {
        "home": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "away": float(np.triu(matrix, 1).sum()),
    }


def blend_with_market(
    model: DixonColes,
    home_key: str,
    away_key: str,
    market_probs: dict[str, float],
    weight: float,
) -> np.ndarray:
    """Verschiebt die Modellmatrix Richtung `weight * Markt + (1-weight) * Modell`.

    `weight` = 1.0 folgt der Quote vollständig, 0.0 ignoriert sie. Sucht dazu
    zwei Faktoren (Heim/Gast), mit denen lambda/mu skaliert werden, sodass die
    resultierende 1X2-Randverteilung dem Blend-Ziel möglichst nahekommt.
    """
    lam0, mu0 = model.expected_goals(home_key, away_key)
    p_model = outcome_probs(model.matrix_from_goals(lam0, mu0))

    target = {k: weight * market_probs[k] + (1 - weight) * p_model[k] for k in p_model}
    total = sum(target.values())
    target = {k: v / total for k, v in target.items()}

    def objective(shifts: np.ndarray) -> float:
        d_home, d_away = shifts
        matrix = model.matrix_from_goals(lam0 * np.exp(d_home), mu0 * np.exp(d_away))
        p = outcome_probs(matrix)
        return sum((p[k] - target[k]) ** 2 for k in p)

    result = minimize(objective, x0=np.zeros(2), method="Nelder-Mead")
    d_home, d_away = result.x
    return model.matrix_from_goals(lam0 * np.exp(d_home), mu0 * np.exp(d_away))
