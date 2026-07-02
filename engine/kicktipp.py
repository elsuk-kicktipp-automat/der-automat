"""Kicktipp-Punktelogik und Erwartungswert-Optimierung.

Kernidee (siehe CONCEPT.md, Schicht 2): Nicht das wahrscheinlichste Ergebnis
tippen, sondern den Tipp mit dem höchsten Punkte-Erwartungswert unter dem
Punkteschema der Runde:

    E[Punkte(Tipp)] = Summe über alle Ergebnisse: P(Ergebnis) * Punkte(Tipp, Ergebnis)
"""

import numpy as np

DEFAULT_SCHEME = {"exact": 4, "goal_diff": 3, "tendency": 2}


def match_points(tip: tuple[int, int], result: tuple[int, int], scheme: dict = DEFAULT_SCHEME) -> int:
    """Kicktipp-Punkte für einen Tipp gegen das reale Ergebnis.

    Exakt > Tordifferenz > Tendenz. Ein nicht-exaktes richtiges Unentschieden
    zählt als richtige Tordifferenz (Kicktipp-Standard).
    """
    tip_h, tip_a = tip
    res_h, res_a = result
    if (tip_h, tip_a) == (res_h, res_a):
        return scheme["exact"]
    if tip_h - tip_a == res_h - res_a:
        return scheme["goal_diff"]
    if np.sign(tip_h - tip_a) == np.sign(res_h - res_a):
        return scheme["tendency"]
    return 0


def expected_points(tip: tuple[int, int], matrix: np.ndarray, scheme: dict = DEFAULT_SCHEME) -> float:
    """Punkte-Erwartungswert eines Tipps über die Ergebnis-Wahrscheinlichkeitsmatrix."""
    size = matrix.shape[0]
    total = 0.0
    for res_h in range(size):
        for res_a in range(size):
            p = matrix[res_h, res_a]
            if p > 0:
                total += p * match_points(tip, (res_h, res_a), scheme)
    return total


def best_tip(
    matrix: np.ndarray, scheme: dict = DEFAULT_SCHEME, max_tip_goals: int = 5
) -> tuple[tuple[int, int], float]:
    """Der Tipp mit maximalem Punkte-Erwartungswert: ((heim, gast), erwartungswert)."""
    best, best_ev = (0, 0), -1.0
    for tip_h in range(max_tip_goals + 1):
        for tip_a in range(max_tip_goals + 1):
            ev = expected_points((tip_h, tip_a), matrix, scheme)
            if ev > best_ev:
                best, best_ev = (tip_h, tip_a), ev
    return best, best_ev


def most_probable_score(matrix: np.ndarray) -> tuple[int, int]:
    """Naive Baseline: das wahrscheinlichste Einzelergebnis."""
    h, a = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(h), int(a)
