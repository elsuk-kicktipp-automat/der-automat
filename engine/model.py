"""Dixon-Coles-Poisson-Modell für Fußball-Ergebnisse.

Pro Team werden Angriffs- und Abwehrstärke geschätzt, dazu Heimvorteil und
die Dixon-Coles-Korrektur (rho) für niedrige Ergebnisse:

    log lambda_heim = mu + heimvorteil + angriff[heim] + abwehr[gast]
    log lambda_gast = mu + angriff[gast] + abwehr[heim]

(abwehr ist "Anfälligkeit": höher = kassiert mehr Tore). Ältere Spiele werden
exponentiell abklingend gewichtet (Dixon & Coles 1997). L2-Regularisierung
hält Teams mit wenig Daten nahe am Ligaschnitt und macht das Modell auch für
Aufsteiger ohne Historie robust (deren Parameter starten beim Ligaschnitt 0).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from .openligadb import Match

RHO_BOUND = 0.3


def _log_tau(hg, ag, lam, mu, rho):
    """Log der Dixon-Coles-Korrektur tau für die Ergebnisse 0:0, 0:1, 1:0, 1:1."""
    tau = np.ones_like(lam)
    tau = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, tau)
    tau = np.where((hg == 0) & (ag == 1), 1 + lam * rho, tau)
    tau = np.where((hg == 1) & (ag == 0), 1 + mu * rho, tau)
    tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
    return np.log(np.clip(tau, 1e-10, None))


@dataclass
class FittedParams:
    team_ids: list[int]
    attack: dict[int, float]
    defense: dict[int, float]
    intercept: float
    home_adv: float
    rho: float


class DixonColes:
    def __init__(self, xi: float = 0.002, l2_penalty: float = 0.1, max_goals: int = 8):
        self.xi = xi
        self.l2_penalty = l2_penalty
        self.max_goals = max_goals
        self.params: FittedParams | None = None

    def fit(self, matches: list[Match], ref_date: datetime) -> FittedParams:
        """Fittet das Modell auf abgeschlossene Spiele, gewichtet relativ zu ref_date.

        Ein bereits vorhandenes Fit-Ergebnis wird als Warmstart genutzt.
        """
        matches = [m for m in matches if m.has_result]
        if not matches:
            raise ValueError("Keine abgeschlossenen Spiele zum Fitten vorhanden.")

        team_ids = sorted({m.home_id for m in matches} | {m.away_id for m in matches})
        idx = {t: i for i, t in enumerate(team_ids)}
        n = len(team_ids)

        hi = np.array([idx[m.home_id] for m in matches])
        ai = np.array([idx[m.away_id] for m in matches])
        hg = np.array([m.home_goals for m in matches], dtype=float)
        ag = np.array([m.away_goals for m in matches], dtype=float)
        days_ago = np.array(
            [max(0.0, (ref_date - m.kickoff_utc).total_seconds() / 86400) for m in matches]
        )
        w = np.exp(-self.xi * days_ago)

        def nll(theta):
            attack = theta[:n]
            defense = theta[n : 2 * n]
            intercept, home_adv, rho = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]
            lam = np.exp(intercept + home_adv + attack[hi] + defense[ai])
            mu = np.exp(intercept + attack[ai] + defense[hi])
            ll = w * (
                _log_tau(hg, ag, lam, mu, rho)
                + hg * np.log(lam) - lam
                + ag * np.log(mu) - mu
            )
            penalty = self.l2_penalty * (attack @ attack + defense @ defense)
            return -ll.sum() + penalty

        x0 = np.zeros(2 * n + 3)
        x0[2 * n] = 0.3  # exp(0.3) ≈ 1.35 Tore, sinnvoller Startwert
        if self.params is not None:
            for t, i in idx.items():
                x0[i] = self.params.attack.get(t, 0.0)
                x0[n + i] = self.params.defense.get(t, 0.0)
            x0[2 * n] = self.params.intercept
            x0[2 * n + 1] = self.params.home_adv
            x0[2 * n + 2] = self.params.rho

        bounds = [(None, None)] * (2 * n + 2) + [(-RHO_BOUND, RHO_BOUND)]
        result = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)

        theta = result.x
        self.params = FittedParams(
            team_ids=team_ids,
            attack={t: theta[idx[t]] for t in team_ids},
            defense={t: theta[n + idx[t]] for t in team_ids},
            intercept=theta[2 * n],
            home_adv=theta[2 * n + 1],
            rho=theta[2 * n + 2],
        )
        return self.params

    def expected_goals(self, home_id: int, away_id: int) -> tuple[float, float]:
        """(lambda_heim, lambda_gast); unbekannte Teams zählen als Ligaschnitt."""
        p = self.params
        if p is None:
            raise ValueError("Modell ist noch nicht gefittet.")
        lam = np.exp(
            p.intercept + p.home_adv + p.attack.get(home_id, 0.0) + p.defense.get(away_id, 0.0)
        )
        mu = np.exp(p.intercept + p.attack.get(away_id, 0.0) + p.defense.get(home_id, 0.0))
        return float(lam), float(mu)

    def score_matrix(self, home_id: int, away_id: int) -> np.ndarray:
        """Wahrscheinlichkeitsmatrix P[heimtore, gasttore] für 0..max_goals."""
        lam, mu = self.expected_goals(home_id, away_id)
        goals = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))

        rho = self.params.rho
        matrix[0, 0] *= max(1 - lam * mu * rho, 1e-10)
        matrix[0, 1] *= max(1 + lam * rho, 1e-10)
        matrix[1, 0] *= max(1 + mu * rho, 1e-10)
        matrix[1, 1] *= max(1 - rho, 1e-10)

        return matrix / matrix.sum()
