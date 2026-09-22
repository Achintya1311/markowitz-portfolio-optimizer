"""Sample mean/covariance estimation, plus how unreliable that estimate is.

Mean-variance optimization needs two inputs per asset: an expected return
and a covariance matrix. Both are usually estimated from the same trailing
window of daily returns used here. The point of this module is not just to
compute them but to size the estimation error on each - the frontier and
tangency portfolio built in later days are only as trustworthy as these
numbers are precise, and for the mean vector in particular, they are not
precise at all over a sample this short.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optimizer.returns import TRADING_DAYS_PER_YEAR, annualize_cov, annualize_mean


def sample_mean_cov(returns_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily sample mean vector and covariance matrix.

    ``returns_matrix`` has shape ``(n_days, n_assets)``. Covariance uses the
    unbiased (``ddof=1``) estimator, matching :func:`numpy.cov`'s default.
    """
    mu_daily = returns_matrix.mean(axis=0)
    cov_daily = np.cov(returns_matrix, rowvar=False, ddof=1)
    return mu_daily, cov_daily


def standard_error_of_annualized_mean(cov_daily: np.ndarray, n_obs: int) -> np.ndarray:
    """Standard error of the annualized mean-return estimator, per asset.

    The annualized mean estimator is ``mu_hat_annual = TRADING_DAYS_PER_YEAR *
    mean(daily returns)``. Since the daily observations are treated as iid,
    ``Var(mean(daily returns)) = sigma_daily^2 / n_obs``, so scaling by
    ``TRADING_DAYS_PER_YEAR`` gives
    ``SE(mu_hat_annual) = TRADING_DAYS_PER_YEAR * sigma_daily / sqrt(n_obs)``.
    This is the number that makes "expected returns are noisy" concrete
    instead of a caveat in prose.
    """
    sigma_daily = np.sqrt(np.diag(cov_daily))
    return TRADING_DAYS_PER_YEAR * sigma_daily / np.sqrt(n_obs)


def mean_t_stats(mu_annual: np.ndarray, se_annual: np.ndarray) -> np.ndarray:
    """How many standard errors each annualized mean estimate is from zero.

    A ratio with ``|t| < 2`` means the point estimate is not statistically
    distinguishable from a zero expected return at conventional confidence -
    common for equity means estimated from a couple of years of daily data,
    and a direct illustration of why a Markowitz frontier is far more
    sensitive to the (badly estimated) mean vector than to the (comparatively
    well estimated) covariance matrix.
    """
    return mu_annual / se_annual


def condition_number(cov_matrix: np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue of a covariance matrix.

    A large condition number means the matrix is close to singular - small
    changes in the estimated returns get amplified into large changes in the
    inverse used by the optimizer, and highly correlated assets (as these
    three NSE names often are) push this up. This is the diagnostic that
    motivates Day 4's Ledoit-Wolf shrinkage comparison: shrinkage exists
    specifically to pull an ill-conditioned sample covariance matrix back
    toward something invertible without this much noise.
    """
    return float(np.linalg.cond(cov_matrix))


@dataclass
class SplitHalfDiagnostic:
    """Sample mean/covariance estimated separately on each half of the window.

    If the two halves disagree a lot, the "true" mean and covariance are not
    stable over the sample period - regime change, not just sampling noise -
    and a single full-window estimate is hiding that instability rather than
    resolving it.
    """

    tickers: list[str]
    mu_annual_first_half: np.ndarray
    mu_annual_second_half: np.ndarray
    cov_annual_first_half: np.ndarray
    cov_annual_second_half: np.ndarray
    mean_shift: np.ndarray = field(init=False)
    cov_frobenius_relative_change: float = field(init=False)

    def __post_init__(self):
        self.mean_shift = self.mu_annual_second_half - self.mu_annual_first_half
        norm_first = np.linalg.norm(self.cov_annual_first_half)
        norm_diff = np.linalg.norm(self.cov_annual_second_half - self.cov_annual_first_half)
        self.cov_frobenius_relative_change = float(norm_diff / norm_first)


def split_half_stability(tickers: list[str], returns_matrix: np.ndarray) -> SplitHalfDiagnostic:
    """Compare mean/covariance estimated on the first vs second half of the sample.

    Splits ``returns_matrix`` in half by row (chronological order is assumed,
    as everywhere else in this pipeline), so the comparison is first-period
    vs second-period, not a random split - the question is regime stability
    over time, not sampling variance within one regime.
    """
    n = returns_matrix.shape[0]
    midpoint = n // 2
    first_half, second_half = returns_matrix[:midpoint], returns_matrix[midpoint:]

    mu_first_daily, cov_first_daily = sample_mean_cov(first_half)
    mu_second_daily, cov_second_daily = sample_mean_cov(second_half)

    return SplitHalfDiagnostic(
        tickers=tickers,
        mu_annual_first_half=annualize_mean(mu_first_daily),
        mu_annual_second_half=annualize_mean(mu_second_daily),
        cov_annual_first_half=annualize_cov(cov_first_daily),
        cov_annual_second_half=annualize_cov(cov_second_daily),
    )


def correlation_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    """Correlation matrix implied by a covariance matrix."""
    sigma = np.sqrt(np.diag(cov_matrix))
    return cov_matrix / np.outer(sigma, sigma)
