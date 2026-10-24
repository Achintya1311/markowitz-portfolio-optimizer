# Pre-registration: which portfolio rules, if any, beat equal weight (written before any result was computed)

**Why.** The repo's headline is that long-only maximum-Sharpe on Ledoit-Wolf covariance loses to 1/N out of sample (+252% net against +427%). The README already says the cause: the sample mean feeding max-Sharpe is statistically indistinguishable from zero, so the optimizer chases noise and trades 58% of the book each fold. That is a finding about one estimator (DeMiguel, Garlappi and Uppal 2009 report the same for many). It is not a finding about portfolio construction, and the repo never walk-forwards the estimators that do not need a mean. This asks whether any of them earn their place against 1/N.

**Fixed in advance, no tuning afterwards:**

- Data and folds: the 29 committed large-cap fixtures, 252-day training window, 21-day hold, non-overlapping folds, as in `optimizer.oos`.
- Candidates (three, all long-only, covariance from Ledoit-Wolf shrinkage on the training window, no expected-return input):
  - **A, minimum variance** (primary).
  - **B, inverse volatility** (weights proportional to 1 / training-window volatility).
  - **C, blend**: half minimum variance, half 1/N.
- Benchmark: equal weight. Max-Sharpe is carried along unchanged as the previous result.
- Costs, identical for every method including equal weight: 0.2075% per unit of one-way L1 turnover (the repo's existing schedule). Turnover is measured from the weights each portfolio has **drifted to** at the end of the previous hold, back to its new target. This corrects the old benchmark, which paid no turnover because it reset to 1/N with no drift.
- Statistics: per-fold net returns, annualised with sqrt(252/21). The Sharpe difference gets a 95% interval from a stationary bootstrap (`arch`, block length chosen by the Politis-White rule, 5,000 resamples, seed 0).

**Pass criteria (all must hold, for the primary candidate A):**

1. Net Sharpe ratio of A is at least that of equal weight (point estimate).
2. The 95% interval of the Sharpe difference (A minus equal weight) is not entirely below zero.
3. Annualised volatility of A is lower than that of equal weight.
4. Maximum drawdown of A is no worse than that of equal weight.
5. Average per-fold turnover of A is at most 20% (one-way L1).
6. Criteria 1 to 4 also hold with a 126-day training window.

**Reporting rules.** Net compounded return is reported for every method but does not gate the result: minimum variance claims lower risk, not higher return. B and C are reported as secondary, with the note that three candidates were tried, so a pass by B or C alone would not meet the pre-registered bar. If A fails any criterion, the conclusion is published as "equal weight is not beaten by this rule on this universe" and no further model changes follow under this pre-registration. Survivorship in the universe (today's large caps) favours all fully invested long-only rules alike and is already disclosed.
