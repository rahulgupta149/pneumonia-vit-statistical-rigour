# Statistical methods

This document defines every test implemented in `pneumonia_stats` and used in
the notebook. All tests are **paired**: every model is scored on exactly the
same images, so per-image correlation is preserved.

Labels: `0 = NORMAL`, `1 = PNEUMONIA`. Significance level α = 0.05, two-sided,
unless stated otherwise.

---

## 1. Wilson score interval

For a proportion with `s` successes out of `n`, the Wilson interval is

```
        p + z²/(2n) ± z·sqrt( p(1-p)/n + z²/(4n²) )
  CI =  -------------------------------------------
                    1 + z²/n
```

where `p = s/n` and `z = 1.95996` for 95%. Unlike the Wald interval, coverage
remains close to nominal when `p` approaches 0 or 1, or when `n` is small.

Used for accuracy, sensitivity and specificity.

**Reference:** Newcombe (1998).

---

## 2. McNemar's test

### 2.1 The paired 2×2 table

Comparing baseline A with challenger B on the same `n` images:

|                | B correct | B wrong |
|----------------|-----------|---------|
| **A correct**  | a         | b (n10) |
| **A wrong**    | c (n01)   | d       |

Only the discordant pairs `b` and `c` carry information about the difference;
images both models classify correctly (or both incorrectly) cancel out.

### 2.2 Test statistic

Exact two-sided test:

```
p = P( X ≤ min(b, c) ) + P( X ≥ max(b, c) ),   X ~ Binomial(b + c, 0.5)
```

This is the primary test because it does not rely on a large-sample
approximation.

The Edwards continuity-corrected χ² statistic is reported alongside:

```
χ² = (|b − c| − 1)² / (b + c),   df = 1
```

### 2.3 Effect sizes

* **Risk difference** — the accuracy difference in percentage points:

  ```
  Δ = (c − b) / n
  ```

  with an asymptotic Wald 95% CI for paired proportions:

  ```
  SE = sqrt( b + c − (c − b)²/n ) / n
  ```

* **Conditional odds ratio** — how much more likely B is to be right on a
  discordant pair:

  ```
  OR = c / b        (> 1 favours B)
  ```

  The interval is obtained by applying an exact Clopper–Pearson interval to
  `c / (b + c)` and transforming the bounds to the odds scale. When `b = 0` or
  `c = 0`, the corresponding bound is reported as infinite.

**Reference:** Dietterich (1998); Edwards (1948).

---

## 3. DeLong's test for two correlated ROC AUCs

For model `k`, let `X` denote scores on positive images (m of them) and `Y`
scores on negative images (n of them). Define the placement values

```
V10_i = (1/n) Σ_j  ψ(X_i, Y_j)        for each positive i
V01_j = (1/m) Σ_i  ψ(X_i, Y_j)        for each negative j

ψ(x, y) = 1 if x > y,  0.5 if x = y,  0 otherwise
```

The AUC is `mean(V10) = mean(V01)`, and its variance is

```
Var(AUC) = Var(V10)/m + Var(V01)/n
```

For two models the covariance is

```
Cov(AUC₁, AUC₂) = Cov(V10¹, V10²)/m + Cov(V01¹, V01²)/n
```

and the test statistic is

```
z = (AUC₂ − AUC₁) / sqrt( Var(AUC₁) + Var(AUC₂) − 2·Cov(AUC₁, AUC₂) )
```

with a two-sided normal p-value.

The implementation uses the fast mid-rank formulation of Sun & Xu (2014),
which is algebraically identical to DeLong's original algorithm but runs in
`O((m+n) log(m+n))`. Ties are handled with mid-ranks, so the AUC matches
`sklearn.metrics.roc_auc_score` exactly.

**References:** DeLong, DeLong & Clarke-Pearson (1988); Sun & Xu (2014).

---

## 4. Paired percentile bootstrap

For `B = 10,000` iterations:

1. Draw `n` indices with replacement from `{1, …, n}`.
2. Score **both** models on the same resampled images.
3. Record `Δ* = metric(B) − metric(A)`.

The 2.5th and 97.5th percentiles of `Δ*` form the 95% interval. The two-sided
bootstrap p-value reported here is

```
p ≈ 2 · min( P(Δ* ≤ 0), P(Δ* ≥ 0) )
```

and is approximate; it is included for completeness, not as a confirmatory test.
Resamples containing a single class have an undefined AUC and are skipped (the
count is reported).

A fixed seed makes the procedure reproducible.

**Reference:** Efron & Tibshirani (1993).

---

## 5. Holm–Bonferroni correction

When the ViT is compared with `k` baselines, each family of tests (McNemar, and
separately DeLong) is corrected as follows:

1. Sort the raw p-values ascending: `p₍₁₎ ≤ … ≤ p₍ₖ₎`.
2. Adjusted value for rank `i`:

   ```
   p̃₍i₎ = max( p̃₍i−1₎, (k − i + 1)·p₍i₎ )
   ```

3. Truncate at 1.

Holm controls the family-wise error rate under arbitrary dependence between
tests, and is uniformly more powerful than plain Bonferroni.

**Reference:** Holm (1979).

---

## 6. Assumptions and interpretation

* **Paired observations.** Both models must be evaluated on identical images.
  The package rejects mismatched test sets rather than silently comparing
  unpaired results.
* **Independent images.** Bootstrap and asymptotic intervals assume images are
  independent samples. Images from the same patient would violate this; the
  dataset does not expose patient identifiers, so this cannot be checked.
* **Fixed test set.** All inference conditions on the cleaned test set. The
  quality-control step is therefore part of the data definition, not of the
  sampling model.
* **Multiple comparisons.** Report the Holm-adjusted p-values when more than one
  baseline is compared. Raw p-values may be shown for transparency but should
  not drive the conclusion.
* **Effect sizes over p-values.** With n = 380, small differences can be
  statistically significant. Report the risk difference and its interval
  alongside every p-value.

---

## References

DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. (1988) 'Comparing the areas
under two or more correlated receiver operating characteristic curves: a
nonparametric approach', *Biometrics*, 44(3), pp. 837–845.

Dietterich, T.G. (1998) 'Approximate statistical tests for comparing supervised
classification learning algorithms', *Neural Computation*, 10(7), pp. 1895–1923.

Edwards, A.L. (1948) 'Note on the "correction for continuity" in testing the
significance of the difference between matched pairs', *Journal of Educational
Psychology*, 39(8), pp. 490–494.

Efron, B. & Tibshirani, R.J. (1993) *An Introduction to the Bootstrap*. New York:
Chapman & Hall.

Holm, S. (1979) 'A simple sequentially rejective multiple test procedure',
*Scandinavian Journal of Statistics*, 6(2), pp. 65–70.

Newcombe, R.G. (1998) 'Two-sided confidence intervals for the single proportion:
comparison of seven methods', *Statistics in Medicine*, 17(8), pp. 857–872.

Sun, X. & Xu, W. (2014) 'Fast implementation of DeLong's algorithm for comparing
the areas under correlated receiver operating characteristic curves', *IEEE
Signal Processing Letters*, 21(11), pp. 1389–1393.
