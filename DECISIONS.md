# Decisions

## Return timing
- Characteristics dated t are paired with the return dated t (same month), not t+1: `mom1m` in the row dated t equals the return dated t−1 (rank corr 0.998), so the row already only uses data up to t−1 and the extra month of lag was unnecessary.
- Known look-ahead kept and disclosed: a firm is in month t's universe only if it has a return in month t, so firms that vanish mid-month (0.57% of stock-months; no delisting returns in the data) are dropped. Sensitivity: keeping each firm for the month after its last return (if it still has characteristics; size cutoff computed without these rows) at −30% or −100% lowers in-sample net Sharpe from 1.57 to 1.49 or 1.45; the strategy holds slightly more of these firms long (0.47% of capital) than short (0.36%), so a negative exit return hurts rather than helps.

## Memberships (τ)
- One formula everywhere, paper eq. 6: Π = softmax(τ·cos), via `SphericalKMeans.transform(X, tau)`; tuning, the factor panel and `transform()` previously used three different formulas.
- *Superseded by the extended grid in Tuning.* τ grid is {1, 3, 10, 30, 100}: τ=1 gives weights 0.99 correlated with the linear-in-cos limit and τ=100 gives weights 0.996 correlated with hard clusters, so the grid spans linear to hard; the old τ=0.35 made memberships near-uniform (avg max 0.04 vs 1/30).
- The linear baseline (τ→0) is run as τ=0.01; no special code. *Superseded: τ=0.01 is now also a grid value, and the default τ=10 is replaced by the tuned value once locked (see Tuning).*

## Spherical k-means
- Empty clusters are reseeded with distinct worst-fit firms (lowest best-match cosine) instead of a constant vector; this never triggered on the cleaned in-sample data (0 of 191 months for K=15/30/35) but was wrong.
- k-means++ samples new centroids ∝ (1 − cos), the correct D² weight on the unit sphere, instead of (1 − cos)², which over-weighted outliers.
- The cold-start fit (first month only) keeps the best of 10 k-means++ runs by total cosine (eq. 4), since that fit seeds all later warm-started months; later months stay warm-started.
- Each month's clusters are fit on the last 3 months of firms stacked together (warm-started), which cuts month-to-month centroid jumps (cos < 0.8) from 44 to 9 in-sample at almost no loss of fit to the current month; 6 and 12 months did not reduce 36-month drift further, so that slow drift is treated as real change in firm types (`notebooks/08`).
- After each fit, centroids are Hungarian-matched to the previous month's and the matched cosines are reported; it never reorders on current data but guards label identity.
- Frozen centroids were rejected because firm types are meant to be re-learned from the changing cross-section.

## Factor construction (Spec B)
- Keep all K factors even though Σ_k c_k w_k = 0 each month: with ‖a‖₁ ≤ 1 the optimiser cannot exploit the dependence (in-sample Sharpe 0.369 with 30 factors vs 0.370 with 29, same gross and turnover); the equal-weight mix of all factors is ~10% gross, so it is never used as a tuning score or benchmark (`notebooks/10`). Re-checked under the risk-budget optimiser at K = 40/50: dropping one factor changes net Sharpe by ≤ 0.02 and stock gross/turnover not at all, and only ~7% of the factor weights lie along the cancelling mix (`notebooks/13`).
- No per-stock cap: the largest single-stock weight in any factor is at most 5.2% of its gross over all in-sample months, K ∈ {15, 30, 35} and τ ∈ {0.01…100} (typically under 1%), so a cap would never bind; instead `compute_weights` raises an error if any stock exceeds 10% (`notebooks/09`).

## Optimiser and costs
- κ is split into `KAPPA_PENALTY` (turnover penalty in the eq. 18 objective, tuned) and `TC_BPS` (realised cost per $ traded, fixed at 10 bp, with 5 and 20 bp as sensitivity), because the paper keeps the penalty (κ_f) and the realised cost (TC^(f), eq. 20) separate and one number cannot be both a tuned knob and a cost estimate.
- The realised cost is charged on single-name turnover Σ|w_t − w_{t−1}| with w_t = W_t a_t (paper eq. 19–20), where last month's holdings are first drifted by their returns, so stocks entering/leaving the universe and price moves are counted; the objective's penalty stays on factor turnover as in eq. 18.
- μ̂ and Σ̂ are estimated from rebuilt history (each month, all past months' factor returns recomputed with that month's centroids), so drift and the yearly accounting-data jumps don't mix different portfolios into one history; rebuilt history predicts next-month factor returns better than stored history at every window tested (`notebooks/11`).
- μ̂ = mean of the last 60 months of rebuilt history: rank IC 0.10 (t 4.3) across factors, positive in both halves (measured at K = 30, τ = 10 before tuning; at the locked K = 40, τ = 1 it is 0.068, Newey-West t 2.1); the expanding mean scored slightly higher (t 4.9) but was rejected because it would apply today's types to ever older firms out-of-sample.
- Σ̂ = exponentially weighted covariance (half-life 12 months) of rebuilt history with Ledoit-Wolf shrinkage: best risk calibration (realised ÷ predicted 1.09 vs 1.12 for 36 months), while 60-month and expanding windows underestimated risk by 18–29% (`notebooks/12`).

- The γ risk term in eq. 18 is replaced by a risk budget: maximise aᵀμ̂ − κ‖Δa‖₁ subject to ex-ante vol ≤ 10% a year, stock gross Σ|W_t a| ≤ 3× capital and |a_k| ≤ 0.6 (20% of the gross budget), because in the γ grid γ above ~300 only shrank the same portfolio (Sharpe unchanged, vol down to 0.07%) and plain Sharpe ignored the leverage needed; for each month this is the eq. 18 solution for some γ, chosen to hit the budget.

## Tuning
- In-sample only (155 trading months, 1993-02 to 2005-12); pick the highest average net Sharpe over a cell and its one-step neighbours among cells positive in both halves (1993–99, 2000–05).
- Rounds: K × τ × γ × κ = 5 × 5 × 6 × 4 (600) with τ ∈ {1…100}, then 7 × 7 × 7 × 4 (1,372) with τ ∈ {0.01…100}, K ≤ 50, γ ≤ 3000 (both *superseded* by the risk budget), then K × τ × κ = 7 × 7 × 4 (196) under the risk budget; 2,168 grid configurations, plus the μ̂/Σ̂ window and feature choices also made on in-sample returns (`notebooks/04`, `07`, `11`, `12`), so the effective number of trials is higher.
- Locked K = 40, τ = 1, κ_penalty = 1 bp (`results/config_locked.json`): in-sample net Sharpe 1.57 (1.76 / 1.33 by half), 16.6% return at 10.6% vol. The rule's pick was K = 50 (1.62) on the same plateau; K = 40 was chosen by judgement because it is off the grid edge, has lower factor turnover, and keeps the factor count nearer Σ̂'s ~35 effective months.
- Σ̂ calibration at the candidate K is fine (realised ÷ predicted risk 1.04–1.06 at K = 40/50, τ = 1) and the largest combined single-stock weight is ≤ 0.7% of capital in every month (`notebooks/13`).
- Disclosure: some feature choices used in-sample return signal (IC tie-breaks, keeping `convind`/`securedind` and the scores). A returns-free variant (ties broken by lower missingness, then the more primitive variable: keeps `pchsale_pchinvt` and `mvel1`, drops `pchsaleinv` and `ill`) gives in-sample net Sharpe 1.61 vs 1.57 with 0.99 correlated monthly returns, so the return-informed choices did not inflate the in-sample result; it will also be run OOS alongside the main run.

## Benchmarks
- Static factor mixes are signed by the same 60-month μ̂ (sign(μ̂)/σ̂ and μ̂/σ̂²) and scaled to the same vol budget, gross cap and costs, because an unsigned equal-weight mix of all factors nearly cancels out.
- The PCA benchmark replaces the centroids with ± the top K/2 principal axes of the same pooled firm vectors, so it has K directions and everything else (τ, Spec B, optimiser, costs) is unchanged except the per-factor cap, which is halved to 0.3 because each ± pair is one bet (factor returns −0.99 correlated) and at 0.6 every month put a net 1.2 on some axis.
- Leak test: with returns shuffled across stocks within each month, the locked strategy's gross Sharpe averages −0.17 over 8 shuffles (range −0.54 to 0.30) vs 1.70 on real data, so no return leaks into features, μ̂/Σ̂ or the accounting.
- Random-direction placebo (8 sets of fixed random unit directions in place of the centroids): in-sample net Sharpe 1.38–1.80 (median ~1.66) vs 1.57, so the k-means directions are not better than random ones before costs; with near-linear memberships any 40 directions span a similar part of characteristic space. K-means wins at high costs (1.07 vs 0.52–0.96 at 50 bp) because its directions are stable and halve stock turnover.
- Costs are reported at 5, 10, 20 and 50 bp per $ traded on the same trades (the optimiser's turnover penalty stays locked).
- *Superseded by the next entry for reporting.* Market benchmarks use raw returns (the data has no risk-free rate), so their Sharpe ratios are overstated relative to the dollar-neutral strategy.
- Market benchmarks are reported as excess returns (minus the one-month T-bill rate from the Ken French 5-factor file, `results/market_excess.csv`); the strategy and other dollar-neutral runs are not adjusted because their shorts fund their longs.

## Data cleaning
All thresholds below were measured on in-sample data (1990–2005) only and are frozen in `src/config.py`. Evidence is in `notebooks/01`–`07`.

- **Universe order:** keep firm-months with a return first, then apply the size (bottom 20% of `mvel1`) and zerotrade filters, so the size cutoff is computed on firms we can actually trade (it was ~3x lower when computed on all firms).
- **Row filter:** drop firm-months missing more than 50% of the 94 raw characteristics; these are firms with no Compustat data (~20% of rows, younger and smaller), and the missing-count distribution is bimodal so any cutoff from 30–70% gives the same result.
- **Sparse columns:** drop the 8 columns with more than 30% missing after the row filter (`realestate`, `rd_sale`, `rd_mve`, `stdacc`, `stdcf`, `orgcap`, `secured`, `grltnoa`); they stay above 30% in both halves of the sample and the next column is at 29%.
- **Binary flags:** drop `sin`, `divi`, `divo`, `rd` (under 10% ones); keep `convind` and `securedind` because they carry return signal and adding them changes k-means clusters no more than a different random seed does.
- **Scores:** keep `ps`, `ms`, `nincr`; they are among the strongest in-sample predictors and correlate at most 0.5 with any other feature.
- **Redundancy:** drop the fewest features so that no pair has |average monthly rank correlation| > 0.85, keeping the one with higher in-sample |IC t-stat| when there is a choice; this drops `betasq`, `pchsale_pchinvt`, `dolvol`, `pchquick`, `mvel1`, `turn`, `retvol`, `quick` (`mvel1` and `dolvol` lose to `ill`; dropping `retvol` alone clears both `maxret` and `baspread`; closest pair below the line is `std_turn`/`zerotrade` at 0.84).
- **Imputation:** rank non-missing values to [−1, 1] each month, then set missing to 0, instead of median-fill then rank, which left a gap around the median.
- **sic2:** metadata only (not a feature), forward-filled within `permno`, unknown = −1; no backward fill because it used future data and recovered only 0.07% of rows.

## Pre-OOS expectations
Written before the single OOS run (2006–2018); with ~150 OOS months a Sharpe estimate has a standard error of ~0.3, so differences under ~0.3 count as equal.
- The locked strategy's net Sharpe will be well below the in-sample 1.57; roughly half (about 0.6–1.0) is expected, since published anomaly returns fall by about half after publication and our settings were tuned on 1990–2005.
- Success means: net Sharpe positive at 10 bp, still positive at 20 bp, and above the inverse-vol mix (the optimiser adds value out of sample).
- k-means ≥ the median random-direction run at 20 bp and 50 bp (its stable directions trade less); about equal at 5–10 bp.
- τ = 0.01 stays about equal to the strategy, and PCA directions stay below it after costs.
- Returns-free features stay about equal to the main features.
- Exits at −30% / −100% lower net Sharpe by around 0.1, as in-sample.
- Correlation with the market stays low (|corr| < 0.3).

## OOS results
- The single OOS run (2006-01 to 2018-11, 155 months) is final: it ran once from tag `pre-oos` (commit in `results/oos_commit.txt`) and the locked strategy will not be re-run or changed; any later analysis of OOS data is exploratory. Numbers are from `results/benchmarks_oos.csv` and `results/benchmarks_costs_oos.csv`.

| Pre-registered expectation | OOS result | |
|---|---|---|
| Net Sharpe about 0.6–1.0 | 0.34 (t ≈ 1.2) | ✗ |
| Positive at 10 bp and at 20 bp | 0.34, 0.23 | ✓ |
| Above the inverse-vol mix | 0.34 vs −0.01 | ✓ |
| ≥ median random run at 20 / 50 bp | 0.23 vs −0.05; −0.12 vs −0.66 | ✓ |
| About equal to random at 5 / 10 bp | 0.40 vs 0.25; 0.34 vs 0.15 | ✓ |
| τ = 0.01 about equal | 0.31 | ✓ |
| PCA below after costs | −0.14 | ✓ |
| Returns-free about equal | 0.39 | ✓ |
| Exits lower Sharpe by about 0.1 | −30%: 0.35; −100%: 0.65 | ✗ |
| \|corr with market\| < 0.3 | −0.06 (equal), −0.04 (value) | ✓ |

- Delisting flip: out of sample the exiting firms sit mostly on the short side (0.51% of capital short vs 0.25% long in the −100% run; about even at −30%), so a −100% exit return raises net Sharpe and is not a conservative stress there; −30% is the case to quote.

## Metrics
- Drawdowns (and the equity curve) use the cumulative sum of monthly returns, not compounding, because the book is long/short at a fixed vol target.
- Beta and correlation use Fama-French Mkt-RF everywhere; our own equal- and value-weighted market series stay only as benchmark rows.
- Gross Sharpe, turnover and cost drag come from the frozen summary files because monthly gross returns were not saved, so gross Sharpe has no confidence interval; replaying the `pre-oos` code to save more columns would be a reproducibility check, decided only if needed.
- Deflated Sharpe (in-sample strategy) uses the spread of net Sharpe across the 196 risk-budget configurations, converted to monthly; N = 2,168 is the conservative bound (treats every trial as independent) and N = 196 is shown alongside; the effective number lies in between because neighbouring grid cells are nearly the same strategy. It is 0.98 (N = 2,168) and 0.99 (N = 196), so the in-sample Sharpe was not mainly luck from the search and the out-of-sample fall points to the change of period; it covers only the grid, not the feature and window choices made on in-sample returns, so it is somewhat optimistic.
- Both an i.i.d. t and a Newey-West t (6 lags) are reported; out of sample they are 1.24 and 1.43, the gap coming from small negative autocorrelations about one standard error from zero, so the Sharpe confidence interval (which matches the i.i.d. t) is the headline.
- The average of the 8 random-direction runs is rescaled to 10% realised vol so its return and drawdown compare with single runs; rescaling leaves its Sharpe unchanged and that Sharpe is still lifted by diversification, so comparisons with random directions use the 8 per-seed results first.
- Factor regressions (market; FF5 + momentum; plus short-term reversal) use net returns of every strategy-type run with Newey-West t-stats (6 lags), written in plain numpy rather than adding a dependency; a one-off check matched statsmodels' HAC t-stats (no small-sample correction) to 1e-13. The out-of-sample regressions were not pre-registered.

## Relative tests
- Monthly gross returns and turnover were not saved, so the `pre-oos` code was replayed once as a reproducibility check (not a new experiment): every run's net return matched the frozen files exactly in every month (pass rule ≤ 1e-10, fixed before running), and the in-sample months are identical whether the run ends in 2005 or 2018, so no later data leaks into earlier months.
- Sharpe differences use a paired stationary block bootstrap (mean block 6 months, 10,000 resamples, fixed seed) with Jobson-Korkie/Memmel as a cross-check. "Better" claims are significant when the 95% CI is above 0, directional when only the point estimate is; "about equal" claims (τ = 0.01, returns-free, random directions at 5–10 bp, as pre-registered) are "no difference" when the CI includes 0.
- K-means is tested against each random seed separately, with Holm's correction within each cost level's 8 seeds judged at 0.025, the same rule as the CIs; switching to 0.05 after seeing the results was rejected. The averaged random series is secondary and shown at 10 bp only, because averaging adds a diversification benefit and its costs are overstated (trades are not netted).

## Diagnostics (exploratory)
- μ̂'s rank IC across factors fell from 0.068 (t 2.1) in-sample to 0.021 (t 0.6) out of sample, positive in 52% of months; 2006–07 still matched in-sample (0.13) and the drop starts in 2008, so the decay is in the return forecasts rather than in costs or risk control.
- Sub-periods at 10 bp (gross, turnover and cost from the verified replay): strategy net Sharpe 0.71 / 0.13 / −0.01 / 0.50 for 2006–08 / 2009–11 / 2012–15 / 2016–18 and 0.37 without 2008; with 35–48 months every 95% CI is about ±1.0–1.15, so these are descriptive only. The CSV covers every run; the README shows a short list, and the row without 2008 has no drawdown because joining around a gap is artificial.
