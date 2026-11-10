# Decisions

## Return timing
- Characteristics dated t are paired with the return dated t (same month), not t+1: `mom1m` in the row dated t equals the return dated t−1 (rank corr 0.998), so the row already only uses data up to t−1 and the extra month of lag was unnecessary.
- Known look-ahead kept and disclosed: a firm is in month t's universe only if it has a return in month t, so firms that vanish mid-month (0.57% of stock-months; no delisting returns in the data) are dropped; the evaluation will re-run with a −30% return on every exit to measure the bias.

## Memberships (τ)
- One formula everywhere, paper eq. 6: Π = softmax(τ·cos), via `SphericalKMeans.transform(X, tau)`; tuning, the factor panel and `transform()` previously used three different formulas.
- τ grid is {1, 3, 10, 30, 100}: τ=1 gives weights 0.99 correlated with the linear-in-cos limit and τ=100 gives weights 0.996 correlated with hard clusters, so the grid spans linear to hard; the old τ=0.35 made memberships near-uniform (avg max 0.04 vs 1/30).
- The linear baseline (τ→0) is run as τ=0.01; no special code. Default τ=10 is a placeholder until tuning.

## Spherical k-means
- Empty clusters are reseeded with distinct worst-fit firms (lowest best-match cosine) instead of a constant vector; this never triggered on the cleaned in-sample data (0 of 191 months for K=15/30/35) but was wrong.
- k-means++ samples new centroids ∝ (1 − cos), the correct D² weight on the unit sphere, instead of (1 − cos)², which over-weighted outliers.
- The cold-start fit (first month only) keeps the best of 10 k-means++ runs by total cosine (eq. 4), since that fit seeds all later warm-started months; later months stay warm-started.
- Each month's clusters are fit on the last 3 months of firms stacked together (warm-started), which cuts month-to-month centroid jumps (cos < 0.8) from 44 to 9 in-sample at almost no loss of fit to the current month; 6 and 12 months did not reduce 36-month drift further, so that slow drift is treated as real change in firm types (`notebooks/08`).
- After each fit, centroids are Hungarian-matched to the previous month's and the matched cosines are reported; it never reorders on current data but guards label identity.
- Frozen centroids were rejected because firm types are meant to be re-learned from the changing cross-section.

## Factor construction (Spec B)
- Keep all K factors even though Σ_k c_k w_k = 0 each month: with ‖a‖₁ ≤ 1 the optimiser cannot exploit the dependence (in-sample Sharpe 0.369 with 30 factors vs 0.370 with 29, same gross and turnover); the equal-weight mix of all factors is ~10% gross, so it is never used as a tuning score or benchmark (`notebooks/10`).
- No per-stock cap: the largest single-stock weight in any factor is at most 5.2% of its gross over all in-sample months, K ∈ {15, 30, 35} and τ ∈ {0.01…100} (typically under 1%), so a cap would never bind; instead `compute_weights` raises an error if any stock exceeds 10% (`notebooks/09`).

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
