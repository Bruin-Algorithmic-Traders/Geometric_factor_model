# Decisions

## Return timing
- Characteristics dated t are paired with the return dated t (same month), not t+1: `mom1m` in the row dated t equals the return dated t−1 (rank corr 0.998), so the row already only uses data up to t−1 and the extra month of lag was unnecessary.

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
