key decisions made:
    1. use polars not pandas
    2. store data as a parquet not csv
    3. data cleaning steps:
        a. remove all rows with missing permno and date as you can't do regression later
        b. remove sic2 as some are missing and it is nonsensical to do this math on a categorical id
        c. remove betasq & absacc & pchsaleinv (duplicate of beta and acc and pchsale_pchinvt), securedind & sin & divi & divo & convind (binary flags that just add dimensions), herf & indmom (industry level constants), rd_mve & rd_sale (kept only rd), ps (composite score)
        d. filter stocks below 20th %ile market cap to remove most microcaps that mess with data and remove illiquid stocks (zerotradingdays > 10)
        e. kept date as int?? maybe dumb but eh for now
    4. spherical k means:
        a. using k means ++ for initialising centroids
        b. setting empty centroid to stock with highest distance
    5. spec b:
        a. w_max val
    6. backtesting:
        a. refitting spherical k means each month
        b. warm-starting centroids (ie, using prev centroids as the initial ones for next month) - use consine sim as sanity check
        c. tune hyperparams on 1990-2005 data (GOT BEST CONFIG OF K = 30, TAU = 0.35)

