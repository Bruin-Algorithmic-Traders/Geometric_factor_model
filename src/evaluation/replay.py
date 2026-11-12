import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from src.config import OOS_END_DATE, RESULTS_DIR
from src.evaluation.benchmarks import benchmark_specs, run_spec

TOLERANCE = 1e-10
REPLAY_MONTHLY_CSV = RESULTS_DIR / "replay_monthly.csv"
REPLAY_CHECK_CSV = RESULTS_DIR / "replay_check.csv"


def main(workers: int = 2):
    """Rerun every benchmark with the unchanged pre-oos code to save monthly gross return, turnover and cost;
    keep the output only if each run's net return matches the frozen files to within TOLERANCE every month."""
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"]:
        os.environ[var] = "1"
    specs = benchmark_specs()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(run_spec, specs.keys(), specs.values(), [OOS_END_DATE] * len(specs))
        monthly = pl.concat([
            r.select("DATE", pl.lit(name).alias("run"), pl.col("strategy_return").alias("net_return"),
                     "gross_return", "turnover", "cost")
            for name, r in zip(specs, results)
        ])

    rows = []
    for sample in ("is", "oos"):
        frozen = pl.read_csv(RESULTS_DIR / f"benchmark_returns_{sample}.csv")
        for name in specs:
            joined = frozen.select("DATE", name).join(monthly.filter(pl.col("run") == name), on="DATE", how="left")
            rows.append({"sample": sample, "run": name, "months": joined.height,
                         "missing": joined["net_return"].null_count(),
                         "max_abs_diff": (joined[name] - joined["net_return"]).abs().max()})
    check = pl.DataFrame(rows).with_columns(passed=(pl.col("missing") == 0) & (pl.col("max_abs_diff") <= TOLERANCE))
    check.write_csv(REPLAY_CHECK_CSV)
    print(check)
    if not check["passed"].all():
        raise RuntimeError("replay does not reproduce the frozen returns; replay_monthly.csv not written")
    monthly.write_csv(REPLAY_MONTHLY_CSV)
