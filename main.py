import argparse


def stage_data():
    from src.data.make_parquet import make_parquet
    from src.data.clean import clean_data
    from src.config import (
        CHARS_CLEAN_DELIST_100_PARQUET, CHARS_CLEAN_DELIST_30_PARQUET, CHARS_CLEAN_RETURNS_FREE_PARQUET,
        RETURNS_FREE_DROPPED_COLS,
    )
    make_parquet()
    clean_data()
    clean_data(RETURNS_FREE_DROPPED_COLS, CHARS_CLEAN_RETURNS_FREE_PARQUET)
    clean_data(out_path=CHARS_CLEAN_DELIST_30_PARQUET, delisting_return=-0.30)
    clean_data(out_path=CHARS_CLEAN_DELIST_100_PARQUET, delisting_return=-1.0)


def stage_tune():
    from src.tuning.tune import main
    main()


def stage_select():
    from src.tuning.tune import report
    report()


def stage_panel():
    from src.factors.build_panel import save_panel
    save_panel()


def stage_benchmarks():
    from src.evaluation.benchmarks import main
    main()


def stage_leaktest():
    from src.evaluation.benchmarks import leak_test
    leak_test()


def stage_metrics():
    from src.evaluation import ff_data, metrics
    ff_data.main()
    metrics.main()


def stage_attribution():
    from src.evaluation.attribution import main
    main()


def stage_replay():
    from src.evaluation.replay import main
    main()


def stage_compare():
    from src.evaluation.compare import main
    main()


def stage_d1():
    from src.evaluation.diagnostics import d1_mu_hat_ic
    d1_mu_hat_ic()


def stage_d4():
    from src.evaluation.diagnostics import d4_subperiods
    d4_subperiods()


def stage_d8():
    from src.evaluation.diagnostics import d8
    d8()


def stage_d3():
    from src.evaluation.diagnostics import d3_by_year
    d3_by_year()


def stage_d7():
    from src.evaluation.diagnostics import d7_rolling_sharpe
    d7_rolling_sharpe()


def stage_d6_d5():
    from src.evaluation.diagnostics import d6_d5
    d6_d5()


def stage_figures():
    from src.evaluation import plots
    plots.equity_curve(); plots.drawdown_chart(); plots.tuning_heatmap(); plots.cost_curve(); plots.ic_by_year()


def stage_readme():
    from src.evaluation.readme_tables import main
    main()


def stage_oos():
    from src.evaluation.benchmarks import run_oos
    run_oos()


STAGES = {
    "data": stage_data,
    "panel": stage_panel,
    "tune": stage_tune,
    "select": stage_select,
    "benchmarks": stage_benchmarks,
    "leaktest": stage_leaktest,
    "oos": stage_oos,
    "metrics": stage_metrics,
    "attribution": stage_attribution,
    "replay": stage_replay,
    "compare": stage_compare,
    "d1": stage_d1,
    "d4": stage_d4,
    "d8": stage_d8,
    "d3": stage_d3,
    "d7": stage_d7,
    "d6_d5": stage_d6_d5,
    "figures": stage_figures,
    "readme": stage_readme,
}


def main():
    parser = argparse.ArgumentParser(description="Geometric firm-type factor model pipeline")
    parser.add_argument("stage", choices=STAGES.keys())
    STAGES[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
