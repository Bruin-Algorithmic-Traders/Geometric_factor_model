import argparse


def stage_data():
    from src.data.make_parquet import make_parquet
    from src.data.clean import clean_data
    make_parquet()
    clean_data()


def stage_tune_factors():
    from src.tuning.tune_factors import evaluate_is_grid
    evaluate_is_grid()


def stage_tune_optimiser():
    from src.tuning.tune_optimiser import main
    main()


def stage_panel():
    from src.factors.build_panel import save_panel
    save_panel()


def stage_backtest():
    from src.portfolio.optimiser import main
    main()


STAGES = {
    "data": stage_data,
    "tune-factors": stage_tune_factors,
    "tune-optimiser": stage_tune_optimiser,
    "panel": stage_panel,
    "backtest": stage_backtest,
}


def main():
    parser = argparse.ArgumentParser(description="Geometric firm-type factor model pipeline")
    parser.add_argument("stage", choices=STAGES.keys())
    STAGES[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
