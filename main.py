import argparse


def stage_data():
    from src.data.make_parquet import make_parquet
    from src.data.clean import clean_data
    make_parquet()
    clean_data()


def stage_tune():
    from src.tuning.tune import main
    main()


def stage_select():
    from src.tuning.tune import report
    report()


def stage_panel():
    from src.factors.build_panel import save_panel
    save_panel()


def stage_backtest():
    from src.portfolio.optimiser import main
    main()


STAGES = {
    "data": stage_data,
    "panel": stage_panel,
    "tune": stage_tune,
    "select": stage_select,
    "backtest": stage_backtest,
}


def main():
    parser = argparse.ArgumentParser(description="Geometric firm-type factor model pipeline")
    parser.add_argument("stage", choices=STAGES.keys())
    STAGES[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
