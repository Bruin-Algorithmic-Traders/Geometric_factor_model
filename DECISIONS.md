# Decisions log

Open notes: dropped cols list (task 4).

## 2026-11-09 — Task 1: folder reorg
- **Decision:** Split code into `src/{data,models,factors,portfolio,tuning,evaluation}`, add `tests/`, `notebooks/`, `docs/paper.pdf`, `results/`; `main.py <stage>` is the single entry point; package imports (`from src...`) everywhere.
- **Why:** `src/backtest/` mixed factor building, optimisation and tuning; bare imports (`from config`, `from optimiser`) only worked from certain folders.
- **Also:** duplicated Sharpe/vol code moved to `src/evaluation/metrics.py`; small result files (png/json/csv) in `results/` are now tracked so the README can show them.
- **Rejected:** folding task 2 (config centralisation) into this commit; empty `benchmarks.py`/`plots.py` stubs before they're needed.
