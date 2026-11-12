from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def _heatmap(ax, grid: pl.DataFrame, row: str, col: str, chosen: dict, row_label: str, col_label: str):
    rows, cols = sorted(grid[row].unique()), sorted(grid[col].unique())
    mat = np.array([[grid.filter((pl.col(row) == r) & (pl.col(col) == c))["net_sharpe"].item() for c in cols] for r in rows])
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=min(0, mat.min()), vmax=mat.max())
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.6 * mat.max() else "black")
    ci, cj = rows.index(chosen[row]), cols.index(chosen[col])
    ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False, edgecolor="#c4462f", lw=2))
    ax.set_xticks(range(len(cols)), [f"{c * 1e4:g} bp" if col == "kappa_penalty" else f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)), [f"{r:g}" for r in rows])
    ax.set_xlabel(col_label); ax.set_ylabel(row_label)
    plt.colorbar(im, ax=ax, label="in-sample net Sharpe")


def tuning_heatmaps(grid: pl.DataFrame, chosen: dict, out_dir: Path):
    """Two slices through the chosen configuration (outlined): K × τ at its κ and τ × κ at its K."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _heatmap(axes[0], grid.filter(pl.col("kappa_penalty") == chosen["kappa_penalty"]), "K", "tau", chosen, "K", "τ")
    axes[0].set_title(f"κ = {chosen['kappa_penalty'] * 1e4:g} bp", fontsize=10)
    _heatmap(axes[1], grid.filter(pl.col("K") == chosen["K"]), "tau", "kappa_penalty", chosen, "τ", "κ_penalty")
    axes[1].set_title(f"K = {chosen['K']}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "tuning_heatmaps.png")
    plt.close(fig)


def benchmark_curves(returns: pl.DataFrame, out_path: Path, target_vol: float = 0.10):
    """Cumulative net return of each series, each rescaled after the fact to the same realised vol for display."""
    dates = returns["DATE"].cast(pl.String).str.to_date("%Y%m%d")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in returns.columns[1:]:
        r = returns[name].to_numpy()
        r = r * target_vol / (r.std(ddof=1) * np.sqrt(12))
        if name.startswith("random_directions"):
            ax.plot(dates, np.cumsum(r), lw=0.8, color="#bbbbbb",
                    label="random directions" if name.endswith("_0") else None)
        else:
            ax.plot(dates, np.cumsum(r), lw=2.2 if name == "strategy" else 1.1,
                    color="black" if name == "strategy" else None, label=name)
    ax.set_ylabel(f"cumulative return at {target_vol:.0%} vol")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (light)
FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def _end_labels(ax, x, ends: list, min_gap: float = 0.11):
    """Direct labels at the right edge, spread apart vertically, each tied to its line end by a thin leader."""
    ends = sorted(ends)
    placed = []
    for y, _ in ends:
        placed.append(max(y, placed[-1] + min_gap) if placed else y)
    for (y, text), y_label in zip(ends, placed):
        ax.annotate(text, (x, y), xytext=(x + np.timedelta64(300, "D"), y_label), va="center", fontsize=8.5,
                    color=INK_2, annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8, "shrinkA": 0, "shrinkB": 2})


def equity_curve(out_path: Path = FIGURES_DIR / "f2_equity_curve.png", target_vol: float = 0.10):
    """Cumulative sum of monthly net returns (10 bp), each line scaled once to 10% realised vol over 1993-2018."""
    from src.config import MARKET_EXCESS_CSV, OOS_START_DATE, RESULTS_DIR
    returns = pl.concat([pl.read_csv(RESULTS_DIR / f"benchmark_returns_{s}.csv") for s in ("is", "oos")])
    returns = returns.join(pl.read_csv(MARKET_EXCESS_CSV).select("DATE", "market_value_excess"), on="DATE").sort("DATE")
    dates = returns["DATE"].cast(pl.String).str.to_date("%Y%m%d").to_numpy()
    curve = lambda c: np.cumsum(returns[c].to_numpy() * target_vol / (returns[c].std() * np.sqrt(12)))
    seeds = np.array([curve(c) for c in returns.columns if c.startswith("random_directions_")])

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    _style(ax)
    ax.fill_between(dates, seeds.min(axis=0), seeds.max(axis=0), color="#d9d8d3", lw=0, label="random directions (8, min–max)")
    lines = [("market_value_excess", "market excess (value-weighted)", SERIES[3], 2),
             ("inverse_vol_mix", "inverse-vol mix", SERIES[2], 2),
             ("pca_directions", "PCA directions", SERIES[1], 2),
             ("linear_tau_0.01", "τ = 0.01", SERIES[0], 2),
             ("strategy", "k-means strategy", INK, 2.6)]
    ends = []
    for col, label, color, lw in lines:
        y = curve(col)
        ax.plot(dates, y, color=color, lw=lw, solid_capstyle="round", label=label)
        ends.append((y[-1], f"{label}  {y[-1]:.1f}"))
    _end_labels(ax, dates[-1], ends)
    boundary = np.datetime64(f"{str(OOS_START_DATE)[:4]}-01-01")
    ax.axvline(boundary, color=INK_2, lw=1)
    ax.text(boundary, ax.get_ylim()[1], "  out of sample →", va="top", fontsize=9, color=INK_2)
    ax.text(boundary, ax.get_ylim()[1], "← in sample  ", va="top", ha="right", fontsize=9, color=INK_2)
    ax.set_ylabel(f"cumulative return, all lines scaled to {target_vol:.0%} vol", color=INK_2, fontsize=9)
    ax.set_title("Strong before 2006, mostly flat after", loc="left", color=INK, fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_2)
    ax.set_xlim(dates[0], dates[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
