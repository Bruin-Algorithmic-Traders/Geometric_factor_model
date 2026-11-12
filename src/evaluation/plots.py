from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
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


def _scaled_returns(target_vol: float = 0.10) -> tuple[np.ndarray, dict]:
    """Dates and each run's monthly net return (plus value-weighted market excess), scaled once to target_vol."""
    from src.config import MARKET_EXCESS_CSV, RESULTS_DIR
    returns = pl.concat([pl.read_csv(RESULTS_DIR / f"benchmark_returns_{s}.csv") for s in ("is", "oos")])
    returns = returns.join(pl.read_csv(MARKET_EXCESS_CSV).select("DATE", "market_value_excess"), on="DATE").sort("DATE")
    dates = returns["DATE"].cast(pl.String).str.to_date("%Y%m%d").to_numpy()
    return dates, {c: returns[c].to_numpy() * target_vol / (returns[c].std() * np.sqrt(12)) for c in returns.columns[1:]}


def _underwater(r: np.ndarray) -> np.ndarray:
    cum = np.cumsum(r)
    return cum - np.maximum.accumulate(np.maximum(cum, 0))


def drawdown_chart(out_path: Path = FIGURES_DIR / "f3_drawdown.png"):
    """Drawdown of the cumulative sum (all lines at 10% vol): strategy, random-seed band and market excess."""
    from src.config import OOS_START_DATE
    dates, r = _scaled_returns()
    seeds = np.array([_underwater(v) for k, v in r.items() if k.startswith("random_directions_")])
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    _style(ax)
    ax.fill_between(dates, seeds.min(axis=0), seeds.max(axis=0), color="#d9d8d3", lw=0, label="random directions (8, min–max)")
    ax.plot(dates, _underwater(r["market_value_excess"]), color=SERIES[3], lw=2, label="market excess (value-weighted)")
    ax.plot(dates, _underwater(r["pca_directions"]), color=SERIES[1], lw=2, label="PCA directions")
    ax.plot(dates, _underwater(r["strategy"]), color=INK, lw=2.6, label="k-means strategy")
    boundary = np.datetime64(f"{str(OOS_START_DATE)[:4]}-01-01")
    ax.axvline(boundary, color=INK_2, lw=1)
    ax.text(boundary, 0.02, "  out of sample →", va="bottom", fontsize=9, color=INK_2)
    ax.text(boundary, 0.02, "← in sample  ", va="bottom", ha="right", fontsize=9, color=INK_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("drawdown, all lines scaled to 10% vol", color=INK_2, fontsize=9)
    ax.set_title("Out of sample, the strategy spent four years under water", loc="left", color=INK, fontsize=13, pad=18)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK_2, ncol=2)
    ax.set_xlim(dates[0], dates[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def tuning_heatmap(out_path: Path = FIGURES_DIR / "f4_tuning_heatmap.png"):
    """In-sample net Sharpe over K × τ at the locked κ, with the locked cell (solid) and the selection rule's
    pick (dashed) outlined."""
    import json
    from src.config import CONFIG_LOCKED_JSON, TUNING_GRID_CSV
    from src.tuning.tune import select
    locked = json.loads(CONFIG_LOCKED_JSON.read_text())["in_sample"]
    grid = pl.read_csv(TUNING_GRID_CSV).filter(pl.col("kappa_penalty") == locked["kappa_penalty"])
    Ks, taus = sorted(grid["K"].unique()), sorted(grid["tau"].unique())
    mat = np.array([[grid.filter((pl.col("K") == k) & (pl.col("tau") == t))["net_sharpe"].item() for t in taus] for k in Ks])

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=mat.max())
    for i in range(len(Ks)):
        for j in range(len(taus)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if mat[i, j] > 0.6 * mat.max() else INK)
    def outline(K, tau, style, label):
        i, j = Ks.index(K), taus.index(tau)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2.2, ls=style, label=label))
    outline(locked["K"], locked["tau"], "-", f"locked: K = {locked['K']}, τ = {locked['tau']:g}")
    pick = select(pl.read_csv(TUNING_GRID_CSV))
    outline(pick["K"], pick["tau"], "--", f"selection rule's pick: K = {pick['K']}, τ = {pick['tau']:g}")
    ax.set_xticks(range(len(taus)), [f"{t:g}" for t in taus])
    ax.set_yticks(range(len(Ks)), [str(k) for k in Ks])
    ax.set_xlabel("τ (low = soft, near-linear memberships; high = hard clusters)", color=INK_2, fontsize=9)
    ax.set_ylabel("K (number of clusters)", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("More clusters and softer memberships scored higher in-sample", loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"in-sample net Sharpe (κ = {locked['kappa_penalty'] * 1e4:g} bp)", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=INK_2, labelsize=8)
    cbar.outline.set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def cost_curve(out_path: Path = FIGURES_DIR / "f5_sharpe_vs_cost.png"):
    """Out-of-sample net Sharpe at 5/10/20/50 bp per $ traded: strategy, PCA, and the random-seed range and median."""
    from src.config import RESULTS_DIR
    costs = pl.read_csv(RESULTS_DIR / "benchmarks_costs_oos.csv")
    cols = [c for c in costs.columns if c.startswith("net_sharpe_")]
    bps = np.array([int(c.removeprefix("net_sharpe_").removesuffix("bp")) for c in cols])
    row = lambda name: costs.filter(pl.col("benchmark") == name).select(cols).to_numpy()[0]
    seeds = costs.filter(pl.col("benchmark").str.starts_with("random_directions_")).select(cols).to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.fill_between(bps, seeds.min(axis=0), seeds.max(axis=0), color="#d9d8d3", lw=0, label="random directions (8, min–max)")
    ax.plot(bps, np.median(seeds, axis=0), color=INK_2, lw=1.5, ls=(0, (4, 3)), label="random directions, median")
    ax.plot(bps, row("pca_directions"), color=SERIES[1], lw=2, marker="o", ms=6, label="PCA directions")
    ax.plot(bps, row("strategy"), color=INK, lw=2.6, marker="o", ms=6, label="k-means strategy")
    ends = [(row("strategy")[-1], f"k-means  {row('strategy')[-1]:.2f}"),
            (np.median(seeds, axis=0)[-1], f"random median  {np.median(seeds, axis=0)[-1]:.2f}"),
            (row("pca_directions")[-1], f"PCA  {row('pca_directions')[-1]:.2f}")]
    for y, text in ends:
        ax.annotate(text, (bps[-1], y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(bps, [f"{b} bp" for b in bps])
    ax.set_xlim(bps[0] - 3, bps[-1])
    ax.set_xlabel("trading cost per $ of stock traded", color=INK_2, fontsize=9)
    ax.set_ylabel("out-of-sample net Sharpe (2006–2018)", color=INK_2, fontsize=9)
    ax.set_title("Out of sample, costs hurt k-means least", loc="left", color=INK, fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def ic_by_year(out_path: Path = FIGURES_DIR / "f7_mu_hat_ic_by_year.png"):
    """Mean monthly rank IC of μ̂ by year, coloured by sample, with each sample's mean as a reference line."""
    from src.config import RESULTS_DIR
    summary = pl.read_csv(RESULTS_DIR / "exploratory" / "d1_mu_hat_ic_summary.csv")
    years = summary.filter(pl.col("group") == "year").with_columns(pl.col("label").cast(pl.Int64))
    means = dict(summary.filter(pl.col("group") == "sample").select("label", "ic_mean").iter_rows())
    is_year = years["label"] <= 2005

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK_2, lw=0.8)
    for mask, color, label, key, span in ((is_year, SERIES[0], "in sample", "is", (1992.6, 2005.5)),
                                         (~is_year, SERIES[1], "out of sample", "oos", (2005.5, 2018.4))):
        sel = years.filter(mask)
        ax.bar(sel["label"], sel["ic_mean"], width=0.6, color=color, label=label)
        ax.hlines(means[key], *span, color=color, lw=1.5, ls=(0, (4, 3)), label=f"{label} mean {means[key]:.3f}")
    ax.axvline(2005.5, color=INK_2, lw=1)
    ax.set_xticks(range(1993, 2019, 2))
    ax.set_xlim(1992.4, 2018.6)
    ax.set_ylabel("mean monthly rank IC of μ̂", color=INK_2, fontsize=9)
    ax.set_title("The factor return forecast became unreliable after 2007", loc="left", color=INK, fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
