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
