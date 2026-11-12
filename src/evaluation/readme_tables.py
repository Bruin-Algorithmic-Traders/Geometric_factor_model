import re

import polars as pl

from src.config import PROJECT_ROOT, RESULTS_DIR

README = PROJECT_ROOT / "README.md"
NAMES = {
    "strategy": "k-means strategy", "linear_tau_0.01": "τ = 0.01 (near-linear memberships)",
    "pca_directions": "PCA directions", "random_seeds_median": "random directions, median of 8",
    "inverse_vol_mix": "inverse-vol mix", "diagonal_mv_mix": "diagonal mean-variance mix",
    "returns_free_features": "returns-free features", "market_value_excess": "market excess (value-weighted)",
}


def _md(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def _num(x: float, fmt: str = ".2f") -> str:
    """Format with a true minus sign and no negative zero."""
    text = f"{x:{fmt}}"
    return text.lstrip("-") if float(text.replace("%", "").replace("+", "")) == 0 else text.replace("-", "−")


def _sharpe_ci(row: dict) -> str:
    return f"{_num(row['net_sharpe'])} [{_num(row['sharpe_ci_low'])}, {_num(row['sharpe_ci_high'])}]"


def _metrics(sample: str) -> dict:
    return {r["run"]: r for r in pl.read_csv(RESULTS_DIR / f"metrics_{sample}.csv").iter_rows(named=True)}


def headline() -> str:
    m = {s: _metrics(s)["strategy"] for s in ("is", "oos")}
    att = {s: {r["term"]: r for r in pl.read_csv(RESULTS_DIR / f"attribution_{s}.csv")
               .filter((pl.col("run") == "strategy") & (pl.col("model") == "ff5_umd")).iter_rows(named=True)}
           for s in ("is", "oos")}
    rows = [
        ["Net Sharpe [95% CI]", *(_sharpe_ci(m[s]) for s in m)],
        ["t-stat (i.i.d. / Newey-West)", *(f"{m[s]['iid_t_stat']:.2f} / {m[s]['nw_t_stat']:.2f}" for s in m)],
        ["Return / vol per year", *(f"{m[s]['ann_return']:.1%} / {m[s]['ann_vol']:.1%}" for s in m)],
        ["Max drawdown / longest spell under water", *(f"{_num(m[s]['max_drawdown'], '.1%')} / {m[s]['max_dd_months']} months" for s in m)],
        ["Worst month", *(_num(m[s]['worst_month'], '.1%') for s in m)],
        ["Cost drag per year / stock turnover per month", *(f"{m[s]['cost_drag_ann']:.1%} / {m[s]['stock_turnover']:.2f}" for s in m)],
        ["Alpha after FF5 + momentum (t)", *(f"{att[s]['alpha_ann']['estimate']:.1%} ({att[s]['alpha_ann']['nw_t']:.2f})" for s in att)],
        ["Correlation with the market", *(_num(m[s]['corr_mkt_rf']) for s in m)],
    ]
    return _md(["", "In sample 1993–2005", "Out of sample 2006–2018"], rows)


def layers() -> str:
    m = {s: _metrics(s) for s in ("is", "oos")}
    def cell(s, run):
        r = m[s][run]
        if run == "random_seeds_median":
            return f"{_num(r['net_sharpe'])} (range {_num(m[s]['random_seeds_min']['net_sharpe'])} to {_num(m[s]['random_seeds_max']['net_sharpe'])})"
        return _sharpe_ci(r)
    runs = ["strategy", "linear_tau_0.01", "random_seeds_median", "pca_directions", "inverse_vol_mix",
            "diagonal_mv_mix", "market_value_excess"]
    return _md(["Net Sharpe at 10 bp", "In sample", "Out of sample"], [[NAMES[r], cell("is", r), cell("oos", r)] for r in runs])


def relative() -> str:
    c = pl.read_csv(RESULTS_DIR / "compare_oos.csv")
    rank = {r["cost_bps"]: r for r in pl.read_csv(RESULTS_DIR / "compare_rank.csv").filter(pl.col("sample") == "oos").iter_rows(named=True)}
    rows = []
    for bench, bps in (("pca_directions", 10), ("pca_directions", 20), ("inverse_vol_mix", 10), ("diagonal_mv_mix", 10),
                       ("linear_tau_0.01", 10), ("returns_free_features", 10)):
        r = c.filter((pl.col("benchmark") == bench) & (pl.col("cost_bps") == bps)).row(0, named=True)
        verdict = "significant, borderline" if r["verdict"] == "significant" and r["ci_low"] < 0.05 else r["verdict"]
        rows.append([f"vs {NAMES[bench]}", f"{bps} bp",
                     f"{_num(r['sharpe_diff'], '+.2f')} [{_num(r['ci_low'], '+.2f')}, {_num(r['ci_high'], '+.2f')}]", verdict])
    for bps in (5, 10, 20, 50):
        seeds = c.filter(pl.col("benchmark").str.starts_with("random_directions_") & (pl.col("cost_bps") == bps))
        sig = (seeds["verdict_holm"] == "significant").sum()
        ahead = (seeds["sharpe_diff"] > 0).sum()
        verdict = (f"ahead of {ahead} of 8; {sig} significant after Holm" if seeds["claim"][0] == "better"
                   else f"no difference from {(seeds['verdict_holm'] == 'no difference').sum()} of 8 after Holm")
        rows.append(["vs each of 8 random direction sets", f"{bps} bp", f"ranks {rank[bps]['rank_among_9']} of 9", verdict])
    return _md(["k-means strategy, out of sample", "Cost", "Sharpe difference [95% CI]", "Verdict"], rows)


def scorecard() -> str:
    m = _metrics("oos")
    costs = {r["benchmark"]: r for r in pl.read_csv(RESULTS_DIR / "benchmarks_costs_oos.csv").iter_rows(named=True)}
    c = pl.read_csv(RESULTS_DIR / "compare_oos.csv")
    rank = {r["cost_bps"]: r for r in pl.read_csv(RESULTS_DIR / "compare_rank.csv").filter(pl.col("sample") == "oos").iter_rows(named=True)}
    get = lambda b, bps: c.filter((pl.col("benchmark") == b) & (pl.col("cost_bps") == bps)).row(0, named=True)
    low_cost = c.filter(pl.col("benchmark").str.starts_with("random_directions_") & pl.col("cost_bps").is_in([5, 10]))
    s, d30, d100 = (m[k]["net_sharpe"] for k in ("strategy", "delisting_minus30", "delisting_minus100"))
    tick = lambda ok: "✓" if ok else "✗"
    rows = [
        ["Net Sharpe roughly 0.6–1.0", f"{s:.2f}", tick(0.6 <= s <= 1.0)],
        ["Positive at 10 bp and 20 bp", f"{costs['strategy']['net_sharpe_10bp']:.2f}, {costs['strategy']['net_sharpe_20bp']:.2f}",
         tick(costs["strategy"]["net_sharpe_10bp"] > 0 and costs["strategy"]["net_sharpe_20bp"] > 0)],
        ["Above the inverse-vol mix", f"{get('inverse_vol_mix', 10)['sharpe_diff']:+.2f}", tick(get("inverse_vol_mix", 10)["sharpe_diff"] > 0)],
        ["At least the median random run at 20 and 50 bp",
         f"{rank[20]['strategy_sharpe']:.2f} vs {rank[20]['seed_median']:.2f}; {rank[50]['strategy_sharpe']:.2f} vs {rank[50]['seed_median']:.2f}",
         tick(all(rank[b]["strategy_sharpe"] >= rank[b]["seed_median"] for b in (20, 50)))],
        ["About equal to random runs at 5–10 bp", f"{(low_cost['verdict_holm'] == 'no difference').sum()} of 16 no difference",
         tick((low_cost["verdict_holm"] == "no difference").all())],
        ["τ = 0.01 about equal", get("linear_tau_0.01", 10)["verdict"], tick(get("linear_tau_0.01", 10)["verdict"] == "no difference")],
        ["PCA below after costs", f"{get('pca_directions', 10)['sharpe_diff']:+.2f}, {get('pca_directions', 20)['sharpe_diff']:+.2f}",
         tick(get("pca_directions", 10)["sharpe_diff"] > 0 and get("pca_directions", 20)["sharpe_diff"] > 0)],
        ["Returns-free features about equal", get("returns_free_features", 10)["verdict"], tick(get("returns_free_features", 10)["verdict"] == "no difference")],
        ["Delisting returns lower Sharpe by about 0.1", f"−30%: {d30 - s:+.2f}; −100%: {d100 - s:+.2f}", tick(d30 < s and d100 < s)],
        ["Low correlation with the market (below 0.3)", f"{m['strategy']['corr_mkt_rf']:+.2f}", tick(abs(m["strategy"]["corr_mkt_rf"]) < 0.3)],
    ]
    rows = [[e, r.replace("-", "−"), t] for e, r, t in rows]
    return _md(["Expectation (written before the run)", "Result", ""], rows)


TABLES = {"headline": headline, "layers": layers, "relative": relative, "scorecard": scorecard}


def main():
    """Replace each <!-- table:name --> ... <!-- /table:name --> block in README.md with the table built from results/."""
    text = README.read_text()
    for name, build in TABLES.items():
        text = re.sub(rf"(<!-- table:{name} -->\n).*?(<!-- /table:{name} -->)", lambda m: m.group(1) + build() + "\n" + m.group(2),
                      text, flags=re.S)
    README.write_text(text)
