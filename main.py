"""
main.py — Run ARGUS v2 model: single trial + Monte Carlo.
"""

from config import load_config, resolve_stochastic
from staffing import load_staffing
from running_costs import load_running_costs
from model import run_single_trial
import numpy as np


def fmt(val):
    if abs(val) >= 1_000_000:
        return f"${val/1e6:>8.2f}M"
    elif abs(val) >= 1_000:
        return f"${val/1e3:>8.1f}K"
    else:
        return f"${val:>8.0f}"


def print_monthly_pnl(results: list[dict]):
    print(f"\n{'='*150}")
    print(f"ARGUS v2 — Monthly P&L")
    print(f"{'='*150}")
    print(
        f"{'Mo':>3} {'Ph':<4} {'Rigs':>4} {'+':>2} {'-':>2} "
        f"{'Svc Rev':>10} {'Inst Rev':>10} {'Tot Rev':>10} "
        f"{'COGS':>10} {'Gross P':>10} "
        f"{'Comp':>10} {'Depr':>8} {'G&A':>8} {'IT':>8} "
        f"{'Profit':>10} {'Cumul':>10}"
    )
    print(f"{'-'*150}")

    for r in results:
        print(
            f"{r['month']:>3} {r['phase']:<4} {r['rig_count']:>4} "
            f"{r['new_rigs']:>2} {r['lost_rigs']:>2} "
            f"{fmt(r['service_revenue']):>10} "
            f"{fmt(r['installation_revenue']):>10} "
            f"{fmt(r['total_revenue']):>10} "
            f"{fmt(r['total_cogs']):>10} "
            f"{fmt(r['gross_profit']):>10} "
            f"{fmt(r['total_compensation']):>10} "
            f"{fmt(r['total_depreciation']):>8} "
            f"{fmt(r['total_ga']):>8} "
            f"{fmt(r['total_it']):>8} "
            f"{fmt(r['profit']):>10} "
            f"{fmt(r['cumulative_profit']):>10}"
        )

    print(f"{'='*150}")


def print_annual_pnl(results: list[dict]):
    print(f"\n{'='*120}")
    print(f"ANNUAL P&L SUMMARY")
    print(f"{'='*120}")
    print(
        f"{'Year':>6} {'Rigs':>6} "
        f"{'Svc Rev':>12} {'Inst Rev':>12} {'Tot Rev':>12} "
        f"{'COGS':>12} {'Gross P':>12} "
        f"{'Comp':>12} {'Depr':>10} {'G&A':>10} {'IT':>10} "
        f"{'Profit':>12}"
    )
    print(f"{'-'*120}")

    n_months = len(results)
    totals = {}

    for year in range(1, (n_months // 12) + 1):
        start = (year - 1) * 12
        end = year * 12
        yr = results[start:end]

        row = {
            "rigs": yr[-1]["rig_count"],
            "svc_rev": sum(r["service_revenue"] for r in yr),
            "inst_rev": sum(r["installation_revenue"] for r in yr),
            "tot_rev": sum(r["total_revenue"] for r in yr),
            "cogs": sum(r["total_cogs"] for r in yr),
            "gross_p": sum(r["gross_profit"] for r in yr),
            "comp": sum(r["total_compensation"] for r in yr),
            "depr": sum(r["total_depreciation"] for r in yr),
            "ga": sum(r["total_ga"] for r in yr),
            "it": sum(r["total_it"] for r in yr),
            "profit": sum(r["profit"] for r in yr),
        }

        # Accumulate totals
        for k, v in row.items():
            if k != "rigs":
                totals[k] = totals.get(k, 0) + v

        print(
            f"{year:>6} {row['rigs']:>6} "
            f"{fmt(row['svc_rev']):>12} {fmt(row['inst_rev']):>12} {fmt(row['tot_rev']):>12} "
            f"{fmt(row['cogs']):>12} {fmt(row['gross_p']):>12} "
            f"{fmt(row['comp']):>12} {fmt(row['depr']):>10} {fmt(row['ga']):>10} {fmt(row['it']):>10} "
            f"{fmt(row['profit']):>12}"
        )

    print(f"{'-'*120}")
    print(
        f"{'TOTAL':>6} {'':>6} "
        f"{fmt(totals['svc_rev']):>12} {fmt(totals['inst_rev']):>12} {fmt(totals['tot_rev']):>12} "
        f"{fmt(totals['cogs']):>12} {fmt(totals['gross_p']):>12} "
        f"{fmt(totals['comp']):>12} {fmt(totals['depr']):>10} {fmt(totals['ga']):>10} {fmt(totals['it']):>10} "
        f"{fmt(totals['profit']):>12}"
    )
    print(f"{'='*120}")

    # Margins
    if totals["tot_rev"] > 0:
        gm = totals["gross_p"] / totals["tot_rev"] * 100
        npm = totals["profit"] / totals["tot_rev"] * 100
        print(f"\n  Gross Margin: {gm:.1f}%  |  Net Margin: {npm:.1f}%")


def run_monte_carlo(config: dict, roles: list[dict], cost_items: list[dict]):
    n_trials = config["simulation"]["n_trials"]
    n_months = config["simulation"]["months"]
    seed = config["simulation"]["seed"]

    print(f"\n  Running Monte Carlo ({n_trials:,} trials)...")

    n_years = n_months // 12
    annual_revenue = np.zeros((n_trials, n_years))
    annual_profit = np.zeros((n_trials, n_years))
    annual_gross_profit = np.zeros((n_trials, n_years))
    annual_rigs = np.zeros((n_trials, n_years))
    cumulative_profit = np.zeros(n_trials)
    breakeven_month = np.full(n_trials, np.nan)

    for trial in range(n_trials):
        trial_rng = np.random.default_rng(seed + trial)
        resolved = resolve_stochastic(config, trial_rng)
        results = run_single_trial(resolved, roles, cost_items, trial_rng)

        for year in range(n_years):
            start = year * 12
            end = (year + 1) * 12
            yr = results[start:end]
            annual_revenue[trial, year] = sum(r["total_revenue"] for r in yr)
            annual_profit[trial, year] = sum(r["profit"] for r in yr)
            annual_gross_profit[trial, year] = sum(r["gross_profit"] for r in yr)
            annual_rigs[trial, year] = yr[-1]["rig_count"]

        cumulative_profit[trial] = sum(r["profit"] for r in results)
        cum = 0
        for r in results:
            cum += r["profit"]
            if cum > 0 and np.isnan(breakeven_month[trial]):
                breakeven_month[trial] = r["month"]

    percentiles = [5, 25, 50, 75, 95]

    print(f"\n{'='*90}")
    print(f"MONTE CARLO RESULTS ({n_trials:,} trials)")
    print(f"{'='*90}")

    print(f"\n  ANNUAL REVENUE")
    print(f"  {'':>12}" + "".join(f"{'Year ' + str(y+1):>14}" for y in range(n_years)))
    print(f"  {'-'*70}")
    for p in percentiles:
        row = f"  {'P' + str(p):<12}"
        for y in range(n_years):
            row += f"{fmt(np.percentile(annual_revenue[:, y], p)):>14}"
        print(row)

    print(f"\n  ANNUAL GROSS PROFIT")
    print(f"  {'':>12}" + "".join(f"{'Year ' + str(y+1):>14}" for y in range(n_years)))
    print(f"  {'-'*70}")
    for p in percentiles:
        row = f"  {'P' + str(p):<12}"
        for y in range(n_years):
            row += f"{fmt(np.percentile(annual_gross_profit[:, y], p)):>14}"
        print(row)

    print(f"\n  ANNUAL NET PROFIT")
    print(f"  {'':>12}" + "".join(f"{'Year ' + str(y+1):>14}" for y in range(n_years)))
    print(f"  {'-'*70}")
    for p in percentiles:
        row = f"  {'P' + str(p):<12}"
        for y in range(n_years):
            row += f"{fmt(np.percentile(annual_profit[:, y], p)):>14}"
        print(row)

    print(f"\n  RIGS AT YEAR END")
    print(f"  {'':>12}" + "".join(f"{'Year ' + str(y+1):>14}" for y in range(n_years)))
    print(f"  {'-'*70}")
    for p in percentiles:
        row = f"  {'P' + str(p):<12}"
        for y in range(n_years):
            row += f"{np.percentile(annual_rigs[:, y], p):>14.0f}"
        print(row)

    print(f"\n  KEY METRICS")
    print(f"  {'-'*50}")

    valid_breakeven = breakeven_month[~np.isnan(breakeven_month)]
    pct_breakeven = len(valid_breakeven) / n_trials * 100

    print(f"  4-Year Cumulative Profit:")
    for p in percentiles:
        print(f"    P{p:<3}: {fmt(np.percentile(cumulative_profit, p))}")

    print(f"\n  Breakeven:")
    print(f"    Trials that break even:  {pct_breakeven:.0f}%")
    if len(valid_breakeven) > 0:
        print(f"    Median breakeven month:  {np.median(valid_breakeven):.0f}")
        print(f"    P5 breakeven month:      {np.percentile(valid_breakeven, 5):.0f}")
        print(f"    P95 breakeven month:     {np.percentile(valid_breakeven, 95):.0f}")

    print(f"\n{'='*90}")


def main():
    config = load_config("config.yaml")
    roles = load_staffing(config["files"]["staffing"])
    cost_items = load_running_costs(config["files"]["running_costs"])

    print(f"\n  Loaded {len(roles)} roles from staffing CSV")
    print(f"  Loaded {len(cost_items)} cost items from running costs CSV")
    print(f"  Projection: {config['simulation']['months']} months")
    print(f"  Trials: {config['simulation']['n_trials']:,}")

    # Single trial
    rng = np.random.default_rng(config["simulation"]["seed"])
    resolved = resolve_stochastic(config, rng)

    market = resolved["market"]
    timeline = resolved["timeline"]
    prod_months = resolved["simulation"]["months"] - timeline["mvp_months"]

    print(f"\n  Sample market parameters:")
    print(f"    Daily rate:           ${market['daily_rate']:,.0f}/day")
    print(f"    Utilization:          {market['utilization_rate']:.0%}")
    print(f"    Rigs added (4yr):     {market['total_rigs_added']:.1f}")
    print(f"      → avg/month:        {market['total_rigs_added']/prod_months:.2f}")
    print(f"    Rigs lost/year:       {market['rigs_lost_per_year']:.1f}")

    results = run_single_trial(resolved, roles, cost_items, rng)
    print_monthly_pnl(results)
    print_annual_pnl(results)

    # Monte Carlo
    run_monte_carlo(config, roles, cost_items)


if __name__ == "__main__":
    main()