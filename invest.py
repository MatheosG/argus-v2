"""
invest.py — Capital-constrained deployment simulation for ARGUS InvestNow.

Given an investment amount, simulates month-by-month deployment where
rigs are only added when the cash pool (investment + cumulative cashflow)
can cover deployment costs plus a safety buffer.

Deployment priority: onshore first (lower cost, earlier timeline),
then offshore when capital allows.
"""

import math
import numpy as np
from staffing import compute_monthly_compensation, get_benefits_multiplier
from running_costs import compute_monthly_running_costs


def run_constrained_trial(
    config: dict,
    roles: list[dict],
    cost_items: list[dict],
    investment: float,
    safety_buffer: float = 50_000,
    priority: str = "onshore_first",  # "onshore_first", "offshore_first", "balanced"
    deterministic: bool = True,
) -> list[dict]:
    """Run a capital-constrained monthly projection.

    The model deploys rigs only when cash_pool >= safety_buffer + deployment_cost.
    Cash pool = investment + cumulative net income.

    Returns list of monthly dicts (same schema as model.run_single_trial)
    with additional fields: cash_pool, rigs_wanted, rigs_blocked.
    """
    n_months = config["simulation"]["months"]
    timeline = config["timeline"]
    comp_config = config["compensation"]

    # Per-class config
    class_cfgs = {}
    for cls_name, cls_cfg in config.get("rig_classes", {}).items():
        mkt = cls_cfg["market"]
        frm = cls_cfg["timeline"]["first_rig_month"]
        prod_months = max(n_months - frm + 1, 1)
        avg_rigs = mkt["total_rigs_added"] / prod_months
        expected_avg = max(mkt["total_rigs_added"] / 2, 1)
        churn = mkt["rigs_lost_per_year"] / 12 / expected_avg

        class_cfgs[cls_name] = {
            "first_rig_month": frm,
            "avg_rigs_per_month": avg_rigs,
            "churn_prob": churn,
            "market": mkt,
            "revenue": cls_cfg["revenue"],
            "max_rigs": int(round(mkt["total_rigs_added"])),
            "total_added_so_far": 0,
        }

    # Determine deployment order
    cls_names = list(class_cfgs.keys())
    if priority == "offshore_first":
        cls_names = sorted(cls_names, key=lambda c: 0 if "off" in c else 1)
    elif priority == "balanced":
        pass  # keep config order
    else:  # onshore_first (default)
        cls_names = sorted(cls_names, key=lambda c: 0 if "on" in c else 1)

    # COGS per rig (from running costs CSV — extract once)
    cogs_per_rig = 7000  # default
    for item in cost_items:
        if item["frequency"] == "onetime" and item["scaling"] == "per_rig":
            cogs_per_rig = item["amount"]
            break

    # Estimate monthly variable cost per rig (internet + depreciation + cloud/IT share)
    monthly_var_per_rig = 800 + 4500 / 24 + 579 / 10 + 1250 / 10  # ~$1,470

    # ─── Month 0 ───
    empty_classes = {cn: {"rigs": 0, "new": 0, "lost": 0,
                          "service_revenue": 0, "installation_revenue": 0,
                          "total_revenue": 0, "active_days_per_rig": 0}
                     for cn in class_cfgs}

    month0 = {
        "month": 0, "phase": "pre",
        "rig_count": 0, "new_rigs": 0, "lost_rigs": 0,
        "by_class": empty_classes,
        "benefits_multiplier": 0,
        "service_revenue": 0, "installation_revenue": 0, "total_revenue": 0,
        "total_compensation": 0, "comp_by_department": {}, "total_headcount": 0,
        "total_cogs": 0, "total_depreciation": 0, "total_ga": 0, "total_it": 0,
        "total_running_costs": 0, "running_by_class": {},
        "gross_profit": 0, "total_opex": 0, "ebitda": 0, "ebit": 0,
        "total_costs": 0, "profit": 0, "cumulative_profit": 0,
        "cash_pool": investment, "rigs_wanted": 0, "rigs_blocked": 0,
        "investment": investment,
    }

    results = [month0]
    prior = month0
    cumulative_profit = 0

    for month in range(1, n_months + 1):
        phase = "mvp" if month <= timeline["mvp_months"] else "prod"

        # Cash pool = investment + all cumulative profit to date
        # This represents total available capital at start of month
        cash_available = investment + cumulative_profit

        # ─── Per-class deployment (capital constrained) ───
        by_class = {}
        total_rigs = 0
        total_new = 0
        total_lost = 0
        total_svc_rev = 0.0
        total_inst_rev = 0.0
        month_rigs_wanted = 0
        month_rigs_blocked = 0
        month_deploy_cost = 0  # track COGS spent this month

        for cls_name in cls_names:
            cc = class_cfgs[cls_name]
            prior_rigs = prior["by_class"][cls_name]["rigs"] if cls_name in prior.get("by_class", {}) else 0

            # Churn (always happens regardless of capital)
            rigs_lost = 0
            if prior_rigs > 0 and cc["churn_prob"] > 0:
                if deterministic:
                    rigs_lost = int(round(prior_rigs * cc["churn_prob"]))
                else:
                    rigs_lost = 0
            after_churn = prior_rigs - rigs_lost

            # How many rigs does the MARKET want to add this month?
            rigs_wanted = 0
            if month >= cc["first_rig_month"] and cc["total_added_so_far"] < cc["max_rigs"]:
                if deterministic:
                    months_active = month - cc["first_rig_month"] + 1
                    target = round(cc["avg_rigs_per_month"] * months_active)
                    rigs_wanted = max(target - cc["total_added_so_far"], 0)
                    if month == cc["first_rig_month"] and cc["total_added_so_far"] == 0:
                        rigs_wanted = max(rigs_wanted, 1)

            month_rigs_wanted += rigs_wanted

            # How many can we AFFORD?
            rigs_added = 0
            for _ in range(rigs_wanted):
                deploy_cost = cogs_per_rig
                # Check: after deploying, will cash pool still be above safety buffer?
                # Estimate next month's burn = current compensation + var costs for (total_rigs+1)
                projected_cash = cash_available - month_deploy_cost - deploy_cost
                if projected_cash >= safety_buffer:
                    rigs_added += 1
                    month_deploy_cost += deploy_cost
                    cc["total_added_so_far"] += 1
                else:
                    month_rigs_blocked += (rigs_wanted - rigs_added)
                    break

            new_count = max(after_churn + rigs_added, 0)
            rev = _compute_rev(cc["market"], cc["revenue"], new_count, rigs_added)

            by_class[cls_name] = {
                "rigs": new_count, "new": rigs_added, "lost": rigs_lost,
                "service_revenue": rev["service_revenue"],
                "installation_revenue": rev["installation_revenue"],
                "total_revenue": rev["total_revenue"],
                "active_days_per_rig": rev["active_days_per_rig"],
            }

            total_rigs += new_count
            total_new += rigs_added
            total_lost += rigs_lost
            total_svc_rev += rev["service_revenue"]
            total_inst_rev += rev["installation_revenue"]

        total_revenue = total_svc_rev + total_inst_rev

        # ─── Costs (same as base model) ───
        benefits = get_benefits_multiplier(
            month, comp_config["benefits_base"], comp_config["benefits_quarterly_increase"])
        comp = compute_monthly_compensation(roles, month, total_rigs, benefits)
        rc = compute_monthly_running_costs(cost_items, phase, total_rigs, total_new)

        total_compensation = comp["total"]
        total_cogs = rc["total_cogs"]
        total_depreciation = rc["total_depreciation"]
        total_ga = rc["by_class"].get("ga", 0)
        total_it = rc["by_class"].get("it_services", 0)
        total_running = rc["total"]

        gross_profit = total_revenue - total_cogs
        total_opex = total_compensation + total_ga + total_it + total_depreciation
        ebitda = gross_profit - total_opex + total_depreciation
        ebit = gross_profit - total_opex
        total_costs = total_compensation + total_running
        profit = total_revenue - total_costs

        cumulative_profit += profit
        # Cash pool = investment + cumulative profit (all-in)
        cash_pool = investment + cumulative_profit

        row = {
            "month": month, "phase": phase,
            "rig_count": total_rigs, "new_rigs": total_new, "lost_rigs": total_lost,
            "by_class": by_class, "benefits_multiplier": benefits,
            "service_revenue": total_svc_rev, "installation_revenue": total_inst_rev,
            "total_revenue": total_revenue,
            "total_compensation": total_compensation,
            "comp_by_department": comp["by_department"],
            "total_headcount": comp["total_headcount"],
            "total_cogs": total_cogs, "total_depreciation": total_depreciation,
            "total_ga": total_ga, "total_it": total_it,
            "total_running_costs": total_running, "running_by_class": rc["by_class"],
            "gross_profit": gross_profit, "total_opex": total_opex,
            "ebitda": ebitda, "ebit": ebit,
            "total_costs": total_costs, "profit": profit,
            "cumulative_profit": cumulative_profit,
            "cash_pool": cash_pool,
            "rigs_wanted": month_rigs_wanted,
            "rigs_blocked": month_rigs_blocked,
            "investment": investment,
        }

        results.append(row)
        prior = row

    return results


def _compute_rev(market, rev_cfg, rig_count, new_rigs):
    util = market["utilization_rate"]
    days = rev_cfg["days_per_month"]
    rate = market["daily_rate"]
    active_days = util * days
    svc = rig_count * active_days * rate
    inst = new_rigs * rev_cfg.get("installation_fee", 0)
    return {"service_revenue": svc, "installation_revenue": inst,
            "total_revenue": svc + inst, "active_days_per_rig": active_days}


def run_investment_comparison(
    config: dict, roles: list[dict], cost_items: list[dict],
    amounts: list[float], safety_buffer: float = 50_000,
    priority: str = "onshore_first",
) -> list[dict]:
    """Run multiple investment scenarios for comparison table."""
    results = []
    for amt in amounts:
        res = run_constrained_trial(config, roles, cost_items, amt, safety_buffer, priority)
        final = res[-1]
        # Find breakeven
        be = None
        for r in res:
            if r["cumulative_profit"] > 0 and r["month"] > 0:
                be = r["month"]
                break

        total_blocked = sum(r["rigs_blocked"] for r in res)

        results.append({
            "investment": amt,
            "final_rigs": final["rig_count"],
            "final_onshore": final["by_class"].get("onshore", {}).get("rigs", 0),
            "final_offshore": final["by_class"].get("offshore", {}).get("rigs", 0),
            "total_revenue": sum(r["total_revenue"] for r in res),
            "cumulative_profit": final["cumulative_profit"],
            "breakeven_month": be,
            "min_cash": min(r["cash_pool"] for r in res),
            "final_cash": final["cash_pool"],
            "total_blocked": total_blocked,
            "moic": (final["cumulative_profit"] + amt) / amt if amt > 0 else 0,
            "results": res,
        })
    return results