"""
model.py — Monthly P&L model for ARGUS.

Supports multiple rig classes (e.g. onshore, offshore), each with
independent market params, timelines, and revenue models.
Shared costs (compensation, cloud, IT) scale on total combined rigs.
"""

import math
import numpy as np
from staffing import compute_monthly_compensation, get_benefits_multiplier
from running_costs import compute_monthly_running_costs


# ─────────────────────────────────────────
# Per-class rig simulation
# ─────────────────────────────────────────
def simulate_class_rigs(
    first_rig_month: int, month: int, current_rigs: int,
    avg_rigs_per_month: float, monthly_churn_prob: float,
    rng: np.random.Generator, deterministic: bool = False,
) -> tuple[int, int, int]:
    """Simulate rig additions and churn for one rig class."""
    if month < first_rig_month:
        return 0, 0, 0

    # Churn
    rigs_lost = 0
    if current_rigs > 0 and monthly_churn_prob > 0:
        if deterministic:
            rigs_lost = int(round(current_rigs * monthly_churn_prob))
        else:
            rigs_lost = rng.binomial(current_rigs, monthly_churn_prob)

    after_churn = current_rigs - rigs_lost

    # Additions
    if deterministic:
        months_active = month - first_rig_month + 1
        target = round(avg_rigs_per_month * months_active)
        prev = round(avg_rigs_per_month * (months_active - 1)) if months_active > 1 else 0
        rigs_added = max(target - prev, 0)
    else:
        rigs_added = rng.poisson(avg_rigs_per_month)

    # First month must have at least 1
    if month == first_rig_month and current_rigs == 0:
        rigs_added = max(rigs_added, 1)

    new_count = max(after_churn + rigs_added, 0)
    return new_count, rigs_added, rigs_lost


def compute_class_revenue(market: dict, rev_cfg: dict, rig_count: int, new_rigs: int) -> dict:
    """Compute revenue for one rig class."""
    util = market["utilization_rate"]
    days = rev_cfg["days_per_month"]
    rate = market["daily_rate"]

    active_days = util * days
    svc_rev = rig_count * active_days * rate
    inst_rev = new_rigs * rev_cfg.get("installation_fee", 0)

    return {
        "service_revenue": svc_rev,
        "installation_revenue": inst_rev,
        "total_revenue": svc_rev + inst_rev,
        "active_days_per_rig": active_days,
    }


# ─────────────────────────────────────────
# Main month computation
# ─────────────────────────────────────────
def compute_month(
    config: dict, roles: list[dict], cost_items: list[dict],
    month: int, prior: dict | None,
    class_params: dict,  # {cls_name: {avg_rigs, churn_prob, first_rig_month, market, revenue}}
    rng: np.random.Generator, deterministic: bool = False,
) -> dict:
    """Compute a single month of the P&L across all rig classes."""
    timeline = config["timeline"]
    comp_config = config["compensation"]
    phase = "mvp" if month <= timeline["mvp_months"] else "prod"

    # --- Per-class rig simulation ---
    by_class = {}
    total_rigs = 0
    total_new = 0
    total_lost = 0
    total_svc_rev = 0.0
    total_inst_rev = 0.0

    for cls_name, cp in class_params.items():
        prior_rigs = prior["by_class"][cls_name]["rigs"] if prior and cls_name in prior.get("by_class", {}) else 0

        rigs, added, lost = simulate_class_rigs(
            cp["first_rig_month"], month, prior_rigs,
            cp["avg_rigs_per_month"], cp["churn_prob"],
            rng, deterministic,
        )

        rev = compute_class_revenue(cp["market"], cp["revenue"], rigs, added)

        by_class[cls_name] = {
            "rigs": rigs, "new": added, "lost": lost,
            "service_revenue": rev["service_revenue"],
            "installation_revenue": rev["installation_revenue"],
            "total_revenue": rev["total_revenue"],
            "active_days_per_rig": rev["active_days_per_rig"],
        }

        total_rigs += rigs
        total_new += added
        total_lost += lost
        total_svc_rev += rev["service_revenue"]
        total_inst_rev += rev["installation_revenue"]

    total_revenue = total_svc_rev + total_inst_rev

    # --- Benefits ---
    benefits = get_benefits_multiplier(
        month, comp_config["benefits_base"], comp_config["benefits_quarterly_increase"])

    # --- Compensation (shared, scales on total rigs) ---
    comp = compute_monthly_compensation(roles, month, total_rigs, benefits)

    # --- Running Costs (shared, scales on total rigs) ---
    rc = compute_monthly_running_costs(cost_items, phase, total_rigs, total_new)

    # --- P&L ---
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

    prior_cum = prior["cumulative_profit"] if prior else 0
    cumulative_profit = prior_cum + profit

    return {
        "month": month,
        "phase": phase,
        "rig_count": total_rigs,
        "new_rigs": total_new,
        "lost_rigs": total_lost,
        "by_class": by_class,
        "benefits_multiplier": benefits,
        "service_revenue": total_svc_rev,
        "installation_revenue": total_inst_rev,
        "total_revenue": total_revenue,
        "total_compensation": total_compensation,
        "comp_by_department": comp["by_department"],
        "total_headcount": comp["total_headcount"],
        "total_cogs": total_cogs,
        "total_depreciation": total_depreciation,
        "total_ga": total_ga,
        "total_it": total_it,
        "total_running_costs": total_running,
        "running_by_class": rc["by_class"],
        "gross_profit": gross_profit,
        "total_opex": total_opex,
        "ebitda": ebitda,
        "ebit": ebit,
        "total_costs": total_costs,
        "profit": profit,
        "cumulative_profit": cumulative_profit,
    }


def _build_class_params(config: dict) -> dict:
    """Pre-compute per-class parameters from resolved config."""
    n_months = config["simulation"]["months"]
    mvp = config["timeline"]["mvp_months"]
    params = {}

    for cls_name, cls_cfg in config.get("rig_classes", {}).items():
        mkt = cls_cfg["market"]
        frm = cls_cfg["timeline"]["first_rig_month"]
        prod_months = n_months - frm + 1
        prod_months = max(prod_months, 1)

        avg_rigs = mkt["total_rigs_added"] / prod_months
        expected_avg = max(mkt["total_rigs_added"] / 2, 1)
        churn = mkt["rigs_lost_per_year"] / 12 / expected_avg

        params[cls_name] = {
            "avg_rigs_per_month": avg_rigs,
            "churn_prob": churn,
            "first_rig_month": frm,
            "market": mkt,
            "revenue": cls_cfg["revenue"],
        }

    return params


def run_single_trial(config: dict, roles: list[dict], cost_items: list[dict],
                     rng: np.random.Generator, deterministic: bool = False) -> list[dict]:
    """Run a full monthly projection with multiple rig classes."""
    n_months = config["simulation"]["months"]
    class_params = _build_class_params(config)

    # Month 0 baseline
    empty_classes = {cn: {"rigs": 0, "new": 0, "lost": 0,
                          "service_revenue": 0, "installation_revenue": 0,
                          "total_revenue": 0, "active_days_per_rig": 0}
                     for cn in class_params}

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
    }

    results = [month0]
    prior = month0

    for month in range(1, n_months + 1):
        row = compute_month(
            config, roles, cost_items, month, prior,
            class_params, rng, deterministic,
        )
        results.append(row)
        prior = row

    return results