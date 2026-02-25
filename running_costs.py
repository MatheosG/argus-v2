"""
running_costs.py — Load running costs CSV and compute monthly field costs.

Cost types:
  - onetime/per_rig: charged once per new rig added (COGS)
  - monthly/per_rig: charged every month per active rig
  - monthly/per_n_rigs: charged every month, scales per N rigs (e.g. 1 per 10 rigs)
  - monthly/fixed: flat monthly cost regardless of rigs
  - monthly/per_rig with scaling_param: depreciation (amount / scaling_param months)

Phase filtering:
  - 'mvp': only active during MVP phase
  - 'prod': only active during production phase
  - 'all' or empty: active in both phases
"""

import csv
import math


def load_running_costs(filepath: str) -> list[dict]:
    """Load running costs CSV into a list of cost item dicts."""
    items = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "cost_class": row["cost_class"].strip(),
                "item": row["item"].strip(),
                "amount": float(row["amount"]),
                "frequency": row["frequency"].strip(),
                "scaling": row["scaling"].strip(),
                "scaling_param": float(row["scaling_param"]) if row["scaling_param"].strip() else None,
                "phase": row["phase"].strip() if row["phase"].strip() else "all",
            }
            items.append(item)
    return items


def is_active_phase(item_phase: str, current_phase: str) -> bool:
    """Check if a cost item is active in the current phase."""
    if item_phase == "all":
        return True
    return item_phase == current_phase


def compute_monthly_running_costs(
    cost_items: list[dict],
    phase: str,
    rig_count: int,
    new_rigs: int,
) -> dict:
    """Compute running costs for a single month.

    Returns dict with:
        - total: total monthly running costs
        - by_class: {cost_class: amount}
        - by_item: [{item, cost_class, amount}]
        - total_cogs: one-time costs for new rigs
        - total_recurring: monthly recurring costs
        - total_depreciation: monthly depreciation
    """
    by_class = {}
    by_item = []
    total = 0.0
    total_cogs = 0.0
    total_recurring = 0.0
    total_depreciation = 0.0

    for item in cost_items:
        if not is_active_phase(item["phase"], phase):
            cost = 0.0
        elif item["frequency"] == "onetime" and item["scaling"] == "per_rig":
            # One-time cost per new rig (COGS)
            cost = new_rigs * item["amount"]
            total_cogs += cost

        elif item["frequency"] == "monthly" and item["scaling"] == "per_rig":
            if item["cost_class"] == "depreciation" and item["scaling_param"] is not None:
                # Depreciation: amount / useful_life_months, per rig
                monthly_dep = item["amount"] / item["scaling_param"]
                cost = rig_count * monthly_dep
                total_depreciation += cost
            else:
                # Regular monthly per-rig cost
                cost = rig_count * item["amount"]
                total_recurring += cost

        elif item["frequency"] == "monthly" and item["scaling"] == "per_n_rigs":
            # Scales per N rigs (e.g. 1 cloud instance per 10 rigs)
            n = item["scaling_param"]
            if rig_count > 0:
                units = math.ceil(rig_count / n)
            else:
                units = 0
            cost = units * item["amount"]
            total_recurring += cost

        elif item["frequency"] == "monthly" and item["scaling"] == "fixed":
            # Fixed monthly cost
            cost = item["amount"]
            total_recurring += cost

        else:
            cost = 0.0

        total += cost

        cls = item["cost_class"]
        by_class[cls] = by_class.get(cls, 0) + cost

        if cost > 0:
            by_item.append({
                "item": item["item"],
                "cost_class": cls,
                "amount": cost,
            })

    return {
        "total": total,
        "by_class": by_class,
        "by_item": by_item,
        "total_cogs": total_cogs,
        "total_recurring": total_recurring,
        "total_depreciation": total_depreciation,
    }