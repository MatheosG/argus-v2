"""
staffing.py — Load staffing CSV and compute monthly compensation.

Two types of roles:
  - Fixed: has headcount, from_month, to_month — charged on schedule
  - Scaling: has rigs_per_hire — headcount = ceil(rig_count / rigs_per_hire)
"""

import csv
import math
from pathlib import Path


def _s(v):
    """Safely strip a CSV value that might be None."""
    return v.strip() if v else ""


def load_staffing(filepath: str) -> list[dict]:
    """Load staffing CSV into a list of role dicts."""
    roles = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys: strip whitespace (handles BOM artifacts)
            row = {k.strip(): v for k, v in row.items() if k is not None}
            role = {
                "department": _s(row.get("department", "")),
                "role": _s(row.get("role", "")),
                "annual_salary": float(row.get("annual_salary", 0)),
                "headcount": float(row["headcount"]) if _s(row.get("headcount")) else 0,
                "from_month": int(row["from_month"]) if _s(row.get("from_month")) else None,
                "to_month": int(row["to_month"]) if _s(row.get("to_month")) else None,
                "rigs_per_hire": float(row["rigs_per_hire"]) if _s(row.get("rigs_per_hire")) else None,
            }
            role["is_scaling"] = role["rigs_per_hire"] is not None
            roles.append(role)
    return roles


def get_benefits_multiplier(month: int, benefits_base: float, quarterly_increase: float) -> float:
    """Compute benefits multiplier for a given month."""
    quarters_elapsed = (month - 1) // 3
    return benefits_base + quarters_elapsed * quarterly_increase


def compute_monthly_compensation(
    roles: list[dict],
    month: int,
    rig_count: int,
    benefits_multiplier: float,
) -> dict:
    """Compute compensation for a single month."""
    by_department = {}
    by_role = []
    total = 0.0
    total_headcount = 0

    for role in roles:
        if role["is_scaling"]:
            if rig_count > 0:
                headcount = math.ceil(rig_count / role["rigs_per_hire"])
            else:
                headcount = 0
        else:
            if role["from_month"] is not None and month < role["from_month"]:
                headcount = 0
            elif role["to_month"] is not None and month > role["to_month"]:
                headcount = 0
            else:
                headcount = role["headcount"]

        monthly_salary = role["annual_salary"] / 12
        monthly_cost = headcount * monthly_salary * benefits_multiplier

        total += monthly_cost
        total_headcount += headcount

        dept = role["department"]
        by_department[dept] = by_department.get(dept, 0) + monthly_cost

        by_role.append({
            "role": role["role"],
            "department": dept,
            "headcount": headcount,
            "monthly_cost": monthly_cost,
        })

    return {
        "total": total,
        "by_department": by_department,
        "by_role": by_role,
        "total_headcount": total_headcount,
    }