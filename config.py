# ============================================================
# ARGUS v2 — Configuration
# ============================================================

simulation:
  n_trials: 5000
  months: 120        # 10-year projection
  seed: 42

# --- File References ---
files:
  staffing: "inputs/staffing.csv"
  running_costs: "inputs/running_costs.csv"

# --- Compensation Rules ---
compensation:
  benefits_base: 1.05
  benefits_quarterly_increase: 0.02

# --- Global Timeline ---
timeline:
  mvp_months: 6

# --- Economics / TEA ---
economics:
  # Annual discount rate for NPV / DCF calculations
  discount_rate:
    distribution: triangular
    params: { low: 0.08, mode: 0.10, high: 0.15 }

# --- Life Cycle Analysis (LCA) ---
# Environmental impact via NPT reduction → diesel savings → CO2 avoided
lca:
  diesel_gal_per_day_onshore: 2000     # World Oil / Canrig (2023)
  diesel_gal_per_day_offshore: 8450    # IPIECA Drilling Rigs (2023) — anchored semi avg
  co2_kg_per_gal: 10.18               # EPA GHG Emission Factors Hub (IPCC 2006)
  npt_baseline_low: 0.20              # GA Drilling / SPE/IADC literature
  npt_baseline_high: 0.30             # upper range of industry NPT
  argus_npt_reduction: 0.03           # conservative 3% of total rig time
  car_co2_per_year_mt: 4.6            # EPA avg passenger vehicle (2022)

# ============================================================
# Rig Classes — each has its own market params, timeline, revenue
# ============================================================
rig_classes:

  onshore:
    timeline:
      first_rig_month: 7
    revenue:
      days_per_month: 30
      installation_fee: 20000
    market:
      total_rigs_added:
        distribution: triangular
        params: { low: 5, mode: 8, high: 10 }
      rigs_lost_per_year:
        distribution: triangular
        params: { low: 0, mode: 1, high: 3 }
      daily_rate:
        distribution: triangular
        params: { low: 150, mode: 200, high: 280 }
      utilization_rate:
        distribution: triangular
        params: { low: 0.27, mode: 0.44, high: 0.67 }

  offshore:
    timeline:
      first_rig_month: 18
    revenue:
      days_per_month: 30
      installation_fee: 10000
    market:
      total_rigs_added:
        distribution: triangular
        params: { low: 51, mode: 220, high: 360 }
      rigs_lost_per_year:
        distribution: triangular
        params: { low: 0, mode: 1, high: 3 }
      daily_rate:
        distribution: triangular
        params: { low: 300, mode: 500, high: 800 }
      utilization_rate:
        distribution: triangular
        params: { low: 0.27, mode: 0.44, high: 0.67 }
