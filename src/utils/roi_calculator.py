# ============================================================================
# Adaptive Data Governance Framework
# src/utils/roi_calculator.py
# ============================================================================
# ROI Calculator for Data Governance Investment.
# Maps data quality improvements to business KPIs (CAC, CLV, RTO).
# ============================================================================

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


# ============================================================================
# DQ Dimension → Business Cost Mapping
# ============================================================================
# Each dimension has a cost-elasticity profile that maps score improvements
# to INR savings.  Values are calibrated for a mid-size Indian e-commerce
# platform (₹5-10 Cr monthly GMV, 100K–500K customers).
# ============================================================================

_DQ_COST_PROFILES: Dict[str, Dict] = {
    "completeness": {
        "description": "Missing fields → wrong deliveries, failed KYC, churn",
        "cost_per_pct_gap": 25_000,     # ₹25K per 1 % below 100 %
        "primary_kpi": "RTO / delivery failure",
        "secondary_kpi": "Customer churn",
    },
    "validity": {
        "description": "Invalid formats / out-of-range → processing errors",
        "cost_per_pct_gap": 18_000,
        "primary_kpi": "Payment failures",
        "secondary_kpi": "Refund processing cost",
    },
    "timeliness": {
        "description": "Stale data → wrong personalisation, stock-outs",
        "cost_per_pct_gap": 30_000,
        "primary_kpi": "Lost revenue (stock-outs)",
        "secondary_kpi": "Decreased conversion",
    },
    "uniqueness": {
        "description": "Duplicates → inflated metrics, double-shipping",
        "cost_per_pct_gap": 15_000,
        "primary_kpi": "Duplicate shipment cost",
        "secondary_kpi": "Reporting accuracy",
    },
    "consistency": {
        "description": "Cross-source discrepancies → reconciliation effort",
        "cost_per_pct_gap": 12_000,
        "primary_kpi": "Manual reconciliation hours",
        "secondary_kpi": "Audit risk",
    },
    "accuracy": {
        "description": "Incorrect values → wrong decisions, customer complaints",
        "cost_per_pct_gap": 22_000,
        "primary_kpi": "Customer complaints",
        "secondary_kpi": "Decision quality",
    },
}


class GovernanceROICalculator:
    """Calculate financial impact of data governance improvements.

    Supports two modes:

    1. **Macro ROI** — High-level before/after metrics (RTO, conversion,
       compliance, ops efficiency).
    2. **Per-dimension ROI** — Maps actual DQ dimension scores
       (completeness, validity, timeliness, …) to granular ₹ savings
       with sensitivity analysis.
    """

    def __init__(
        self,
        monthly_revenue: float,
        customer_base: int,
        avg_order_value: float,
        rto_rate_before: float,
        rto_rate_after: float,
        cac: float,
        operational_hours_saved: float,
        dq_scores_before: Optional[Dict[str, float]] = None,
        dq_scores_after: Optional[Dict[str, float]] = None,
    ):
        self.monthly_revenue = monthly_revenue
        self.customer_base = customer_base
        self.avg_order_value = avg_order_value
        self.rto_rate_before = rto_rate_before
        self.rto_rate_after = rto_rate_after
        self.cac = cac
        self.operational_hours_saved = operational_hours_saved

        # Per-dimension DQ scores (0-100 scale)
        self.dq_scores_before = dq_scores_before or {}
        self.dq_scores_after = dq_scores_after or {}

    # ------------------------------------------------------------------
    # Individual benefit calculators
    # ------------------------------------------------------------------

    def calculate_rto_savings(self) -> Dict[str, float]:
        """Cost savings from reduced Return-to-Origin rates."""
        monthly_orders = self.monthly_revenue / self.avg_order_value
        rto_before = monthly_orders * self.rto_rate_before
        rto_after = monthly_orders * self.rto_rate_after
        rto_reduction = rto_before - rto_after

        # RTO costs ~150 % of order value (logistics + refund)
        savings_per_month = rto_reduction * self.avg_order_value * 1.5
        return {
            "monthly_savings": savings_per_month,
            "annual_savings": savings_per_month * 12,
            "rto_reduction_count": rto_reduction,
        }

    def calculate_personalization_uplift(
        self,
        conversion_rate_before: float = 0.02,
        conversion_rate_after: float = 0.025,
    ) -> Dict[str, float]:
        """Revenue uplift from better data-driven personalisation."""
        monthly_visitors = self.customer_base * 3
        additional = monthly_visitors * (conversion_rate_after - conversion_rate_before)
        monthly_uplift = additional * self.avg_order_value
        return {
            "monthly_uplift": monthly_uplift,
            "annual_uplift": monthly_uplift * 12,
            "additional_conversions": additional,
        }

    def calculate_clv_impact(
        self,
        avg_purchase_frequency: float = 4.0,
        avg_customer_lifespan_years: float = 3.0,
        retention_improvement_pct: float = 0.05,
    ) -> Dict[str, float]:
        """Revenue impact from improved CLV via better data quality.

        Better data governance → fewer wrong deliveries → higher
        retention → increased Customer Lifetime Value.

        Parameters
        ----------
        avg_purchase_frequency : float
            Average purchases per customer per year.
        avg_customer_lifespan_years : float
            Average active lifespan of a customer.
        retention_improvement_pct : float
            Expected retention improvement from governance (default 5 %).
        """
        clv_before = (
            self.avg_order_value * avg_purchase_frequency
            * avg_customer_lifespan_years
        )
        clv_after = (
            self.avg_order_value * avg_purchase_frequency
            * (avg_customer_lifespan_years * (1 + retention_improvement_pct))
        )
        clv_uplift_per_customer = clv_after - clv_before
        total_clv_uplift = clv_uplift_per_customer * self.customer_base

        return {
            "clv_before": clv_before,
            "clv_after": clv_after,
            "clv_uplift_per_customer": clv_uplift_per_customer,
            "total_clv_uplift": total_clv_uplift,
            "annual_clv_impact": total_clv_uplift / avg_customer_lifespan_years,
        }

    def calculate_compliance_cost_avoidance(self) -> Dict[str, float]:
        """Estimated cost avoidance from DPDP Act 2023 compliance."""
        # Conservative: 1 % probability of ₹10 Cr penalty
        expected_penalty_avoided = 10_00_00_000 * 0.01  # ₹10 lakh
        # 5 % customer churn from a potential data breach
        churn_cost = self.customer_base * 0.05 * self.cac
        return {
            "penalty_avoidance": expected_penalty_avoided,
            "churn_cost_avoidance": churn_cost,
            "total_compliance_value": expected_penalty_avoided + churn_cost,
        }

    def calculate_operational_efficiency(
        self, hourly_rate: float = 2000,
    ) -> Dict[str, float]:
        """Cost savings from automated governance replacing manual work."""
        monthly = self.operational_hours_saved * hourly_rate
        return {
            "monthly_savings": monthly,
            "annual_savings": monthly * 12,
            "hours_saved": self.operational_hours_saved,
        }

    # ------------------------------------------------------------------
    # Per-dimension DQ → Cost mapping
    # ------------------------------------------------------------------

    def calculate_dq_dimension_roi(
        self,
        cost_profiles: Optional[Dict[str, Dict]] = None,
    ) -> List[Dict]:
        """Map DQ dimension score improvements to ₹ savings.

        For each dimension where we have before/after scores, compute:
        - Gap reduction (pct points gained)
        - Monthly / annual savings based on cost-per-pct-gap
        - Natural-language attribution sentence

        Parameters
        ----------
        cost_profiles : dict, optional
            Custom cost profiles.  Falls back to ``_DQ_COST_PROFILES``.

        Returns
        -------
        list[dict]  One entry per dimension with savings breakdown.
        """
        profiles = cost_profiles or _DQ_COST_PROFILES
        results: List[Dict] = []

        for dim, profile in profiles.items():
            before = self.dq_scores_before.get(dim)
            after = self.dq_scores_after.get(dim)
            if before is None or after is None:
                continue

            gap_before = 100.0 - before
            gap_after = 100.0 - after
            gap_reduction = gap_before - gap_after  # positive = improvement
            cost_per_pct = profile["cost_per_pct_gap"]

            monthly_saving = gap_reduction * cost_per_pct
            annual_saving = monthly_saving * 12

            attribution = (
                f"Improving {dim} from {before:.1f}% → {after:.1f}% "
                f"saves ₹{annual_saving:,.0f}/year "
                f"(primary impact: {profile['primary_kpi']})"
            )

            results.append({
                "dimension": dim,
                "score_before": before,
                "score_after": after,
                "gap_reduction_pct": round(gap_reduction, 2),
                "monthly_saving": round(monthly_saving, 2),
                "annual_saving": round(annual_saving, 2),
                "primary_kpi": profile["primary_kpi"],
                "secondary_kpi": profile["secondary_kpi"],
                "attribution": attribution,
            })

        return results

    def sensitivity_analysis(
        self,
        dimension: str,
        score_range: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Show how costs change as a DQ dimension score varies.

        Parameters
        ----------
        dimension : str
            One of completeness, validity, timeliness, uniqueness,
            consistency, accuracy.
        score_range : list[float]
            Scores to evaluate.  Defaults to [70, 75, 80, 85, 90, 95, 100].

        Returns
        -------
        pd.DataFrame  with columns: score, gap, monthly_cost, annual_cost
        """
        if score_range is None:
            score_range = [70, 75, 80, 85, 90, 95, 100]

        profile = _DQ_COST_PROFILES.get(dimension)
        if not profile:
            raise ValueError(f"Unknown dimension: {dimension}")

        rows = []
        for score in score_range:
            gap = 100.0 - score
            monthly = gap * profile["cost_per_pct_gap"]
            rows.append({
                "score": score,
                "gap_pct": gap,
                "monthly_cost": round(monthly, 2),
                "annual_cost": round(monthly * 12, 2),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Consolidated report
    # ------------------------------------------------------------------

    def generate_roi_report(self) -> pd.DataFrame:
        """Generate a comprehensive ROI report as a DataFrame."""
        rto = self.calculate_rto_savings()
        pers = self.calculate_personalization_uplift()
        comp = self.calculate_compliance_cost_avoidance()
        ops = self.calculate_operational_efficiency()
        clv = self.calculate_clv_impact()

        implementation_cost = 50_00_000   # ₹50 lakhs (one-time)
        annual_maintenance = 10_00_000    # ₹10 lakhs/year

        total_annual = (
            rto["annual_savings"]
            + pers["annual_uplift"]
            + comp["total_compliance_value"]
            + ops["annual_savings"]
            + clv["annual_clv_impact"]
        )

        # Add per-dimension DQ savings if available
        dim_results = self.calculate_dq_dimension_roi()
        dq_annual = sum(r["annual_saving"] for r in dim_results)
        total_annual += dq_annual

        three_year_benefit = total_annual * 3
        three_year_cost = implementation_cost + annual_maintenance * 3
        net_roi = ((three_year_benefit - three_year_cost) / three_year_cost) * 100
        payback_months = implementation_cost / (total_annual / 12)

        data = {
            "Metric": [
                "RTO Cost Savings (Annual)",
                "Personalisation Revenue Uplift (Annual)",
                "CLV Uplift Impact (Annual)",
                "Compliance Cost Avoidance (Annual)",
                "Operational Efficiency Savings (Annual)",
            ],
            "Value (₹)": [
                f"₹{rto['annual_savings']:,.0f}",
                f"₹{pers['annual_uplift']:,.0f}",
                f"₹{clv['annual_clv_impact']:,.0f}",
                f"₹{comp['total_compliance_value']:,.0f}",
                f"₹{ops['annual_savings']:,.0f}",
            ],
        }

        # Per-dimension rows
        for r in dim_results:
            data["Metric"].append(
                f"  DQ: {r['dimension'].title()} ({r['score_before']:.0f}%→{r['score_after']:.0f}%)"
            )
            data["Value (₹)"].append(f"₹{r['annual_saving']:,.0f}")

        if dim_results:
            data["Metric"].append("DQ Dimension Savings Sub-total")
            data["Value (₹)"].append(f"₹{dq_annual:,.0f}")

        data["Metric"].extend([
            "Total Annual Benefit",
            "",
            "Implementation Cost (One-time)",
            "Annual Maintenance Cost",
            "3-Year Total Cost",
            "",
            "3-Year Total Benefit",
            "3-Year Net ROI (%)",
            "Payback Period (Months)",
        ])
        data["Value (₹)"].extend([
            f"₹{total_annual:,.0f}",
            "",
            f"₹{implementation_cost:,.0f}",
            f"₹{annual_maintenance:,.0f}",
            f"₹{three_year_cost:,.0f}",
            "",
            f"₹{three_year_benefit:,.0f}",
            f"{net_roi:.1f}%",
            f"{payback_months:.1f} months",
        ])
        return pd.DataFrame(data)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    calc = GovernanceROICalculator(
        monthly_revenue=5_00_00_000,
        customer_base=100_000,
        avg_order_value=2500,
        rto_rate_before=0.15,
        rto_rate_after=0.08,
        cac=500,
        operational_hours_saved=200,
        dq_scores_before={
            "completeness": 82, "validity": 78, "timeliness": 70,
            "uniqueness": 88, "consistency": 75, "accuracy": 80,
        },
        dq_scores_after={
            "completeness": 95, "validity": 93, "timeliness": 91,
            "uniqueness": 97, "consistency": 92, "accuracy": 94,
        },
    )
    report = calc.generate_roi_report()
    print(report.to_string(index=False))

    print("\n\n--- Per-Dimension Attribution ---")
    for r in calc.calculate_dq_dimension_roi():
        print(f"  • {r['attribution']}")

    print("\n--- Sensitivity: Completeness ---")
    print(calc.sensitivity_analysis("completeness").to_string(index=False))
