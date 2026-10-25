from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

PLATFORM_MIN_LOAN: float = 500.0
PLATFORM_MAX_LOAN: float = 40000.0
BENCHMARK_ANNUAL_APR: float = 0.115  # 11.5% representative LendingClub APR


def calculate_amortized_installment(
    principal: float,
    term_months: int,
    annual_apr: float = BENCHMARK_ANNUAL_APR,
) -> float:
    """
    Computes standard monthly amortized loan payment:
    A = P * [r(1+r)^n] / [(1+r)^n - 1]
    """
    if principal <= 0 or term_months <= 0:
        return 0.0

    monthly_rate = annual_apr / 12.0
    if monthly_rate <= 0:
        return round(principal / term_months, 2)

    factor = math.pow(1.0 + monthly_rate, term_months)
    installment = principal * (monthly_rate * factor) / (factor - 1.0)
    return round(float(installment), 2)


def evaluate_solvency_and_guardrails(
    loan_amnt: float,
    annual_inc: float,
    term: int,
    installment: float,
    dti: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool, List[str]]:
    """
    Computes custom financial integrity & solvency metrics and determines
    whether Tier-1 policy knockout rules are triggered.

    Returns:
        (solvency_dict, warnings_list, is_knockout, knockout_reasons)
    """
    warnings: List[Dict[str, Any]] = []
    knockout_reasons: List[str] = []

    # 1. Compute baseline amortized installment
    amortized_inst = calculate_amortized_installment(loan_amnt, term)

    # 2. Financial integrity checks
    total_scheduled_repayment = installment * term
    coverage_ratio = total_scheduled_repayment / loan_amnt if loan_amnt > 0 else 1.0

    # 3. Monthly cashflow analysis
    gross_monthly_income = annual_inc / 12.0 if annual_inc > 0 else 0.0
    existing_debt_payments = gross_monthly_income * (dti / 100.0) if dti > 0 else 0.0
    disposable_cashflow = gross_monthly_income - existing_debt_payments - installment
    debt_service_burden_pct = (installment * 12.0 / annual_inc * 100.0) if annual_inc > 0 else 100.0
    salary_multiplier = loan_amnt / annual_inc if annual_inc > 0 else 999.0

    # Flag 1: Impossible Amortization (Total payments < principal)
    if coverage_ratio < 0.98 and loan_amnt > 0:
        warnings.append({
            "code": "IMPOSSIBLE_AMORTIZATION",
            "field": "installment",
            "message": (
                f"Mathematically invalid repayment: Total scheduled payments (${total_scheduled_repayment:,.2f}) "
                f"do not cover the requested principal (${loan_amnt:,.2f}). "
                f"Fair market installment is ${amortized_inst:,.2f}/mo."
            ),
        })
        knockout_reasons.append("Scheduled payments do not cover borrowed principal.")

    # Flag 2: Platform Ceiling ($40,000 max for LendingClub)
    if loan_amnt > PLATFORM_MAX_LOAN:
        warnings.append({
            "code": "EXCEEDS_PLATFORM_CAP",
            "field": "loan_amnt",
            "message": (
                f"Requested loan amount (${loan_amnt:,.2f}) exceeds the platform's "
                f"${PLATFORM_MAX_LOAN:,.0f} maximum borrowing ceiling."
            ),
        })
        knockout_reasons.append(f"Exceeds institutional maximum loan limit (${PLATFORM_MAX_LOAN:,.0f}).")

    # Flag 3: Severe Debt Burden (>40% of gross income on this single loan)
    if debt_service_burden_pct > 40.0:
        warnings.append({
            "code": "SEVERE_DEBT_BURDEN",
            "field": "installment",
            "message": (
                f"Monthly payment consumes {debt_service_burden_pct:.1f}% of gross income "
                f"(underwriting ceiling is 40.0%)."
            ),
        })
        knockout_reasons.append(f"Excessive debt burden ({debt_service_burden_pct:.1f}% of income).")

    # Flag 4: Negative Disposable Cashflow
    if disposable_cashflow < 0 and gross_monthly_income > 0:
        warnings.append({
            "code": "NEGATIVE_DISPOSABLE_CASHFLOW",
            "field": "dti",
            "message": (
                f"Insolvency Warning: Borrower's monthly debt payments exceed gross monthly earnings "
                f"by ${abs(disposable_cashflow):,.2f}/mo."
            ),
        })
        knockout_reasons.append("Negative monthly disposable cash flow after debt payments.")

    # Flag 5: High Unsecured Leverage (Loan > 2.5x annual salary)
    if salary_multiplier > 2.5 and annual_inc > 0:
        warnings.append({
            "code": "HIGH_SALARY_MULTIPLIER",
            "field": "loan_amnt",
            "message": (
                f"High leverage: Unsecured personal loan request is {salary_multiplier:.1f}x "
                f"annual salary (norm is < 1.0x)."
            ),
        })

    is_knockout = len(knockout_reasons) > 0

    solvency_metrics = {
        "repayment_coverage_ratio": round(coverage_ratio, 3),
        "disposable_monthly_cashflow": round(disposable_cashflow, 2),
        "debt_service_burden_pct": round(debt_service_burden_pct, 2),
        "salary_multiplier": round(salary_multiplier, 2),
        "amortized_installment": amortized_inst,
        "policy_knockout": is_knockout,
        "policy_reasons": knockout_reasons,
    }

    return solvency_metrics, warnings, is_knockout, knockout_reasons
