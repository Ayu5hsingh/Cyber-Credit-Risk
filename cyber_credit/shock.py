"""The cyber incident shock: an incident cost, as a share of revenue, charged to one year's ratios (04_modelling §11).

Used by 04_modelling.ipynb and app/engine.py, so the notebook's headroom numbers and the app's are computed by the same code.
"""
import numpy as np

TAX = 0.25        # flat tax rate on the cost
MAX_COST = 0.30   # largest cost searched, share of revenue


def assets_over_capital_employed(frame):
    """Total assets / capital employed per row, implied by ROCE / (EBIT margin x asset turnover)."""
    k = frame.returnOnCapitalEmployed / (frame.ebitPerRevenue * frame.assetTurnover)
    return k.where((k > 0.8) & (k < 5))


def fit_reference(frame):
    """Training-set constants the shock needs: TA/CE median, FCF/OCF floor, equity-multiplier median."""
    return {"tace": float(assets_over_capital_employed(frame).median()),
            "fcf_floor": float(frame.freeCashFlowOperatingCashFlowRatio.quantile(0.01)),
            "equity_multiplier": float(frame.companyEquityMultiplier.median())}


def incident_shock(frame, cost, ref, debt_funded=True, tax=TAX):
    """Charge an incident cost (share of revenue, scalar or one per row) to each firm's ratios, row by row."""
    d = frame.copy()
    cost = np.broadcast_to(np.asarray(cost, dtype=float), len(d))
    turnover = d.assetTurnover.clip(lower=0)
    after_tax = cost - tax * np.minimum(cost, d.pretaxProfitMargin.fillna(0).clip(lower=0))  # shield only on existing profit

    for col in ["pretaxProfitMargin", "operatingProfitMargin", "ebitPerRevenue"]:
        d[col] = d[col] - cost
    d["netProfitMargin"] -= after_tax
    d["operatingCashFlowSalesRatio"] -= after_tax
    d["returnOnAssets"] -= after_tax * turnover
    d["returnOnCapitalEmployed"] -= cost * turnover * assets_over_capital_employed(frame).fillna(ref["tace"])
    d["returnOnEquity"] -= after_tax * turnover * d.companyEquityMultiplier.fillna(ref["equity_multiplier"]).clip(lower=0)
    if debt_funded:
        d["debtRatio"] += after_tax * turnover   # new debt = after-tax cost

    old = frame.freeCashFlowOperatingCashFlowRatio   # FCF/OCF with capex fixed, floored once cash flow is gone
    capex = (1 - old) * frame.operatingCashFlowSalesRatio
    ocf = d.operatingCashFlowSalesRatio
    new = (1 - capex / ocf.where(ocf > 1e-6)).fillna(ref["fcf_floor"]).clip(lower=ref["fcf_floor"])
    d["freeCashFlowOperatingCashFlowRatio"] = np.where(old.isna(), np.nan, np.minimum(old, new.where(cost > 0, old)))

    d["roe_negative"] = ((d.roe_negative == 1) | (d.returnOnEquity < 0)).astype(int)
    d["ocf_negative"] = ((d.ocf_negative == 1) | (d.operatingCashFlowSalesRatio < 0)).astype(int)
    return d
