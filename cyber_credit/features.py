"""Credit features built from the statement ratios: the candidates 03_features tests, and the step that computes them in the model.

The data carry ratios only, never amounts, but many ratios share a denominator (revenue, total assets, current liabilities), so
they combine into the measures a credit analyst reads: interest coverage, the price lenders charge, cash flow against what is owed,
the stability of profit. Each candidate has a credit reason, an expected direction and a shock rule:

- "recompute": computed inside the saved model from the ratios (CreditFeatures, the first step of the pipeline), so a cyber shock
  that moves those ratios moves the feature too, always towards more risk;
- "hold": a precomputed input column (input_features), held at today's value by the shock, because the shock would move it the
  wrong way (the implied interest rate falls when debt rises and interest is unchanged) or it describes history, not this year.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Ratios any shock moves (04 §11 and the app's attack types), plus liquidity: protected, never dropped by feature selection
PROTECTED = ["pretaxProfitMargin", "operatingProfitMargin", "ebitPerRevenue", "netProfitMargin", "operatingCashFlowSalesRatio",
             "returnOnAssets", "returnOnCapitalEmployed", "returnOnEquity", "debtRatio", "freeCashFlowOperatingCashFlowRatio",
             "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding", "assetTurnover", "fixedAssetTurnover"]
PROTECTED_FLAGS = ["roe_negative", "ocf_negative", "negative_equity"]   # moved by the shock / used by the grade rule
REMOVABLE = {   # the other inputs, tested for removal in groups (a ratio with the flag that marks its placeholder values)
    "gross margin": ["grossProfitMargin", "cogs_not_reported"],
    "payables turnover": ["payablesTurnover", "payables_zero"],
    "DSO-zero flag": ["dso_zero"],
    "debt / equity": ["debtEquityRatio"],
    "equity multiplier": ["companyEquityMultiplier"],
    "effective tax rate": ["effectiveTaxRate"],
}


@dataclass(frozen=True)
class Candidate:
    label: str        # readable name, as on the slides
    family: str       # what credit question it answers
    formula: str      # how it is built from the ratios
    why: str          # the credit reason
    direction: int    # +1 more = riskier, -1 more = safer, 0 no fixed direction (credit logic, checked on data in 03)
    shock: str        # "recompute" (inside the model) or "hold" (input column)


CANDIDATES = {
    # leverage and coverage: can the company pay?
    "coverage": Candidate("Interest coverage", "leverage and coverage", "operating margin / (operating margin - pre-tax margin)",
                          "how many times operating profit pays the interest (Moody's EBITA / interest)", -1, "recompute"),
    "implied_rate": Candidate("Implied interest rate", "leverage and coverage",
                              "(operating margin - pre-tax margin) x asset turnover / debt ratio",
                              "interest per unit of liabilities: what lenders charge shows what they think", 1, "hold"),
    "ocf_liab": Candidate("Operating cash flow / liabilities", "leverage and coverage", "OCF/sales x asset turnover / debt ratio",
                          "cash from operations against what is owed (Moody's CFO / debt)", -1, "hold"),
    "fcf_liab": Candidate("Free cash flow / liabilities", "leverage and coverage", "FCF/OCF x OCF/sales x asset turnover / debt ratio",
                          "cash left after investment against what is owed (Moody's FCF / debt)", -1, "hold"),
    "lt_liab_ta": Candidate("Long-term liabilities / assets", "leverage and coverage",
                            "debt ratio - current liabilities / assets (from ROCE)", "the closest the data get to long-term debt", 1, "hold"),
    "equity_liab": Candidate("Equity / liabilities", "leverage and coverage", "(1 - debt ratio) / debt ratio",
                             "the owners' cushion against what is owed", -1, "recompute"),
    # profitability and cash quality
    "ebit_ta": Candidate("Operating profit / assets", "profitability", "operating margin x asset turnover",
                         "operating return on assets (an Altman Z-score ingredient)", -1, "recompute"),
    "accruals": Candidate("Accruals / assets", "earnings quality", "(net margin - OCF/sales) x asset turnover",
                          "profit not backed by cash (Sloan)", 0, "hold"),
    "cash_conversion": Candidate("Cash conversion", "earnings quality", "OCF/sales / operating margin",
                                 "how much operating profit turns into cash", 0, "hold"),
    "opex_burden": Candidate("Overheads / revenue", "profitability", "gross margin - operating margin",
                             "weight of SG&A and R&D: fixed costs that stay when sales stop", 0, "hold"),
    # balance-sheet shape and liquidity
    "wc_ta": Candidate("Working capital / assets", "liquidity", "(current ratio - 1) x current liabilities / assets",
                       "short-term cushion (an Altman Z-score ingredient)", 0, "hold"),
    "cash_runway_days": Candidate("Cash runway (days)", "liquidity",
                                  "cash ratio x current liabilities / assets / asset turnover / (1 - operating margin) x 365",
                                  "days of costs cash covers with no sales: what a ransomware outage tests", -1, "hold"),
    "dpo_days": Candidate("Days payables outstanding", "liquidity", "365 x (1 - gross margin) / payables turnover",
                          "how long suppliers wait to be paid", 0, "hold"),
    "capex_rev": Candidate("Capex / revenue", "investment", "(1 - FCF/OCF) x OCF/sales", "how much the company reinvests", 0, "hold"),
    "ppe_ta": Candidate("Fixed assets / assets", "investment", "asset turnover / fixed asset turnover",
                        "asset intensity: plants and equipment against a light business", 0, "hold"),
    "intangible_other_ta": Candidate("Intangible and other assets / assets", "investment",
                                     "1 - current assets / assets - fixed assets / assets",
                                     "goodwill-heavy balance sheets, often after acquisitions", 0, "hold"),
    "z_partial": Candidate("Altman Z'' (without retained earnings)", "classic scores",
                           "6.56 x working capital / assets + 6.72 x operating profit / assets + 1.05 x equity / liabilities",
                           "the classic bankruptcy score, less the term these data cannot build", -1, "hold"),
    # stability and trend: the company's own history (rows dated up to this one only)
    "op_margin_vol": Candidate("Operating margin volatility", "stability", "sd of the company's operating margin over its statements so far",
                               "jumpy profit is riskier; ratings look through the cycle", 1, "hold"),
    "d_op_margin": Candidate("Change in operating margin", "stability", "operating margin minus its previous statement's",
                             "is profit getting better or worse?", 0, "hold"),
    "d_debt_ratio": Candidate("Change in debt ratio", "stability", "debt ratio minus its previous statement's",
                              "is the company borrowing more?", 0, "hold"),
    "d_coverage": Candidate("Change in interest coverage", "stability", "coverage minus its previous statement's",
                            "is the interest burden getting heavier?", 0, "hold"),
    # against the sector: medians learned on the training rows inside the model
    "op_margin_vs_sector": Candidate("Operating margin vs sector", "sector-relative", "operating margin - sector median",
                                     "good for its industry? (Moody's scorecards are sector-specific)", -1, "recompute"),
    "debt_ratio_vs_sector": Candidate("Debt ratio vs sector", "sector-relative", "debt ratio - sector median",
                                      "leveraged for its industry?", 1, "recompute"),
    "coverage_vs_sector": Candidate("Interest coverage vs sector", "sector-relative", "coverage - sector median",
                                    "interest burden for its industry?", -1, "recompute"),
    "ocf_sales_vs_sector": Candidate("Operating cash flow / sales vs sector", "sector-relative", "OCF/sales - sector median",
                                     "cash generation for its industry?", -1, "recompute"),
}
SECTOR_RELATIVE = {"op_margin_vs_sector": "operatingProfitMargin", "debt_ratio_vs_sector": "debtRatio",
                   "coverage_vs_sector": "coverage", "ocf_sales_vs_sector": "operatingCashFlowSalesRatio"}
HISTORY = ["op_margin_vol", "d_op_margin", "d_debt_ratio", "d_coverage"]


def _ratio(num, den, lo, hi, where=None):
    """num / den clipped to [lo, hi]; missing where den is not positive or `where` is False."""
    ok = (den > 0) if where is None else ((den > 0) & where)
    return (num / den.where(ok)).clip(lo, hi)


def current_liabilities_share(f):
    """Current liabilities / total assets, from ROCE = EBIT / (assets - current liabilities); missing where that is unstable."""
    k = f.returnOnCapitalEmployed / (f.ebitPerRevenue * f.assetTurnover)   # assets / capital employed
    return 1 - 1 / k.where((k > 1) & (k < 5))


def coverage(f):
    """Operating profit / interest, interest read as operating minus pre-tax margin: 0 when operating profit is not positive,
    50 (the cap) when there is no net interest burden (a gap of 0.1% of revenue or less)."""
    op, interest = f.operatingProfitMargin, f.operatingProfitMargin - f.pretaxProfitMargin
    out = pd.Series(np.where(op <= 0, 0.0, np.where(interest <= 0.001, 50.0, op / interest.where(interest > 0.001))), index=f.index)
    return out.clip(0, 50).where(op.notna() & f.pretaxProfitMargin.notna())


def row_feature(name, f):
    """One row-wise candidate, from the ratios of each row only (no statistics learned from data). Reads only the ratios it needs."""
    op, turn, dr, ocf = (f.get(c) for c in ["operatingProfitMargin", "assetTurnover", "debtRatio", "operatingCashFlowSalesRatio"])
    cl_ta = current_liabilities_share(f) if name in ("lt_liab_ta", "wc_ta", "cash_runway_days", "intangible_other_ta", "z_partial") else None
    if name == "coverage":
        return coverage(f)
    if name == "implied_rate":
        return ((op - f.pretaxProfitMargin) * turn / dr.where(dr > 0)).clip(-0.05, 0.3)
    if name == "ocf_liab":
        return (ocf * turn / dr.where(dr > 0)).clip(-1, 2)
    if name == "fcf_liab":
        return (f.freeCashFlowOperatingCashFlowRatio * ocf * turn / dr.where(dr > 0)).clip(-1, 1)
    if name == "lt_liab_ta":
        return (dr - cl_ta).clip(0, 1.5)
    if name == "equity_liab":
        return ((1 - dr) / dr.where(dr > 0)).clip(-1, 5)
    if name == "ebit_ta":
        return (op * turn.clip(lower=0)).clip(-0.5, 0.5)
    if name == "accruals":
        return ((f.netProfitMargin - ocf) * turn).clip(-0.5, 0.5)
    if name == "cash_conversion":
        return _ratio(ocf, op, -2, 5, where=op > 0.01)
    if name == "opex_burden":
        return (f.grossProfitMargin - op).clip(-0.2, 1)
    if name == "wc_ta":
        return ((f.currentRatio - 1) * cl_ta).clip(-0.5, 0.8)
    if name == "cash_runway_days":
        return (f.cashRatio * cl_ta / turn.where(turn > 0) / (1 - op).clip(lower=0.05) * 365).clip(0, 1000)
    if name == "dpo_days":
        return (365 * (1 - f.grossProfitMargin) / f.payablesTurnover.where(f.payablesTurnover > 0)).clip(0, 400)
    if name == "capex_rev":
        return ((1 - f.freeCashFlowOperatingCashFlowRatio) * ocf).clip(-0.2, 0.6)
    if name == "ppe_ta":
        return (turn / f.fixedAssetTurnover.where(f.fixedAssetTurnover > 0)).clip(0, 1.2)
    if name == "intangible_other_ta":
        return (1 - f.currentRatio * cl_ta - row_feature("ppe_ta", f)).clip(-0.2, 1)
    if name == "z_partial":
        return (6.56 * row_feature("wc_ta", f) + 6.72 * row_feature("ebit_ta", f) + 1.05 * row_feature("equity_liab", f))
    raise KeyError(name)


def history_features(f):
    """Features from each company's own statements dated up to this row, never later ones: operating-margin volatility (sd, at
    least two statements) and the change since the previous statement. Needs Name and Date; returned in the input's row order."""
    s = f.assign(_cov=coverage(f)).sort_values(["Name", "Date"], kind="stable")
    g = s.groupby("Name", sort=False)
    prev = g[["operatingProfitMargin", "debtRatio", "_cov"]].shift()
    out = pd.DataFrame({
        "op_margin_vol": g.operatingProfitMargin.transform(lambda x: x.expanding().std()).clip(0, 0.3),
        "d_op_margin": (s.operatingProfitMargin - prev.operatingProfitMargin).clip(-0.3, 0.3),
        "d_debt_ratio": (s.debtRatio - prev.debtRatio).clip(-0.3, 0.3),
        "d_coverage": (s._cov - prev._cov).clip(-30, 30)}, index=s.index)
    return out.loc[f.index]


def input_features(f, names):
    """The "hold" candidates among `names`, as input columns: row-wise ones from each row's ratios, history ones from the company's
    earlier statements. Learns nothing from the data, so it can run on all rows before any split."""
    names = [n for n in names if CANDIDATES[n].shock == "hold"]
    hist = history_features(f) if any(n in HISTORY for n in names) else None
    return pd.DataFrame({n: hist[n] if n in HISTORY else row_feature(n, f) for n in names}, index=f.index)


class CreditFeatures(BaseEstimator, TransformerMixin):
    """First step of the model: adds the "recompute" credit features to the frame, from its ratios, so that a shocked frame is
    scored with features that follow the shock. Sector medians (for the sector-relative features) are learned in fit(), on the
    training rows only; a sector not seen in training uses the overall median."""
    def __init__(self, names=()):
        self.names = names

    def _row(self, X):
        return {n: row_feature(n, X) for n in self.names if n not in SECTOR_RELATIVE}

    def fit(self, X, y=None):
        bases = {SECTOR_RELATIVE[n] for n in self.names if n in SECTOR_RELATIVE}
        frame = X.assign(coverage=coverage(X)) if "coverage" in bases else X
        self.medians_ = {b: (frame.groupby("Sector")[b].median().to_dict(), float(frame[b].median())) for b in bases}
        return self

    def transform(self, X):
        X = X.copy()
        for n, v in self._row(X).items():
            X[n] = v
        for n in self.names:
            if n in SECTOR_RELATIVE:
                b = SECTOR_RELATIVE[n]
                by_sector, overall = self.medians_[b]
                base = X[b] if b in X.columns else coverage(X)
                X[n] = base - X.Sector.map(by_sector).astype(float).fillna(overall)
        return X
