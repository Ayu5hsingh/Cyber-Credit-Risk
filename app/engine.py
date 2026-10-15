"""Engine for the cyber stress-test app: the cross-fitted models of 03 §11, the incident shock, headroom and SHAP.

Every company is scored by the one of five fold models that never saw it, exactly as in 03 ("Cyber headroom for every
company"). The fold models are refits of the saved pipeline (models/shadow_rating.pkl), so preprocessing, model size,
calibrator, screen and grade scale all come from that file. app/check_engine.py checks this against results/.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import cloudpickle
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data" / "processed", ROOT / "models"

TAX = 0.25        # flat tax rate on the cost (03 §11)
MAX_COST = 0.30   # largest cost searched, share of revenue (03 §11)
CALL = 0.5        # a model classification as SG

ANCHORS = pd.DataFrame([
    ("Norsk Hydro", 2019, 0.0040, "LockerGoga ransomware; plants switched to manual operation", "NOK 550-650m H1 2019 / NOK 149.8bn revenue, gross"),
    ("Maersk", 2017, 0.0081, "NotPetya malware wiped IT systems worldwide", "USD 200-300m / USD 30.9bn revenue, insurance not stated"),
    ("MGM Resorts", 2023, 0.0068, "Social engineering, then ransomware; hotel and casino systems down", "~USD 110m / USD 16.2bn revenue, gross"),
    ("UnitedHealth", 2024, 0.0077, "Ransomware at Change Healthcare; US claims processing disrupted", "USD 3.07bn (10-K) / USD 400.3bn revenue"),
    ("Merck", 2017, 0.0115, "NotPetya malware; manufacturing and sales disrupted", "USD 177m lost-sales profit + USD 285m net costs / USD 40.1bn revenue, partly net"),
    ("Marks & Spencer", 2025, 0.0217, "Ransomware via social engineering; online orders paused for weeks", "~GBP 300m operating profit / GBP 13.8bn revenue, estimated, gross of insurance"),
], columns=["incident", "year", "share_of_revenue", "attack", "basis"])

LABELS = {
    "currentRatio": "Current ratio", "quickRatio": "Quick ratio", "cashRatio": "Cash ratio",
    "daysOfSalesOutstanding": "Days sales outstanding", "netProfitMargin": "Net profit margin",
    "pretaxProfitMargin": "Pre-tax profit margin", "grossProfitMargin": "Gross profit margin",
    "operatingProfitMargin": "Operating profit margin", "returnOnAssets": "Return on assets",
    "returnOnCapitalEmployed": "Return on capital employed", "returnOnEquity": "Return on equity",
    "assetTurnover": "Asset turnover", "fixedAssetTurnover": "Fixed-asset turnover", "debtEquityRatio": "Debt / equity",
    "debtRatio": "Debt ratio", "effectiveTaxRate": "Effective tax rate",
    "freeCashFlowOperatingCashFlowRatio": "Free cash flow / operating cash flow", "companyEquityMultiplier": "Equity multiplier",
    "ebitPerRevenue": "EBIT margin", "operatingCashFlowSalesRatio": "Operating cash flow / sales", "payablesTurnover": "Payables turnover",
    "roe_negative": "Flag: loss-making", "negative_equity": "Flag: negative equity", "ocf_negative": "Flag: negative operating cash flow",
    "cogs_not_reported": "Flag: cost of goods not reported", "payables_zero": "Flag: payables turnover 0", "dso_zero": "Flag: DSO 0",
}


# --- the shock, as in 03 §11 (kept identical; check_engine.py reproduces the notebook's headroom and migration numbers) ---

def fit_reference(frame):
    """Training-set constants the shock needs: TA/CE median, FCF/OCF floor, equity-multiplier median."""
    return {"tace": float(assets_over_capital_employed(frame).median()),
            "fcf_floor": float(frame.freeCashFlowOperatingCashFlowRatio.quantile(0.01)),
            "equity_multiplier": float(frame.companyEquityMultiplier.median())}


def assets_over_capital_employed(frame):
    """Total assets / capital employed per row, implied by ROCE / (EBIT margin x asset turnover)."""
    k = frame.returnOnCapitalEmployed / (frame.ebitPerRevenue * frame.assetTurnover)
    return k.where((k > 0.8) & (k < 5))


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


# --- attack types: which ratios each attack moves, with the accounting kept consistent (app only, not in 03) ---

ATTACKS = {
    "Operations shutdown": "Ransomware stops plants, logistics or stores: sales are lost for some days while costs carry on.",
    "Data breach": "Customer data stolen: one-off response, legal and regulatory costs.",
    "Billing / payments disruption": "Invoicing and collections stop: cash comes in late, plus direct response costs.",
    "Destructive malware / IT rebuild": "Systems wiped (NotPetya-style): an outage, then IT replaced as new capital spending.",
}
CAPS = {"lost": 0.5, "cost": MAX_COST, "delay": 180.0, "capex": MAX_COST}   # largest size searched for the line, per channel


@dataclass(frozen=True)
class Attack:
    """One incident, in shares of a normal year's revenue: lost = share of the year's sales lost (days / 365), cost = one-off
    costs through the P&L, accrued = those costs provisioned and paid later, delay = extra days to collect receivables,
    capex = IT rebuild capitalised; how it is funded, the insurance recovery on the P&L hit, and the tax rate."""
    lost: float = 0.0
    cost: float = 0.0
    accrued: bool = False
    delay: float = 0.0
    capex: float = 0.0
    cash_funded: bool = False
    recovery: float = 0.0
    tax: float = TAX

    def scaled(self, m):
        """The same attack m times as large (every channel in proportion)."""
        return replace(self, lost=self.lost * m, cost=self.cost * m, delay=self.delay * m, capex=self.capex * m)

    def max_scale(self):
        """The largest multiple searched for the line: the first channel to reach its cap."""
        caps = [CAPS[k] / v for k, v in (("lost", self.lost), ("cost", self.cost), ("delay", self.delay), ("capex", self.capex)) if v > 0]
        return min(caps) if caps else 0.0


def current_liabilities(frame):
    """Current liabilities per unit of revenue, implied by receivables / (quick ratio - cash ratio); NaN where that is unstable."""
    receivables = frame.daysOfSalesOutstanding / 365
    gap = frame.quickRatio - frame.cashRatio
    return (receivables / gap.where(gap > 0.02)).where(frame.daysOfSalesOutstanding > 0).where(lambda x: x.between(0.01, 2))


def attack_shock(frame, attack, ref, gross_margin, liabilities):
    """Apply an attack to each firm's ratios. All amounts are per unit of a normal year's revenue; per-row arrays are allowed for
    every channel. With only `cost`, paid and debt-funded, it equals incident_shock (03 §11). Returns (shocked frame, impacts)."""
    d, n = frame.copy(), len(frame)
    arr = lambda x: np.broadcast_to(np.asarray(x, dtype=float), n)
    L, c, delay, capex = np.minimum(arr(attack.lost), 0.9), arr(attack.cost), arr(attack.delay), arr(attack.capex)
    kept, tax = 1 - attack.recovery, attack.tax

    turnover = frame.assetTurnover.clip(lower=0)
    assets = (1 / turnover.where(turnover > 0)).to_numpy()
    g = np.clip(np.asarray(gross_margin, dtype=float), 0.05, 1.0)   # profit lost per unit of sales lost (as 03's outage days)

    lost_profit, charged = L * g * kept, c * kept
    pnl = lost_profit + charged                                                        # pre-tax hit
    after_tax = pnl - tax * np.minimum(pnl, frame.pretaxProfitMargin.fillna(0).clip(lower=0).to_numpy())   # shield on existing profit
    paid_now = lost_profit + np.where(attack.accrued, 0.0, charged)
    cash_pnl = np.where(pnl > 0, after_tax * paid_now / np.where(pnl > 0, pnl, 1), 0.0)
    sales = 1 - L
    receivables_build = delay / 365 * sales
    ocf_hit = cash_pnl + receivables_build

    # margins and cash flow per unit of a normal year's revenue, as in 03: the lost profit is charged like a cost, so an
    # attack can never lift a margin; the lost sales themselves show in asset turnover below
    for col in ["pretaxProfitMargin", "operatingProfitMargin", "ebitPerRevenue"]:
        d[col] = frame[col] - pnl
    d["netProfitMargin"] = frame.netProfitMargin - after_tax
    d["operatingCashFlowSalesRatio"] = frame.operatingCashFlowSalesRatio - ocf_hit
    d["returnOnAssets"] = frame.returnOnAssets - after_tax * turnover
    d["returnOnCapitalEmployed"] = frame.returnOnCapitalEmployed - pnl * turnover * assets_over_capital_employed(frame).fillna(ref["tace"])
    d["returnOnEquity"] = frame.returnOnEquity - after_tax * turnover * frame.companyEquityMultiplier.fillna(ref["equity_multiplier"]).clip(lower=0)

    # funding: cash leaves first when cash-funded, the rest is new debt; a provision is a new liability
    outflow = ocf_hit + capex
    cash = (frame.cashRatio * liabilities).fillna(0).clip(lower=0).to_numpy()
    cash_used = np.minimum(outflow, cash) if attack.cash_funded else np.zeros(n)
    new_liabilities = outflow - cash_used + np.where(attack.accrued, charged, 0.0)
    new_assets = capex + receivables_build - cash_used
    debt_ratio = np.fmax((frame.debtRatio * assets + new_liabilities) / (assets + new_assets), frame.debtRatio)
    # with negative equity (debt ratio >= 1), new assets matched by new debt would lower the ratio: there, as in 03, only the
    # new liabilities count, so the ratio never falls as the attack grows
    d["debtRatio"] = np.where(frame.debtRatio < 1, debt_ratio, frame.debtRatio + new_liabilities * turnover)

    old = frame.freeCashFlowOperatingCashFlowRatio               # FCF/OCF with the old capex plus the rebuild, floored once cash flow is gone
    spend = (1 - old) * frame.operatingCashFlowSalesRatio + capex
    ocf = frame.operatingCashFlowSalesRatio - ocf_hit
    new = (1 - spend / ocf.where(ocf > 1e-6)).fillna(ref["fcf_floor"]).clip(lower=ref["fcf_floor"])
    hit = (pnl > 0) | (ocf_hit > 0) | (capex > 0)
    d["freeCashFlowOperatingCashFlowRatio"] = np.where(old.isna(), np.nan, np.minimum(old, new.where(hit, old)))

    # ratios the model is not monotone in: computed for display, fed to the model only on request
    moved = hit | (delay > 0)
    d["assetTurnover"] = np.where(moved & np.isfinite(assets), sales / (assets + new_assets), frame.assetTurnover)
    d["fixedAssetTurnover"] = np.where(moved & (frame.fixedAssetTurnover > 0), sales / (1 / frame.fixedAssetTurnover + capex), frame.fixedAssetTurnover)
    d["daysOfSalesOutstanding"] = frame.daysOfSalesOutstanding + delay
    cl = liabilities.to_numpy()
    cl_after = cl + np.where(attack.accrued, charged, 0.0)
    for col, extra in (("cashRatio", 0.0), ("quickRatio", receivables_build), ("currentRatio", receivables_build)):
        d[col] = np.where(moved, (frame[col] * cl - cash_used + extra) / cl_after, frame[col])

    d["roe_negative"] = ((frame.roe_negative == 1) | (d.returnOnEquity < 0)).astype(int)
    d["ocf_negative"] = ((frame.ocf_negative == 1) | (d.operatingCashFlowSalesRatio < 0)).astype(int)
    return d, {"pnl": pnl, "cash_out": outflow, "lost_profit": lost_profit}


# --- the cross-fitted models ---

@dataclass
class Fold:
    model: object            # fitted pipeline: preprocessing + monotone LightGBM
    ref: dict                # shock constants from this fold's fitting rows
    gross_margin: pd.Series  # median gross margin by sector on the fitting rows (outage days)
    liabilities: pd.Series   # median current liabilities per unit of revenue by sector on the fitting rows
    explainer: object


class Engine:
    def __init__(self):
        self.shadow = cloudpickle.loads((MODELS / "shadow_rating.pkl").read_bytes())
        schema = self.shadow.schema
        self.features, self.flags = schema["features"], schema["flags"]
        self.columns = self.features + self.flags + [f"Sector={s}" for s in self.shadow.sectors]
        self.screen = self.shadow.screen
        self.cuts, self.grade_names = self.shadow.grades["cuts"], self.shadow.grades["names"]
        mono = self.shadow.model.named_steps["model"].monotone_constraints
        self.unconstrained = [f for f, m in zip(self.columns, mono) if m == 0 and f in self.features]

        data = pd.read_csv(DATA / "firm_years_clean.csv", parse_dates=["Date"]).reset_index(drop=True)
        self.folds, fold_of = [], {}
        split = StratifiedGroupKFold(5, shuffle=True, random_state=11).split(data, data.SG, groups=data.Name)   # 03 §11
        for i, (fit_rows, score_rows) in enumerate(split):
            fit = data.iloc[fit_rows]
            model = clone(self.shadow.model).fit(fit, fit.SG.values)
            cl = current_liabilities(fit)
            self.folds.append(Fold(model, fit_reference(fit),
                                   fit.groupby("Sector").grossProfitMargin.median().reindex(self.shadow.sectors).fillna(fit.grossProfitMargin.median()),
                                   cl.groupby(fit.Sector).median().reindex(self.shadow.sectors).fillna(cl.median()),
                                   shap.TreeExplainer(model.named_steps["model"])))
            fold_of.update(dict.fromkeys(data.Name.iloc[score_rows], i))
        companies = data.sort_values("Date").groupby("Name").tail(1).copy()   # latest statement per company
        companies["fold"] = companies.Name.map(fold_of)
        companies["p_sg"] = np.nan
        companies["headroom_screen"] = np.nan
        companies["headroom_call"] = np.nan
        for i in range(len(self.folds)):
            rows = companies[companies.fold == i]
            companies.loc[rows.index, "p_sg"] = self.p_sg(rows, i)
            companies.loc[rows.index, "headroom_screen"] = self.headroom(rows, i, self.screen)
            companies.loc[rows.index, "headroom_call"] = self.headroom(rows, i, CALL)
        companies["grade"] = self.grade(companies, companies.p_sg)
        self.companies = companies.set_index("Name", drop=False)

    # scoring
    def raw(self, frame, fold):
        return self.folds[fold].model.predict_proba(frame)[:, 1]

    def p_sg(self, frame, fold):
        """Calibrated P(SG) from the fold model that never saw these companies."""
        return self.shadow.calibrator(self.raw(frame, fold))

    def grade(self, frame, p, rule=False):
        """Grade index from P(SG), 0 = IG-1 ... 4 = SG-2, as in 03's migration table; rule=True adds the saved model's floor
        (negative equity means SG-1 or worse), which an incident cannot change because the shock leaves equity's sign alone."""
        g = np.searchsorted(np.asarray(self.cuts), np.asarray(p, dtype=float), side="right")
        if rule:
            g = np.maximum(g, np.where(frame.negative_equity.to_numpy() == 1, self.grade_names.index("SG-1"), 0))
        return g

    def headroom(self, frame, fold, threshold, debt_funded=True, tax=TAX, tol=1e-6):
        """Smallest cost (share of revenue) at which P(SG) reaches the threshold: 0 if already there, inf if never (03 §11)."""
        ref = self.folds[fold].ref
        crosses = lambda f, c: self.p_sg(incident_shock(f, c, ref, debt_funded, tax), fold) >= threshold
        out = np.full(len(frame), np.nan)
        out[self.p_sg(frame, fold) >= threshold] = 0.0
        out[np.isnan(out) & ~crosses(frame, MAX_COST)] = np.inf
        todo = np.flatnonzero(np.isnan(out))
        lo, hi = np.zeros(len(todo)), np.full(len(todo), MAX_COST)
        while (hi - lo).max(initial=0) > tol:
            mid = (lo + hi) / 2
            crossed = crosses(frame.iloc[todo], mid)
            hi, lo = np.where(crossed, mid, hi), np.where(crossed, lo, mid)
        out[todo] = hi
        return out

    # one company
    def row(self, name):
        return self.companies.loc[[name]]

    # attacks on one company
    def attacked(self, name, attack, feed_all=False):
        """(before, after with every ratio moved, after as fed to the model, impacts, fold) for one company and one or more attacks.
        `attack` may hold per-row arrays; feed_all=False keeps the ratios the model is not monotone in at their values before."""
        r = self.row(name)
        fold = int(r.fold.iloc[0])
        f = self.folds[fold]
        n = max(np.size(attack.lost), np.size(attack.cost), np.size(attack.delay), np.size(attack.capex))
        before = r.loc[r.index.repeat(n)].reset_index(drop=True)
        gm = before.grossProfitMargin.fillna(before.Sector.map(f.gross_margin)).fillna(f.gross_margin.median())
        cl = current_liabilities(before).fillna(before.Sector.map(f.liabilities)).fillna(f.liabilities.median())
        after, impacts = attack_shock(before, attack, f.ref, gm, cl)
        fed = after.copy()
        if not feed_all:
            fed[self.unconstrained] = before[self.unconstrained]
        return before, after, fed, impacts, fold

    def gross_margin_used(self, name):
        """Profit lost per unit of sales lost, as attack_shock uses it (for turning days into a share of revenue)."""
        r = self.row(name)
        f = self.folds[int(r.fold.iloc[0])]
        gm = r.grossProfitMargin.fillna(r.Sector.map(f.gross_margin)).fillna(f.gross_margin.median()).to_numpy()
        return float(np.clip(gm, 0.05, 1.0)[0])

    def attack_scenario(self, name, attack, feed_all=False):
        """Before and after one attack: P(SG), grade, raw score, SHAP, ratios as moved and as the model sees them, impacts."""
        before, after, fed, impacts, fold = self.attacked(name, attack, feed_all)
        f = self.folds[fold]
        pre = f.model.named_steps["pre"]
        out = {"fold": fold, "base": float(np.ravel(f.explainer.expected_value)[-1]),
               "impacts": {k: float(v[0]) for k, v in impacts.items()}}
        for key, frame, shown in (("before", before, before), ("after", fed, after)):
            X = pre.transform(frame)
            sv = f.explainer.shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            p = float(self.p_sg(frame, fold)[0])
            out[key] = {"p_sg": p, "grade": int(self.grade(frame, [p])[0]), "raw": float(self.raw(frame, fold)[0]),
                        "shap": pd.Series(sv[0], index=self.columns), "seen": pd.Series(X[0], index=self.columns), "given": shown.iloc[0]}
        winsor = pre.named_transformers_["num"].named_steps["winsor"]
        out["clip_lo"], out["clip_hi"] = winsor.lo_.set_axis(self.features), winsor.hi_.set_axis(self.features)
        return out

    def attack_curve(self, name, attack, scales, feed_all=False):
        """P(SG) as the attack is scaled by each multiple in `scales`."""
        scales = np.asarray(scales, dtype=float)
        a = replace(attack, lost=attack.lost * scales, cost=attack.cost * scales, delay=attack.delay * scales, capex=attack.capex * scales)
        _, _, fed, _, fold = self.attacked(name, a, feed_all)
        return self.p_sg(fed, fold)

    def attack_line(self, name, attack, threshold, feed_all=False, grid=241, tol=1e-4):
        """The multiple of the attack at which P(SG) first reaches the threshold: 0 if already there, inf if not within the caps.
        Scans a grid first, so it also works when unconstrained ratios are fed and P(SG) need not rise steadily."""
        top = attack.max_scale()
        if top == 0:
            return np.inf
        scales = np.linspace(0, top, grid)
        p = self.attack_curve(name, attack, scales, feed_all)
        if p[0] >= threshold:
            return 0.0
        above = np.flatnonzero(p >= threshold)
        if not len(above):
            return np.inf
        lo, hi = scales[above[0] - 1], scales[above[0]]
        while hi - lo > tol * top:
            mid = (lo + hi) / 2
            if self.attack_curve(name, attack, [mid], feed_all)[0] >= threshold:
                hi = mid
            else:
                lo = mid
        return float(hi)
