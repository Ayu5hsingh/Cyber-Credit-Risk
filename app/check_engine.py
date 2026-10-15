"""Check that the app's engine reproduces 04's cross-fitted numbers (results/numbers_of_record.json). Run from the repo root:
    .venv/bin/python app/check_engine.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import ANCHORS, ROOT, Engine, incident_shock  # noqa: E402

NOR = json.loads((ROOT / "results" / "numbers_of_record.json").read_text())
eng = Engine()
co = eng.companies
below = co[co.p_sg_cf < eng.screen]
ig = below[below.SG == 0]
want = NOR["headroom_all_companies"]

got = {"companies": len(co), "below_screen": len(below), "below_screen_rated_sg": int(below.SG.sum()), "rated_ig_below_screen": len(ig)}
assert got == {k: want[k] for k in got}, (got, want)
assert round(float(np.median(ig.headroom_screen_cf)), 4) == want["to_screen_rated_ig"]["median"]
assert round(float(np.median(ig.headroom_call_cf)), 4) == want["to_psg_0_5_rated_ig"]["median"]
for a in ANCHORS.itertuples():
    assert int((ig.headroom_screen_cf <= a.share_of_revenue).sum()) == want["anchor_crossing"][a.incident]["count"], a.incident

for a in ANCHORS.itertuples():
    after = np.empty(len(co), int)
    for f in sorted(co.fold_cf.unique()):
        m = (co.fold_cf == f).to_numpy()
        after[m] = eng.grade(co[m], eng.p_sg(incident_shock(co[m], a.share_of_revenue, eng.folds[f].ref), f))
    drop = float(np.mean(after > co.grade_cf.to_numpy()))
    assert round(drop, 3) == NOR["cyber_migration"]["by_incident"][a.incident]["drop 1+ grade"], (a.incident, drop)

print(f"engine reproduces 04: {got}; median headroom to the screen {np.median(ig.headroom_screen_cf):.2%}, "
      f"M&S-scale crossings {int((ig.headroom_screen_cf <= 0.0217).sum())} of {len(ig)}; grade drops match for all six incidents")

# The holdout companies, scored by the saved model as on the slides, reproduce 04's holdout check.
ho = co[co.holdout]
assert len(ho) == NOR["headroom_companies"] and np.allclose(ho.p_sg, eng.shadow.score(ho).p_sg)
ho_ig = ho[(ho.p_sg < eng.screen) & (ho.SG == 0)]
want_ho = NOR["headroom_to_screen_rated_ig"]
assert len(ho_ig) == want_ho["companies"] and round(float(np.median(ho_ig.headroom_screen)), 4) == want_ho["median"]
assert int((ho_ig.headroom_screen <= 0.0217).sum()) == want_ho["count_le_ms"]
print(f"holdout: {len(ho)} companies scored by the saved model; {len(ho_ig)} rated IG below the screen, median headroom "
      f"{np.median(ho_ig.headroom_screen):.2%}, {int((ho_ig.headroom_screen <= 0.0217).sum())} cross at M&S scale (as in 04)")

# The attack types (app only): a paid, debt-funded data breach is 04's shock exactly; no attack ever lowers P(SG).
import pandas as pd  # noqa: E402
from engine import Attack  # noqa: E402

model_cols = eng.inputs + eng.flags   # the input columns (computed features are made inside the model)
for name in co.index:
    f = int(co.loc[name, "fold"])
    before, after, fed, _, fold = eng.attacked(name, Attack(cost=0.0217))
    ref04 = incident_shock(before, 0.0217, eng.folds[f].ref)
    pd.testing.assert_frame_equal(fed[model_cols], ref04[model_cols], check_dtype=False)
    _, after0, fed0, _, _ = eng.attacked(name, Attack())
    pd.testing.assert_frame_equal(after0[model_cols], before[model_cols], check_dtype=False)   # size zero changes nothing

# Balance-sheet identity: for every attack and funding, equity falls by exactly the after-tax loss (assets change - liabilities change).
cases = [Attack(lost=30 / 365), Attack(cost=0.02), Attack(cost=0.02, accrued=True), Attack(cost=0.0077, delay=10),
         Attack(lost=14 / 365, capex=0.0025, recovery=0.3, tax=0.35)]
sane = co[(co.debtRatio < 1) & (co.assetTurnover > 0) & (co.assetTurnover <= 20)].index
worst = 0.0
for name in sane:
    for base in cases:
        for cash in (False, True):
            a = Attack(**{**base.__dict__, "cash_funded": cash})
            before, after, _, imp, _ = eng.attacked(name, a)
            assets = 1 / before.assetTurnover.iloc[0]
            equity_before = assets * (1 - before.debtRatio.iloc[0])
            equity_after = (assets + imp["new_assets"][0]) * (1 - after.debtRatio.iloc[0])
            worst = max(worst, abs(equity_before - imp["after_tax"][0] - equity_after))
assert worst < 1e-9, worst
print(f"balance sheet: equity falls by exactly the after-tax loss for {len(sane)} companies x 5 attacks x 2 fundings (worst gap {worst:.1e})")

defaults = {"Operations shutdown": Attack(lost=30 / 365), "Data breach, provisioned": Attack(cost=0.01, accrued=True),
            "Data breach, paid, 40% insured, 35% tax": Attack(cost=0.01, recovery=0.4, tax=0.35),
            "Billing / payments disruption": Attack(cost=0.0077, delay=10), "Destructive malware / IT rebuild": Attack(lost=14 / 365, capex=0.0025)}
scales = np.linspace(0, 1, 41)
for kind, attack in defaults.items():
    for funding in (False, True):
        a = Attack(**{**attack.__dict__, "cash_funded": funding})
        top = a.max_scale()
        bad = [n for n in co.index if (np.diff(eng.attack_curve(n, a, scales * top)) < -1e-9).any()]
        assert not bad, (kind, funding, bad[:5])
print("attack types: a paid, debt-funded breach (25% tax, no insurance) equals 04's shock for all 592 companies; size zero changes nothing;"
      " with the liquidity toggle off, P(SG) never falls as any attack grows (5 attack set-ups, debt- or cash-funded)")
