"""Check that the app's engine reproduces 03's cross-fitted numbers (results/numbers_of_record.json). Run from the repo root:
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
below = co[co.p_sg < eng.screen]
ig = below[below.SG == 0]
want = NOR["headroom_all_companies"]

got = {"companies": len(co), "below_screen": len(below), "below_screen_rated_sg": int(below.SG.sum()), "rated_ig_below_screen": len(ig)}
assert got == {k: want[k] for k in got}, (got, want)
assert round(float(np.median(ig.headroom_screen)), 4) == want["to_screen_rated_ig"]["median"]
assert round(float(np.median(ig.headroom_call)), 4) == want["to_psg_0_5_rated_ig"]["median"]
for a in ANCHORS.itertuples():
    assert int((ig.headroom_screen <= a.share_of_revenue).sum()) == want["anchor_crossing"][a.incident]["count"], a.incident

for a in ANCHORS.itertuples():
    after = np.empty(len(co), int)
    for f in range(len(eng.folds)):
        m = (co.fold == f).to_numpy()
        after[m] = eng.grade(co[m], eng.p_sg(incident_shock(co[m], a.share_of_revenue, eng.folds[f].ref), f))
    drop = float(np.mean(after > co.grade.to_numpy()))
    assert round(drop, 3) == NOR["cyber_migration"]["by_incident"][a.incident]["drop 1+ grade"], (a.incident, drop)

print(f"engine reproduces 03: {got}; median headroom to the screen {np.median(ig.headroom_screen):.2%}, "
      f"M&S-scale crossings {int((ig.headroom_screen <= 0.0217).sum())} of {len(ig)}; grade drops match for all six incidents")

# The attack types (app only): a paid, debt-funded data breach is 03's shock exactly; no attack ever lowers P(SG).
import pandas as pd  # noqa: E402
from engine import Attack  # noqa: E402

model_cols = eng.features + eng.flags
for name in co.index:
    f = int(co.loc[name, "fold"])
    before, after, fed, _, fold = eng.attacked(name, Attack(cost=0.0217))
    ref03 = incident_shock(before, 0.0217, eng.folds[f].ref)
    pd.testing.assert_frame_equal(fed[model_cols], ref03[model_cols], check_dtype=False)
    _, after0, fed0, _, _ = eng.attacked(name, Attack())
    pd.testing.assert_frame_equal(after0[model_cols], before[model_cols], check_dtype=False)   # size zero changes nothing

defaults = {"Operations shutdown": Attack(lost=30 / 365), "Data breach": Attack(cost=0.01, accrued=True),
            "Billing / payments disruption": Attack(cost=0.0077, delay=10), "Destructive malware / IT rebuild": Attack(lost=14 / 365, capex=0.0025)}
scales = np.linspace(0, 1, 41)
for kind, attack in defaults.items():
    for funding in (False, True):
        a = Attack(**{**attack.__dict__, "cash_funded": funding})
        top = a.max_scale()
        bad = [n for n in co.index if (np.diff(eng.attack_curve(n, a, scales * top)) < -1e-9).any()]
        assert not bad, (kind, funding, bad[:5])
print("attack types: a paid, debt-funded breach equals 03's shock for all 592 companies; size zero changes nothing;"
      " P(SG) never falls as any of the four attacks grows (debt- or cash-funded)")
