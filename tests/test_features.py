"""The credit features (cyber_credit/features.py): formulas, edge cases, and that they learn nothing they should not."""
import numpy as np
import pandas as pd
import pytest

from cyber_credit.features import CANDIDATES, PROTECTED, PROTECTED_FLAGS, REMOVABLE, CreditFeatures, coverage, history_features, \
    input_features, row_feature


def frame(**cols):
    return pd.DataFrame(cols)


def test_coverage_edge_cases():
    f = frame(operatingProfitMargin=[0.20, 0.20, -0.05, 0.10, 0.10, np.nan],
              pretaxProfitMargin=[0.15, 0.2005, -0.10, 0.12, 0.099, 0.05])
    # 4x; no net interest (gap <= 0.1% of revenue) -> cap 50; loss -> 0; pre-tax above operating -> cap 50; gap 0.001 -> cap; missing
    assert np.allclose(coverage(f), [4.0, 50.0, 0.0, 50.0, 50.0, np.nan], equal_nan=True)


def test_implied_rate_is_interest_over_liabilities():
    # revenue 100, assets 200 (turnover 0.5), liabilities 120 (debt ratio 0.6), interest 3 (operating 10%, pre-tax 7%)
    f = frame(operatingProfitMargin=[0.10], pretaxProfitMargin=[0.07], assetTurnover=[0.5], debtRatio=[0.6])
    assert row_feature("implied_rate", f).iloc[0] == pytest.approx(3 / 120)


def test_history_uses_only_earlier_statements():
    f = frame(Name=["A"] * 4 + ["B"] * 2,
              Date=pd.to_datetime(["2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01", "2013-01-01", "2014-01-01"]),
              operatingProfitMargin=[0.10, 0.20, 0.30, 0.90, 0.05, 0.05], pretaxProfitMargin=[0.05] * 6, debtRatio=[0.5] * 6)
    h = history_features(f)
    assert np.isnan(h.op_margin_vol.iloc[0])                                    # one statement: no volatility
    assert h.op_margin_vol.iloc[2] == pytest.approx(np.std([0.10, 0.20, 0.30], ddof=1))
    later = f.copy(); later.loc[3, "operatingProfitMargin"] = -5.0            # change only the latest statement of A
    assert np.allclose(history_features(later).op_margin_vol.iloc[:3], h.op_margin_vol.iloc[:3], equal_nan=True)
    assert h.d_op_margin.iloc[1] == pytest.approx(0.10) and np.isnan(h.d_op_margin.iloc[4])
    shuffled = f.sample(frac=1, random_state=0)                                 # row order does not matter
    pd.testing.assert_frame_equal(history_features(shuffled).loc[f.index], h)


def test_input_features_learn_nothing_from_other_rows(clean):
    names = [n for n, c in CANDIDATES.items() if c.shock == "hold"]
    full = input_features(clean, names)
    keep = clean.Name.isin(clean.Name.drop_duplicates().sample(100, random_state=0))
    pd.testing.assert_frame_equal(input_features(clean[keep], names), full[keep])


def test_sector_medians_come_from_fit_only():
    train = frame(Sector=["Energy", "Energy", "Technology"], operatingProfitMargin=[0.1, 0.3, 0.5])
    test = frame(Sector=["Energy", "Unknown"], operatingProfitMargin=[0.2, 0.2])
    out = CreditFeatures(["op_margin_vs_sector"]).fit(train).transform(test)
    assert np.allclose(out.op_margin_vs_sector, [0.2 - 0.2, 0.2 - 0.3])   # Energy median 0.2; unseen sector -> overall median 0.3


def test_recompute_features_never_move_the_wrong_way_under_a_cost(clean):
    """A cost lowers operating and pre-tax margins by the same amount: every 'recompute' feature must move towards more risk."""
    names = [n for n, c in CANDIDATES.items() if c.shock == "recompute"]
    cf = CreditFeatures(names).fit(clean)
    before = cf.transform(clean)
    hit = clean.assign(operatingProfitMargin=clean.operatingProfitMargin - 0.03, pretaxProfitMargin=clean.pretaxProfitMargin - 0.03,
                       operatingCashFlowSalesRatio=clean.operatingCashFlowSalesRatio - 0.02, debtRatio=clean.debtRatio + 0.01)
    after = cf.transform(hit)
    for n in names:
        move = (after[n] - before[n]) * CANDIDATES[n].direction
        assert (move.dropna() >= -1e-12).all(), n


def test_protected_and_removable_cover_the_inputs(schema):
    removable = {c for g in REMOVABLE.values() for c in g}
    assert set(PROTECTED) | (removable & set(schema["features"])) == set(schema["features"])
    assert set(PROTECTED_FLAGS) | (removable & set(schema["flags"])) == set(schema["flags"])
    assert not set(PROTECTED) & removable
