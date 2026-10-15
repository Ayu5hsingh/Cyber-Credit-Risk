"""The incident shock (cyber_credit/shock.py): accounting properties that hold for every company, whatever the model."""
import numpy as np
import pandas as pd
import pytest

from cyber_credit.shock import TAX, fit_reference, incident_shock


@pytest.fixture(scope="module")
def latest(data):
    return data.sort_values("Date").groupby("Name").tail(1).reset_index(drop=True)


@pytest.fixture(scope="module")
def ref(data):
    return fit_reference(data)


def test_zero_cost_changes_nothing(latest, ref):
    pd.testing.assert_frame_equal(incident_shock(latest, 0.0, ref), latest, check_dtype=False)


def test_bigger_cost_never_helps(latest, ref):
    """Margins, returns and cash flow only fall, the debt ratio only rises, the loss flags only switch on."""
    steps = [incident_shock(latest, c, ref) for c in np.linspace(0, 0.3, 7)]
    for a, b in zip(steps, steps[1:]):
        for col in ["pretaxProfitMargin", "operatingProfitMargin", "ebitPerRevenue", "netProfitMargin", "operatingCashFlowSalesRatio",
                    "returnOnAssets", "returnOnCapitalEmployed", "returnOnEquity", "freeCashFlowOperatingCashFlowRatio"]:
            assert ((b[col] <= a[col] + 1e-12) | a[col].isna()).all(), col
        assert ((b.debtRatio >= a.debtRatio - 1e-12) | a.debtRatio.isna()).all()
        for flag in ["roe_negative", "ocf_negative"]:
            assert (b[flag] >= a[flag]).all()


def test_held_features_stay_at_todays_value(latest, ref, spec):
    held = spec["added"]
    shocked = incident_shock(latest, 0.05, ref)
    pd.testing.assert_frame_equal(shocked[held], latest[held])


def test_tax_shield_only_on_existing_profit(latest, ref):
    rows = latest.head(3).assign(pretaxProfitMargin=[0.20, 0.02, -0.05])
    drop = rows.netProfitMargin - incident_shock(rows, 0.10, ref).netProfitMargin
    assert np.allclose(drop, [0.10 * (1 - TAX), 0.10 - TAX * 0.02, 0.10])
