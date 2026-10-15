"""The saved model (models/shadow_rating.pkl): it matches 03's feature choice, reproduces 04's holdout numbers, keeps its input
contract, and never reads a bigger cyber loss as safer."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cyber_credit.shock import MAX_COST, fit_reference, incident_shock


def test_model_uses_the_features_03_chose(model, spec, selection):
    assert model.metadata["added_features"] == spec["added"] == selection["final"]["added"]
    assert model.metadata["removed_inputs"] == spec["removed"] == selection["final"]["removed"]
    assert model.schema["inputs"] == spec["numeric"] and model.schema["computed"] == spec["computed"]
    assert not set(spec["removed"]) & set(model.schema["features"] + model.schema["flags"])


def test_holdout_numbers_of_record(model, holdout, record):
    p = model.score(holdout).p_sg
    assert round(roc_auc_score(holdout.SG, p), 3) == record["headline"]["holdout_auc"]
    latest = holdout.sort_values("Date").groupby("Name").tail(1)
    assert round(roc_auc_score(latest.SG, model.score(latest).p_sg), 3) == record["holdout_latest_company"]["auc"]
    assert model.screen == pytest.approx(record["screen"]["threshold_calibrated_psg"], abs=5e-4)


def test_input_contract(model, holdout):
    rows = holdout.head(3)
    for broken in (rows.drop(columns=model.schema["inputs"][-1]), rows.assign(debtRatio=np.inf), rows.assign(roe_negative=2)):
        with pytest.raises(ValueError):
            model.score(broken)


def test_a_bigger_loss_never_lowers_p_sg(model, holdout, data):
    latest = holdout.sort_values("Date").groupby("Name").tail(1)
    train = data[~data.Name.isin(set(holdout.Name))]
    ref = fit_reference(train)
    p = np.column_stack([model.score(incident_shock(latest, c, ref)).p_sg for c in np.linspace(0, MAX_COST, 31)])
    assert (np.diff(p, axis=1) >= -1e-9).all()
