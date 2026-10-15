"""Shared fixtures. Run from the repository root with the environment of the README (on macOS, LightGBM needs libomp):
    pytest tests
"""
import json
import sys
from pathlib import Path

import cloudpickle
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA, RESULTS = ROOT / "data" / "processed", ROOT / "results"


@pytest.fixture(scope="session")
def schema():
    return json.loads((DATA / "schema.json").read_text())


@pytest.fixture(scope="session")
def spec():
    return json.loads((DATA / "features.json").read_text())


@pytest.fixture(scope="session")
def clean():
    return pd.read_csv(DATA / "firm_years_clean.csv", parse_dates=["Date"])


@pytest.fixture(scope="session")
def data():
    return pd.read_csv(DATA / "firm_years_features.csv", parse_dates=["Date"])


@pytest.fixture(scope="session")
def record():
    return json.loads((RESULTS / "numbers_of_record.json").read_text())


@pytest.fixture(scope="session")
def selection():
    return json.loads((RESULTS / "feature_selection.json").read_text())


@pytest.fixture(scope="session")
def model():
    return cloudpickle.loads((ROOT / "models" / "shadow_rating.pkl").read_bytes())


@pytest.fixture(scope="session")
def holdout(data):
    names = set(pd.read_csv(RESULTS / "holdout_companies.csv").Name)
    return data[data.Name.isin(names)].reset_index(drop=True)
