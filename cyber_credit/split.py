"""The holdout: one fifth of the companies, set aside before any choice is made (04 §2). Used by 03_features and 04_modelling."""
from sklearn.model_selection import StratifiedGroupKFold


def holdout_fifth(frame):
    """One fifth of the companies, stratified on IG/SG: the first fold of a seeded company-grouped split."""
    _, rows = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(frame, frame.SG, groups=frame.Name))
    return set(frame.Name.iloc[rows])


def holdout_companies(data):
    """The holdout companies. It was drawn before Finance came back into scope, so Finance companies are split by the same rule on
    their own and added, which keeps the original 120 locked away."""
    finance = data.Sector == "Finance"
    return holdout_fifth(data[~finance].reset_index(drop=True)) | holdout_fifth(data[finance].reset_index(drop=True))
