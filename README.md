# Cyber headroom: how much cyber damage can a company's credit absorb?

Case study on cyber credit risk: the question is how a cyber incident affects a company's creditworthiness. Only a few incidents matter for credit, so it is a rare-event problem, like default prediction.

The data hold financial statements and ratings, but no incidents, so the incident-to-credit link cannot be learned from them. They do support the financial side of the question:
1. **The engine:** a candidate screening model that estimates, for any company with financial statements, the calibrated probability of a speculative-grade (SG) rating, P(SG). It works for rated companies and for unrated ones (suppliers, vendors, private firms), where no rating exists to look up.
2. **Cyber headroom:** a stress test of how large an incident, as a share of revenue, each company could absorb before it lands on the review list (or before the model would classify it as SG). It is a model-based scenario, not a downgrade forecast.
3. **The rare-event view:** IG → SG migration, judged with rare-event measures.

It is a triage screen for analysts, not a rating opinion and not a probability of default. It is validated on rated companies it never saw; whether it holds for unrated private firms still needs its own validation. The slide outline for the presentation is in [`docs/presentation_outline.md`](docs/presentation_outline.md).

**Recommended policy: the review budget.** Analysts review the riskiest 30% of companies. The review screen (calibrated P(SG) ≥ 0.30) gives cyber headroom a line to measure distance to.

## Results
All numbers come from `results/numbers_of_record.json` (model), `results/eda_numbers.json` (data) and `results/cleaning_numbers.json` (cleaning). Portfolio numbers count companies on their latest statement; rows are called firm-years.

**The cyber result: every company, each scored by a model that never saw it (cross-fitted)**

| | |
|---|---|
| Who is measured | 575 companies on their latest statement; 194 sit below the review screen, of which 162 are rated investment grade (IG) |
| Cyber headroom, rated IG | the median company absorbs **6.04%** of revenue [5.48%, 7.11%] before landing on the review list |
| At reported incident sizes | Norsk Hydro scale (0.40% of revenue): 5 of 162 land on the review list. Maersk, MGM, UnitedHealth scale (0.68%–0.81%): 12–15 of 162. Merck (1.15%): 18 of 162. **M&S (2.17%, estimated, gross of insurance): 31 of 162** (19%, exact 95% interval 13%–26%) |
| To a speculative-grade classification | at M&S scale, **3 of 162** would reach the model's classification at calibrated P(SG) ≥ 0.5 (exact 95% interval 0.4%–5.3%; not an agency decision); median headroom to it 10.17% |
| By sector (descriptive) | median headroom from 4.1% (Capital Goods, 26 companies) to 7.0% (Consumer Non-Durables, 15) across sectors with 10+ companies; the intervals overlap, so no sector is clearly more exposed. Table in `03_modelling.ipynb` §10 |
| Check on the 120 holdout companies | median 4.80% [3.27%, 6.69%]; 7 of 29 at M&S scale (10%–44%); consistent with the result above |

**The engine and the rare-event view**

| | |
|---|---|
| Data | 2,028 firm-years from 592 companies after cleaning (405 duplicate rows and the corrupt Company 590 row removed); 1,978 / 575 in scope (Finance excluded); 43% SG |
| Model | LightGBM with monotone constraints, 200 trees × 5 leaves: the smallest model within one standard error of the best grouped-CV AUC |
| IG vs SG on companies never seen | **Holdout** (120 companies, fixed before modelling; the independent holdout estimate): AUC 0.76 [0.68, 0.83], PR-AUC 0.75 (no skill 0.43). **Grouped CV** (5 folds × 3 seeds; development evidence that may contain model-selection optimism, because the model size was chosen on the same folds; its size is not measured): AUC 0.81 ± 0.03, PR-AUC 0.78 |
| Holdout, one row per company (latest statement, the unit the review works on) | AUC 0.79 [0.70, 0.86]; PR-AUC 0.77 (no skill 0.46) |
| Review budget (recommended) | reviewing the riskiest 30% of holdout companies catches 49% [42%, 62%] of SG companies at 75% precision (random: 30%) |
| Leakage a naive set-up would hide | random split with duplicates 0.87 → without duplicates 0.86 → grouped by company 0.81 |
| Overfitting | AUC on training companies 0.91, CV 0.81, holdout 0.76 (the earlier 400 × 15 model: 1.00 / 0.81 / 0.75) |
| Hardest boundary, BBB vs BB | AUC 0.69 (CV), 0.67 (holdout) |
| Calibration (holdout) | approximate. Firm-years: slope 0.88 [0.60, 1.18], intercept −0.15 [−0.57, +0.22], Brier 0.19; the largest miss is the middle fifth (0.40 predicted vs 0.27 actual). One row per company: slope 0.96 [0.62, 1.45], intercept −0.24 [−0.71, +0.19]; the largest miss is the second fifth (0.30 predicted vs 0.17 actual). Both intercept intervals include zero, so there is no clear evidence of over- or under-prediction overall |
| By sector (descriptive) | CV AUC from 0.69 (Public Utilities, 49 training companies) to 0.92 (Health Care, 43) across the 7 sectors with at least 30 training companies. Outside that range: Miscellaneous 0.67 (14 companies). CV uses firm-years and has no intervals; every holdout sector result is exploratory (5–21 companies each). Table in `03_modelling.ipynb` §6 |
| Stability (20 bootstrap refits; calibrator and screen threshold held fixed) | sensitivity to which companies were in training, given those choices, not the uncertainty of the whole pipeline. Per-company spread of P(SG): median sd 0.078, 90th percentile 0.118. 72% of holdout companies keep the same screen decision in at least 90% of refits; of the 33 less stable ones, 29 are within 0.10 of the screen (median distance 0.042, against 0.338 for the stable ones) |
| Company-weighted training (challenger) | +0.001 AUC vs the champion (sd of the difference 0.010): no gain beyond noise, not adopted |
| Out of time (exploratory) | trained on ratings dated up to 2014 (mostly 2012–14). Energy companies the model never saw, 2015–16: AUC 0.71 (19 companies; 2015: 0.60 on 10, 2016: 0.79 on 16). All sectors, new companies: 0.82. A nowcast: 2016 statements already show the crash |
| Rare-event view: IG → SG migration (exploratory) | a rare event: 27 of 438 IG pairs crossed to SG at the next rating (base rate 6.2%). Point-in-time scores (trained only on years before the scored year): PR-AUC 0.12 [0.07, 0.22], about 1.9× the base rate; reviewing the riskiest 20% of IG statements catches 10 of 27 movers (1.8× lift); AUC 0.64 [0.52, 0.75]. A modest signal, up to about a year ahead; not shown within BBB alone (AUC 0.60 [0.47, 0.73], the interval reaches 0.5) |
| Cyber headroom on the holdout only (the check) | 37 of 120 holdout companies are below the screen (29 rated IG, 8 rated SG). For the 29 rated IG, the median absorbs 4.80% of revenue [3.27%, 6.69%]; 7 of 29 (exact 95% interval 10%–44%) would land on the review list at M&S scale; 0 of 29 (0%–12%) would reach the model's classification at P(SG) ≥ 0.5 |
| Headroom sensitivity (holdout) | one fixed population (the 25 rated-IG holdout companies below the lowest screen tested), one assumption changed per row: base 5.33%; tax 15–35%: 4.70%–5.94%; not debt-funded: 5.93%; screen −0.03 / +0.03: 4.18% / 6.08%. 4–7 of 25 companies cross at M&S scale. Where the screen sits moves the answer most |

Intervals in brackets are 95% company-bootstrap intervals. Crossing counts use exact (Clopper–Pearson) binomial intervals: one row per company removes repeated-company dependence (independence across companies is an assumption), and the interval stays informative when none cross. Quote headroom as counts ("31 of 162"); read the anchors as tail scenarios: they are large, widely reported incidents. The cross-fitted result reuses the calibrator and screen fitted on the training companies.

## Limitations
- **Unrated firms.** Validated on rated, listed companies (2012–16). Applicability to unrated private firms, and calibration for them, need their own validation.
- **Statement timing is unknown.** Each row carries the rating date only, so look-ahead bias cannot be fully ruled out.
- **The time-based tests are historical diagnostics.** Each model is refitted on earlier years, but the inputs, constraints and model size were chosen with all years in view.
- **Liquidity.** Higher liquidity ratios read as riskier. Market access is a possible explanation; the data cannot identify the mechanism.
- **Calibration is approximate:** fitted on fold models trained on 80% of the companies, with companies weighted by their number of firm-years. Read P(SG) as a ranking with an approximate level.
- **Cyber headroom is a scenario.** Its intervals leave out uncertainty in the model, the accounting assumptions and how an incident's cost lands. The debt-funded shock raises the debt ratio but holds debt/equity and the equity multiplier fixed (to keep the score monotone), so it understates the leverage effect. Liquidity and sector are not shocked.
- **What the data lack:** firm size, default outcomes (the labels are agency opinions) and any cyber data.

## Appendix: secondary outputs
Kept in the notebook appendix, not in the main story. **Why the main product is IG/SG and not the full rating** (what was tested, how it did, limitations, what would improve it): [`docs/why_ig_sg_not_rating.md`](docs/why_ig_sg_not_rating.md).

| | |
|---|---|
| Review screen (calibrated P(SG) ≥ 0.30; set to catch 85% of SG training companies on their latest statement, out of fold) | flags 69% [61%, 78%] of holdout companies and catches 85% [76%, 94%] of SG ones |
| Full-rating attempt: grade band (6 bands, "AA and above" … "CCC and below") | exact grade 42% (CV) and 34% (holdout), against 33% for always answering BBB and 57% for two agencies on the same company; within one grade 90% (CV) and 85% (holdout), against 76% for always BBB (holdout). It predicts BBB or BB for 77% of holdout firm-years |
| Tiers (holdout companies) | SG grade 62 companies (65% actually SG); IG grade with a screen alert 22 (32%); IG grade, no alert 36 (22%) |
| Cost rule (review if P(SG) ≥ 1/k) | k = 2 flags 49% and catches 76% of SG companies; k = 4 flags 78% and catches 89% |

## Validation and monitoring roadmap (future work)
**Backtesting before deployment (walk-forward).** Train and select the model on data up to a cutoff year; calibrate and set the policy on earlier validation data; freeze the whole pipeline; score the next period using only information available then; compare with the labels that arrive later; move the cutoff forward and repeat. Report existing companies and new companies separately. The current data (mostly 2012–16, no publication dates) support only a limited prototype.

**Monitoring after deployment.**

| Layer | Measures | Response |
|---|---|---|
| Data quality | missing inputs, impossible ratios (the 02 rules), duplicates, statement age | repair the feed or withhold affected scores |
| Population drift | sector mix, ratio distributions, missingness | find which segment changed; check it is still in scope |
| Prediction drift | P(SG) distribution, share in the screen, review-list size | real credit change or a model/data problem? |
| Performance, when labels arrive | AUC, PR-AUC, catch rate at the review budget, by sector | compare with reference performance and its uncertainty |
| Calibration | calibration slope and intercept, Brier, predicted vs observed SG share | recalibrate if the ranking holds but the level drifts |
| Business effectiveness | review volume, analyst overrides, findings, missed risky companies | revisit the review policy |

- **Drift triggers investigation, not retraining.** In 2015–16 Energy inputs and scores shifted sharply, and the model tracked the shock: drift is not failure. Retraining follows sustained deterioration in ranking or catch rate, and a retrained challenger replaces the model only after it beats it on later-period data and on the relevant sectors.
- **Labels may never arrive for unrated firms.** Agree a validation source up front: later ratings where they appear, structured analyst assessments, or defaults as a separate check.
- **Audit a random sample of unflagged firms.** If analysts review only flagged firms, missed risk can never be measured.
- **Actions:** a broken feed is fixed; drift with stable performance is watched; a level shift is recalibrated; a broken ranking leads to a validated retrain; a new population needs new data and validation.

## How to read it
- **Grades, not notches.** The data carry letter grades only (AAA … D); a notch is one step on the modified rating scale (BBB− → BB+ is one notch), so we claim no notch precision.
- **A credit rating** is an opinion on relative credit risk: how likely a company is to default and how much lenders would lose if it did. The model estimates the probability of a speculative-grade rating, not a probability of default.
- **What it is for:** a transparent screen trained on rating labels, with a cyber stress layer. It complements a probability-of-default (PD) model rather than replacing it.
- **In deployment**, rank companies by P(SG) × criticality (spend, data access, single point of failure); the model provides the first factor.
- **Not benchmarked:** there is no outlook or watchlist data, so the early signal is not compared with the agencies' own early-warning tools.
- **Quoting:** quote numbers as computed, to two decimals; do not round ranges up.

## Using the saved model
`03_modelling.ipynb` saves the fitted pipeline, calibrator, screen, input schema and version metadata as `models/shadow_rating.pkl`. Unpickling runs code, so load only a copy you trust.

**Input: processed rows, not raw data.** `score()` expects the output of `02_data_cleaning.ipynb` (cleaned rows with the rebuilt ratios and flags, like `data/processed/firm_years_clean.csv`); cleaning is not part of the artifact. It rejects missing or non-numeric input columns, infinite values and flags other than 0/1, and warns when a sector was not seen in training (that row is scored with no sector effect).
```python
import cloudpickle, pandas as pd
model = cloudpickle.loads(open("models/shadow_rating.pkl", "rb").read())
firm_years = pd.read_csv("data/processed/firm_years_clean.csv", parse_dates=["Date"])   # processed, not raw
firm_years = firm_years[firm_years.Sector != "Finance"]                                   # Finance is out of scope
scores = model.score(firm_years)
scores.p_sg, scores.screen             # calibrated P(SG) and the screen flag
model.metadata                         # model size, screen, training data hash, package versions
```

## Layout
There are no `.py` files: all code lives in the three notebooks, and assert cells check every cleaning rule, count and stress-test property.
```
data/                 raw CSV as delivered
data/processed/       firm_years_clean.csv (2,028 firm-years), firm_years_naive.csv (duplicates kept, for the leakage baseline),
                      schema.json (the input lists, defined once); written by 02
notebooks/
  01_eda.ipynb            the raw data and its problems (reads the raw CSV only)
  02_data_cleaning.ipynb  row-wise cleaning and features; writes data/processed/
  03_modelling.ipynb      validation, model size, calibration, review budget, sectors, stability, SHAP, out of time,
                          migration as a rare event, cyber headroom (holdout check, sensitivity, every company cross-fitted,
                          by sector), the saved model; appendix: the full-rating attempt, tiers, cost rule
models/               shadow_rating.pkl, the scoring artifact (03)
results/              numbers_of_record.json (03), eda_numbers.json (01), cleaning_numbers.json (02), holdout_companies.csv
figures/              figures written by the notebooks (eda_* from 01, fig_* from 03)
docs/                 presentation_outline.md: the 20-minute slide outline, with speaker notes and Q&A
                      why_ig_sg_not_rating.md: why the model predicts IG vs SG and not the full rating
```

## Run
Tested on Python 3.13.16 (macOS, Apple silicon) with the versions pinned in `requirements.txt`. scikit-learn must be ≥ 1.8 (earlier versions shuffle grouped splits differently). On macOS, LightGBM needs `libomp`: install it (`brew install libomp`) or point the loader at the copy shipped in scikit-learn's wheel, as below. Run the notebooks in order, 01 → 02 → 03:
```bash
pip install -r requirements.txt
cd notebooks && \
export OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 && \
export DYLD_FALLBACK_LIBRARY_PATH="$(python -c 'import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')" && \
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb && \
jupyter nbconvert --to notebook --execute --inplace 02_data_cleaning.ipynb && \
jupyter nbconvert --to notebook --execute --inplace 03_modelling.ipynb
```
01 and 02 take seconds; 03 takes 2–5 minutes (mostly the model-size grid). All three are deterministic, numbers and figures.
