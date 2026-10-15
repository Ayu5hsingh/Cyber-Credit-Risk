# Cyber headroom: how much cyber damage can a company's credit absorb?

Case study on cyber credit risk: the question is how a cyber incident affects a company's creditworthiness. Only a few incidents matter for credit, so it is a rare-event problem, like default prediction.

The data hold financial statements and ratings, but no incidents, so the incident-to-credit link cannot be learned from them. They do support the financial side of the question:
1. **The engine:** a candidate screening model that estimates, for any company with financial statements, the calibrated probability of a speculative-grade (SG) rating, P(SG). It works for rated companies and for unrated ones (suppliers, vendors, private firms), where no rating exists to look up.
2. **Cyber headroom:** a stress test of how large an incident, as a share of revenue, each company could absorb before it lands on the review list (or before the model would classify it as SG). It is a model-based scenario, not a downgrade forecast.
3. **The rare-event view:** IG → SG migration, judged with rare-event measures.
4. **Internal grades:** P(SG) grouped into six internal grades (IG-1, IG-2, IG-3 below P(SG) 0.5; SG-1, SG-2, SG-3 above it), so the cyber result also reads as a migration table: how many companies drop a grade after an incident.

The model reads the ratios the way a credit analyst does: `03_features.ipynb` builds 25 credit features from them (interest coverage, the price lenders charge, cash flow against what is owed, the stability of profit, …) and selects, on training companies only, the two that help: the **implied interest rate** and **operating-margin volatility**. It also drops seven inputs that add nothing.

It is a triage screen for analysts, not a rating opinion and not a probability of default. It is validated on rated companies it never saw; whether it holds for unrated private firms still needs its own validation.

**Recommended policy: the review budget.** Analysts review the riskiest 30% of companies. The review screen (calibrated P(SG) ≥ 0.28) gives cyber headroom a line to measure distance to.

## Results
All numbers come from `results/numbers_of_record.json` (model), `results/feature_selection.json` (credit features), `results/eda_numbers.json` (data) and `results/cleaning_numbers.json` (cleaning). Portfolio numbers count companies on their latest statement; rows are called firm-years.

**The cyber result: every company, each scored by a model that never saw it (cross-fitted)**

| | |
|---|---|
| Who is measured | 592 companies on their latest statement; 215 sit below the review screen, of which 177 are rated investment grade (IG) |
| Cyber headroom, rated IG | the median company absorbs **6.36%** of revenue [5.47%, 7.45%] before landing on the review list |
| At reported incident sizes | Norsk Hydro scale (0.40% of revenue): 9 of 177 land on the review list. Maersk, MGM, UnitedHealth scale (0.68%–0.81%): 16–18 of 177. Merck (1.15%): 21 of 177. **M&S (2.17%, estimated, gross of insurance): 34 of 177** (19%, exact 95% interval 14%–26%) |
| To a speculative-grade classification | at M&S scale, **5 of 177** would reach the model's classification at calibrated P(SG) ≥ 0.5 (exact 95% interval 0.9%–6.5%; not an agency decision); median headroom to it 10.10% |
| By sector (descriptive) | median headroom from 2.09% (Consumer Durables, 11 companies) to 10.20% (Miscellaneous, 10) across the 8 sectors with 10+ companies; every pair of intervals overlaps, so no sector is shown to differ. Table in `04_modelling.ipynb` §11 |
| Check on the 123 holdout companies | median 6.29% [4.89%, 8.89%]; 8 of 34 at M&S scale (11%–41%); consistent with the result above |

**Credit features** (`03_features.ipynb`, figure `fig_feature_selection.png`; training companies only)

| | |
|---|---|
| Candidates | 25 features built from the ratios, each with a credit reason: leverage and coverage (interest coverage, implied interest rate, cash flow / liabilities, …), profitability, earnings quality, liquidity, investment, a partial Altman Z'', the company's own history (margin volatility, changes), and position against its sector |
| Filter (each alone) | 12 pass: at most half missing, separates IG/SG on its own (company-bootstrap interval of the AUC excludes 0.5), agrees with the credit direction, not a near-copy (rank correlation below 0.9) of an input the model has or a stronger candidate |
| Forward selection | starting from the model before (21 ratios, 6 flags, sector; CV AUC 0.812): the **implied interest rate** enters first (+0.012), **operating-margin volatility** second (+0.005); coverage would add +0.001 more (it reads the same interest line). The one-SE rule, as for the model's size, stops at two |
| Backward elimination | of the inputs no cyber attack moves: gross margin, debt / equity, the equity multiplier, payables turnover and three placeholder flags go (debt / equity and the equity multiplier are the debt ratio written another way); the effective tax rate stays. One one-SE line for the whole search |
| Result | 17 statement ratios, 2 credit features, 3 flags and sector. CV AUC 0.812 → **0.829** (better in 13 of 15 folds; in 20 of 20 resamples of the training companies); holdout 0.769 → **0.799** |
| Checks | both added features beat shuffled "shadow" copies and noise on permutation importance; the implied interest rate is the most important input (permutation and SHAP); a bigger cyber loss never lowers the score (469 training companies, 0–30% of revenue) |
| Caveat | lenders charge by risk, so for a rated company part of the implied rate follows the rating itself; for an unrated company it is the lenders' real view. Both added features are held at today's value by the cyber shock (no interest on new debt, no spread widening), which understates that channel |

**Internal grades and the cyber migration table** (`04_modelling.ipynb` §12, figure `fig_cyber_migration.png`)

| | |
|---|---|
| Grades | six, by calibrated P(SG): **IG-1** below 0.15 (strong), **IG-2** 0.15–0.3 (solid), **IG-3** 0.3–0.5 (near the line), **SG-1** 0.5–0.7 (near the line), **SG-2** 0.7–0.85 (weak), **SG-3** 0.85 and above (very weak). Round cut-offs, the best set on a 0.05 grid, keep 99.5% of the fit of the monotone optimal binning on out-of-fold training scores (optimum 0.173 / 0.308 / 0.501 / 0.68 / 0.879). Six grades because, on training companies only, six is the most whose SG rate rises grade by grade in at least 90% of 25 company splits (4, 5, 6: 100%; 7: 84%); before the credit features the same rule gave five. A screening scale: regulatory master scales need at least 7 grades plus default |
| Checks on the holdout | SG rate by grade 0.10, 0.33, 0.32, 0.44, 0.71, 0.92: every step rises except IG-2 to IG-3, out of order by one point (overlapping intervals); five of six grades pass the one-sided Jeffreys test, IG-2 fails it (p = 0.007: its SG rate, 0.33, is above the 0.22 its P(SG) predicts); AUC on the grades 0.784 against 0.799 on P(SG); ECB Herfindahl index 0.031 (0 = evenly spread) |
| Portfolio (592 companies) | 99 / 123 / 120 / 95 / 70 / 85 companies. Named after the decision, not agency letters: IG-1 companies are mostly rated BBB (39%) and A (38%), SG-3 mostly B (45%) |
| Cyber migration | at M&S scale, **31.1% of companies drop at least one grade and 4.9% (29 companies) drop two or more**; Merck scale 16.9% (8 drop two); Norsk Hydro scale 6.9% (2 drop two). The grades nearest the line are the most fragile: 44% of IG-3 and 44% of SG-1 companies drop at least one grade at M&S scale |
| A normal year, for reference | the same company a year later: 46% keep their grade, 28% go down, 26% go up (764 company-years). An M&S-scale incident moves more companies down than a normal year does (31% against 28%), and all in one direction |
| Rule on top of the model | negative equity: SG-1 or worse. Judged on training companies only, it moves 10 companies (24 firm-years) the model graded below SG-1 (mean P(SG) 0.35, 62% rated SG). A second candidate, losses with negative operating cash flow, never changed a grade and is not adopted |

**The engine and the rare-event view**

| | |
|---|---|
| Data | 2,028 firm-years from 592 companies after cleaning (405 duplicate rows and the corrupt Company 590 row removed), all in scope (Finance included); 43% SG |
| Inputs | 17 statement ratios, the implied interest rate and operating-margin volatility (`03_features.ipynb`), 3 flags and sector; 15 of them monotone |
| Model | LightGBM with monotone constraints, 200 trees × 5 leaves: the smallest model within one standard error of the best grouped-CV AUC |
| IG vs SG on companies never seen | **Holdout** (123 companies, fixed before modelling, scored by the saved model file; the independent holdout estimate): AUC 0.80 [0.72, 0.87], PR-AUC 0.78 (no skill 0.43). **Grouped CV** (5 folds × 3 seeds; development evidence that may contain model-selection optimism, because the model size and the features were chosen on the same folds; its size is not measured): AUC 0.83 ± 0.02, PR-AUC 0.80 |
| Holdout, one row per company (latest statement, the unit the review works on) | AUC 0.82 [0.74, 0.89]; PR-AUC 0.80 (no skill 0.46) |
| Review budget (recommended) | reviewing the riskiest 30% of holdout companies catches 54% [46%, 64%] of SG companies at 81% precision (random: 30%) |
| Leakage a naive set-up would hide | random split with duplicates 0.87 → without duplicates 0.86 → grouped by company 0.83 |
| Overfitting | AUC on training companies 0.91, CV 0.83, holdout 0.80 (the earlier 400 × 15 model: 1.00 / 0.83 / 0.79) |
| Hardest boundary, BBB vs BB | AUC 0.71 (CV), 0.69 (holdout) |
| Calibration (holdout) | approximate. Firm-years: slope 0.87 [0.63, 1.17], calibration-in-the-large −0.11 [−0.57, +0.33], Brier 0.18; the largest miss is the fourth fifth (0.64 predicted vs 0.46 actual). One row per company: slope 0.92 [0.66, 1.32], calibration-in-the-large −0.27 [−0.71, +0.20], Brier 0.17; the largest miss is the middle fifth (0.47 predicted vs 0.32 actual). Calibration-in-the-large is the intercept with the slope fixed at 1 (0 = right on average, below 0 = P(SG) runs high). Both intervals include zero, so there is no clear evidence of over- or under-prediction overall |
| By sector (descriptive) | CV AUC from 0.76 (Public Utilities, 49 training companies) to 0.90 (Health Care, 43) across the 7 sectors with at least 30 training companies; the five smaller sectors fall inside that range (0.79–0.84). CV uses firm-years and has no intervals; every holdout sector result is exploratory. Table in `04_modelling.ipynb` §7 |
| Stability (20 bootstrap refits; calibrator and screen threshold held fixed) | sensitivity to which companies were in training, given those choices, not the uncertainty of the whole pipeline. Per-company spread of P(SG): median sd 0.082, 90th percentile 0.121. 75% of holdout companies keep the same screen decision in at least 90% of refits; of the 31 less stable ones, 25 are within 0.10 of the screen (median distance 0.035, against 0.352 for the stable ones) |
| Company-weighted training (challenger) | −0.003 AUC vs the champion (sd of the difference 0.011): no gain beyond noise, not adopted |
| Out of time (exploratory) | trained on ratings dated up to 2014 (mostly 2012–14). Energy companies the model never saw, 2015–16: AUC 0.76 (19 companies; 2015: 0.63 on 10, 2016: 0.88 on 16). All sectors, new companies: 0.83. A nowcast: 2016 statements already show the crash |
| Rare-event view: IG → SG migration (exploratory) | a rare event: 27 of 453 IG pairs crossed to SG at the next rating (base rate 6.0%). Point-in-time scores (trained only on years before the scored year): PR-AUC 0.15 [0.08, 0.25], about 2.5× the base rate; reviewing the riskiest 20% of IG statements catches 12 of 27 movers (2.2× lift); AUC 0.68 [0.56, 0.79]. A modest signal, up to about a year ahead; within BBB alone there is no clear signal (AUC 0.64 [0.50, 0.77], the interval reaches 0.5) |
| Cyber headroom on the holdout only (the check) | 41 of 123 holdout companies are below the screen (34 rated IG, 7 rated SG). For the 34 rated IG, the median absorbs 6.29% of revenue [4.89%, 8.89%]; 8 of 34 (exact 95% interval 11%–41%) would land on the review list at M&S scale; 0 of 34 (0%–10%) would reach the model's classification at P(SG) ≥ 0.5 |
| Headroom sensitivity (holdout) | one fixed population (the 30 rated-IG holdout companies below the lowest screen tested), one assumption changed per row: base 6.90%; tax 15–35%: 6.42%–7.16%; not debt-funded: 7.14%; screen −0.03 / +0.03: 6.82% / 7.28%. 5–7 of 30 companies cross at M&S scale |

Intervals in brackets are 95% company-bootstrap intervals. Crossing counts use exact (Clopper–Pearson) binomial intervals: one row per company removes repeated-company dependence (independence across companies is an assumption), and the interval stays informative when none cross. Quote headroom as counts ("34 of 177"); read the anchors as tail scenarios: they are large, widely reported incidents. The cross-fitted result reuses the calibrator and screen fitted on the training companies.

## Limitations
- **Unrated firms.** Validated on rated, listed companies (2012–16). Applicability to unrated private firms, and calibration for them, need their own validation. If their SG share differs from the 43% here, shift the calibrator's intercept by logit(their share) − logit(43%) and confirm it on a labelled sample: the ranking carries over, but the screen and the grade cut-offs are in P(SG) units and move with the level.
- **Statement timing is unknown.** Each row carries the rating date only, so look-ahead bias cannot be fully ruled out.
- **The time-based tests are historical diagnostics.** Each model is refitted on earlier years, but the inputs, constraints and model size were chosen with all years in view.
- **Liquidity.** Higher liquidity ratios read as riskier. Market access is a possible explanation; the data cannot identify the mechanism.
- **Calibration is approximate:** fitted on fold models trained on 80% of the companies, with companies weighted by their number of firm-years. Read P(SG) as a ranking with an approximate level.
- **Cyber headroom is a scenario.** Its intervals leave out uncertainty in the model, the accounting assumptions and how an incident's cost lands. The shock holds the implied interest rate and margin volatility at today's value (no interest on the new debt, no spread widening), so it understates that channel. Liquidity and sector are not shocked.
- **The implied interest rate** is a proxy (the gap between operating and pre-tax profit also holds other non-operating items; payables count as liabilities), and for rated companies part of the price lenders charge follows the rating itself.
- **Six grades, not seven.** The internal scale is a screening scale, below the 7 grades plus default that regulatory master scales require; these data support six. On the holdout, IG-3 comes out one point below IG-2 and IG-2 fails the Jeffreys test. Grades near a cut-off are sensitive: 59% of training companies keep their grade across the three CV seeds.
- **What the data lack:** firm size, default outcomes (the labels are agency opinions) and any cyber data.

## Appendix: secondary outputs
Kept in the notebook appendix, not in the main story.

| | |
|---|---|
| Review screen (calibrated P(SG) ≥ 0.28; set to catch 85% of SG training companies on their latest statement, out of fold) | flags 67% [59%, 75%] of holdout companies and catches 88% [79%, 96%] of SG ones |
| Full-rating attempt: grade band (6 bands, "AA and above" … "CCC and below") | exact grade 43% (CV) and 42% (holdout), against 34% for always answering BBB and 56% for two agencies on the same company; within one grade 90% (CV) and 89% (holdout), against 77% for always BBB (holdout). It predicts BBB or BB for 68% of holdout firm-years |
| Tiers (holdout companies) | SG grade 59 companies (69% actually SG); IG grade with a screen alert 23 (35%); IG grade, no alert 41 (17%) |
| Cost rule (review if P(SG) ≥ 1/k) | k = 2 flags 46% and catches 71% of SG companies; k = 4 flags 74% and catches 95% |

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
| Internal grades | SG rate by grade (still rising grade by grade?), Jeffreys test per grade, concentration (Herfindahl), moves between grades against the normal year (MWB) | re-fit the cut-offs with the same binning if a grade's level drifts; move toward 7 grades as data allow |
| Business effectiveness | review volume, analyst overrides (rate, direction, later outcome), findings, missed risky companies | revisit the review policy; many overrides in one direction mean the model needs fixing |

- **Drift triggers investigation, not retraining.** In 2015–16 Energy inputs and scores shifted sharply, and the model tracked the shock: drift is not failure. Retraining follows sustained deterioration in ranking or catch rate, and a retrained challenger replaces the model only after it beats it on later-period data and on the relevant sectors.
- **Labels may never arrive for unrated firms.** Agree a validation source up front: later ratings where they appear, structured analyst assessments, or defaults as a separate check.
- **Audit a random sample of unflagged firms.** If analysts review only flagged firms, missed risk can never be measured.
- **Actions:** a broken feed is fixed; drift with stable performance is watched; a level shift is recalibrated; a broken ranking leads to a validated retrain; a new population needs new data and validation.

## How to read it
- **Grades, not notches.** The data carry letter grades only (AAA … D); a notch is one step on the modified rating scale (BBB− → BB+ is one notch), so we claim no notch precision.
- **Internal grades are not ratings.** IG-1 … SG-3 are named after the IG/SG decision, not agency letters, because the model cannot place a company on the exact letter; each grade carries its observed SG rate.
- **A credit rating** is an opinion on relative credit risk: how likely a company is to default and how much lenders would lose if it did. The model estimates the probability of a speculative-grade rating, not a probability of default.
- **What it is for:** a transparent screen trained on rating labels, with a cyber stress layer. It complements a probability-of-default (PD) model rather than replacing it.
- **In deployment**, rank companies by P(SG) × criticality (spend, data access, single point of failure); the model provides the first factor.
- **Not benchmarked:** there is no outlook or watchlist data, so the early signal is not compared with the agencies' own early-warning tools.
- **Quoting:** quote numbers as computed, to two decimals; do not round ranges up.

## Using the saved model
`04_modelling.ipynb` saves the fitted pipeline, calibrator, screen, internal grade scale, input schema and version metadata as `models/shadow_rating.pkl`. Unpickling runs code, so load only a copy you trust.

**Input: processed rows, not raw data.** `score()` expects the output of `03_features.ipynb` (cleaned rows with the rebuilt ratios, flags and the two credit features, like `data/processed/firm_years_features.csv`); cleaning and those features are not part of the artifact. It rejects missing or non-numeric input columns, infinite values and flags other than 0/1, and warns when a sector was not seen in training (that row is scored with no sector effect).
```python
import cloudpickle, pandas as pd
model = cloudpickle.loads(open("models/shadow_rating.pkl", "rb").read())
firm_years = pd.read_csv("data/processed/firm_years_features.csv", parse_dates=["Date"])   # processed, not raw
scores = model.score(firm_years)
scores.p_sg, scores.screen             # calibrated P(SG) and the screen flag
scores.grade, scores.grade_rule        # internal grade (IG-1 … SG-3) and whether the negative-equity rule set it
model.metadata                         # model size, screen, grade cut-offs, training data hash, package versions
```

## Layout
The analysis lives in the four notebooks, and assert cells check every cleaning rule, count and stress-test property. Code the notebooks and the app share (the incident shock, the credit features, the holdout split, winsorising) lives in `cyber_credit/`; `tests/` checks it, and the saved model, against `results/`.
```
data/                 raw CSV as delivered
data/processed/       firm_years_clean.csv (2,028 firm-years), firm_years_naive.csv (duplicates kept, for the leakage baseline),
                      schema.json (the input lists, defined once): written by 02; firm_years_features.csv and
                      firm_years_naive_features.csv (the same rows plus the selected credit features) and features.json
                      (the model's inputs): written by 03
notebooks/
  01_eda.ipynb            the raw data and its problems (reads the raw CSV only)
  02_data_cleaning.ipynb  row-wise cleaning and flags; writes data/processed/
  03_features.ipynb       credit features from the ratios and their selection (filter, forward selection, backward elimination,
                          shadow, stability and shock checks), on training companies only
  04_modelling.ipynb      validation, model size, model choice; calibration, review screen, grade scale and the saved model
                          (training companies only); the holdout, scored by the saved file: review budget, sectors, stability;
                          SHAP, out of time, migration as a rare event, cyber headroom (holdout check, sensitivity,
                          every company cross-fitted, by sector), internal grades on the holdout and the cyber migration table;
                          appendix: the full-rating attempt, tiers, cost rule
models/               shadow_rating.pkl, the scoring artifact (04)
cyber_credit/         shared code for the notebooks and the app: the incident shock (shock.py), the credit features (features.py),
                      the holdout split (split.py), winsorising (preprocess.py)
tests/                pytest checks of the shared code and of the saved model against results/
app/                  the cyber stress-test app (Streamlit): engine.py, app.py, check_engine.py, run.sh
results/              numbers_of_record.json (04), feature_selection.json (03), eda_numbers.json (01), cleaning_numbers.json (02),
                      holdout_companies.csv
figures/              figures written by the notebooks when run (eda_* from 01, fig_feature_selection from 03, fig_* from 04);
                      not kept in the repository
```

## Run
Tested on Python 3.13.16 (macOS, Apple silicon) with the versions pinned in `requirements.txt`. scikit-learn must be ≥ 1.8 (earlier versions shuffle grouped splits differently). On macOS, LightGBM needs `libomp`: install it (`brew install libomp`) or point the loader at the copy shipped in scikit-learn's wheel, as below. Run the notebooks in order, 01 → 02 → 03 → 04:
```bash
pip install -r requirements.txt
cd notebooks && \
export OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 && \
export DYLD_FALLBACK_LIBRARY_PATH="$(python -c 'import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')" && \
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb && \
jupyter nbconvert --to notebook --execute --inplace 02_data_cleaning.ipynb && \
jupyter nbconvert --to notebook --execute --inplace 03_features.ipynb && \
jupyter nbconvert --to notebook --execute --inplace 04_modelling.ipynb
```
01 and 02 take seconds; 03 takes about 2 minutes on 10 cores (its model fits run side by side, each single-threaded, so the numbers do not depend on the core count); 04 takes 2–5 minutes (mostly the model-size grid). All four are deterministic, numbers and figures. Then run the tests from the repository root: `pytest tests`.

## The stress-test app
Pick a company and one of four attack types: **operations shutdown** (days of lost sales), **data breach** (one-off costs, paid now or
provisioned), **billing / payments disruption** (direct costs plus collections still late at year-end) or **destructive malware / IT rebuild**
(an outage plus capitalised IT spending), each sizeable to a reported incident. Each attack moves the ratios it would hit in the accounts;
the page shows P(SG), internal grade, the SHAP drivers before and after, and **the line**: how big the attack can get before the company
lands on the review list and before the model calls it SG, in days or % of revenue (and money, if a revenue figure is entered).

- Every company is scored by the one of five cross-fitted models that never saw it, as in 04's every-company headroom; the models are
  refits of `models/shadow_rating.pkl`, so preprocessing, calibrator, screen and grade scale come from the saved file.
- A paid, debt-funded data breach is exactly 04's shock, so the deck's numbers hold for it. The other attack types are app-only
  explorations; the mapping from attack to ratios is an assumption, set out on the page.
- Ratios the model is not monotone in (liquidity, days sales outstanding, turnover) are shown but held at today's value by default: the
  model reads more cash as riskier, so feeding a cash drain can make an attack look like an improvement. A toggle feeds them for Q&A.
- `check_engine.py` checks that the app reproduces 04's headroom and migration numbers, that the breach equals 04's shock for all 592
  companies, and that P(SG) never falls as any attack grows.
```bash
pip install -r app/requirements.txt
app/run.sh                      # opens http://localhost:8501; sets the libomp path as above
python app/check_engine.py      # with the same environment as the notebooks; about 90 seconds
```
