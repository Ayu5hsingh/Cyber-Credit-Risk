# Cyber headroom: how much cyber damage can a company's credit absorb?

Case study on cyber credit risk: the question is how a cyber incident affects a company's creditworthiness. Only a few incidents matter for credit, so it is a rare-event problem, like default prediction.

The data hold financial statements and ratings, but no incidents, so the incident-to-credit link cannot be learned from them. They do support the financial side of the question:
1. **The engine:** a candidate screening model that estimates, for any company with financial statements, the calibrated probability of a speculative-grade (SG) rating, P(SG). It works for rated companies and for unrated ones (suppliers, vendors, private firms), where no rating exists to look up.
2. **Cyber headroom:** a stress test of how large an incident, as a share of revenue, each company could absorb before it lands on the review list (or before the model would classify it as SG). It is a model-based scenario, not a downgrade forecast.
3. **The rare-event view:** IG → SG migration, judged with rare-event measures.
4. **Internal grades:** P(SG) grouped into five internal grades (IG-1, IG-2, XO, SG-1, SG-2), so the cyber result also reads as a migration table: how many companies drop a grade after an incident.

It is a triage screen for analysts, not a rating opinion and not a probability of default. It is validated on rated companies it never saw; whether it holds for unrated private firms still needs its own validation. The slide outline for the presentation is in [`docs/presentation_outline.md`](docs/presentation_outline.md).

**Recommended policy: the review budget.** Analysts review the riskiest 30% of companies. The review screen (calibrated P(SG) ≥ 0.31) gives cyber headroom a line to measure distance to.

## Results
All numbers come from `results/numbers_of_record.json` (model), `results/eda_numbers.json` (data) and `results/cleaning_numbers.json` (cleaning). Portfolio numbers count companies on their latest statement; rows are called firm-years.

**The cyber result: every company, each scored by a model that never saw it (cross-fitted)**

| | |
|---|---|
| Who is measured | 592 companies on their latest statement; 208 sit below the review screen, of which 169 are rated investment grade (IG) |
| Cyber headroom, rated IG | the median company absorbs **6.52%** of revenue [5.50%, 7.19%] before landing on the review list |
| At reported incident sizes | Norsk Hydro scale (0.40% of revenue): 4 of 169 land on the review list. Maersk, MGM, UnitedHealth scale (0.68%–0.81%): 6–8 of 169. Merck (1.15%): 13 of 169. **M&S (2.17%, estimated, gross of insurance): 26 of 169** (15%, exact 95% interval 10%–22%) |
| To a speculative-grade classification | at M&S scale, **1 of 169** would reach the model's classification at calibrated P(SG) ≥ 0.5 (exact 95% interval 0.01%–3.3%; not an agency decision); median headroom to it 10.23% |
| By sector (descriptive) | median headroom from 4.74% (Capital Goods, 26 companies) to 7.81% (Technology, 19) across sectors with 10+ companies. Only one pair of intervals does not overlap: Capital Goods [3.39%, 6.25%] sits below Public Utilities [6.49%, 8.05%]; with seven sectors compared and no correction for that, read it as a lead, not a finding. Table in `03_modelling.ipynb` §11 |
| Check on the 123 holdout companies | median 5.47% [4.48%, 7.75%]; 6 of 30 at M&S scale (8%–39%); consistent with the result above |

**Internal grades and the cyber migration table** (`03_modelling.ipynb` §12, figure `fig_cyber_migration.png`)

| | |
|---|---|
| Grades | five, by calibrated P(SG): **IG-1** below 0.2 (strong), **IG-2** 0.2–0.4 (solid), **XO** 0.4–0.6 (crossover, the BBB/BB border), **SG-1** 0.6–0.8 (weak), **SG-2** 0.8 and above (very weak). Round cut-offs that keep 97.3% of the fit of the monotone optimal binning on out-of-fold training scores (optimum 0.173 / 0.269 / 0.42 / 0.74). Five grades because, on training companies only, five is the most whose SG rate rises grade by grade in at least 90% of 25 company splits (4: 100%, 5: 96%, 6: 84%, 7: 76%). A screening scale: regulatory master scales need at least 7 grades plus default |
| Checks on the holdout | SG rate by grade 0.21, 0.27, 0.35, 0.67, 0.86, rising every grade; four of five grades pass the one-sided Jeffreys test, IG-1 fails it at 5% (p = 0.04: its SG rate, 0.21, is higher than its P(SG) predicts); AUC on the grades 0.756 against 0.769 on P(SG); ECB Herfindahl index 0.047 (0 = evenly spread) |
| Portfolio (592 companies) | 136 / 144 / 116 / 100 / 96 companies. Named after the decision, not agency letters: IG-1 companies are mostly rated BBB (42%) and A (35%), SG-2 mostly B (44%) |
| Cyber migration | at M&S scale, **32.6% of companies drop at least one grade and 3.4% (20 companies) drop two or more**; Merck scale 19.8% (4 drop two); Norsk Hydro scale 7.1% (1 drops two). XO is the most fragile grade: 53% of XO companies drop at least one grade at M&S scale |
| A normal year, for reference | the same company a year later: 54% keep their grade, 26% go down, 20% go up (764 company-years). An M&S-scale incident moves more companies down than a normal year does (33% against 26%), and all in one direction |
| Rule on top of the model | negative equity: SG-1 or worse. Judged on training companies only, it moves 13 companies (23 firm-years) the model graded below SG-1 (mean P(SG) 0.42, 57% rated SG). A second candidate, losses with negative operating cash flow, never changed a grade and is not adopted |

**The engine and the rare-event view**

| | |
|---|---|
| Data | 2,028 firm-years from 592 companies after cleaning (405 duplicate rows and the corrupt Company 590 row removed), all in scope (Finance included); 43% SG |
| Model | LightGBM with monotone constraints, 200 trees × 5 leaves: the smallest model within one standard error of the best grouped-CV AUC |
| IG vs SG on companies never seen | **Holdout** (123 companies, fixed before modelling, scored by the saved model file; the independent holdout estimate): AUC 0.77 [0.69, 0.84], PR-AUC 0.75 (no skill 0.43). **Grouped CV** (5 folds × 3 seeds; development evidence that may contain model-selection optimism, because the model size was chosen on the same folds; its size is not measured): AUC 0.81 ± 0.03, PR-AUC 0.78 |
| Holdout, one row per company (latest statement, the unit the review works on) | AUC 0.79 [0.71, 0.87]; PR-AUC 0.78 (no skill 0.46) |
| Review budget (recommended) | reviewing the riskiest 30% of holdout companies catches 52% [44%, 62%] of SG companies at 78% precision (random: 30%) |
| Leakage a naive set-up would hide | random split with duplicates 0.87 → without duplicates 0.85 → grouped by company 0.81 |
| Overfitting | AUC on training companies 0.91, CV 0.81, holdout 0.77 (the earlier 400 × 15 model: 1.00 / 0.81 / 0.75) |
| Hardest boundary, BBB vs BB | AUC 0.70 (CV), 0.69 (holdout) |
| Calibration (holdout) | approximate. Firm-years: slope 0.88 [0.64, 1.20], calibration-in-the-large −0.15 [−0.58, +0.29], Brier 0.19; the largest miss is the middle fifth (0.40 predicted vs 0.28 actual). One row per company: slope 0.95 [0.66, 1.38], calibration-in-the-large −0.28 [−0.69, +0.16]; the largest miss is the middle fifth (0.49 predicted vs 0.36 actual). Calibration-in-the-large is the intercept with the slope fixed at 1 (0 = right on average, below 0 = P(SG) runs high). Both intervals include zero, so there is no clear evidence of over- or under-prediction overall |
| By sector (descriptive) | CV AUC from 0.72 (Public Utilities, 49 training companies) to 0.92 (Health Care, 43) across the 7 sectors with at least 30 training companies. Smaller sectors below that range: Consumer Durables 0.68 (16 companies), Finance 0.68 (14). CV uses firm-years and has no intervals; every holdout sector result is exploratory (5–21 companies each). Table in `03_modelling.ipynb` §7 |
| Stability (20 bootstrap refits; calibrator and screen threshold held fixed) | sensitivity to which companies were in training, given those choices, not the uncertainty of the whole pipeline. Per-company spread of P(SG): median sd 0.087, 90th percentile 0.126. 71% of holdout companies keep the same screen decision in at least 90% of refits; of the 36 less stable ones, 27 are within 0.10 of the screen (median distance 0.063, against 0.316 for the stable ones) |
| Company-weighted training (challenger) | −0.001 AUC vs the champion (sd of the difference 0.011): no gain beyond noise, not adopted |
| Out of time (exploratory) | trained on ratings dated up to 2014 (mostly 2012–14). Energy companies the model never saw, 2015–16: AUC 0.73 (19 companies; 2015: 0.66 on 10, 2016: 0.78 on 16). All sectors, new companies: 0.83. A nowcast: 2016 statements already show the crash |
| Rare-event view: IG → SG migration (exploratory) | a rare event: 27 of 453 IG pairs crossed to SG at the next rating (base rate 6.0%). Point-in-time scores (trained only on years before the scored year): PR-AUC 0.13 [0.07, 0.25], about 2.2× the base rate; reviewing the riskiest 20% of IG statements catches 11 of 27 movers (2.0× lift); AUC 0.67 [0.56, 0.77]. A modest signal, up to about a year ahead; within BBB alone it is weaker (AUC 0.64 [0.51, 0.75], the interval just clears 0.5) |
| Cyber headroom on the holdout only (the check) | 38 of 123 holdout companies are below the screen (30 rated IG, 8 rated SG). For the 30 rated IG, the median absorbs 5.47% of revenue [4.48%, 7.75%]; 6 of 30 (exact 95% interval 8%–39%) would land on the review list at M&S scale; 0 of 30 (0%–12%) would reach the model's classification at P(SG) ≥ 0.5 |
| Headroom sensitivity (holdout) | one fixed population (the 28 rated-IG holdout companies below the lowest screen tested), one assumption changed per row: base 5.79%; tax 15–35%: 5.78%–6.07%; not debt-funded: 6.40%; screen −0.03 / +0.03: 4.79% / 6.11%. 3–6 of 28 companies cross at M&S scale. Where the screen sits moves the answer most |

Intervals in brackets are 95% company-bootstrap intervals. Crossing counts use exact (Clopper–Pearson) binomial intervals: one row per company removes repeated-company dependence (independence across companies is an assumption), and the interval stays informative when none cross. Quote headroom as counts ("26 of 169"); read the anchors as tail scenarios: they are large, widely reported incidents. The cross-fitted result reuses the calibrator and screen fitted on the training companies.

## Limitations
- **Unrated firms.** Validated on rated, listed companies (2012–16). Applicability to unrated private firms, and calibration for them, need their own validation. If their SG share differs from the 43% here, shift the calibrator's intercept by logit(their share) − logit(43%) and confirm it on a labelled sample: the ranking carries over, but the screen and the grade cut-offs are in P(SG) units and move with the level.
- **Statement timing is unknown.** Each row carries the rating date only, so look-ahead bias cannot be fully ruled out.
- **The time-based tests are historical diagnostics.** Each model is refitted on earlier years, but the inputs, constraints and model size were chosen with all years in view.
- **Liquidity.** Higher liquidity ratios read as riskier. Market access is a possible explanation; the data cannot identify the mechanism.
- **Calibration is approximate:** fitted on fold models trained on 80% of the companies, with companies weighted by their number of firm-years. Read P(SG) as a ranking with an approximate level.
- **Cyber headroom is a scenario.** Its intervals leave out uncertainty in the model, the accounting assumptions and how an incident's cost lands. The debt-funded shock raises the debt ratio but holds debt/equity and the equity multiplier fixed (to keep the score monotone), so it understates the leverage effect. Liquidity and sector are not shocked.
- **Five grades, not seven.** The internal scale is a screening scale, below the 7 grades plus default that regulatory master scales require; these data support five. Grades near a cut-off are sensitive: 59% of training companies keep their grade across the three CV seeds.
- **What the data lack:** firm size, default outcomes (the labels are agency opinions) and any cyber data.

## Appendix: secondary outputs
Kept in the notebook appendix, not in the main story. **Why the main product is IG/SG and not the full rating** (what was tested, how it did, limitations, what would improve it): [`docs/why_ig_sg_not_rating.md`](docs/why_ig_sg_not_rating.md).

| | |
|---|---|
| Review screen (calibrated P(SG) ≥ 0.31; set to catch 85% of SG training companies on their latest statement, out of fold) | flags 69% [61%, 77%] of holdout companies and catches 86% [76%, 94%] of SG ones |
| Full-rating attempt: grade band (6 bands, "AA and above" … "CCC and below") | exact grade 41% (CV) and 37% (holdout), against 34% for always answering BBB and 56% for two agencies on the same company; within one grade 89% (CV) and 85% (holdout), against 77% for always BBB (holdout). It predicts BBB or BB for 76% of holdout firm-years |
| Tiers (holdout companies) | SG grade 64 companies (67% actually SG); IG grade with a screen alert 21 (24%); IG grade, no alert 38 (21%) |
| Cost rule (review if P(SG) ≥ 1/k) | k = 2 flags 49% and catches 75% of SG companies; k = 4 flags 76% and catches 91% |

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
- **Internal grades are not ratings.** IG-1 … SG-2 are named after the IG/SG decision, not agency letters, because the model cannot place a company on the exact letter; each grade carries its observed SG rate.
- **A credit rating** is an opinion on relative credit risk: how likely a company is to default and how much lenders would lose if it did. The model estimates the probability of a speculative-grade rating, not a probability of default.
- **What it is for:** a transparent screen trained on rating labels, with a cyber stress layer. It complements a probability-of-default (PD) model rather than replacing it.
- **In deployment**, rank companies by P(SG) × criticality (spend, data access, single point of failure); the model provides the first factor.
- **Not benchmarked:** there is no outlook or watchlist data, so the early signal is not compared with the agencies' own early-warning tools.
- **Quoting:** quote numbers as computed, to two decimals; do not round ranges up.

## Using the saved model
`03_modelling.ipynb` saves the fitted pipeline, calibrator, screen, internal grade scale, input schema and version metadata as `models/shadow_rating.pkl`. Unpickling runs code, so load only a copy you trust.

**Input: processed rows, not raw data.** `score()` expects the output of `02_data_cleaning.ipynb` (cleaned rows with the rebuilt ratios and flags, like `data/processed/firm_years_clean.csv`); cleaning is not part of the artifact. It rejects missing or non-numeric input columns, infinite values and flags other than 0/1, and warns when a sector was not seen in training (that row is scored with no sector effect).
```python
import cloudpickle, pandas as pd
model = cloudpickle.loads(open("models/shadow_rating.pkl", "rb").read())
firm_years = pd.read_csv("data/processed/firm_years_clean.csv", parse_dates=["Date"])   # processed, not raw
scores = model.score(firm_years)
scores.p_sg, scores.screen             # calibrated P(SG) and the screen flag
scores.grade, scores.grade_rule        # internal grade (IG-1 … SG-2) and whether the negative-equity rule set it
model.metadata                         # model size, screen, grade cut-offs, training data hash, package versions
```

## Layout
All analysis code lives in the three notebooks, and assert cells check every cleaning rule, count and stress-test property. The only `.py` files are the stress-test app in `app/`.
```
data/                 raw CSV as delivered
data/processed/       firm_years_clean.csv (2,028 firm-years), firm_years_naive.csv (duplicates kept, for the leakage baseline),
                      schema.json (the input lists, defined once); written by 02
notebooks/
  01_eda.ipynb            the raw data and its problems (reads the raw CSV only)
  02_data_cleaning.ipynb  row-wise cleaning and features; writes data/processed/
  03_modelling.ipynb      validation, model size, model choice; calibration, review screen, grade scale and the saved model
                          (training companies only); the holdout, scored by the saved file: review budget, sectors, stability;
                          SHAP, out of time, migration as a rare event, cyber headroom (holdout check, sensitivity,
                          every company cross-fitted, by sector), internal grades on the holdout and the cyber migration table;
                          appendix: the full-rating attempt, tiers, cost rule
models/               shadow_rating.pkl, the scoring artifact (03)
app/                  the cyber stress-test app (Streamlit): engine.py, app.py, check_engine.py, run.sh
results/              numbers_of_record.json (03), eda_numbers.json (01), cleaning_numbers.json (02), holdout_companies.csv
figures/              figures written by the notebooks (eda_* from 01, fig_* from 03)
figures/slides/       slide versions of the data findings, one color rule for the deck (01 section 15; flags.png from 02)
docs/                 presentation_outline.md: the 20-minute slide outline, with speaker notes and Q&A
                      slides_segments_1_2.md: business case and data slides, each with its visual, notes and standards
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

## The stress-test app
Pick a company and one of four attack types: **operations shutdown** (days of lost sales), **data breach** (one-off costs, paid now or
provisioned), **billing / payments disruption** (direct costs plus collections still late at year-end) or **destructive malware / IT rebuild**
(an outage plus capitalised IT spending), each sizeable to a reported incident. Each attack moves the ratios it would hit in the accounts;
the page shows P(SG), internal grade, the SHAP drivers before and after, and **the line**: how big the attack can get before the company
lands on the review list and before the model calls it SG, in days or % of revenue (and money, if a revenue figure is entered).

- Every company is scored by the one of five cross-fitted models that never saw it, as in 03's every-company headroom; the models are
  refits of `models/shadow_rating.pkl`, so preprocessing, calibrator, screen and grade scale come from the saved file.
- A paid, debt-funded data breach is exactly 03's shock, so the deck's numbers hold for it. The other attack types are app-only
  explorations; the mapping from attack to ratios is an assumption, set out on the page.
- Ratios the model is not monotone in (liquidity, days sales outstanding, turnover) are shown but held at today's value by default: the
  model reads more cash as riskier, so feeding a cash drain can make an attack look like an improvement. A toggle feeds them for Q&A.
- `check_engine.py` checks that the app reproduces 03's headroom and migration numbers, that the breach equals 03's shock for all 592
  companies, and that P(SG) never falls as any attack grows.
```bash
pip install -r app/requirements.txt
app/run.sh                      # opens http://localhost:8501; sets the libomp path as above
python app/check_engine.py      # with the same environment as the notebooks; about 90 seconds
```
