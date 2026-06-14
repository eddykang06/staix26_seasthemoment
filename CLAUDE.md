# CLAUDE.md — Award B Autonomous Data-Analysis Agent

When given the single prompt **"Do the data analysis"**, execute the workflow
below end-to-end with **no further human input**. The held-out dataset is of an
**undisclosed domain**: it may not be about overdoses, may have different column
names, a different target, different file names, and a different task type. Every
rule here is **domain-agnostic**. Any concrete column name in this file
(`gtrends_*`, `jurisdiction`, `period_id`, …) is an *illustrative example only* —
never assume it exists. Detect everything at runtime from the data itself.

> **Scope.** Build everything from the files under `data/` alone. **Ignore
> `experts/` and `SEAStheMoment_STAIX26_submission.ipynb`** — those are the
> Award A Kaggle pipeline, hardcoded to the original overdose domain. Do not
> reuse their features, models, or column names.

> **Environment.** CPU-only, no GPU. Network is available only to `pip install`
> Python packages; it must **never** be used to download any dataset or model.
> Before modeling, ensure these import; install any that are missing:
> `pip install -q lightgbm xgboost catboost scikit-learn reportlab matplotlib pandas numpy scipy`

---

## The one rule that overrides everything

**There must always be a valid `submission.csv` on disk.** Write a schema-correct
baseline (per the template, filled with the train target's per-group median for
regression or mode for classification) **before any modeling**, and only ever
*overwrite* it with something verified to be better. If any later step crashes,
times out, or produces NaN, the last good file survives and still scores. A
mediocre valid submission beats a perfect one that never gets written.

---

## Hard constraints

| Constraint | Value |
|---|---|
| External data / models | **Forbidden.** Use only files under `data/`. No pretrained weights, no downloaded embeddings. |
| GPU | **Not available.** CPU-only models. |
| Token budget | ≤ 1,000,000 tokens total. Be economical (see *Token economy*). |
| Wall-clock | ≤ 2 hours. Budget it (see *Time budget*). |
| Determinism | Fixed `SEED = 42` everywhere; the run must reproduce. |
| Submission schema | Must match the template **exactly** — same id column, same target column, same row set and order, correct dtype, **zero NaN**. |

---

## Operating rules

**Determinism.** Set `SEED = 42` and apply it to `random`, `numpy`, and every
model (`random_state`/`seed`/`bagging_seed`). No time-seeded randomness.

**Token economy.** Never print or read whole files/frames. Inspect with
`df.shape`, `df.dtypes`, `df.head(3)`, `df.isna().mean()`, `df.nunique()`,
`df[col].describe()`. Read a file once; reuse the variable. Sample (`.sample`,
`nrows=`) when a peek suffices. Keep model logs quiet (`verbose=-1`,
`verbose_eval=False`).

**Time budget (≈2 h).** Record `t0 = time.time()` at the start.
1. Baseline submission + EDA — first ~5 min.
2. Feature engineering + a single LightGBM with CV — next ~30–40 min.
3. Add XGBoost, then CatBoost, ensembling — while time remains.
4. Report — last ~10 min.
Check elapsed time before each heavy stage. If past ~75 min, stop adding models
and finalize. If a stage is slow, cut folds (5→3) or drop the slowest model.
Never start a stage you cannot finish inside the budget.

**Fail isolation.** Wrap every model fit and the report in `try/except`. On
failure: log a one-line reason, keep the best submission so far, and continue.
A single broken group or model must never abort the run.

**No leakage.** Fit every transform, encoder, imputer, and scaler on **train
folds only**, then apply to validation/test. Never fit on the prediction rows.
Never build a feature from the target of the row being predicted (no
current-period target in lags/rolling/EWMA); temporal features may use **past
periods only**.

---

## Step-by-step workflow

### 1 — Discover and read the data description
Locate the description (`data/DATA_DESCRIPTION.md`, else `Data_Description.md`,
else any `*escription*.md`/`README` under `data/`). If none exists, infer from
the files. Record, **by reading, not assuming**:
- the **target** column name and dtype;
- the **id / key** columns (the submission id, plus any join/group keys);
- the **submission template** path and its exact columns;
- **suppression semantics** (does `NaN` mean suppressed or truly missing?);
- any stated **metric** and any **constraints** between rows (e.g. nesting,
  non-negativity).

### 2 — Load the template first, then the data
Load the submission template — it defines the exact rows and id values you must
predict. Then discover and load every CSV under `data/` (glob `train/`, `val/`,
`test/`, and the root). For each: print `shape`, `dtypes`, and NaN rates. Find
the join keys by matching column names across files.

### 3 — Write the fail-safe baseline submission
Compute the train target's median (regression) or mode (classification), per
group if a grouping/category column exists in the template, else globally.
Fill the template's target with it, enforce any stated constraint, assert
**zero NaN** and **row count == template**, and write `submission.csv` **now**.
This is the floor you will improve on.

### 4 — Infer the task
- **Type:** float/continuous target → regression (default metric MAE);
  integer-with-few-values / bool / string / categorical → classification
  (binary → AUC/logloss, multiclass → multi-logloss). Honor any metric the
  description states.
- **Grouping:** if the template has a category/type/segment column, train one
  model per group value.
- **Temporal:** if a period/date-like column exists, recover order (numeric or
  parseable date; else leave unordered) and enable past-only lag features only
  when there is more than one time step.

### 5 — Feature engineering (generic, by column *type* — skip what doesn't apply)
| Column type (detected, not named) | Transform |
|---|---|
| Numeric | median-impute (train medians); `log1p` for non-negative right-skewed columns |
| Low-cardinality categorical | native model categorical handling, or frequency/one-hot encoding fit on train |
| High-cardinality categorical / ID | frequency (count) encoding |
| Free text (long strings) | TF-IDF (≤5000 feats, `sublinear_tf`) → TruncatedSVD (~20 comps) |
| Datetime | decompose into ordinal / year / month / day-of-week / cyclical sin-cos |
| Temporal order present | `lag_1`, `lag_2` per (group keys) sorted by period rank, **past only** |
| Constant / duplicate columns | drop |
Align train and prediction feature matrices to the **same columns**.

### 6 — Train and validate
For each target group (or once if ungrouped):
1. Drop rows with NaN target.
2. Cross-validate: `GroupKFold` on the temporal/group key if one exists, else
   `KFold(shuffle=True, random_state=SEED)` (`StratifiedKFold` for
   classification). 5 folds, or 3 if time is tight.
3. Train **LightGBM** first (`objective` matched to the task,
   `early_stopping_rounds=100`). Then **XGBoost**, then **CatBoost** if time
   allows — each in its own `try/except`.
4. Ensemble = mean of out-of-fold predictions across the models that succeeded.
5. Print the OOF metric (MAE / AUC / logloss) per group.
Report the mean OOF metric across groups.

### 7 — Finalize the submission
Start from the template id column. Merge predictions by id. For any row still
missing, fall back to the group (or global) baseline from step 3. Apply stated
constraints (clip to ≥0 only if the target is a non-negative quantity; enforce
nesting/monotonic relations if specified). Then, only if it is at least as
complete and valid as the current file, **overwrite** `submission.csv`. Assert:
exact template columns, row count and id set identical to the template, target
dtype correct, **zero NaN**.

### 8 — Write `report.pdf` (wrapped in `try/except` — never let it block the submission)
Use `reportlab.platypus`. Minimal code, ~5 sections:
1. **Dataset overview** — shapes, NaN/suppression rates, task type, metric.
2. **EDA** — target distribution (per group) and a feature-correlation heatmap.
3. **CV results** — OOF metric per group and per model family.
4. **Feature importance** — top-15 LightGBM importances per group.
5. **Submission preview** — `head(10)` of `submission.csv`.

---

## Error recovery
- Description path missing → try alternatives, else infer from the files.
- Stated file path missing → glob for it under `data/`, `train/`, `val/`, root.
- Target/id column not found by name → infer: the template's non-id column is
  the target; the column shared with train that is unique per row is the id.
- Temporal order unrecoverable → skip lag features silently.
- A model errors on a group → log it, keep that group's baseline, continue.
- Past ~75 min → drop CatBoost (then XGBoost), finalize with what you have.
- Report fails → log it and stop; the submission already exists and stands.

---

## Output checklist (verify before stopping)
- [ ] `submission.csv` exists, has exactly the template's id + target columns,
      identical row set, correct dtype, and **0 NaN**.
- [ ] Every model fit and the report ran inside `try/except`; the run did not
      abort on any single failure.
- [ ] `report.pdf` exists and opens (if reporting succeeded).
- [ ] No external dataset or model was downloaded.
- [ ] `SEED = 42` applied throughout; result is reproducible.
- [ ] Stayed under 2 h wall-clock and 1,000,000 tokens.
