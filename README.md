# STAI-X Challenge 2026 — MoE Pipeline + Eddy XGBoost Expert

Predict `rate_per_10000_ed_visits`, the suspected nonfatal overdose ED-visit rate per 10,000 total ED visits, for held-out validation rows across **6 periods × 51 jurisdictions × 3 drug categories**.

**Metric:** block-averaged MAE, averaged across `all_drugs`, `all_opioids`, and `all_stimulants`.

This fork contains the full team Mixture-of-Experts (MoE) submission pipeline. My personal contribution is the **XGBoost tree expert** in `experts/eddy/src`, which produces `expert_eddy.csv` for the final ensemble.

---

## Full-repo pipeline context

```text
SEAStheMoment_STAIX26_submission.ipynb
  ├─ runs four isolated expert pipelines under experts/
  ├─ loads each expert_*.csv prediction file
  ├─ combines experts with inverse-MAE weights by drug category
  ├─ applies nonnegative + nesting constraints
  └─ writes final submission.csv
```

| Expert | Approach | Output |
|---|---|---|
| Jasmine | healthcare-informed LightGBM | `expert_jasmine.csv` |
| Lenny | multimodal transformer | `expert_lenny.csv` / `expert_transformer.csv` |
| William | classical statistics + temporal models | `expert_william.csv` |
| **Eddy** | **tuned XGBoost with tabular/text/image/time features** | **`expert_eddy.csv`** |

---

## My XGBoost expert

```text
train/val covariates + overdose labels + MAT-density PNGs
        │
        ▼
data_loader.py      load CSVs/images, split targets by overdose category
        │
        ▼
features.py         build tabular, text, image, and shifted rolling features
        │
        ▼
predict.py          train 3 tuned XGBRegressor models
        │
        ▼
expert_eddy.csv     schema-aligned expert predictions for the MoE combiner
```

**Model design:** one `XGBRegressor` per target category.  
**Objective:** `reg:absoluteerror`, evaluated with MAE.  
**Validation:** `GroupKFold` grouped by `period_id` to reduce temporal leakage.  
**Tuning:** Optuna search utilities with final parameters frozen in `config.py`.  
**Integration:** `run_expert.py` provides a subprocess-safe entrypoint for the root MoE notebook.

---

## Feature engineering choices

- **Tabular:** region, period date ordering, weather interaction, and Google Trends aggregates.
- **Text:** compact regex count features from `state_doh_release` for drug, opioid, stimulant, and statistical mentions.
- **Image:** MAT-density PNGs summarized into intensity statistics and hotspot counts after border/background cleanup.
- **Temporal:** jurisdiction-level 3- and 12-period rolling mean/std features, always shifted with `shift(1)`.

---

## My files

```text
experts/eddy/
├── run_expert.py              # subprocess-safe expert entrypoint
└── src/
    ├── data_loader.py         # train/val/sample loading + PNG sidecars
    ├── features.py            # multimodal feature pipeline
    ├── predict.py             # end-to-end train → predict → CSV
    ├── config.py              # tuned XGBoost hyperparameters
    └── tuning.py              # Optuna CV for XGBoost/LightGBM comparison
```

---

## Quick start

Run the full MoE pipeline from the repo root:

```bash
jupyter lab SEAStheMoment_STAIX26_submission.ipynb
```

Run only my XGBoost expert:

```bash
python experts/eddy/run_expert.py data expert_eddy.csv
```

On Kaggle:

```bash
python experts/eddy/run_expert.py /kaggle/input/staix-challenge expert_eddy.csv
```

---
## Engineering highlights

Key implementation choices:

- Built a clean XGBoost pipeline with separate modules for data loading, feature engineering, model configuration, tuning, and inference.
- Combined tabular covariates, public-health text signals, MAT-density image summaries, and past-only temporal features into one lightweight feature set.
- Trained category-specific regressors for all-drugs, opioids, and stimulants so each target could learn its own nonlinear patterns.
- Used grouped cross-validation by period to reduce temporal leakage in a time-dependent panel setting.
- Generated predictions against the provided submission template so the expert output remains compatible with the root MoE combiner.
