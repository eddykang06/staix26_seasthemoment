# STAI-X Challenge 2026: MoE Pipeline + XGBoost Expert

## Objective

This repository contains a Mixture-of-Experts (MoE) forecasting pipeline submitted to the [2026 STAI-X Challenge](https://statsupai.org/STAIX2026/challenge.html), hosted by the Harvard Departments of Statistics and Biostatistics. The objective was to predict `rate_per_10000_ed_visits`, the suspected nonfatal overdose ED-visit rate per 10,000 total ED visits, for held-out validation rows across **6 time periods × 51 jurisdictions × 3 drug categories**. The dataset consisted of tabular, text, and image data from a real public health dataset. 

**Metric:** block-averaged MAE, averaged across `all_drugs`, `all_opioids`, and `all_stimulants`.

This fork contains the full team MoE submission pipeline. My personal contribution is the **XGBoost tree expert** in `experts/eddy/src`, which produces `expert_eddy.csv` for the final ensemble.

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
| Jasmine | Healthcare-informed LightGBM | `expert_jasmine.csv` |
| Lenny | Multimodal transformer | `expert_lenny.csv` / `expert_transformer.csv` |
| William | Classical statistics + temporal models | `expert_william.csv` |
| **Eddy** | **Tuned XGBoost with tabular/text/image/time features** | **`expert_eddy.csv`** |

---

## My XGBoost expert

```text
train/val covariates + overdose labels + MAT-density PNGs
        │
        v
data_loader.py      load CSVs/images, split targets by overdose category
        │
        v
features.py         build tabular, text, image, and shifted rolling features
        │
        v
predict.py          train 3 tuned XGBRegressor models
        │
        v
expert_eddy.csv     schema-aligned expert predictions for the MoE combiner
```

**Model design:** One XGBoost regression model per target category.  
**Objective:** Mean out-of-fold MAE, averaged across 3 target categories.
**Validation:** 4-fold cross-validation grouped by time-period to minimize temporal leakage.  
**Tuning:** Optuna search utilities with final parameters frozen in `config.py`.  
**Integration:** `run_expert.py` provides a subprocess-safe entrypoint for the root MoE notebook.

---

## Engineering highlights
Feature engineering:
- **Tabular:** region, period date ordering, weather interaction, and Google Trends aggregates.
- **Text:** compact regex count features from Department of Health release text for drug, opioid, stimulant, and statistical mentions.
- **Image:** MAT-density PNGs summarized into intensity statistics and hotspot counts after border/background cleanup.
- **Temporal:** state-level 3- and 12-period rolling mean/std features.

Training and tuning:
- 3 category-specific regression models, tuned separately for category-dependent prediction.
- Grouped cross-validation by time period to reduce temporal leakage, facilitating generalization to future forecasting.
- 
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

