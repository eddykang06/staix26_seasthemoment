# STAI-X Challenge 2026: MoE Pipeline + XGBoost Expert

## Objective

This repository contains a Mixture-of-Experts forecasting pipeline submitted to the [2026 STAI-X Challenge](https://statsupai.org/STAIX2026/challenge.html), hosted by the Harvard Departments of Statistics and Biostatistics. The objective was to predict `rate_per_10000_ed_visits`, the suspected nonfatal overdose emergency department (ED) visit rate per 10,000 total ED visits, for held-out validation rows across 6 time periods × 51 jurisdictions × 3 drug categories. The dataset consisted of tabular, text, and image data from a real public health dataset. 

The evaluation metric was block-averaged MAE, averaged across `all_drugs`, `all_opioids`, and `all_stimulants`.

This fork contains the full team submission pipeline. My personal contribution is the XGBoost tree expert in `experts/eddy/src`, which produces `expert_eddy.csv` for the final ensemble prediction.

## Full pipeline context

| Expert | Approach | 
|---|---|
| Jasmine | Healthcare-informed LightGBM |
| Lenny | Multimodal transformer |
| William | Classical statistics + temporal models |
| **Eddy** | **Tuned XGBoost** |

The final submission notebook `SEAStheMoment_STAIX26_submission.ipynb` perfoms the followings:
- Runs four isolated expert pipelines under `experts/`
- Loads predictions from each expert into separate csvs
- Combines expert prediction with inverse-MAE weighting by drug category

## My files

```text
experts/eddy/
├── run_expert.py              # Subprocess-safe expert entrypoint
└── src/
    ├── data_loader.py         # Train/val/sample loading + PNG sidecars
    ├── features.py            # Multimodal feature engineering pipeline
    ├── predict.py             # End-to-end train → predict → CSV
    ├── config.py              # Tuned XGBoost hyperparameters
    └── tuning.py              # Time-grouped cross-validation for hyperparameter tuning
```

## XGBoost expert description
Model:
- One XGBoost regression model for each prediction target category
- Objective: Mean out-of-fold MAE, averaged across 3 target categories

Feature engineering:
- Tabular: Region, period date ordering, weather interaction, and Google Trends aggregates
- Text: Compact regex count features from Department of Health release text for drug, opioid, stimulant, and statistical mentions
- Image: MAT-density PNGs summarized into intensity statistics and hotspot counts after border/background cleanup
- Temporal: State-level 3- and 12-period rolling mean/std features

Training and tuning:
- 3 category-specific regression models, tuned separately for category-dependent prediction
- 4-fold time-grouped cross-validation to reduce temporal leakage, facilitating generalization to future forecasting


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

