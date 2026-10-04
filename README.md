# Kaggle Playground S6E9: Predicting Electric Vehicle Purchases

Final result: **78th of 3,575 teams (top 2.2%)**, AUC **0.94566** on the hidden test set, 7 submissions.
Metric: ROC AUC. Task: predict `Will_Buy_EV` from 12 customer features.

## Approach

The training data was synthetic, generated from a real dataset of 10,000 rows, and it kept fingerprints of the generator:

- income capped at 30,000 on about 9% of rows, commute distance floored at 5 km on about 22%
- income values repeat so often (13,214 distinct values in 668,665 rows) that they behave like IDs
- digit and rounding patterns of the numbers carry signal
- a hand-derived logistic formula scores 0.9377 AUC on its own

Out-of-fold AUC at each stage (from my experiment logs):

| Stage | OOF AUC |
|---|---|
| Basic features, LightGBM / XGBoost | 0.9421 |
| + generator-pattern and income-group features | 0.9456 |
| + windowed target encodings | 0.9459 |
| + finer windows, 10 folds, 4 models averaged | 0.9464 |
| + blend with public community predictions | 0.9466 |

Target encodings are computed out-of-fold inside the cross-validation folds so the validation score stays honest.

## Files

- `common.py` loads the data, adds light features and builds the 5-fold split (seed 42)
- `fe2.py` `add_art()`: features from the generator's digit, modulo, count and commute-decimal patterns
- `run_final6.py` the main model: LightGBM / XGBoost with out-of-fold target encodings (exact and binned), group aggregates and a logistic margin feature
- `stack_ext2.py` nested hill-climbing rank-average of my models with public out-of-fold predictions

## How to run

```
pip install -r requirements.txt
python run_final6.py lgb 0 10 10 1 255
python run_final6.py xgb 0 10 10 1 255
python run_final6.py lgb 1 10 10 1 255
python run_final6.py xgb 1 10 10 1 255
python stack_ext2.py
```

Arguments of `run_final6.py`: model (`lgb` or `xgb`), seed, folds to run, number of folds, extra-key encodings on/off, bin size. Predictions are written to `preds/`.

## Data (not included)

The competition data is not part of this repository. Download it with the Kaggle CLI:

```
kaggle competitions download -c playground-series-s6e9
```

Put `train.csv` and `test.csv` next to the scripts. `run_final6.py` also reads `original.csv`, the public dataset the competition data was generated from (see the competition's data page for the source).

## Credit

The final blend uses public out-of-fold predictions from other participants' Kaggle notebooks, placed under `ext/` when stacking:

- `kps6e09-generator-aware-ridge-logistic-regression` and `kps6e09-xgb-sample` (heuljax)
- `s6e9-xgboost-window-encodings-0-946-cv` (blamerx)
- `s6e9-six-feature-views-oof-library`
- `s6e9-lr-margin-gbdt-oof-stack-lb-0-94675`

They carried most of the blend weight. My own models reach about 0.9464 out-of-fold alone, and the blend reaches 0.9466. Thanks to all of them.
