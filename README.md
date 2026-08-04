# Survival Analysis — All-Cause Mortality Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.28.0-lightgrey.svg)](https://shap.readthedocs.io/)

Mid-term report for the **GHPIM** course.

This project predicts **all-cause mortality** from a NHANES metabolic-syndrome cohort using nine classical machine-learning classifiers, compares them on the same train/test split, and uses **SHAP** to explain which clinical measurements drive the predictions.

All of the work lives in a single notebook: [GHPIM.ipynb](GHPIM.ipynb).

---

## Dataset

`nhanes_metSyn_20220316.csv` — 2,796 participants × 45 columns (an identical `.xlsx` copy is included).

| | |
| --- | --- |
| **Target** | `allcausedeath` — 0 = alive, 1 = died during follow-up |
| **Class balance** | 2,228 alive (79.7 %) vs 565 deceased (20.2 %) |
| **Feature groups** | Demographics (`AGE`, `sex`, `race`), anthropometrics (`BMI`, `WAIST`, `waist_hip_ratio`, skinfolds `BMPTRI`/`BMPSUB`/`BMPSUP`/`BMPTHI`), vitals (`HR`, `SBP`, `DBP`), blood counts (`HB`, `HCT`, `PLT`), lipids (`TCHO`, `TG`, `LDL`, `HDL`), metabolic and inflammatory markers (`GLU`, `CRP`, `uric_acid`, `fibriongen`), liver and renal function (`GOT`, `GPT`, `T_bilirubin`, `Creatinine`, `BUN`, `MDRD_eGFR`, `ckd`), medication and history flags (`diabetes_told`, `anti_hypertensive`, `Smoking`), and the metabolic-syndrome labels `ATP_MS` / `IDF_MS` |

### Preprocessing

1. Every column's missing values are imputed with that column's **mean**; `ckd` is filled with `0` instead.
2. Columns dropped before training: `SEQN` (identifier), `ApoA`, `ApoB`, `insulin`, `oral_hypoglycemic_agent`, and `followupmonth` (leaks the outcome), leaving **38 features**.
3. `train_test_split(test_size=0.1, random_state=42)` → **2,516 training** / **280 test** samples.

---

## Models

Nine classifiers are trained on identical data:

| Category | Models |
| --- | --- |
| Simple | Logistic Regression (`lbfgs`, `max_iter=4000`), Decision Tree, Gaussian Naive Bayes, SVM (RBF, `probability=True`) |
| Ensemble | Random Forest (100 trees), Bagging (100 estimators), AdaBoost (100 estimators), XGBoost (`n_estimators=100`, `learning_rate=0.3`), LightGBM (`is_unbalance=True`) |

Each is evaluated with train accuracy, test accuracy, a hand-rolled **10-fold cross-validation** (`KFold(shuffle=False)`, reporting mean/max/min), **sensitivity** and **specificity** from the confusion matrix, and **AUC** from the ROC curve.

An `XGBRegressor` is also fitted at the top of the notebook as a quick regression sanity check (R² 0.99 train vs 0.26 test — clearly overfitted, and not used further).

---

## Results

### 1. Accuracy and discrimination

| Model | Train acc. | Test acc. | 10-fold CV (mean) | CV min–max | Sensitivity | Specificity | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.86 | 0.83 | 0.85 | 0.81 – 0.89 | 0.46 | 0.93 | **0.81** |
| Decision Tree | 0.98 | 0.78 | 0.78 | 0.73 – 0.83 | 0.54 | 0.84 | 0.66 |
| Naive Bayes | 0.82 | 0.78 | 0.82 | 0.77 – 0.88 | **0.58** | 0.84 | 0.77 |
| SVM | 0.80 | 0.79 | 0.80 | 0.75 – 0.84 | 0.00 | 1.00 | 0.77 |
| Random Forest | 0.98 | 0.83 | **0.86** | 0.82 – 0.88 | 0.39 | 0.94 | **0.83** |
| Bagging | 0.98 | **0.85** | 0.85 | 0.80 – 0.90 | 0.47 * | 0.92 * | 0.82 |
| AdaBoost | 0.88 | 0.82 | 0.84 | 0.80 – 0.90 | 0.47 * | 0.92 * | 0.77 |
| XGBoost | 0.98 | **0.85** | 0.85 | 0.81 – 0.89 | 0.47 | 0.92 | 0.80 |
| LightGBM | 0.98 | 0.82 | 0.85 | 0.81 – 0.89 | 0.56 | 0.90 | 0.79 |

\* See *Known issues* — the Bagging and AdaBoost cells reuse the previous model's predictions, so those two rows are not trustworthy.

<p align="center">
  <img src="result/output2.png" alt="Fig. 1 — Accuracy comparison across models">
</p>

### 2. ROC curves

Random Forest (0.83), Bagging (0.82) and Logistic Regression (0.81) give the best ranking performance; the single Decision Tree is far behind at 0.66.

<p align="center">
  <img src="result/output.png" alt="Fig. 2 — ROC curves for all models">
</p>

### 3. SHAP explanations (XGBoost)

**Waterfall** — how each feature pushes a single participant's prediction away from the baseline.

<p align="center">
  <img src="result/output3.png" width="800" alt="Fig. 3 — SHAP waterfall plot for one participant">
</p>

**Mean |SHAP|** — global feature importance. `AGE` dominates by a wide margin, followed by `GPT`, `HCT`, `fibriongen` and `HB`.

<p align="center">
  <img src="result/output4.png" width="800" alt="Fig. 4 — Mean absolute SHAP value per feature">
</p>

**Beeswarm** — direction of each effect. High `AGE` (red) pushes the prediction strongly toward death, while low `HB` and low `MDRD_eGFR` also increase risk.

<p align="center">
  <img src="result/output5.png" width="800" alt="Fig. 5 — SHAP beeswarm summary plot">
</p>

---

## Requirements

```bash
pip install pandas numpy scikit-learn==1.4.1 xgboost lightgbm shap==0.28.0 matplotlib jupyter
```

`openpyxl` is only needed if you want to read the `.xlsx` copy of the dataset instead of the CSV.

## How to run

```bash
git clone https://github.com/rainday1029/Suvive-Analyze.git
cd Suvive-Analyze
jupyter notebook GHPIM.ipynb
```

Run the cells top to bottom. The notebook reads `./nhanes_metSyn_20220316.csv` relative to the working directory, so start Jupyter from the repository root. Figures are displayed inline; the versions in [result/](result/) were exported manually.

---

## Interpretation

- **Accuracy is misleading here.** Predicting "alive" for everyone already scores ~0.80 because only 20 % of the cohort died. The SVM does exactly that — sensitivity 0.00, specificity 1.00 — yet still reports 0.79 test accuracy. AUC, sensitivity and specificity are the numbers to read.
- **All tree ensembles overfit**, reaching 0.98 training accuracy against ~0.83 test accuracy. Depth limits, pruning or stronger regularisation would help.
- **Sensitivity is low across the board** (0.39 – 0.58). For a mortality-screening use case, missing more than half of the deaths matters more than the headline accuracy; class weighting, resampling, or lowering the decision threshold would trade specificity for recall.
- **Age dominates the model.** SHAP puts `AGE` far ahead of every other feature, with liver enzyme `GPT`, haematocrit `HCT`, `fibriongen`, `HB` and renal function `MDRD_eGFR` making up the rest of the signal — consistent with the clinical literature on all-cause mortality.

---

## Known issues

- **Bagging and AdaBoost metrics are wrong.** Both cells compute `predicted = model.predict(X_test)` but then pass `y_preds=pred` — the variable still holding the *XGBoost* predictions — into `Calculate_Sensitivity_Specificity()`. That is why both report an identical 0.47 / 0.92. Change the argument to `predicted` to fix it.
- **No feature scaling.** Logistic Regression and the RBF SVM are fitted on raw clinical units, which is why Logistic Regression needs `max_iter=4000` to converge and the SVM collapses to the majority class.
- **Cross-validation uses `shuffle=False`**, so the folds follow the file order rather than being randomised, and it is run on the training split only.
- **Three rows have a missing `allcausedeath`.** They are mean-imputed to ≈0.2 and then cast with `.astype('int')`, which silently turns them into label 0.
- **Mean imputation ignores the outcome split**, so test-set statistics leak into the training features. Imputing inside a `Pipeline` after the split would be cleaner.
- The dropped-column check `if i not in 'ckd'` is a substring test, not an equality test. It happens to behave correctly for this dataset, but any future column named `c`, `k`, `d`, `ck` or `kd` would be skipped.
- Cell 11 (`PipelineProfiler`) raises `AttributeError` — it expects an auto-sklearn model and does not work with `XGBClassifier`. It can be deleted.

---

## Tools

| Package | Version |
| --- | --- |
| scikit-learn | 1.4.1 |
| SHAP | 0.28.0 |
| xgboost / lightgbm | latest |
| pandas / numpy / matplotlib | latest |

---

## License

MIT License

Copyright (c) 2023 rainday1029 and RexJian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
