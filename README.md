# Loan Default Model Comparison

A reproducible analysis of 50,000 historical loan records. It compares a class-weighted logistic regression with a small neural network, then examines how decision thresholds change false negatives and false positives.

## What the analysis uses

The target is `loan_is_bad`. The 18 predictors are application or origination fields: requested loan amount, interest rate, instalment, income, employment length, debt-to-income ratio, credit-history indicators, account counts, revolving balance and utilisation, term, sub-grade, home ownership, verification status and loan purpose.

Payment amounts, payment dates, recoveries, collections and other post-origination fields are excluded. Those fields can reveal how a loan performed and would leak outcome information into a model intended to assess risk at application time. The earlier 97.25% recall headline came from a pipeline that included post-origination payment fields; it should not be presented as a valid application-time result.

The records cover May 2012 to February 2013. The analysis uses a stratified random split, so the results are a model comparison on this dataset, not a time-forward validation or evidence of performance on today's lending population.

## Method

1. Split records into training, calibration, threshold-validation and test sets at 70%, 10%, 10% and 10%, stratified by outcome.
2. Impute missing numeric values with training medians, standardise them, impute missing categories and one-hot encode them. Fit preprocessing on the training set only.
3. Fit logistic regression and a four-hidden-layer neural network on the same training observations, using balanced class weights. The neural network reserves 10% of the model-development split for early stopping.
4. Calibrate each model's probabilities on a separate calibration split.
5. Select decision thresholds on the threshold-validation split under illustrative false-negative:false-positive count-cost ratios of 5:1, 10:1 and 20:1.
6. Report final metrics on the untouched test split.

The ratios are scenario assumptions, not observed financial losses, lender-approved costs or lending-policy recommendations. The project does not implement or evaluate a real lending strategy.

## Results from the checked run

The overall default rate was 15.63%. The test set contained 5,000 records, including 781 defaults. Both models had modest ranking performance and were close to each other.

| Model | Test ROC AUC | Test Brier score | Test result at 0.50 threshold |
|---|---:|---:|---|
| Logistic regression | 0.6623 | 0.1259 | Predicted no defaults; 0% recall |
| Neural network | 0.6564 | 0.1265 | Predicted no defaults; 0% recall |

A constant prediction equal to the test default rate has a Brier score of about 0.1318. Both calibrated models improve on that benchmark slightly, but their ROC AUC shows limited ability to rank loans by default risk. A 0.50 threshold is therefore not useful here; changing the threshold trades false negatives against false positives.

Thresholds were chosen on validation data and then evaluated once on the test data. The cost values below are weighted error counts, not money.

| Model | Assumed FN:FP ratio | Validation-selected threshold | Test recall | Test precision | Test false negatives | Test false positives | Test weighted error count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 5:1 | 0.17 | 57.62% | 23.96% | 331 | 1,428 | 3,083 |
| Neural network | 5:1 | 0.17 | 59.54% | 23.66% | 316 | 1,500 | 3,080 |
| Logistic regression | 10:1 | 0.11 | 85.53% | 18.81% | 113 | 2,883 | 4,013 |
| Neural network | 10:1 | 0.09 | 90.91% | 17.86% | 71 | 3,265 | 3,975 |
| Logistic regression | 20:1 | 0.05 | 98.72% | 16.10% | 10 | 4,017 | 4,217 |
| Neural network | 20:1 | 0.07 | 98.21% | 16.47% | 14 | 3,890 | 4,170 |

The lower thresholds catch more defaults but flag many non-default loans. The neural network does not consistently outperform logistic regression. No threshold in this exercise should be treated as an actual lending rule.

## Run the project

Place `loan_data.xlsx` beside the script, then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "numpy==1.26.4" "pandas==2.2.3" "scikit-learn==1.6.1" "tensorflow-cpu==2.16.2" openpyxl
python "loan default project.py"
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

The script writes `outputs/model_comparison.csv` and `outputs/threshold_scenarios.csv`.

## Limits

This is a historical portfolio-data exercise with a random holdout from the same short period. It is not a production credit model, does not establish fairness or regulatory compliance, and does not demonstrate professional lending-strategy ownership. The results should be explained with these limits in view.
