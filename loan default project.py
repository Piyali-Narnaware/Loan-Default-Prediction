"""Leakage-aware loan-default model comparison.

Run from the project folder with: python "loan default project.py"
The test split is used only for final evaluation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "loan_data.xlsx"
OUTPUT_DIR = PROJECT_DIR / "outputs"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

raw = pd.read_excel(DATA_PATH)
if "loan_is_bad" not in raw.columns:
    raise ValueError("Expected target column 'loan_is_bad' was not found.")

target = raw["loan_is_bad"].astype(int)

# Only variables available at loan origination are predictors. Payment,
# recovery and collection fields are deliberately excluded to prevent leakage.
numeric_features = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "emp_length",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
]
categorical_features = [
    "term", "sub_grade", "home_ownership", "verification_status", "purpose",
]
feature_names = numeric_features + categorical_features
missing_columns = sorted(set(feature_names) - set(raw.columns))
if missing_columns:
    raise ValueError(f"Required predictor columns are missing: {missing_columns}")

X = raw[feature_names].copy()
for column in numeric_features:
    X[column] = pd.to_numeric(X[column], errors="coerce")
for column in categorical_features:
    X[column] = X[column].astype("string").fillna("Missing").astype(str)

# Fixed, stratified 70/10/10/10 split: model fitting, probability calibration,
# threshold selection, and final testing use separate rows.
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, target, test_size=0.30, random_state=SEED, stratify=target
)
X_cal, X_remaining, y_cal, y_remaining = train_test_split(
    X_remaining, y_remaining, test_size=2 / 3, random_state=SEED,
    stratify=y_remaining,
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_remaining, y_remaining, test_size=0.50, random_state=SEED,
    stratify=y_remaining,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)
X_train_ready = preprocessor.fit_transform(X_train).astype("float32")
X_cal_ready = preprocessor.transform(X_cal).astype("float32")
X_validation_ready = preprocessor.transform(X_validation).astype("float32")
X_test_ready = preprocessor.transform(X_test).astype("float32")

# Both models use the same original training rows and balanced class weights.
class_counts = np.bincount(y_train)
class_weights = {
    0: len(y_train) / (2 * class_counts[0]),
    1: len(y_train) / (2 * class_counts[1]),
}
logistic = LogisticRegression(
    max_iter=3000, class_weight="balanced", random_state=SEED
)
logistic.fit(X_train_ready, y_train)

neural_network = Sequential([
    Input(shape=(X_train_ready.shape[1],)),
    Dense(30, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(30, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(15, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(15, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(1, activation="sigmoid"),
])
neural_network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
)
neural_network.fit(
    X_train_ready,
    y_train.to_numpy(),
    validation_split=0.10,
    epochs=100,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
    verbose=0,
)


def calibrate_probabilities(raw_calibration, y_calibration, raw_values):
    """Fit a sigmoid calibration map on the separate calibration split."""
    epsilon = 1e-6
    calibration_scores = np.clip(raw_calibration, epsilon, 1 - epsilon)
    calibration_logits = np.log(calibration_scores / (1 - calibration_scores)).reshape(-1, 1)
    values = np.clip(raw_values, epsilon, 1 - epsilon)
    value_logits = np.log(values / (1 - values)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=SEED)
    calibrator.fit(calibration_logits, y_calibration)
    return calibrator.predict_proba(value_logits)[:, 1]


raw_probabilities = {
    "Logistic regression": {
        "calibration": logistic.predict_proba(X_cal_ready)[:, 1],
        "validation": logistic.predict_proba(X_validation_ready)[:, 1],
        "test": logistic.predict_proba(X_test_ready)[:, 1],
    },
    "Neural network": {
        "calibration": neural_network.predict(X_cal_ready, verbose=0).ravel(),
        "validation": neural_network.predict(X_validation_ready, verbose=0).ravel(),
        "test": neural_network.predict(X_test_ready, verbose=0).ravel(),
    },
}
probabilities = {}
for model_name, values in raw_probabilities.items():
    probabilities[model_name] = {
        split_name: calibrate_probabilities(
            values["calibration"], y_cal, values[split_name]
        )
        for split_name in ("validation", "test")
    }


def get_metrics(y_true, predicted_probabilities, threshold):
    predicted = (predicted_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "specificity": tn / (tn + fp),
        "roc_auc": roc_auc_score(y_true, predicted_probabilities),
        "brier_score": brier_score_loss(y_true, predicted_probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


comparison_rows = []
scenario_rows = []
threshold_grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
for model_name, splits in probabilities.items():
    test_probabilities = splits["test"]
    comparison_rows.append({
        "model": model_name,
        "evaluation": "held-out test at 0.50 threshold",
        **get_metrics(y_test.to_numpy(), test_probabilities, 0.50),
    })

    # Relative error-cost ratios are user-selected scenarios, not measured
    # financial losses or lending policy recommendations.
    for false_negative_cost in (5, 10, 20):
        validation_options = []
        for threshold in threshold_grid:
            validation_metrics = get_metrics(
                y_validation.to_numpy(), splits["validation"], threshold
            )
            cost_units = (
                false_negative_cost * validation_metrics["false_negatives"]
                + validation_metrics["false_positives"]
            )
            validation_options.append((cost_units, threshold))
        validation_cost, selected_threshold = min(
            validation_options, key=lambda option: (option[0], option[1])
        )
        test_metrics = get_metrics(
            y_test.to_numpy(), test_probabilities, selected_threshold
        )
        scenario_rows.append({
            "model": model_name,
            "illustrative_FN_to_FP_cost_ratio": f"{false_negative_cost}:1",
            "threshold_selected_on_validation": selected_threshold,
            "validation_cost_units": int(validation_cost),
            "test_cost_units": int(
                false_negative_cost * test_metrics["false_negatives"]
                + test_metrics["false_positives"]
            ),
            **test_metrics,
        })

OUTPUT_DIR.mkdir(exist_ok=True)
comparison = pd.DataFrame(comparison_rows)
scenarios = pd.DataFrame(scenario_rows)
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
scenarios.to_csv(OUTPUT_DIR / "threshold_scenarios.csv", index=False)

print(f"Rows: {len(raw):,}; default rate: {target.mean():.4%}")
print(
    f"Split: train {len(y_train):,}; calibration {len(y_cal):,}; "
    f"threshold validation {len(y_validation):,}; test {len(y_test):,}"
)
print(f"Encoded features: {X_train_ready.shape[1]}")
print("\nHeld-out test metrics at threshold 0.50")
print(comparison[[
    "model", "accuracy", "precision", "recall", "f1", "roc_auc", "brier_score",
    "true_negatives", "false_positives", "false_negatives", "true_positives",
]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("\nThreshold scenarios; cost ratios are illustrative assumptions")
print(scenarios[[
    "model", "illustrative_FN_to_FP_cost_ratio",
    "threshold_selected_on_validation", "precision", "recall",
    "false_negatives", "false_positives", "test_cost_units",
]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"\nSaved outputs to {OUTPUT_DIR}")
