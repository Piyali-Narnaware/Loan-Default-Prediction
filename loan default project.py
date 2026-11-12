# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:47:52 2026

@author: Piyali
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from tensorflow.keras.layers import Dense, Dropout, Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from collections import Counter
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import random as python_random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Set seed for reproducibility
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

# Load data
data = pd.read_excel('loan_data.xlsx')

# Columns to drop are id, date (it does not help us understand anything)
# Invoice column is being dropped too because it repeated. 
# Some of the columns like total_credit_rv has more than 10k NAs, which will cause loss of data if it is kept and NA is dropped later on.
columns_to_drop = [
    'id', 'member_id', 'last_pymnt_d', 'last_credit_pull_d',
    'mths_since_last_major_derog', 'mths_since_last_delinq', 'emp_title',
    'tot_coll_amt', 'tot_cur_bal', 'mths_since_last_record', 'funded_amnt_inv',
    'grade', 'pymnt_plan', 'desc', 'purpose', 'title', 'addr_state',
    'total_pymnt_inv', 'collection_recovery_fee', 'next_pymnt_d',
    'collections_12_mths_ex_med', 'policy_code', 'issue_d', 'earliest_cr_line',
    'emp_length', 'recoveries', 'total_rec_late_fee', 'total_credit_rv',
    'loan_status', 'verification_status'
]

# Drop the columns not useful for deep learning
data_cleaned = data.drop(columns=columns_to_drop)

# loan_is_bad, transform into numeric
data_cleaned['loan_is_bad'] = data_cleaned['loan_is_bad'].apply(lambda x: 0 if x == 0 else 1)

# Drop the xx behind the firdt three number so zip code can be use to train the model
data_cleaned['zip_code'] = data_cleaned['zip_code'].str.extract(r'(\d+)').astype(int)

# Transform different home_ownership into numeric using one_hot encoding, it is 0 if FALSE , 1 if TRUE
data_cleaned = pd.get_dummies(data_cleaned, columns=['home_ownership'], prefix='home')

# Nominal ordering for sub_grade. 
sub_grade_mapping = {
    "A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5,
    "B1": 6, "B2": 7, "B3": 8, "B4": 9, "B5": 10,
    "C1": 11, "C2": 12, "C3": 13, "C4": 14, "C5": 15,
    "D1": 16, "D2": 17, "D3": 18, "D4": 19, "D5": 20,
    "E1": 21, "E2": 22, "E3": 23, "E4": 24, "E5": 25,
    "F1": 26, "F2": 27, "F3": 28, "F4": 29, "F5": 30,
    "G1": 31, "G2": 32, "G3": 33, "G4": 34, "G5": 35
}

# Mapped the sub_grade numeric values into the column sub_grade
data_cleaned['sub_grade'] = data_cleaned['sub_grade'].map(sub_grade_mapping)

# Remove NAs in revol_util
data_cleaned = data_cleaned.dropna(subset=['revol_util'])

# Dropped the unwanted extra home_columns
data_cleaned = data_cleaned.drop(['home_OTHER', 'home_NONE'], axis=1)

# Converting TRUE to 1 and FALSE to 0. 
data_cleaned['home_OWN'] = data_cleaned['home_OWN'].apply(lambda x: 0 if x == 0 else 1)

data_cleaned['home_RENT'] = data_cleaned['home_RENT'].apply(lambda x: 0 if x == 0 else 1)

data_cleaned['home_MORTGAGE'] = data_cleaned['home_MORTGAGE'].apply(lambda x: 0 if x == 0 else 1)

# Prepare features and labels, features are the variables to train the model, labels are the outcome of loan_is_bad
features = data_cleaned.drop(columns=['loan_is_bad']).values
labels = data_cleaned['loan_is_bad'].values
#print(features[0:20])
#print(labels[0:20])

# Since we need to use SMOTEENN, let's split using sklearn (could use TensorFlow but keeping as is for SMOTEENN compatibility)
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)

# From the temporary dataset split half to test and another half to validation
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize features - TensorFlow's Normalization layer expects batches, so we normalize using sklearn here for simplicity
# Normalize all features in X_train for SMOTEENN application
normalizer = Normalization(axis=-1)  # Normalizes along the features axis
normalizer.adapt(X_train)  # Adapt to the training data

# Apply normalization to X_train and X_test, both training and testing variables (excluding outcome variable) needs to be normalised to reduce the range
X_train_normalized = normalizer(X_train)
X_test_normalized = normalizer(X_test)
X_val_normalized = normalizer(X_val)

# Apply SMOTEENN for both over-sampling and under-sampling on the scaled training data
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train_normalized, y_train) #bothsampling on the training set for both the features and the label

# Convert the resampled data back into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled)).batch(64) 
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_normalized, y_test)).batch(64) # test set only normalised for the features and label remain unchanged
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_normalized, y_val)).batch(64)

# Initial learning rate
initial_learning_rate = 0.001

# Define the learning rate schedule. Adjust the parameters as needed.
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)

# Use the learning rate schedule in the optimizer.
optimizer = Adam(learning_rate=lr_schedule) 

# Model definition
model = Sequential([
    Dense(30, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_resampled.shape[1],)),
    Dropout(0.5),
    Dense(30, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(15, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(15, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalseNegatives()])

# Train the model
history = model.fit(train_dataset, epochs=100, validation_data = val_dataset) # the train_dataset contain x_resampled, and y_resampled namely the variables and outcome

# Model evaluation
evaluation = model.evaluate(test_dataset, verbose=2) 

# Extract and print metrics
accuracy, precision, recall, fn = evaluation[1], evaluation[2], evaluation[3], evaluation[4]
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, False Negatives: {fn}')

# Plotting the training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')

# Lower epoch higher false negative cases. 

# Generate predictions for the confusion matrix and FNR calculation
predictions = model.predict(X_test_normalized) # This code fundamentally use the trained model to make predictions on the normalised test dataset)
predicted_classes = (predictions > 0.5).astype(int) # Converting the probabilities into class labels of 0 or 1. If probability > 0.5 then it is 1, 0 otherwise

test_features, test_labels = next(iter(test_dataset))

# Print out the predicted and the real values for comparison
for pred, real in zip(predicted_classes, test_labels):
    print(f"Predicted: {pred[0]};    Real: {real}")

# Generate confusion matrix
cm = confusion_matrix(y_test, predicted_classes) #Computes the confusion matrix using the actual result (y_test) and the predicted result from predicted_classes

# Calculate True Positives (TP) and False Negatives (FN)
TP = cm[1, 1] # true positive is at the 1st row and 1st column of cm 
FN = cm[1, 0]

# Calculate and print False Negative Rate (FNR)
FNR = FN / (FN + TP) 
print(f'False Negative Rate (Miss Rate): {FNR:.4f}')


# DECISION-FOCUSED EVALUATION
# Thresholds are selected on validation data. The test set is used only for final
# evaluation. Relative false-negative/false-positive costs below are illustrative
# scenarios, not observed lender losses or financial estimates.

def classification_metrics(y_true, probabilities, threshold):
    probabilities = np.asarray(probabilities).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else np.nan,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


# Train an interpretable baseline on the original training rows.
# The neural network above uses SMOTEENN; the logistic baseline uses class weights.
X_train_array = np.asarray(X_train_normalized, dtype=np.float32)
X_val_array = np.asarray(X_val_normalized, dtype=np.float32)
X_test_array = np.asarray(X_test_normalized, dtype=np.float32)
y_train_array = np.asarray(y_train).astype(int)
y_val_array = np.asarray(y_val).astype(int)
y_test_array = np.asarray(y_test).astype(int)

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
logistic_model.fit(X_train_array, y_train_array)

model_probabilities = {
    "Neural network": {
        "validation": model.predict(X_val_array, verbose=0).reshape(-1),
        "test": np.asarray(predictions).reshape(-1),
    },
    "Logistic regression": {
        "validation": logistic_model.predict_proba(X_val_array)[:, 1],
        "test": logistic_model.predict_proba(X_test_array)[:, 1],
    },
}

# Compare both models at the default threshold on the untouched test set.
comparison_rows = []
for model_name, probability_sets in model_probabilities.items():
    row = {"model": model_name, "evaluation": "test at 0.50 threshold"}
    row.update(classification_metrics(y_test_array, probability_sets["test"], 0.50))
    comparison_rows.append(row)

# Select each threshold using validation data only. Relative costs are scenario
# inputs and must not be presented as actual portfolio loss estimates.
threshold_rows = []
candidate_thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
for model_name, probability_sets in model_probabilities.items():
    validation_probabilities = probability_sets["validation"]
    test_probabilities = probability_sets["test"]
    for false_negative_cost in (5, 10, 20):
        false_positive_cost = 1
        validation_candidates = []
        for threshold in candidate_thresholds:
            val_metrics = classification_metrics(
                y_val_array, validation_probabilities, threshold
            )
            scenario_cost = (
                false_negative_cost * val_metrics["false_negatives"]
                + false_positive_cost * val_metrics["false_positives"]
            )
            validation_candidates.append((scenario_cost, threshold, val_metrics))
        best_validation_cost, selected_threshold, best_validation_metrics = min(
            validation_candidates, key=lambda item: (item[0], item[1])
        )

        test_metrics = classification_metrics(
            y_test_array, test_probabilities, selected_threshold
        )
        test_scenario_cost = (
            false_negative_cost * test_metrics["false_negatives"]
            + false_positive_cost * test_metrics["false_positives"]
        )
        threshold_rows.append({
            "model": model_name,
            "false_negative_to_false_positive_cost_ratio": (
                f"{false_negative_cost}:1 illustrative"
            ),
            "validation_selected_threshold": selected_threshold,
            "validation_scenario_cost_units": int(best_validation_cost),
            "validation_recall": best_validation_metrics["recall"],
            "validation_precision": best_validation_metrics["precision"],
            "test_scenario_cost_units": int(test_scenario_cost),
            **test_metrics,
        })

# Export logistic-regression coefficients as directional, model-based explanations.
# Coefficients are not causal effects or a substitute for a lending policy review.
feature_names = data_cleaned.drop(columns=["loan_is_bad"]).columns
coefficient_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": logistic_model.coef_[0],
})
coefficient_table["absolute_coefficient"] = coefficient_table["coefficient"].abs()
coefficient_table = coefficient_table.sort_values(
    "absolute_coefficient", ascending=False
)

# Produce quantile-bin calibration points for both models on the held-out test set.
calibration_rows = []
for model_name, probability_sets in model_probabilities.items():
    observed_rate, mean_probability = calibration_curve(
        y_test_array,
        probability_sets["test"],
        n_bins=10,
        strategy="quantile",
    )
    for bin_number, (mean_predicted, observed) in enumerate(
        zip(mean_probability, observed_rate), start=1
    ):
        calibration_rows.append({
            "model": model_name,
            "quantile_bin": bin_number,
            "mean_predicted_probability": float(mean_predicted),
            "observed_default_rate": float(observed),
        })

output_directory = Path("outputs")
output_directory.mkdir(parents=True, exist_ok=True)
pd.DataFrame(comparison_rows).to_csv(
    output_directory / "model_comparison.csv", index=False
)
pd.DataFrame(threshold_rows).to_csv(
    output_directory / "threshold_cost_scenarios.csv", index=False
)
coefficient_table.to_csv(
    output_directory / "logistic_regression_coefficients.csv", index=False
)
pd.DataFrame(calibration_rows).to_csv(
    output_directory / "calibration_bins.csv", index=False
)

print("\nDecision-focused evaluation")
print("Models: neural network and class-weighted logistic regression")
print("Thresholds selected on validation data; final metrics use the held-out test set")
print("False-negative cost ratios are illustrative scenario assumptions only")
print(pd.DataFrame(comparison_rows)[
    ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]
].to_string(index=False))
print(f"Analysis outputs saved in: {output_directory.resolve()}")

