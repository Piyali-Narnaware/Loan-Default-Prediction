# Loan Default Prediction using Deep Learning

A deep learning model that predicts loan default risk using borrower financial attributes. Built with TensorFlow/Keras, incorporating SMOTEENN for class imbalance handling and L2 regularization to prevent overfitting.

## Project Overview

This project develops a binary classification model to predict whether a borrower will default on a loan. The imbalanced dataset is addressed using SMOTEENN (combined over-sampling and under-sampling), and the neural network is trained with an exponential learning rate decay schedule.

## Model Architecture

- Input layer with 30 neurons (ReLU, L2 regularization)
- Dropout layer (50%)
- Hidden layer with 30 neurons (ReLU, L2 regularization)
- Dropout layer (50%)
- Hidden layer with 15 neurons (ReLU, L2 regularization)
- Hidden layer with 15 neurons (ReLU, L2 regularization)
- Output layer with 1 neuron (Sigmoid)

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 93.10% |
| Precision | 69.66% |
| Recall | 97.25% |
| False Negative Rate | 2.75% |

### Why Recall?

In loan default prediction, missing a borrower who will default (false negative) is far more costly than flagging a good borrower as risky (false positive). **Recall** measures the proportion of actual defaults correctly identified, making it the primary metric. A 97.25% recall means the model captures nearly all defaulters, minimizing financial loss from bad loans.

## Dependencies

- Python 3.11+
- TensorFlow
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- matplotlib
- openpyxl

## Usage

```bash
python "loan default project.py"
```

## Dataset

The dataset (`loan_data.xlsx`) contains borrower information and loan performance data. Features include credit history, loan characteristics, and demographic information.

## Training Curves

![Training and Validation Loss Curves](loss_curves.png)
