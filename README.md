# Network Anomaly Detection

## Overview
This project trains a model to learn how to predict whether traffic is normal or not.

## Dataset
NSL_KDD is the dataset containing train and test files on network traffic. The dataset is converted into binary classification of either normal or any attack type.

## Approach
- Loaded and cleaned the NSL-KDD dataset
- One-hot encoded categorical features (protocol type, service, flag)
- Trained a Decision Tree classifier
- Evaluated performance on a separate test set

## Results
Decision Tree (`random_state=42`), evaluated on `KDDTest+` (22,544 records):

| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Normal | 0.69      | 0.96   | 0.80     |
| Attack | 0.96      | 0.67   | 0.79     |

Overall accuracy: **80%**.


## Limitations
The model only catches 67% of attacks and misses 33% of real attacks in the test set.

## How to Run
From the project root:

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m nad.train
```

On macOS/Linux, set the path with `export PYTHONPATH=src` instead.

This trains the model, prints the classification report on the test set, and saves the trained pipeline to `models/decision_tree.joblib`.

