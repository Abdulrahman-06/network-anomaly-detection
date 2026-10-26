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
| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Normal | 0.69      | 0.97   | 0.81     |
| Attack | 0.97      | 0.68   | 0.80     |

## Limitations
It was noticed that Attack traffic only catches 68% of malicious traffic and misses 32% of real attacks.

## How to Run
```bash
pip install -r requirements.txt
python explore_data.py
```
