# Network Anomaly Detection

A simple machine-learning project that learns the difference between normal and attack network traffic, then tests how well it spots attacks it has not seen before.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), a standard benchmark for network intrusion detection.

- `KDDTrain+.txt`: 125,973 records, used for training
- `KDDTest+.txt`: 22,544 records, used for testing. It contains attack types that do not appear in the training data.

## Approach

1. Load the data and apply the NSL-KDD column names.
2. Drop the `difficulty_level` column.
3. Turn the attack labels into a binary label: `0` = normal traffic, `1` = attack.
4. One-hot encode the categorical features (protocol type, service, flag).
5. Train a decision tree classifier (scikit-learn) on the training set.
6. Evaluate it on the separate test set.

## Results

Evaluated on `KDDTest+` (22,544 records):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal (0) | 0.68 | 0.96 | 0.80 |
| Attack (1) | 0.96 | 0.66 | 0.78 |

Overall accuracy: **79%**.

The model is very precise when it flags an attack (96%), but it only catches 66% of them. That is expected on this test set, because it contains attack types the model never saw in training.

## Limitations and next steps

- Recall on attacks is the main weakness. Next I want to try other models, such as a random forest, and tune the decision tree.
- Test on more datasets and, later, on live traffic.
- Break the results down by attack type instead of one attack class.

## Run it

```
pip install -r requirements.txt
python explore_data.py
```

The script prints the dataset shape, a preview of the processed data and the classification report.
