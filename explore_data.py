import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from nad.data import load_nsl_kdd


dataframe = load_nsl_kdd("data/raw/KDDTrain+.txt")
print(dataframe.shape)
print(dataframe["is_attack"].value_counts(normalize=True))
unique_counts = dataframe.nunique()
print(unique_counts[unique_counts == 1])
dataframe = pd.get_dummies(dataframe, columns=["protocol_type", "service", "flag"])

test_dataframe = load_nsl_kdd("data/raw/KDDTest+.txt")
print(test_dataframe.shape)
test_dataframe = pd.get_dummies(test_dataframe, columns=["protocol_type", "service", "flag"])

y = dataframe["is_attack"]
X = dataframe.drop(columns=["class", "is_attack"])

y_test = test_dataframe["is_attack"]
X_test = test_dataframe.drop(columns=["class", "is_attack"])
X_test = X_test.reindex(columns=X.columns, fill_value=0)

model = DecisionTreeClassifier()
model.fit(X, y)
print("Model trained")
predictions = model.predict(X_test)

print(dataframe.shape)
print(dataframe.head())
print(dataframe[["class", "is_attack"]].head(10))
print(X.shape)
print(y.shape)
print(classification_report(y_test, predictions))


