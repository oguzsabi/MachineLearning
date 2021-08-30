import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(y)
# print(X)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit(X[:, 1:3]).transform(X[:, 1:3])
# print(X[:, 1:3])
# imputer.fit(X[:, 1:3])
# print(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X[:, 1:3])

# print(X)

# Encoding categorical data

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # independent values must be numpy array for later use
# print(X)

le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# Splitting the dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Feature scaling

sc = StandardScaler()
# fit() only calculates the mean and the std of the given dataset, transform() applies the standardisation formula.
X_train[:, -2:] = sc.fit_transform(X_train[:, 3:])
X_test[:, -2:] = sc.transform(X_test[:, 3:])  # test data also needs to be scaled
print(X_train)
print(X_test)
