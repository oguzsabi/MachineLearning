import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
X = np.array(ct.fit_transform(X)) # independent values must be numpy array for later use
# print(X)

le = LabelEncoder()
y = le.fit_transform(y)
print(y)
