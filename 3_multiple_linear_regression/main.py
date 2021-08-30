import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# Splitting the dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
# print(y_pred - y_test)

np.set_printoptions(precision=2)
print(
    np.concatenate(
        (
            y_pred.reshape(len(y_pred), 1),
            y_test.reshape(len(y_test), 1),
            abs((y_pred - y_test)).reshape(len(y_pred), 1)
        ),
        1
    )
)

print(lr.predict([[1, 0, 0, 160000, 130000, 300000]]))  # single prediction
print(lr.intercept_)  # b0
print(lr.coef_)  # coefficients
