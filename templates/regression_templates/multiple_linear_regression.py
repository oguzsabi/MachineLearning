import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)
print(
    f'Differences:\n',
    np.concatenate(
        (
            y_pred.reshape(len(y_pred), 1),
            y_test.reshape(len(y_test), 1),
            abs((y_pred - y_test)).reshape(len(y_pred), 1),
        ),
        1
    )
)

r2 = r2_score(y_test, y_pred)
print(f'R2 Score: {r2}')

adj_r2 = 1 - (1 - r2) * (len(X_train) - 1) / (len(X_train) - len(X_train[0]) - 1)
