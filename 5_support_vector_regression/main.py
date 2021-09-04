import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)

y = y.reshape(len(y), 1)
# print(y)

# Feature Scaling

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

print(X)
print(y)

y = y.flatten()
print(y)

svr = SVR(kernel='rbf')
svr.fit(X, y)

y_pred = svr.predict(sc_X.transform([[6.5]]))  # We have to transform our 6.5 value as well
print(y_pred)

y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)

# Polynomial Linear Regression

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(svr.predict(X)), color='orange')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(svr.predict(X)), color='blue')
plt.title('Salary vs Position Level (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(svr.predict(sc_X.transform(X_grid))), color='blue')
plt.title('Salary vs Position Level (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
