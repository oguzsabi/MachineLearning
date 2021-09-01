import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)

# Simple Linear Regression

lr = LinearRegression()
lr.fit(X, y)

y_pred = lr.predict(X)

plt.scatter(X, y, color='red')
plt.scatter(X, y_pred, color='orange')
plt.plot(X, y_pred, color='blue')
plt.title('Salary vs Position Level (Linear)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# print(lr.intercept_)
# print(lr.coef_)
print(f'Linear model prediction: {lr.predict([[6.5]])}')

# Polynomial Linear Regression

pr = PolynomialFeatures(degree=4)
X_poly = pr.fit_transform(X)

lr2 = LinearRegression()
lr2.fit(X_poly, y)

y_pred2 = lr2.predict(X_poly)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
# X = X_grid
# y_pred2 = lr2.predict(pr.fit_transform(X))
plt.scatter(X, y_pred2, color='orange')
plt.plot(X, y_pred2, color='blue')
plt.title('Salary vs Position Level (Polynomial)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print(f'Polynomial model prediction: {lr2.predict(pr.fit_transform([[6.5]]))}')
