import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)

rfr = RandomForestRegressor(n_estimators=50, random_state=0)  # Each tree is an estimator

rfr.fit(X, y)
print(rfr.predict([[6.5]]))

plt.scatter(X, y, color='red')
plt.plot(X, rfr.predict(X), color='blue')
plt.title('Salary vs Position Level (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, rfr.predict(X_grid), color='blue')
plt.title('Salary vs Position Level (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
