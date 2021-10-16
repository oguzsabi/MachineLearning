import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel='rbf', random_state=0)  # rbf kernel performs better
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(
    np.concatenate(
        (
            y_pred.reshape(len(y_pred), 1),
            y_test.reshape(len(y_test), 1),
        ),
        1
    )
)

cm = confusion_matrix(y_test, y_pred)
print(cm)
# print(f'0 correct: {cm[0, 0]}   1 correct: {cm[1, 1]}\n0 wrong: {cm[1, 0]}   1 wrong: {cm[0, 1]}')

a_score = accuracy_score(y_test, y_pred)
print(a_score)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print('Accuracy: {:.2f}%'.format(accuracies.mean() * 100))
print('Standard Deviation: {:.2f}%'.format(accuracies.std() * 100))
