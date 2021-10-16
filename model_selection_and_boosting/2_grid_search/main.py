import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=0))
# pipe = Pipeline([
#     ('sc', StandardScaler()),
#     ('svc', SVC(kernel='rbf', random_state=0))
# ])
parameters = [
    {'svc__C': [0.25, 0.5, 0.75, 1], 'svc__kernel': ['linear']},
    {
          'svc__C': [0.25, 0.5, 0.75, 1],
          'svc__kernel': ['rbf'],
          'svc__gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
]
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=parameters,
    scoring='accuracy',
    cv=10,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

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

# parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
#               {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
# grid_search = GridSearchCV(
#     estimator=classifier,
#     param_grid=parameters,
#     scoring='accuracy',
#     cv=10,
#     n_jobs=-1,
# )
# grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)
