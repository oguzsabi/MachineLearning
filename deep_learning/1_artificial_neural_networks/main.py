import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = Sequential()
ann.add(Dense(units=6, activation='relu'))
ann.add(Dense(units=6, activation='relu'))
ann.add(Dense(units=1, activation='sigmoid'))
# ann.add(Dense(units=1, activation='softmax'))  # for non-binary categorical data

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # for nonbinary categorical data

ann.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)
y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5

# Single customer prediction
single_customer = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
single_customer[:, 2] = le.fit_transform(single_customer[:, 2])
single_customer = np.array(ct.transform(single_customer))
single_customer = sc.transform(single_customer)

single_pred = ann.predict(single_customer)
print(single_pred)
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

a_score = accuracy_score(y_test, y_pred)
print(a_score)
