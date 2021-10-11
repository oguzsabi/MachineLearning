import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download('stopwords')

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = []
all_stopwords_but_not = stopwords.words('english')
all_stopwords_but_not.remove('not')
all_stopwords_but_not.remove('isn\'t')

for i in range(0, len(dataset['Review'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords_but_not)]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Naive Bayes
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# Random Forest
# classifier = RandomForestClassifier(n_estimators=38, criterion='entropy', random_state=0)
# classifier.fit(X_train, y_train)

# Decision Tree
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier.fit(X_train, y_train)

# Kernel SVM
# classifier = SVC(kernel='rbf', random_state=0)
# classifier.fit(X_train, y_train)

# Linear SVM
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# K Nearest Neighbors
# classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# classifier.fit(X_train, y_train)

# Logistic Regression
# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)


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

a_score = accuracy_score(y_test, y_pred)
print(a_score)

precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
f1_score = 2 * precision * recall / (precision + recall)
print(f1_score)

# Single prediction
new_review = 'Not so great'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)
