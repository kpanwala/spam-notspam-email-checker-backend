#email span or not

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('spam.csv',encoding='latin-1')
dataset.columns =['result', 'one','two','three','four']

n,m=dataset.shape
    
dataset['final'] = dataset['one'].fillna('') + dataset['two'].fillna('') + dataset['three'].fillna('') + dataset['four'].fillna('')
X=dataset['final']
# y = dataset['result']

# max_features = 1500

import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, n):
    review = re.sub('[^a-zA-Z]', ' ', dataset['final'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,0]

for i in range(0,len(y)):
    if y[i]=='ham':
        y[i]=1
    else:
        y[i]=0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train.astype(int), y_train.astype(int))

# Predicting the Test set results
y_pred = classifier.predict(X_test.astype(int))

# confusion matrix using naive bayes classifier with accuracy of 86.54 %
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred1 = regressor.predict(X_test)
for i in range(0,len(y_pred1)):
    if y_pred1[i]>0.5:
        y_pred1[i]=1
    else:
        y_pred1[i]=0

y_test = y_test.astype(int) 

# confusion matrix using random forest classifier with accuracy of 98.02 %
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train.astype(int), y_train.astype(int))

# Predicting the Test set results
y_pred2 = classifier1.predict(X_test.astype(int))

# Making the Confusion Matrix for logistic regression with accuracy of 97.84 %
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test.astype(int), y_pred2.astype(int))

