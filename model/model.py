import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
import random
import csv


filename = "../data/cardio_train.csv"
df = pd.read_csv(filename, sep = ";")

names = [
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
]
classifiers = [
    GaussianProcessClassifier(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
]


#Train-val-test split 
X = df.drop('cardio', axis = 1)
y = df['cardio']
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.metrics import classification_report, accuracy_score
for name, classifier in zip(names, classifiers):
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(name, score)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test,y_pred))
    print('Accuracy: {} %'.format(100*accuracy_score(y_test, y_pred)))
