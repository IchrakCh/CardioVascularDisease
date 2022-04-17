import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
import pickle



names = [
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def train(filename):

    df = pd.read_csv(filename, sep = ";")

    #Train - Test split 
    X = df.drop('cardio', axis = 1)
    y = df['cardio']
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    #Choose the model with the best accuracy 

    models = []
    infoModel = {}    
    accuracies = []
    chosenModel = {}

    for name, classifier in zip(names, classifiers):
        classifier.fit(X_train, y_train)
        #score = classifier.score(X_test, y_test)
        #print(name, score)
        y_pred = classifier.predict(X_test)
        infoModel['classifier'] = classifier
        infoModel['Accuracy']= format(100*accuracy_score(y_test, y_pred))
        models.append(infoModel.copy())

    for dict in models:
        accuracies.append(dict['Accuracy'])
        if max(accuracies) == dict["Accuracy"]:
            chosenModel = dict["classifier"]
    
    #Retrain the chosen model 
    chosenModel.fit(X_train, y_train)
    y_pred = chosenModel.predict(X_test)

    #Save the model with pickle 
    filenameModel= "../model/saved_model.pickle"
    pickle.dump(chosenModel, open(filenameModel, 'wb'))

    #Return metrics of the chosen model 
    nameModel = "Name : " + str(chosenModel)
    accuracy = "Accuracy : " +format(100*accuracy_score(y_test, y_pred))
    metrics = classification_report(y_test,y_pred, output_dict = True)

    return nameModel, accuracy, metrics 

def predict(input):
    loaded_model = pickle.load(open("../model/saved_model.pickle", 'rb'))
    result = loaded_model.predict(input)
    if (result == 1):
        return 1,"You have a risk of a cardio vascular disease"
    else:
        return 0, "You're okay for now but stay safe"
