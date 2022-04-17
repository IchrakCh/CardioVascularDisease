import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
from sklearn import metrics
import pickle
import random
import csv

filename = "../data/cardio_train.csv"

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

trainingResult = train("../data/cardio_train.csv")

def predict(input):
    loaded_model = pickle.load(open("../model/saved_model.pickle", 'rb'))
    result = loaded_model.predict(input)
    if (result == 1):
        return "You have a risk of a cardio vascular disease"
    else:
        return "You're okay for now but stay safe"


result = {
  "age": 33,
  "height": 167,
  "weight": 67,
  "gender": 2,
  "ap_hi": 44,
  "ap_lo": 55,
  "cholestrol": 56,
  "gluc": 45,
  "smoke": True,
  "alco": True,
  "active": True
}



def setId(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        all_lines = list(reader)
        newId = int(all_lines[-1][0])+1
        csvfile.close()
    return newId

input = []
input.insert(0,setId("../data/cardio_train.csv"))
listResult = list(result.values())
for value in listResult :
  input.append(value)
print(input)

Xnew = []
Xnew.append(input)
XnewArray = np.array(Xnew)
print(predict(XnewArray))
