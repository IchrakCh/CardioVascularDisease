import re
from xmlrpc.client import boolean 
from fastapi import FastAPI
from matplotlib.cbook import ls_mapper 
from pydantic import BaseModel
from typing import Optional
import numpy as np
import csv
import pickle
from sklearn import metrics

from starlette.responses import FileResponse
from os.path import exists
import sys
sys.path.append('../model')

from model import train, predict


class Features(BaseModel):
    # Objective Features 
    age : int 
    gender : int
    height : int
    weight : float
    
    # Examination Features 
    #   For Cholestrol & glucose levels :
    #       1: normal / 2:above normal / 3: well above normal
    ap_hi : int 
    ap_lo : int 
    cholestrol : int
    gluc : int
    # Sujbective Features 
    smoke : bool
    alco : bool
    active : bool
    # Target Variable
    #cardio : bool
    
# Fonctions utiles pour rajouter les données au csv 
def setId(filename):
    with open(filename, 'r', newline= '') as f:
        reader = csv.reader(f, delimiter=';')
        all_lines = list(reader)
        newId = int(all_lines[-1][0])+1
        f.close()
    return newId

def addToCsv(filename, input):
    with open(filename, 'a', newline='') as f:
        writer_object = csv.writer(f, delimiter=';')
        writer_object.writerow(input)
        f.close()
app = FastAPI()

@app.get('/')

async def home():
    return {"message" : "CardioVascular Disease Detection API"}

# Improving the dataset by adding a new line to it 
@app.put('/api/dataset/improve')

async def addExamination(features: Features):
    
    result = {**features.dict()}
    # add informations into the dataset !

    input = []
    input.append(setId("../data/cardio_train.csv"))
    
    listResult = list(result.values())

    for value in listResult :
        input.append(value)

    Xnew = []
    Xnew.append(input)
    XnewArray = np.array(Xnew)
    resultPrediction = predict(XnewArray)
    input.append(resultPrediction[0])
    print(input)
    addToCsv("../data/cardio_train.csv",input)

# Retraining the ML model using the new lines added to it 
@app.post('/api/model/retrain')

async def retrain():
    name, accuracy, metrics= train("../data/cardio_train.csv")
    print(name, accuracy)
    print(metrics["0"])
    return {"message" : "Model Trained succesfully"}


# Get prediction from the model 
@app.put('/api/model/predict')

async def getprediction(features: Features):
    result = {**features.dict()}
    input = []
    input.append(setId("../data/cardio_train.csv"))
    listResult = list(result.values())
    for value in listResult :
        input.append(value)
    Xnew = []
    Xnew.append(input)
    XnewArray = np.array(Xnew)
    resultPrediction = predict(XnewArray)
    input.append(resultPrediction[0])
    print(input)
    addToCsv("../data/cardio_train.csv",input)
    return resultPrediction[1]

# Get metrics from the model 
@app.get('/api/model/metrics')

async def getMetrics():
    nameModel, accuracy,  metrics = train("../data/cardio_train.csv")
    return nameModel, accuracy, metrics["0"]

# Get the pickel of the model for other uses 
@app.get('/api/model')

async def getModel():
    file_location = "../model/saved_model.pickle"
    if exists(file_location):
        return FileResponse(path = file_location,filename="saved_model.pickle")
    else: 
        name, metrics = train("../data/cardio_train.csv")
        return FileResponse(path = file_location,filename="saved_model.pickle")