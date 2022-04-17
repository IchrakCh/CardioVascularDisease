import re
from xmlrpc.client import boolean 
from fastapi import FastAPI
from matplotlib.cbook import ls_mapper 
from pydantic import BaseModel
from typing import Optional
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
    height : int
    weight : float
    gender : int
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
    
# Fonctions utiles pour traitement des données 
def setId(filename,input):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        all_lines = list(reader)
        newId = int(all_lines[-1][0])+1
        csvfile.close()
    return newId

def addToCsv(filename, input):
    inputRow = input.insert(0,setId(filename, input))
    with open(filename, 'a', newline='') as f:
        reader = csv.reader(csvfile, delimiter=';')
        all_lines = list(reader)
        newId = int(all_lines[-1][0])+1
        writer_object = csv.writer(f)
        writer_object.writerow(inputRow)
        f.close()

app = FastAPI()

@app.get('/')

async def home():
    return {"message" : "CardioVascular Disease Detection API"}

# Improving the dataset by adding a new line to it 
@app.put('/api/dataset/improve')

async def addExamination(features: Features):
    
    result = {**features.dict()}
    print(type(result))
    input = list(result.values())
    print(type(input), len(input))
    # add informations into the dataset !
    #addToCsv("../data/cardio_train.csv",input)

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
    input = list(result.values())
    print(len(input))
    return input

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