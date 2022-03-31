# import joblib #to save the model (not like pickel)
#from ctypes.wintypes import BOOLEAN
import re
from xmlrpc.client import boolean 
from fastapi import FastAPI 
from pydantic import BaseModel
from typing import Optional
import csv
import pickle

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
    cardio : bool
    
app = FastAPI()

#filename_Model= "../model/"
#loaded_model = pickle.load(open(filename_Model,'rb'))

# def prediction_combi(combi):
#     label = loaded_model.predict([combi])[0]
#     combi_proba = loaded_model.predict_proba([combi])

#     return {'label': label, 'winning_proba':combi_proba[0][1]}


@app.get('/')

async def home():
    return {"message" : "CardioVascular Disease Detection API"}

# Improving the dataset by adding a new line to it 
@app.put('/api/dataset/improve')

async def addExamination(features: Features):
    result = {**features.dict()}
    # add informations into the dataset !
    return result

# Retraining the ML model using the new lines added to it 
@app.post('/api/model/retrain')

async def retrain():
    #retrain the model and pickel it (delete the old one)
    return "model retrained"


# Get prediction from the model 
@app.get('/api/model/predict')

async def getprediction(features: Features):
    return "prediction"

# Get metrics from the model 
@app.get('/api/model/metrics')

async def getMetrics(features: Features):
    return "metrics"

# Get the pickel of the model for other uses 
@app.get('/api/model')

async def getModel(features: Features):
    return "model"